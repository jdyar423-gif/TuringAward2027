"""The chat entry point (bes.decode.respond).

A character-level tiny tokenizer, an explicit table model and a tiny chat format written here
drive the whole pipeline; prompts are captured by a spy model and compared with renderings
the test builds itself. The harness factory is exercised on a smoke-sized checkpoint written
with ``bes.train.checkpoint``.
"""
from __future__ import annotations

import dataclasses
import string
from pathlib import Path

import numpy as np
import pytest

from bes.config import DecodeCfg
from bes.decode import respond as R
from bes.decode.lm import TableLM, TinyTokenizer
from bes.exact import constraints as C

REPO = Path(__file__).resolve().parents[2]
PIECES = list(string.ascii_letters + string.digits + " .,!?'\n-:;\"()")


class TinyChat:
    """``BOS (ROLE content END)* ASSISTANT`` over a tiny tokenizer; display up to ``END``."""

    supports_library = True

    def __init__(self, tok):
        self.tok = tok
        self.sp = tok.special_tokens
        self.end_id = self.sp["<|end|>"]

    def render(self, messages):
        ids = [self.sp["<|bos|>"]]
        for m in messages:
            ids += [self.sp[f"<|{m['role']}|>"], *self.tok.encode(m["content"]), self.end_id]
        return ids + [self.sp["<|assistant|>"]]

    def display(self, tokens):
        toks = list(tokens)
        if self.end_id in toks:
            toks = toks[:toks.index(self.end_id)]
        return self.tok.decode_bytes([t for t in toks if t < self.tok.n_ordinary]).decode()


class Spy:
    """Wraps a language model and records every prefilled prompt."""

    def __init__(self, lm):
        self.lm, self.prompts = lm, []
        self.vocab_size = lm.vocab_size

    def prefill(self, prompt, k):
        self.prompts.append(list(prompt))
        return self.lm.prefill(prompt, k)

    def step(self, tokens, active=None):
        return self.lm.step(tokens, active)

    def reorder(self, anc):
        self.lm.reorder(anc)

    def close(self):
        self.lm.close()


@dataclasses.dataclass
class FakePassage:
    title: str
    text: str


class FakeLibrary:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def retrieve(self, query):
        self.calls.append(query)
        hit = self.answers.get(query)
        return None if hit is None else FakePassage(hit[0], hit[1])


def setup(seed=0, library=None, context=4096, cfg=None):
    tok = TinyTokenizer(PIECES)
    sp = tok.special_tokens
    boost = {sp["<|end|>"]: 1.5, PIECES.index("."): 1.0, PIECES.index(" "): 1.5}
    lm = Spy(TableLM.random(tok.vocab_size, order=2, seed=seed, boost=boost))
    cfg = cfg or DecodeCfg(particles=3, max_new_tokens=40, temperature=1.0, top_p=1.0)
    resp = R.Responder(lambda n: lm, tok, cfg, library=library, chat=TinyChat(tok), context=context)
    return tok, lm, resp


USER = [{"role": "user", "content": "Answer in exactly 2 sentences: why is it dark at night?"}]


# ---------------------------------------------------------------------------- settings
def test_settings_resolution():
    cfg = DecodeCfg(max_new_tokens=384, particles=6, temperature=0.6)
    r = R.resolve(cfg, {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05,
                        "repeat_penalty": 1.0, "max_tokens": 1024, "system_prompt": None})
    d = r.decode
    assert (d.temperature, d.top_p, d.top_k, d.min_p) == (0.8, 0.95, 40, 0.05)
    assert d.max_new_tokens == 384 and d.particles == 6 and r.potentials and not r.ignored
    assert R.resolve(cfg, {"max_tokens": 100}).decode.max_new_tokens == 100
    assert R.resolve(cfg, {"max_tokens": 100, "max_new_tokens": 200}).decode.max_new_tokens == 200
    r = R.resolve(cfg, {"particles": "4", "use_library": "false", "potentials": "0",
                        "resample_at": "token", "band": False, "utility": "lcs"})
    assert r.decode.particles == 4 and r.decode.use_library is False and not r.potentials
    assert r.decode.resample_at == "token" and not r.band and r.utility == "lcs"
    g = R.resolve(cfg, {"temperature": 0.0})
    assert g.greedy and g.decode.temperature == 1.0
    ig = R.resolve(cfg, {"repeat_penalty": 1.1, "use_critic": True, "think_budget": 5})
    assert dict(ig.ignored) == {"repeat_penalty": 1.1, "use_critic": True, "think_budget": 5}
    assert ig.decode.use_critic is False
    with pytest.raises(ValueError):
        R.resolve(cfg, {"particles": 0})


def test_clean_messages():
    m = R.clean_messages([{"role": "system", "content": "S"},
                          {"role": "library", "content": "old passage"},
                          {"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert m == [{"role": "system", "content": "S"}, {"role": "user", "content": "a\n\nb"}]
    for bad in ([], [{"role": "assistant", "content": "x"}],
                [{"role": "user", "content": "x"}, {"role": "system", "content": "s"},
                 {"role": "user", "content": "y"}],
                [{"role": "robot", "content": "x"}], [{"role": "user", "content": 3}]):
        with pytest.raises(ValueError):
            R.clean_messages(bad)


# ---------------------------------------------------------------------------- the prompt
class WordChat:
    """One token per word plus one per role marker and per turn end (lengths by hand)."""

    supports_library = True

    def render(self, messages):
        out = ["BOS"]
        for m in messages:
            out += [m["role"].upper()] + m["content"].split() + ["END"]
        return out + ["ASSISTANT"]


def test_fit_prompt_drops_in_the_stated_order():
    chat = WordChat()
    sys_msg = {"role": "system", "content": "be kind"}                  # 4 tokens
    old = [{"role": "user", "content": "one two three"},                # 5
           {"role": "assistant", "content": "four five"},               # 4
           {"role": "user", "content": "six"},                          # 3
           {"role": "assistant", "content": "seven eight"}]             # 4
    last = {"role": "user", "content": " ".join(f"w{i}" for i in range(20))}   # 22
    psg = "p1 p2 p3"                                                    # 5
    msgs = [sys_msg] + old + [last]
    full = 1 + 4 + 5 + 4 + 3 + 4 + 5 + 22 + 1
    ids, info = R.fit_prompt(chat, msgs, psg, full)
    assert len(ids) == full and info["dropped_turns"] == 0
    ids, info = R.fit_prompt(chat, msgs, psg, full - 1)                 # the oldest exchange
    assert info["dropped_turns"] == 2 and "one" not in ids and "six" in ids
    ids, info = R.fit_prompt(chat, msgs, psg, full - 9 - 1)             # both exchanges
    assert info["dropped_turns"] == 4 and "six" not in ids and "p1" in ids
    ids, info = R.fit_prompt(chat, msgs, psg, 1 + 4 + 22 + 1)           # then the passage
    assert info["dropped_passage"] and "p1" not in ids and "be" in ids
    ids, info = R.fit_prompt(chat, msgs, psg, 1 + 22 + 1)               # then the system turn
    assert info["dropped_system"] and "be" not in ids and len(ids) == 24
    ids, info = R.fit_prompt(chat, msgs, psg, 15)                       # then the user's middle
    assert info["cut_user"] and len(ids) <= 15 and "w0" in ids and "w19" in ids
    assert ids[-1] == "ASSISTANT" and ids[0] == "BOS"


# ------------------------------------------------------------------------------ replies
def test_reply_contract_and_determinism():
    tok, lm, resp = setup()
    out = resp.reply(USER, {"temperature": 1.0}, seed=3)
    assert set(out) == {"text", "finish_reason", "prompt_tokens", "completion_tokens",
                        "logprob_sum", "logprob_count", "extra"}
    assert isinstance(out["text"], str) and out["finish_reason"] in ("stop", "length")
    assert out["prompt_tokens"] == len(lm.prompts[-1]) and out["completion_tokens"] >= 1
    assert out["logprob_count"] == out["completion_tokens"] and out["logprob_sum"] <= 0
    ex = out["extra"]
    assert ex["particles"] == 3 and ex["constraints"] == ["sentences: exactly 2"]
    assert ex["rule"] in ("strict", "relax_band", "relax_degenerate", "best_finished", "unfinished")
    again = resp.reply(USER, {"temperature": 1.0}, seed=3)
    assert again == out
    others = {resp.reply(USER, {"temperature": 1.0}, seed=s)["text"] for s in range(4, 8)}
    assert len(others | {out["text"]}) > 1


def test_potentials_make_every_finished_particle_satisfy_the_constraint():
    tok, lm, resp = setup(seed=1)
    spec = C.compile(USER[0]["content"])
    for seed in range(4):
        out = resp.reply(USER, {"particles": 4, "max_new_tokens": 60}, seed=seed)
        if out["finish_reason"] == "stop":
            assert not C.verify(spec, out["text"]), out["text"]
            assert not out["extra"]["violations"]
        assert out["extra"]["enforced"] == ["sentences: exactly 2"]


def test_the_reply_is_the_selected_particles_display_text():
    from bes.decode import select as Sel
    from bes.decode import smc
    from bes.decode import sampling as S

    tok, lm, resp = setup(seed=2)
    out = resp.reply(USER, {"particles": 3}, seed=9)
    # rerun the decoder and the selection by hand with the same inputs
    prompt = lm.prompts[-1]
    spec = C.compile(USER[0]["content"])
    prog = resp.program(spec, R.resolve(resp.cfg, {"particles": 3}))
    res = smc.decode(lm, prompt, n_particles=3, max_new_tokens=40, seed=9,
                     settings=S.SamplingSettings(temperature=1.0, top_p=1.0), program=prog,
                     tok=tok, no_repeat_ngram=resp.cfg.no_repeat_ngram,
                     max_copy_run=resp.cfg.max_copy_run)
    chat = TinyChat(tok)
    cands = [Sel.Candidate(chat.display(p.tokens).strip(), float(res.weights[j]), p.finish,
                           len(p.tokens)) for j, p in enumerate(res.particles)]
    sel = Sel.select(cands, spec, USER[0]["content"], budget=40)
    assert out["text"] == sel.text and out["extra"]["rule"] == sel.rule
    p = res.particles[sel.index]
    assert out["logprob_sum"] == pytest.approx(sum(p.token_logp))


def test_library_passage_goes_right_before_the_last_user_turn():
    q = "Who was Ada?"
    lib = FakeLibrary({q: ("Ada", "Ada was a writer.")})
    tok, lm, resp = setup(library=lib)
    msgs = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."},
            {"role": "library", "content": "An old passage."}, {"role": "user", "content": q}]
    out = resp.reply(msgs, {"particles": 2, "max_new_tokens": 12}, seed=0)
    sp = tok.special_tokens
    want = TinyChat(tok).render([msgs[0], msgs[1], {"role": "library", "content": "Ada was a writer."},
                                 msgs[3]])
    assert lm.prompts[-1] == want and lib.calls == [q]
    assert lm.prompts[-1].count(sp["<|library|>"]) == 1 and out["extra"]["library"] == "Ada"
    resp.reply(msgs, {"particles": 2, "max_new_tokens": 12, "use_library": False}, seed=0)
    assert sp["<|library|>"] not in lm.prompts[-1] and lib.calls == [q]
    # no passage for this query: the gate said no
    resp.reply([{"role": "user", "content": "Write a poem."}], {"particles": 2, "max_new_tokens": 8}, 0)
    assert sp["<|library|>"] not in lm.prompts[-1]


def test_old_turns_are_dropped_to_fit_the_context():
    tok, lm, resp = setup(context=120, cfg=DecodeCfg(particles=2, max_new_tokens=20,
                                                     temperature=1.0, top_p=1.0))
    msgs = [{"role": "user", "content": "First question about the sea and sky."},
            {"role": "assistant", "content": "An answer about the sea and the sky."},
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "Second answer."},
            {"role": "user", "content": "Last one: say hi."}]
    out = resp.reply(msgs, {}, seed=0)
    prompt = lm.prompts[-1]
    assert len(prompt) <= 100 and out["extra"]["fit"]["dropped_turns"] == 2
    assert prompt == TinyChat(tok).render(msgs[2:])


def test_our_chat_display_shows_calculator_spans():
    from bes.config import CALC, CALC_END, EQ
    from bes.tokenizer import load

    tok = load()
    chat = R.OurChat(tok)
    ids = tok.encode("It is ") + [CALC] + tok.encode("2+3") + [EQ] + tok.encode("5") + \
        [CALC_END] + tok.encode(".") + [chat.end_id]
    assert chat.display(ids) == "It is 2+3 = 5."
    ids2 = chat.render([{"role": "user", "content": "Hi"}])
    from bes.data.chat_format import render_conversation
    assert ids2 == [int(t) for t in render_conversation([{"role": "user", "content": "Hi"}], tok,
                                                        add_generation_prompt=True)[0]]


# --------------------------------------------------------------------- harness factory
def _save_smoke(root: Path, seed: int, cfg):
    import jax

    from bes.config import config_hash
    from bes.model import model as M
    from bes.train import checkpoint as ck

    params = M.init_params(cfg.model, jax.random.PRNGKey(seed))
    tree = {"state": {"step": np.int32(seed), "params": params,
                      "opt_state": {"m": jax.tree.map(np.zeros_like, params["head"])},
                      "consts": M.make_consts(cfg.model)},
            "rng": jax.random.PRNGKey(7)}
    return ck.save(root, tree, step=seed, config_hash=config_hash(cfg))


def test_our_generator_follows_the_harness_contract(tmp_path):
    from bes.config import load_config
    from bes.eval.proxy.common import GenSettings
    from bes.eval.proxy.run_ours import CallableBackend

    config = str(REPO / "configs" / "smoke.py")
    over = "model.n_loops=2,decode.particles=2,decode.max_new_tokens=8"
    cfg = load_config(config, over.split(","))
    _save_smoke(tmp_path / "ckpt", 1, cfg)
    gen = R.our_generator(checkpoint=str(tmp_path), config=config, overrides=over)
    assert gen.key.startswith("ours-step1-") and gen.checkpoint["consts"] == "stored"
    msgs = [{"role": "user", "content": "Say hello in one sentence."}]
    out = gen(msgs, {"temperature": 0.6, "top_p": 0.9, "max_tokens": 8}, 5)
    assert isinstance(out["text"], str) and out["completion_tokens"] <= 8
    assert out == gen(msgs, {"temperature": 0.6, "top_p": 0.9, "max_tokens": 8}, 5)
    reply = CallableBackend(gen).generate(msgs, GenSettings(max_tokens=8), 5)
    assert reply.text == reply.text.strip() and reply.finish_reason in ("stop", "length")
    same = R.our_generator(checkpoint=str(tmp_path / "ckpt"), config=config, overrides=over)
    assert same.key == gen.key
    _save_smoke(tmp_path / "ckpt", 2, cfg)                          # a newer checkpoint
    newer = R.our_generator(checkpoint=str(tmp_path), config=config, overrides=over)
    assert newer.key != gen.key and newer.key.startswith("ours-step2-")
    with pytest.raises(ValueError):
        R.load_checkpoint(tmp_path, dataclasses.replace(cfg.model, head_rank=32))
    with pytest.raises(FileNotFoundError):
        R.find_checkpoint(tmp_path / "missing")
