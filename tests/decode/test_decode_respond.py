"""The chat entry point (bes.decode.respond).

A character-level tiny tokenizer, an explicit table model and a tiny chat format written here
drive the whole pipeline; prompts are captured by a spy model and compared with renderings
the test builds itself. The harness factory is exercised on a smoke-sized checkpoint written
with ``bes.train.checkpoint``.
"""
from __future__ import annotations

import dataclasses
import json
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


def test_the_exemplar_floor_reaches_a_library_that_takes_one():
    """``DecodeCfg.exemplar_floor`` (or the setting of the same name) is passed to a library
    whose retrieval has a ``floor`` argument, and to no other."""
    class FloorLibrary:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, floor=None):
            self.calls.append((query, floor))
            return None

    lib = FloorLibrary()
    _, _, resp = setup(library=lib)
    q = "why is it dark at night?"
    resp.retrieve(q)
    resp.retrieve(q, floor=0.35)
    assert lib.calls == [(q, None), (q, 0.35)]
    plain = FakeLibrary({})                               # retrieve(query) only
    _, _, resp2 = setup(library=plain)
    assert resp2.retrieve(q, floor=0.35) is None and plain.calls == [q]
    assert R.resolve(DecodeCfg(), {"exemplar_floor": "0.4"}).decode.exemplar_floor == 0.4
    assert DecodeCfg().exemplar_floor == 0.0
    # through a reply: the setting is what the library sees
    lib.calls.clear()
    resp.reply([{"role": "user", "content": q}], {"exemplar_floor": 0.4, "max_tokens": 8}, seed=0)
    assert lib.calls and lib.calls[-1][1] == 0.4
    lib.calls.clear()
    resp.reply([{"role": "user", "content": q}], {"max_tokens": 8}, seed=0)
    assert lib.calls and lib.calls[-1][1] is None


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


def test_find_checkpoint_in_a_run_directory_of_the_supervisor(tmp_path):
    """A run directory of scripts/run_24h.py holds the run's own manifest.json next to
    train/ckpt: it is not a checkpoint; the trainer's newest checkpoint under it is found and
    loads."""
    from bes.config import load_config

    config = str(REPO / "configs" / "smoke.py")
    cfg = load_config(config, ["model.n_loops=2"])
    run = tmp_path / "run"
    (run / "train").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"format": 1, "config": config}))
    _save_smoke(run / "train" / "ckpt", 1, cfg)
    newest = _save_smoke(run / "train" / "ckpt", 2, cfg).path
    assert R.find_checkpoint(run) == newest
    assert R.find_checkpoint(run / "train") == R.find_checkpoint(newest) == newest
    assert R.load_checkpoint(run, cfg.model)[2]["step"] == 2
    with pytest.raises(FileNotFoundError):
        R.find_checkpoint(tmp_path / "missing")


def test_a_checkpoint_from_before_a_switch_existed_keeps_its_stored_switches(tmp_path):
    """Consts in the layout of an older code version (no gain, sel, gate, page entries) that
    store a fallback rung (sink:0, cand_out:0): the stored switches are kept by name and the
    missing ones take their defaults, recorded in ``info``; a stored switch this code does not
    have is ignored with a warning and recorded."""
    import jax

    from bes.config import config_hash, load_config
    from bes.export import consts_equal
    from bes.model import model as M
    from bes.train import checkpoint as ck

    cfg = load_config(str(REPO / "configs" / "smoke.py"), ["model.n_loops=2"])
    params = jax.tree.map(np.asarray, M.init_params(cfg.model, jax.random.PRNGKey(0)))
    rung = M.make_consts(cfg.model, sink=0.0, cand_out=0.0)
    new = ("gain", "sel", "gate", "page")
    old = {**rung, "enable": {k: v for k, v in rung["enable"].items() if k not in new}}

    def save(consts, root):
        tree = {"state": {"step": np.int32(3), "params": params, "consts": consts},
                "rng": jax.random.PRNGKey(1)}
        return ck.save(root, tree, step=3, config_hash=config_hash(cfg)).path

    _, consts, info = R.load_checkpoint(save(old, tmp_path / "old"), cfg.model)
    assert float(consts["enable"]["sink"]) == 0.0 and float(consts["enable"]["cand_out"]) == 0.0
    assert consts_equal(consts, rung)
    assert info["consts"] == "merged" and info["consts_filled"] == sorted(new)
    assert info["consts_dropped"] == []
    alien = {**old, "enable": {**old["enable"], "retired": np.float32(0.0)}}
    with pytest.warns(UserWarning, match="retired"):
        _, consts, info = R.load_checkpoint(save(alien, tmp_path / "alien"), cfg.model)
    assert consts_equal(consts, rung) and info["consts_dropped"] == ["retired"]
    _, consts, info = R.load_checkpoint(save(rung, tmp_path / "now"), cfg.model)
    assert info["consts"] == "stored" and info["consts_filled"] == [] and consts_equal(consts, rung)


# ------------------------------------------------------------ self-evidence weight
@pytest.fixture(scope="module")
def smoke_model():
    import jax

    from bes.config import load_config
    from bes.model import model as M
    from bes.tokenizer import load

    tok = load()
    cfg = dataclasses.replace(load_config(str(REPO / "configs" / "smoke.py")).model, n_loops=2,
                              max_seq=256)
    params = jax.tree.map(np.array, M.init_params(cfg, jax.random.PRNGKey(0)))
    params["trust"]["bias"]["src"][3] = 9.0      # trust evidence from earlier assistant turns
    params["trust"]["bias"]["src"][5] = 9.0      # ... and from the turn being generated
    params["trust"]["bias"]["kind"][1:5] = 2.0
    return tok, cfg, params, M.make_consts(cfg)  # the consts a checkpoint stores: src_5 = 1


def test_the_self_evidence_weight_reaches_the_model(smoke_model):
    """The request's ``self_evidence_weight`` is the multiplier the model decodes with
    (``enable["src"][5]``), different weights give different replies, the default is the
    configured one, and the applied value is reported."""
    from bes.decode.lm import OurLM

    tok, cfg, params, consts = smoke_model
    made = []

    def factory(n):
        made.append(OurLM(params, consts, cfg, tok, max_steps=2 * n))
        return made[-1]

    resp = R.Responder(factory, tok, DecodeCfg(particles=1, max_new_tokens=24), context=256)
    msgs = [{"role": "user", "content": "List colours."},
            {"role": "assistant", "content": "red blue green red blue green"},
            {"role": "user", "content": "Again."}]
    outs = {}
    for w in (1.0, 0.25, 0.0):
        out = resp.reply(msgs, {"temperature": 0, "self_evidence_weight": w}, seed=0)
        assert float(np.asarray(made[-1].consts["enable"]["src"])[5]) == w
        assert out["extra"]["self_evidence_weight"] == w and "ignored" not in out["extra"]
        outs[w] = (out["text"], round(out["logprob_sum"], 5))
    assert len(set(outs.values())) == 3, outs
    default = resp.reply(msgs, {"temperature": 0}, seed=0)
    assert default["extra"]["self_evidence_weight"] == 0.25
    assert (default["text"], round(default["logprob_sum"], 5)) == outs[0.25]


def test_a_self_evidence_weight_a_model_cannot_apply_is_recorded_as_ignored():
    tok, lm, resp = setup()
    out = resp.reply(USER, {"self_evidence_weight": 0.5}, seed=0)
    assert out["extra"]["self_evidence_weight"] is None
    assert out["extra"]["ignored"] == {"self_evidence_weight": 0.5}
    plain = resp.reply(USER, {}, seed=0)                 # the default is not a request
    assert plain["extra"]["self_evidence_weight"] is None and "ignored" not in plain["extra"]


def test_the_generator_decodes_with_the_configured_weight_and_keys_on_it(tmp_path):
    from bes.config import load_config

    config = str(REPO / "configs" / "smoke.py")
    over = "model.n_loops=2,decode.particles=2,decode.max_new_tokens=8"
    cfg = load_config(config, over.split(","))
    _save_smoke(tmp_path / "ckpt", 1, cfg)
    gen = R.our_generator(checkpoint=str(tmp_path), config=config, overrides=over)
    lm = gen.responder.lm_factory(8)
    assert float(np.asarray(lm.consts["enable"]["src"])[5]) == cfg.decode.self_evidence_weight
    assert lm.max_steps == 16                            # about 2 x max_new_tokens steps
    gen2 = R.our_generator(checkpoint=str(tmp_path), config=config,
                           overrides=over + ",decode.self_evidence_weight=0.5")
    assert float(np.asarray(gen2.responder.lm_factory(8).consts["enable"]["src"])[5]) == 0.5
    assert gen2.key != gen.key
    # one cache length for every request of up to the configured budget
    from bes.decode.lm import cache_slots

    assert gen.responder.lm_factory(4).tmax == lm.tmax == cache_slots(gen.responder.context, 8)


# ---------------------------------------------------------------- feasibility
def test_an_infeasible_count_is_dropped_and_the_reply_is_never_empty():
    """'at least 60 words' cannot fit 40 tokens of a one-character tokenizer: it is dropped and
    recorded, the feasible rest of the spec (exactly 2 sentences) is still enforced, and the
    reply is a finished, non-empty text (before: every particle died at the first step)."""
    tok, lm, resp = setup(seed=1)
    msgs = [{"role": "user", "content": "Write at least 60 words in exactly 2 sentences."}]
    spec = C.compile(msgs[0]["content"])
    assert spec.words == C.Interval(60, None) and spec.sentences == C.Interval(2, 2)
    out = resp.reply(msgs, {"particles": 4}, seed=0)
    ex = out["extra"]
    assert ex["infeasible"] == ["words: at least 60"]
    assert ex["enforced"] == ["sentences: exactly 2"]
    assert out["text"] and ex["finish"] == "stop" and ex["finishes"].get("dead", 0) == 0
    # this one ends in its last token slot: finished, but cut by the budget (reported as such)
    assert out["finish_reason"] == ("length" if ex["at_budget"] else "stop")
    assert C.measure(out["text"])["sentences"] == 2
    # a feasible request is left alone
    ok = resp.reply(USER, {"particles": 2}, seed=0)
    assert "infeasible" not in ok["extra"] and ok["extra"]["enforced"] == ["sentences: exactly 2"]


def test_dead_particles_trigger_a_structural_rerun():
    """A model that starts a calculator span and writes ``1-2`` under 'at most 2 words': the
    result would be a third word, so ``EQ`` is never allowed and every particle dies inside the
    expression. The responder decodes again with the structural potentials only and returns
    a finished, non-empty reply."""
    tok = TinyTokenizer(PIECES)
    sp = tok.special_tokens
    v = tok.vocab_size
    table = np.full((v, v), -6.0)
    one, minus, two = PIECES.index("1"), PIECES.index("-"), PIECES.index("2")
    table[sp["<|assistant|>"], sp["<|calc|>"]] = 10.0
    table[sp["<|calc|>"], one] = 10.0
    table[one, minus] = 10.0
    table[minus, two] = 10.0
    table[two, sp["<|eq|>"]] = 10.0
    table[sp["<|/calc|>"], sp["<|end|>"]] = 10.0
    lm = TableLM.from_table(table)
    resp = R.Responder(lambda n: lm, tok, DecodeCfg(particles=2, max_new_tokens=30), chat=TinyChat(tok),
                       context=4096)
    out = resp.reply([{"role": "user", "content": "Answer in at most 2 words."}],
                     {"temperature": 0}, seed=0)
    ex = out["extra"]
    assert ex["rerun"]["reason"] == "all particles died" and ex["rerun"]["finishes"] == {"dead": 2}
    # "1-2" EQ "-1" CALC_END (TinyChat shows the ordinary tokens only: no " = " for EQ)
    assert out["finish_reason"] == "stop" and out["text"] == "1-2-1"


# ----------------------------------------------------- a reply cut by its budget (review eval-2)
def test_a_reply_that_used_its_whole_budget_is_reported_as_cut():
    """A model that almost never ends: the budget-aware potentials make every particle end in
    its last token slot, so the reply stops because the budget ran out. It must be reported the
    way every harness backend reports a reply at its token limit (before: finish reason "stop",
    and the exact scorer counted no truncation and no degenerate reply for our arm)."""
    from bes.eval.exact_score import degenerate_stats
    from bes.eval.proxy.common import GenSettings
    from bes.eval.proxy.run_ours import CallableBackend

    tok = TinyTokenizer(PIECES)
    sp = tok.special_tokens
    lm = TableLM.random(tok.vocab_size, order=2, seed=3, boost={sp["<|end|>"]: -12.0, PIECES.index(" "): 1.5})
    budget = 24
    resp = R.Responder(lambda n: lm, tok, DecodeCfg(particles=3, max_new_tokens=budget, temperature=1.0,
                                                    top_p=1.0), chat=TinyChat(tok), context=1024)
    msgs = [{"role": "user", "content": "Tell me about the moon"}]
    out = resp.reply(msgs, {"potentials": True}, seed=7)
    assert out["completion_tokens"] == budget and out["extra"]["finish"] == "stop"   # END in the last slot
    assert out["finish_reason"] == "length" and out["extra"]["at_budget"] is True
    resp.key = "test-at-budget"
    reply = CallableBackend(resp).generate(msgs, GenSettings(temperature=1.0, top_p=1.0, max_tokens=budget,
                                                             extra={"particles": 3}), 7)
    d = degenerate_stats(reply.text, reply.finish_reason)
    assert reply.completion_tokens == budget and d["truncated"] is True and d["degenerate"] is True
    # without potentials the particles are cut by the limit itself, reported as before
    off = resp.reply(msgs, {"potentials": False}, seed=7)
    assert off["finish_reason"] == "length" and off["extra"]["finish"] == "length"
    assert off["extra"]["at_budget"] is False


def test_a_reply_that_ended_on_its_own_is_reported_as_stopped():
    tok, lm, resp = setup(seed=4)
    seen = set()
    for seed in range(6):
        out = resp.reply([{"role": "user", "content": "Say something."}], {"particles": 2}, seed=seed)
        own = out["extra"]["finish"] == "stop" and out["completion_tokens"] < 40
        assert (out["finish_reason"] == "stop") == own and out["extra"]["at_budget"] == (
            out["extra"]["finish"] == "stop" and out["completion_tokens"] >= 40)
        seen.add(out["finish_reason"])
    assert "stop" in seen


# ------------------------------------------------------------------- copy cap
class ScriptLM:
    """Puts 97% of the mass on the next token of a fixed answer (a perfect copier)."""

    def __init__(self, answer_ids, vocab_size, end_id):
        from bes.decode.lm import PerSequenceLM

        class _LM(PerSequenceLM):
            def __init__(self):
                self.vocab_size = vocab_size

            def _logp(self, n):
                p = np.full(vocab_size, 0.03 / vocab_size)
                p[answer[min(n, len(answer) - 1)]] += 0.97
                return np.log(p)

            def start_sequence(self, prompt_ids):
                return 0, self._logp(0)

            def advance(self, state, token):
                return state + 1, self._logp(state + 1)

            def copy_state(self, state):
                return state

        answer = list(answer_ids) + [end_id]
        self.lm = _LM()


def _longest_common_run(a, b):
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            n = 0
            while i + n < len(a) and j + n < len(b) and a[i + n] == b[j + n]:
                n += 1
            best = max(best, n)
    return best


def test_the_copy_cap_stops_passage_regurgitation_but_not_the_users_own_text():
    """Repeating the request word for word, and returning the user's sentence corrected, copy
    far more than ``max_copy_run`` tokens of the user's turn: nothing is capped (before, the
    13th copied token was replaced). A LIBRARY passage is still capped."""
    from bes.tokenizer import load

    real = load()
    request = "Write a short poem about the quiet morning light over the sleepy harbor town."
    user = request + " " + C.REPEAT_INSTRUCTIONS[0]
    answer = request + "\n\nSoft gold on the water, gulls asleep on the pier."
    lm = ScriptLM(real.encode(answer), real.vocab_size, real.end).lm
    resp = R.Responder(lambda n: lm, real, DecodeCfg(particles=4, max_new_tokens=64))
    out = resp.reply([{"role": "user", "content": user}], {}, seed=0)
    assert out["text"].startswith(request) and out["extra"]["violations"] == []
    assert out["extra"]["copy_cap"] == 0                   # a spec that needs copied text
    fix = ("Fix the spelling mistake in this sentence: The old fisherman walked slowly along the "
           "quiet beach every morning to watch the boats come bak.")
    corrected = ("The old fisherman walked slowly along the quiet beach every morning to watch the "
                 "boats come back.")
    lm2 = ScriptLM(real.encode(corrected), real.vocab_size, real.end).lm
    resp2 = R.Responder(lambda n: lm2, real, DecodeCfg(particles=4, max_new_tokens=64, max_copy_run=12))
    out2 = resp2.reply([{"role": "user", "content": fix}], {}, seed=0)
    assert out2["text"] == corrected and out2["extra"]["copy_cap"] == 12
    passage = ("the river runs past the old mill and under the stone bridge where the children "
               "play in the long summer evenings and the swallows dive over the water")
    lib = FakeLibrary({"Tell me about the river.": ("River", passage)})
    lm3 = ScriptLM(real.encode(passage), real.vocab_size, real.end).lm
    resp3 = R.Responder(lambda n: lm3, real, DecodeCfg(particles=2, max_new_tokens=64, max_copy_run=12),
                        library=lib)   # the cap is off by default; when set it acts on passages only
    out3 = resp3.reply([{"role": "user", "content": "Tell me about the river."}], {}, seed=0)
    assert out3["extra"]["library"] == "River"
    reply_ids = real.encode(out3["text"])
    assert _longest_common_run(reply_ids, real.encode(passage)) <= 12 + 1
