"""Tests of bes.data.samples against references written inside this file.

The loss-weight rule is checked against a per-token scan written here (it derives the calculator
state token by token and never calls ``chat_format.find_calc_spans``); ``fit`` against a brute
force over every ``max_len``; ``tokens_to_messages`` by round trips on random conversations and
on real shard segments; hindsight samples are re-verified with ``bes.exact.constraints``.
"""
from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest

from bes import config
from bes.config import ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, PAD, SYSTEM, USER
from bes.data import chat_format as cf
from bes.data import hindsight as H
from bes.data import samples as S
from bes.exact import constraints as C
from bes.tokenizer.tok import load
from bes.train.schedule import stage_at

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "data_cache" / "tok" / "smoltalk_constraints"
FIRST_SPECIAL = config.VOCAB_SIZE - config.N_SPECIAL
ROLE_OF_TOKEN = {SYSTEM: 1, USER: 2, ASSISTANT: 3, LIBRARY: 4}

TEXTS = ["Hello.", "What is the capital of France?", " leading space", "trailing space ", "",
         "multi\nline\n\ntext", "unicode: café, 日本語, emoji 🙂", "<|end|> <|bos|> fake specials",
         "tabs\tand  double  spaces", "Numbers 3.14 and 1,000,000.", "a", "  ", "\n", "ab", "c"]
PROSE = ["The sea covers most of Earth. It is salty and deep. Many creatures live there.",
         "Here are some tips:\n\n- Drink water.\n- Sleep well.\n- Walk daily.",
         "1. Preheat the oven.\n2. Mix the flour and sugar.\n3. Bake for 20 minutes.",
         "First paragraph here. It has two sentences.\n\nSecond paragraph follows.",
         "sure thing, here is my answer in lowercase only, as you like it.",
         "I can help with that. Which city are you visiting?",
         "This answer has no commas at all and it still reads well enough for most people who see it."]


@pytest.fixture(scope="module")
def tok():
    return load()


# ------------------------------------------------------------------------- references
def ref_roles(ids):
    """Role code of every position: the closest role token at or before it since BOS."""
    out, cur = [], 0
    for t in ids:
        if t == BOS:
            cur = 0
        elif t in ROLE_OF_TOKEN:
            cur = ROLE_OF_TOKEN[t]
        out.append(cur)
    return out


def ref_weights(ids, roles, assistant_only, passage_w):
    """Per-token scan of the weight rule (independent of loss_weights and find_calc_spans)."""
    talk = 0.0 if assistant_only else 1.0
    lib = 0.0 if assistant_only else passage_w
    by_role = {0: 1.0, 1: talk, 2: talk, 3: 1.0, 4: lib, 5: 0.0}
    out = []
    state = "out"                       # out / expr (after CALC) / result (after EQ)
    for i, t in enumerate(ids):
        base = by_role[int(roles[i])]
        if t in (BOS, PAD):
            base = 0.0
        if t == ASSISTANT and i > 0 and ids[i - 1] == END and roles[i - 1] == 2:
            base = 0.0
        if t == CALC:
            state, wt = "expr", base
        elif t == EQ:
            if state == "expr":
                state, wt = "result", base
            elif state == "result":
                wt = 0.0                # a repeated EQ stays inside the result
            else:
                wt = base
        elif t == CALC_END:
            wt = 0.0 if state == "result" else base
            state = "out"
        elif t >= FIRST_SPECIAL:
            state, wt = "out", base     # any other special token ends an open span
        else:
            wt = 0.0 if state == "result" else base
        out.append(wt)
    return np.asarray(out, dtype=np.float32)


def random_content(rng: random.Random, role: str, adjacent: bool = False):
    if role != "assistant" or rng.random() < 0.4:
        return rng.choice(TEXTS + PROSE)
    parts, last_text = [], False
    for _ in range(rng.randint(1, 5)):
        if rng.random() < 0.45:
            parts.append({"type": "calc", "expr": rng.choice(["1+2", " 3 * 4", "10/4 ", "(5-3)", ""]),
                          "result": rng.choice(["3", "12", "2.5", " 1024", "2"])})
            last_text = False
        elif adjacent or not last_text:
            parts.append({"type": "text", "text": rng.choice(TEXTS + PROSE)})
            last_text = True
    return parts or "plain"


def random_conversation(rng: random.Random, adjacent: bool = False, prompt: bool = False):
    conv = []
    if rng.random() < 0.3:
        conv.append({"role": "system", "content": rng.choice(TEXTS)})
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.3:
            conv.append({"role": "library", "content": rng.choice(TEXTS + PROSE)})
        conv.append({"role": "user", "content": rng.choice(TEXTS + PROSE)})
        conv.append({"role": "assistant", "content": random_content(rng, "assistant", adjacent)})
    if prompt or rng.random() < 0.2:
        if rng.random() < 0.3:
            conv.append({"role": "library", "content": "passage"})
        conv.append({"role": "user", "content": rng.choice(TEXTS)})
    return conv


def random_samples(tok, n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.25:
            out.append(S.document_sample(" ".join(rng.choice(PROSE) for _ in range(rng.randint(0, 3))), tok))
        else:
            out.append(S.conversation_sample(random_conversation(rng), tok))
    return out


# ------------------------------------------------------------------------- weight rule
@pytest.mark.parametrize("assistant_only", [False, True])
@pytest.mark.parametrize("passage_w", [0.25, 1.0])
def test_loss_weights_match_per_token_reference_on_conversations(tok, assistant_only, passage_w):
    rule = S.WeightRule(assistant_only=assistant_only, passage_w=passage_w)
    rng = random.Random(1)
    for _ in range(150):
        conv = random_conversation(rng, prompt=rng.random() < 0.2)
        prompt = conv[-1]["role"] == "user" and rng.random() < 0.5
        ids, roles = cf.render_conversation(conv, tok, add_generation_prompt=prompt)
        assert roles.tolist() == ref_roles(ids.tolist())
        got = S.loss_weights(ids, roles, rule)
        assert got.dtype == np.float32 and got.shape == ids.shape
        np.testing.assert_array_equal(got, ref_weights(ids.tolist(), roles.tolist(), assistant_only, passage_w))


def test_loss_weights_match_reference_on_random_token_soup():
    rng = np.random.default_rng(3)
    alphabet = np.array([BOS, PAD, SYSTEM, USER, ASSISTANT, LIBRARY, END, CALC, EQ, CALC_END,
                         5, 77, 300, 8000, FIRST_SPECIAL + 12])
    for _ in range(3000):
        ids = rng.choice(alphabet, size=int(rng.integers(1, 40))).tolist()
        roles = rng.integers(0, 6, size=len(ids)).tolist() if rng.random() < 0.5 else ref_roles(ids)
        for ao in (False, True):
            rule = S.WeightRule(assistant_only=ao, passage_w=0.25)
            np.testing.assert_array_equal(S.loss_weights(ids, roles, rule), ref_weights(ids, roles, ao, 0.25))


def test_loss_weight_table_by_hand(tok):
    conv = [{"role": "system", "content": "S"}, {"role": "library", "content": "P"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": [{"type": "text", "text": "A"},
                                              {"type": "calc", "expr": "1+1", "result": "2"}]}]
    s = S.conversation_sample(conv, tok)
    ids = s.ids.tolist()
    e = tok.encode
    expect = ([BOS, SYSTEM, *e("S"), END, LIBRARY, *e("P"), END, USER, *e("Q"), END, ASSISTANT,
               *e("A"), CALC, *e("1+1"), EQ, *e("2"), CALC_END, END])
    assert ids == expect
    w = dict(zip(range(len(ids)), s.w.tolist()))
    pos = {name: ids.index(t) for name, t in (("sys", SYSTEM), ("lib", LIBRARY), ("user", USER),
                                              ("asst", ASSISTANT), ("calc", CALC), ("eq", EQ),
                                              ("calc_end", CALC_END))}
    assert w[0] == 0.0 and w[pos["sys"]] == 1.0 and w[pos["lib"]] == 0.25 and w[pos["user"]] == 1.0
    assert w[pos["asst"]] == 0.0 and w[pos["calc"]] == 1.0 and w[pos["eq"]] == 1.0
    assert w[pos["eq"] + 1] == 0.0 and w[pos["calc_end"]] == 0.0 and w[len(ids) - 1] == 1.0
    tail = s.with_rule(S.WeightRule(assistant_only=True))
    assert tail.w[pos["sys"]] == 0.0 and tail.w[pos["lib"]] == 0.0 and tail.w[pos["user"] + 1] == 0.0
    assert tail.w[pos["calc"] + 1] == 1.0 and tail.w[-1] == 1.0
    doc = S.document_sample("plain text", tok, S.WeightRule(assistant_only=True))
    assert doc.w[0] == 0.0 and (doc.w[1:] == 1.0).all()           # raw text keeps weight 1


def test_rule_at_stage():
    cfg = config.Cfg()
    assert S.WeightRule.at(stage_at(0.5, cfg)) == S.WeightRule(False, cfg.data.passage_loss_w)
    assert S.WeightRule.at(stage_at(0.95, cfg)).assistant_only


# -------------------------------------------------------------------- samples
def test_sample_validation(tok):
    s = S.make_sample([BOS, 5, 6], [0, 0, 0], source="x", kind="base")
    assert s.ids.dtype == np.int32 and s.roles.dtype == np.int8 and s.w.dtype == np.float32
    assert not s.ids.flags.writeable and not s.w.flags.writeable
    assert len(s) == 3 and not s.is_conversation and s.n_targets == 2.0
    for ids, roles in [([], []), ([5, 6], [0, 0]), ([BOS, BOS], [0, 0]), ([BOS, 9000], [0, 0]),
                       ([BOS, 5], [0]), ([BOS, 5], [0, 9]), ([BOS, -1], [0, 0])]:
        with pytest.raises(ValueError):
            S.make_sample(ids, roles)
    with pytest.raises(ValueError):
        S.from_tokens([BOS, USER, 5, END])                      # a conversation needs its roles
    seg = np.array([BOS, 7, 8], dtype=np.uint16)
    assert S.from_tokens(seg).ids.tolist() == [BOS, 7, 8]
    conv = S.conversation_sample([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], tok)
    again = S.from_tokens(conv.ids.astype(np.uint16), conv.roles, source="s", kind="chat")
    assert again.key() == conv.key() and again.is_conversation


# ------------------------------------------------------------------ tokens_to_messages
def test_round_trip_random_conversations(tok):
    rng = random.Random(7)
    for _ in range(300):
        prompt = rng.random() < 0.25
        conv = random_conversation(rng, prompt=prompt)
        if prompt and conv[-1]["role"] != "user":
            prompt = False
        ids, _ = cf.render_conversation(conv, tok, add_generation_prompt=prompt)
        got = S.tokens_to_messages(ids, tok, add_generation_prompt=prompt)
        assert got == S.canonical_messages(conv)
        again, _ = cf.render_conversation(got, tok, add_generation_prompt=prompt)
        assert again.tolist() == ids.tolist()


def test_round_trip_with_adjacent_text_parts(tok):
    """Adjacent text parts whose concatenation encodes differently still round trip token by
    token (the parts are re-split)."""
    pairs = [(a, b) for a in ["ab", "Hello wor", "x ", "12", "naï"] for b in ["c", "ld", " y", "34", "ve"]
             if tok.encode(a + b) != tok.encode(a) + tok.encode(b)]
    assert pairs, "expected some pairs that BPE merges across the boundary"
    rng = random.Random(8)
    for a, b in pairs:
        parts = [{"type": "text", "text": a}, {"type": "text", "text": b},
                 {"type": "calc", "expr": "1+1", "result": "2"}, {"type": "text", "text": a},
                 {"type": "text", "text": b}]
        conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": parts}]
        ids, _ = cf.render_conversation(conv, tok)
        got = S.tokens_to_messages(ids, tok)
        assert cf.render_conversation(got, tok)[0].tolist() == ids.tolist()
    for _ in range(100):
        conv = random_conversation(rng, adjacent=True)
        ids, _ = cf.render_conversation(conv, tok)
        assert cf.render_conversation(S.tokens_to_messages(ids, tok), tok)[0].tolist() == ids.tolist()


@pytest.mark.parametrize("bad", [
    [],
    [USER, 5, END],                                    # no BOS
    [BOS, USER, 5],                                    # no END
    [BOS, USER, 5, CALC, 6, END, ASSISTANT, 7, END],   # special token in a user turn
    [BOS, USER, 5, END, ASSISTANT, CALC, 6, EQ, 7, END],               # open calculator span
    [BOS, USER, 5, END, ASSISTANT, 6, EQ, 7, CALC_END, END],           # stray EQ
    [BOS, USER, 5, END, ASSISTANT, CALC, 6, CALC_END, END],            # span without EQ
    [BOS, ASSISTANT, 5, END],                          # assistant first
    [BOS, USER, 5, END, USER, 6, END],                 # two user turns in a row
    [BOS, LIBRARY, 5, END],                            # ends with a library turn
    [BOS, 5, USER, 6, END],                            # text outside a turn
    [BOS, USER, 5, END, BOS, USER, 6, END],            # BOS inside
    [BOS],                                             # no message
    [BOS, USER, 0xE9, END],                            # invalid UTF-8 (a lone continuation byte)
])
def test_tokens_to_messages_rejects_what_no_conversation_renders_to(tok, bad):
    with pytest.raises(ValueError):
        S.tokens_to_messages(bad, tok)


def test_generation_prompt_round_trip(tok):
    conv = [{"role": "user", "content": "Hi"}]
    ids, _ = cf.render_conversation(conv, tok, add_generation_prompt=True)
    assert S.tokens_to_messages(ids, tok, add_generation_prompt=True) == conv
    with pytest.raises(ValueError):
        S.tokens_to_messages(ids, tok)
    with pytest.raises(ValueError):
        S.tokens_to_messages(cf.render_conversation(conv, tok)[0], tok, add_generation_prompt=True)


@pytest.mark.skipif(not (SHARD_DIR / "manifest.jsonl").exists(), reason="no tokenised shard in data_cache")
def test_round_trip_on_real_shard_segments(tok):
    from bes.data.shards import ShardSet

    shards = ShardSet(SHARD_DIR)
    picks = np.linspace(0, len(shards) - 1, 300).astype(int)
    for j in picks.tolist():
        seg, roles = np.asarray(shards.segment(j)), np.asarray(shards.segment_roles(j))
        messages = S.tokens_to_messages(seg, tok)
        ids, r = cf.render_conversation(messages, tok)
        assert ids.tolist() == seg.tolist() and r.tolist() == roles.tolist()
        s = S.from_tokens(seg, roles, source="smoltalk_constraints", kind="chat")
        np.testing.assert_array_equal(s.w, ref_weights(seg.tolist(), roles.tolist(), False, 0.25))


# ------------------------------------------------------------------------------- fit
def ref_cuts(ids):
    """Every prefix length that ends with the END of an assistant turn (brute force)."""
    cuts = []
    for k in range(1, len(ids) + 1):
        if ids[k - 1] != END:
            continue
        j = k - 2
        while j >= 0 and ids[j] not in ROLE_OF_TOKEN and ids[j] != BOS:
            j -= 1
        if j >= 0 and ids[j] == ASSISTANT:
            cuts.append(k)
    return cuts


def complete_turns(ids):
    """True when ids = BOS then complete ROLE ... END turns (no turn is cut)."""
    depth = 0
    for t in ids[1:]:
        if t in ROLE_OF_TOKEN:
            if depth:
                return False
            depth = 1
        elif t == END:
            if not depth:
                return False
            depth = 0
    return depth == 0


def test_fit_brute_force_over_every_max_len(tok):
    for s in random_samples(tok, 60, seed=5):
        ids = s.ids.tolist()
        cuts = ref_cuts(ids)
        for max_len in range(0, len(s) + 3):
            head, rest = S.fit(s, max_len)
            if s.is_conversation:
                ok = [k for k in cuts if k <= max_len]
                assert rest is None
                if not ok:
                    assert head is None
                    continue
                assert head is not None and len(head) == ok[-1] <= max_len
                assert head.ids.tolist() == ids[:len(head)]
                assert complete_turns(head.ids.tolist())
                assert ref_roles(head.ids.tolist())[-1] == 3 and head.ids[-1] == END
                np.testing.assert_array_equal(head.w, s.w[:len(head)])
                np.testing.assert_array_equal(head.roles, s.roles[:len(head)])
                assert (head.source, head.kind, head.tags) == (s.source, s.kind, s.tags)
            else:
                if len(s) <= max_len:
                    assert head is s and rest is None
                    continue
                if max_len < 2:
                    assert head is None and rest is s
                    continue
                assert len(head) == max_len and head.ids.tolist() == ids[:max_len]
                assert rest.ids[0] == BOS and rest.ids[1:].tolist() == ids[max_len:]
                assert rest.w[0] == 0.0 and (rest.w[1:] == 1.0).all() and not rest.is_conversation


def test_fit_chunks_a_long_document(tok):
    doc = S.document_sample(" ".join(PROSE) * 4, tok)
    for m in (2, 3, 17, 64, 1024):
        chunks, cur = [], doc
        while cur is not None:
            head, cur = S.fit(cur, m)
            chunks.append(head)
        assert all(len(c) <= m and c.ids[0] == BOS for c in chunks)
        assert sum(len(c) - 1 for c in chunks) == len(doc) - 1
        assert np.concatenate([c.ids[1:] for c in chunks]).tolist() == doc.ids[1:].tolist()


def test_fit_never_ends_on_a_user_turn(tok):
    conv = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "library", "content": "p"}, {"role": "user", "content": "q2"}]
    s = S.conversation_sample(conv, tok)
    head, _ = S.fit(s, 10 ** 6)
    assert S.tokens_to_messages(head.ids, tok) == conv[:2]
    only_user = S.conversation_sample([{"role": "user", "content": "q"}], tok)
    assert S.fit(only_user, 10 ** 6) == (None, None)


# ---------------------------------------------------------------------------- windows
def ref_windows(conv, tok, max_len):
    """Brute force on messages: ``conv`` = [system] + exchanges ``[library] user assistant``
    (a last exchange may lack its assistant); pieces = [system] + a run of whole exchanges, the
    longest that fits ``max_len`` rendered tokens, the first starting at the first exchange, each
    next one at the exchange after the previous piece; an exchange that cannot fit on its own is
    skipped. ``[conv]`` when it fits and ends with an assistant turn."""
    head = conv[:1] if conv and conv[0]["role"] == "system" else []
    groups, cur = [], []
    for m in conv[len(head):]:
        cur.append(m)
        if m["role"] == "assistant":
            groups.append(cur)
            cur = []
    complete = len(groups)                          # a trailing group without assistant is dropped

    def length(msgs):
        return len(cf.render_conversation(msgs, tok)[0])

    if not cur and length(conv) <= max_len:
        return [conv]
    out, i = [], 0
    while i < complete:
        k = 0
        while i + k < complete and length(head + sum(groups[i:i + k + 1], [])) <= max_len:
            k += 1
        if k == 0:
            i += 1
            continue
        out.append(head + sum(groups[i:i + k], []))
        i += k
    return out


def long_conversation(rng: random.Random, n: int):
    conv = [{"role": "system", "content": rng.choice(TEXTS[:4])}] if rng.random() < 0.5 else []
    for _ in range(n):
        if rng.random() < 0.25:
            conv.append({"role": "library", "content": rng.choice(PROSE)})
        conv.append({"role": "user", "content": rng.choice(TEXTS + PROSE)})
        conv.append({"role": "assistant", "content": " ".join(rng.choice(PROSE) for _ in range(rng.randint(1, 3)))})
    return conv


def test_fit_windows_match_a_brute_force_on_messages(tok):
    """fit_windows: the prefix, then the windows that keep the system turn and start at the
    exchange after the previous piece (never at an assistant turn or at a library turn without
    its user turn), each the longest that fits, against a reference on message lists."""
    rng = random.Random(21)
    seen_windows = 0
    for trial in range(40):
        conv = long_conversation(rng, rng.randint(1, 6)) if trial % 4 else random_conversation(rng)
        s = S.conversation_sample(conv, tok, source="x", kind="chat", tags=("t",))
        for max_len in sorted({rng.randint(4, len(s) + 5) for _ in range(6)} | {len(s)}):
            pieces = S.fit_windows(s, max_len)
            want = [S.canonical_messages(p) for p in ref_windows(conv, tok, max_len)]
            assert [S.tokens_to_messages(p.ids, tok) for p in pieces] == want, (trial, max_len)
            for j, p in enumerate(pieces):
                assert len(p) <= max_len and p.ids[-1] == END and p.ids[0] == BOS
                np.testing.assert_array_equal(p.w, S.loss_weights(p.ids, p.roles, p.rule))
                assert (p.source, p.kind) == ("x", "chat") and p.tags[0] == "t"
                assert ("window" in p.tags) == (not s.ids[:len(p)].tolist() == p.ids.tolist())
                seen_windows += "window" in p.tags
            head, _ = S.fit(s, max_len)
            if head is not None:
                assert pieces[0].ids.tolist() == head.ids.tolist()
    assert seen_windows > 30


def test_window_keeps_the_system_turn_and_starts_at_an_exchange(tok):
    conv = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "library", "content": "p2"}, {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"}, {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"}]
    s = S.conversation_sample(conv, tok)
    starts = S.exchange_starts(s.ids)
    assert [int(s.ids[x]) for x in starts] == [USER, LIBRARY, USER]
    assert S.window(s, starts[0]) is s
    w = S.window(s, starts[1])
    assert S.tokens_to_messages(w.ids, tok) == [conv[0]] + conv[3:] and "window" in w.tags
    assert S.tokens_to_messages(S.window(s, starts[2]).ids, tok) == [conv[0]] + conv[6:]
    for bad in (starts[1] + 1, int(np.flatnonzero(s.ids == ASSISTANT)[0]), 0):
        with pytest.raises(ValueError):
            S.window(s, bad)


def test_fit_draw_takes_the_prefix_or_a_later_window(tok):
    """fit_draw: a conversation that fits is kept; otherwise the prefix with probability
    1 - later, else a later window drawn uniformly (a window whenever the prefix does not exist);
    deterministic in the generator."""
    rng = random.Random(5)
    conv = long_conversation(rng, 6)
    s = S.conversation_sample(conv, tok)
    max_len = len(s) // 3
    pieces = S.fit_windows(s, max_len)
    assert len(pieces) >= 3
    keys = [p.key() for p in pieces]
    assert S.fit_draw(s, len(s), np.random.default_rng(0)) is s
    assert all(S.fit_draw(s, max_len, np.random.default_rng(i), later=0.0).key() == keys[0] for i in range(20))
    n = 3000
    got = [keys.index(S.fit_draw(s, max_len, S.rng_for("draw", i), later=0.5).key()) for i in range(n)]
    share = np.bincount(got, minlength=len(keys)) / n
    assert abs(share[0] - 0.5) < 4 * (0.25 / n) ** 0.5
    for j in range(1, len(keys)):
        p = 0.5 / (len(keys) - 1)
        assert abs(share[j] - p) < 4 * (p * (1 - p) / n) ** 0.5
    assert S.fit_draw(s, max_len, S.rng_for("draw", 7)).key() == S.fit_draw(s, max_len, S.rng_for("draw", 7)).key()
    long_first = [{"role": "user", "content": "x " * 400}, {"role": "assistant", "content": "y " * 400},
                  {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    s2 = S.conversation_sample(long_first, tok)
    out = S.fit_draw(s2, 100, np.random.default_rng(0), later=0.0)
    assert S.fit(s2, 100)[0] is None and S.tokens_to_messages(out.ids, tok) == long_first[2:]
    doc = S.document_sample(" ".join(PROSE), tok)
    assert S.fit_draw(doc, 10, np.random.default_rng(0)).ids.tolist() == doc.ids[:10].tolist()


# ------------------------------------------------------------------------- hindsight
def hindsight_conversations(rng: random.Random, n: int):
    users = ["Tell me something.", "What do you think?", "Help me with this", "Write about it:\n\nsome topic"]
    out = []
    for _ in range(n):
        conv = []
        if rng.random() < 0.3:
            conv.append({"role": "system", "content": "Be helpful."})
        for _ in range(rng.randint(1, 3)):
            conv += [{"role": "user", "content": rng.choice(users)},
                     {"role": "assistant", "content": rng.choice(PROSE)}]
        if rng.random() < 0.2:
            conv[-1] = {"role": "assistant", "content": [{"type": "text", "text": "It costs "},
                                                         {"type": "calc", "expr": "3*4", "result": "12"},
                                                         {"type": "text", "text": " dollars in total."}]}
        out.append(conv)
    return out


def check_hindsight(original: S.Sample, out: S.Sample, tok) -> None:
    before, after = S.tokens_to_messages(original.ids, tok), S.tokens_to_messages(out.ids, tok)
    assert len(before) == len(after) and after[:-2] == before[:-2] and after[-1] == before[-1]
    user, response = after[-2]["content"], H.response_text(after[-1]["content"], tok)
    assert user.startswith(before[-2]["content"].rstrip()) and user != before[-2]["content"]
    spec = C.compile(user)
    assert not spec.empty and C.verify(spec, response) == []
    assert "hindsight" in out.tags and (out.source, out.kind) == (original.source, original.kind)
    np.testing.assert_array_equal(out.w, S.loss_weights(out.ids, out.roles, out.rule))
    # every token outside the last user turn is unchanged
    spans = [sp for sp in S.turn_spans(original.ids) if sp[0] == USER]
    start, stop = spans[-1][1], spans[-1][2]
    tail = len(original) - stop
    assert out.ids[:start].tolist() == original.ids[:start].tolist()
    assert out.ids[len(out) - tail:].tolist() == original.ids[stop:].tolist()


def test_apply_hindsight_outputs_reverify(tok):
    rng = random.Random(11)
    produced = 0
    for i, conv in enumerate(hindsight_conversations(rng, 120)):
        s = S.conversation_sample(conv, tok, source="smol_smoltalk", kind="chat")
        out = S.apply_hindsight(s, S.rng_for("test", i), tok)
        if out is None:
            continue
        produced += 1
        check_hindsight(s, out, tok)
    assert produced >= 80
    doc = S.document_sample("text", tok)
    assert S.apply_hindsight(doc, np.random.default_rng(0), tok) is None
    prompt_only = S.conversation_sample([{"role": "user", "content": "q"}], tok)
    assert S.apply_hindsight(prompt_only, np.random.default_rng(0), tok) is None


def test_apply_hindsight_max_len_and_determinism(tok):
    conv = [{"role": "user", "content": "Tell me something."}, {"role": "assistant", "content": PROSE[0]}]
    s = S.conversation_sample(conv, tok)
    a = S.apply_hindsight(s, S.rng_for(0, "h"), tok)
    b = S.apply_hindsight(s, S.rng_for(0, "h"), tok)
    assert a is not None and a.key() == b.key()
    assert S.apply_hindsight(s, S.rng_for(0, "h"), tok, max_len=len(a) - 1) is None
    assert S.apply_hindsight(s, S.rng_for(0, "h"), tok, max_len=len(a)).key() == a.key()


@pytest.mark.skipif(not (SHARD_DIR / "manifest.jsonl").exists(), reason="no tokenised shard in data_cache")
def test_apply_hindsight_on_real_conversations(tok):
    from bes.data.shards import ShardSet

    shards = ShardSet(SHARD_DIR)
    produced = 0
    for j in np.linspace(0, len(shards) - 1, 120).astype(int).tolist():
        s = S.from_tokens(np.asarray(shards.segment(j)), np.asarray(shards.segment_roles(j)),
                          source="smoltalk_constraints", kind="chat")
        out = S.apply_hindsight(s, S.rng_for("real", j), tok)
        if out is not None:
            produced += 1
            check_hindsight(s, out, tok)
    assert produced >= 20


# ----------------------------------------------------------------------- determinism
def test_rng_for_is_counter_based():
    a = S.rng_for(1, "x", (2, 3)).integers(0, 2 ** 62, size=4)
    b = S.rng_for(1, "x", (2, 3)).integers(0, 2 ** 62, size=4)
    assert a.tolist() == b.tolist()
    for other in [(1, "x", (2, 4)), (1, "y", (2, 3)), (2, "x", (2, 3)), ("1", "x", (2, 3)),
                  (1, "x", 2, 3), (1.0, "x", (2, 3)), (True, "x", (2, 3))]:
        assert S.rng_for(*other).integers(0, 2 ** 62, size=4).tolist() != a.tolist(), other
    assert S.key_digest(np.int64(5)) == S.key_digest(5)
    with pytest.raises(TypeError):
        S.key_digest(object())


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 64, 1000, 4097])
def test_permute_index_is_a_keyed_bijection(n):
    perm = [S.permute_index(i, n, "k", 1) for i in range(n)]
    assert sorted(perm) == list(range(n))
    assert perm == [S.permute_index(i, n, "k", 1) for i in range(n)]
    if n >= 7:
        assert perm != [S.permute_index(i, n, "k", 2) for i in range(n)]
        assert perm != list(range(n))
    with pytest.raises(IndexError):
        S.permute_index(n, n)
