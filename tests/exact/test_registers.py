"""Tests of the registers and clocks (bes.exact.registers).

``brute_force`` recomputes every position from scratch: it walks back to the segment start, the
last BOS and the turn start, re-runs the byte-level transducer over the bytes of the turn (END
flushes, EQ after CALC reads as " = ", other special tokens add nothing), recounts the segment
clocks turn by turn and encodes the registers with its own arithmetic. It shares no code with
the vectorised ``compute`` except the byte-level step function.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pytest

from bes import config
from bes.config import REG_FIELDS
from bes.data.chat_format import display_text, render_conversation
from bes.exact import instr_table as it
from bes.exact import registers as R
from bes.exact import transducer as tr
from bes.tokenizer.tok import load

ROLE_CODE = {config.SYSTEM: 1, config.USER: 2, config.ASSISTANT: 3, config.LIBRARY: 4}
SPECIALS = set(range(config.VOCAB_SIZE - config.N_SPECIAL, config.VOCAB_SIZE))
NAMES = [name for name, _ in REG_FIELDS]
CARDS = dict(REG_FIELDS)


@pytest.fixture(scope="module")
def tok():
    return load()


@pytest.fixture(scope="module")
def table(tok):
    return it.load(tok)


# ------------------------------------------------------------------ brute-force reference
def ref_rem_count(target: int, count: int) -> int:
    if target < 0:
        return 0
    k = target - count
    return 1 if k < 0 else 2 + min(k, 31)


def ref_rem_word(target: int, count: int) -> int:
    if target < 0:
        return 0
    k = target - count
    if k < 0:
        return 1
    if k < 16:
        return 2 + k
    for value, upper in zip(range(18, 25), (32, 64, 128, 256, 512, 1024, 2048)):
        if k < upper:
            return value
    return 25


def next_calc(c: int, v: int) -> int:
    if v == config.CALC:
        return 1
    if v == config.EQ:
        return 2 if c in (1, 2) else 0
    return 0 if v in SPECIALS else c


def brute_force(tokens: list[int], seg: list[int], targets, tok):
    """Per position: (raw counter dict, register dict, clock list), all from scratch."""
    memo: dict[tuple[int, int], tuple] = {}
    out = []
    n = len(tokens)
    for t in range(n):
        r = t
        while r > 0 and seg[r - 1] == seg[r]:
            r -= 1
        begin = max([j for j in range(r, t + 1) if tokens[j] == config.BOS], default=r)
        turn = max(j for j in range(begin, t + 1) if tokens[j] in ROLE_CODE or j == begin)
        calc, calc_before = 0, {}
        for j in range(r, t + 1):
            calc_before[j] = calc
            calc = next_calc(calc, tokens[j])

        def run_turn(a: int, b: int) -> tuple:
            if (a, b) not in memo:
                first = a + 1 if (tokens[a] == config.BOS or tokens[a] in ROLE_CODE) else a
                state = tr.INIT
                words = sents = lines = items = wis = depth = 0
                for j in range(first, b + 1):
                    v = tokens[j]
                    if v == config.END:
                        events = [tr.flush(state)]
                    elif v == config.EQ and calc_before[j] == 1:
                        events = tr.trace(b" = ", state)
                    elif v in SPECIALS:
                        events = []
                    else:
                        events = tr.trace(tok.token_bytes(v), state)
                    for state, ev in events:
                        if ev.sentence:
                            sents, wis = sents + 1, 0
                        words, wis = words + ev.word, wis + ev.word
                        lines, items = lines + ev.newline, items + ev.item
                        depth = max(0, depth + ev.bracket)
                memo[(a, b)] = (state, words, sents, lines, items, wis, depth)
            return memo[(a, b)]

        state, words, sents, lines, items, wis, depth = run_turn(turn, t)
        starts = [begin] + [j for j in range(begin + 1, t + 1) if tokens[j] in ROLE_CODE]
        seg_sents = seg_lines = 0
        for k, a in enumerate(starts):
            b = starts[k + 1] - 1 if k + 1 < len(starts) else t
            res = run_turn(a, b)
            seg_sents, seg_lines = seg_sents + res[2], seg_lines + res[3]
        role = ROLE_CODE.get(tokens[turn], 0)
        turn_idx = sum(tokens[j] in ROLE_CODE for j in range(begin, t + 1))
        ntok = t - turn
        tgt = [int(x) for x in targets[t]] if (targets is not None and role == 3) else [-1, -1, -1]
        raw = dict(words=words, sentences=sents, lines=lines, items=items, tokens=ntok,
                   word_in_sent=wis, depth=depth, seg_sentences=seg_sents, seg_lines=seg_lines,
                   turn_idx=turn_idx, role=role, calc=calc)
        reg = dict(role=role, fence=state.fence, inline=state.inline, quote=state.quote,
                   depth=min(depth, 7), calc=calc, tok_in_turn=min((1 + ntok).bit_length() - 1, 11),
                   word_in_sent=min(wis, 63), sent_in_turn=min(sents, 63),
                   line_in_turn=min(lines, 63), item_in_turn=min(items, 63),
                   turn_idx=min(turn_idx, 15), rem_sent=ref_rem_count(tgt[0], sents),
                   rem_item=ref_rem_count(tgt[1], items), rem_word=ref_rem_word(tgt[2], words))
        out.append((raw, reg, [seg_sents, seg_lines, ntok], state))
    return out


def assert_matches_brute_force(regs: R.Registers, b: int, tokens, seg, targets, tok):
    machine = tr.compile_machine()
    ref = brute_force(list(map(int, tokens)), list(map(int, seg)), targets, tok)
    for t, (raw, reg, clock, state) in enumerate(ref):
        got_raw = {k: int(regs.raw[k][b, t]) for k in raw}
        assert got_raw == raw, (t, got_raw, raw)
        assert int(regs.raw["state"][b, t]) == machine.index(state), t
        got_reg = dict(zip(NAMES, regs.reg[b, t].tolist()))
        assert got_reg == reg, (t, got_reg, reg)
        assert regs.clock[b, t].tolist() == clock, t


# ---------------------------------------------------------------------- random sequences
def interesting_ids(tok) -> list[int]:
    marks = set(b"`\n\"'.!?()[]{}*_-+0123456789 \t") | {0xE2}
    return [v for v in range(tok.n_ordinary) if marks & set(tok.token_bytes(v))]


def random_row(rng: random.Random, tok, pool: list[int], length: int):
    """Random packed row: segments (mostly starting with BOS), malformed turns, calculator
    tokens, padding; returns (tokens, seg, targets)."""
    tokens, seg, targets = [], [], []
    sid = -1
    while len(tokens) < length:
        if tokens and rng.random() < 0.15:                    # padding to the end of the row
            n = length - len(tokens)
            tokens += [config.PAD] * n
            seg += [sid + 1] * n
            targets += [[-1, -1, -1]] * n
            break
        sid += 1
        body = [config.BOS] if rng.random() < 0.85 else []
        for _ in range(rng.randint(1, 70)):
            u = rng.random()
            if u < 0.09:
                body.append(rng.choice(config.ROLE_TOKENS))
            elif u < 0.14:
                body.append(config.END)
            elif u < 0.17:
                body.append(config.CALC)
            elif u < 0.20:
                body.append(config.EQ)
            elif u < 0.22:
                body.append(config.CALC_END)
            elif u < 0.23:
                body.append(config.BOS)
            elif u < 0.24:
                body.append(rng.choice([config.PAD, config.VOCAB_SIZE - 1]))
            elif u < 0.70:
                body.append(rng.choice(pool))
            else:
                body.append(rng.randrange(tok.n_ordinary))
        tgt = [rng.choice([-1, -1, 0, 1, 3, 9, 40]), rng.choice([-1, 0, 2, 5]),
               rng.choice([-1, -1, 0, 7, 15, 16, 31, 200, 5000])]
        tokens += body
        seg += [sid] * len(body)
        targets += [tgt if rng.random() < 0.9 else [rng.randint(-1, 50) for _ in range(3)]
                    for _ in body]
    return (np.array(tokens[:length], np.int32), np.array(seg[:length], np.int32),
            np.array(targets[:length], np.int32))


def random_batch(seed: int, tok, n_rows: int, length: int):
    rng = random.Random(seed)
    pool = interesting_ids(tok)
    rows = [random_row(rng, tok, pool, length) for _ in range(n_rows)]
    return tuple(np.stack(x) for x in zip(*rows))


def test_compute_equals_brute_force_on_random_rows(tok, table):
    tokens, seg, targets = random_batch(0, tok, 24, 140)
    regs = R.compute(tokens, seg, targets, table=table)
    for b in range(tokens.shape[0]):
        assert_matches_brute_force(regs, b, tokens[b], seg[b], targets[b], tok)


def conversation_rows(tok):
    convs = [
        [{"role": "system", "content": "Answer briefly.\n```\nunclosed fence"},
         {"role": "user", "content": 'Name (two "fruits. Use `code'},
         {"role": "assistant", "content": "1. Apple\n2. Pear.\n- done\nDone. Bye!"},
         {"role": "library", "content": "Pear\n\nA pear is a fruit (genus Pyrus)."},
         {"role": "user", "content": "And 3.5 times 2?"},
         {"role": "assistant", "content": [{"type": "text", "text": "It is "},
                                           {"type": "calc", "expr": "3.5*2", "result": "7"},
                                           {"type": "text", "text": ". Indeed..."}]}],
        [{"role": "user", "content": "```python\nx = (1,\n```\nWhy?"},
         {"role": "assistant", "content": "Because… it is.\n\n• one\n• two"}],
    ]
    out = []
    for conv in convs:
        ids, _ = render_conversation(conv, tok)
        out.append(ids)
    out.append(render_conversation(convs[1], tok, add_generation_prompt=False)[0][:9])
    return out


def test_compute_equals_brute_force_on_conversations(tok, table):
    rows = conversation_rows(tok)
    length = max(len(r) for r in rows) + 7
    tokens = np.full((len(rows), length), config.PAD, np.int32)
    seg = np.ones((len(rows), length), np.int32)
    targets = np.full((len(rows), length, 3), -1, np.int32)
    for b, r in enumerate(rows):
        tokens[b, : len(r)] = r
        seg[b, : len(r)] = 0
        targets[b, : len(r)] = [2, 3, 20]
    regs = R.compute(tokens, seg, targets, table=table)
    for b in range(len(rows)):
        assert_matches_brute_force(regs, b, tokens[b], seg[b], targets[b], tok)


# --------------------------------------------------------------- incremental and batch
def new_segments(seg: np.ndarray) -> np.ndarray:
    ns = np.zeros(seg.shape, bool)
    ns[:, 0] = True
    ns[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return ns


def test_incremental_equals_compute(tok, table):
    tokens, seg, targets = random_batch(1, tok, 6, 500)
    regs = R.compute(tokens, seg, targets, table=table)
    inc = R.Incremental(tokens.shape[0], table=table)
    ns = new_segments(seg)
    for t in range(tokens.shape[1]):
        reg, clock = inc.step(tokens[:, t], targets[:, t], new_segment=ns[:, t])
        assert np.array_equal(reg, regs.reg[:, t]), t
        assert np.array_equal(clock, regs.clock[:, t]), t
        assert reg.dtype == np.int16 and clock.dtype == np.int32
        raw = inc.raw()
        for k in R.RAW_FIELDS:
            assert np.array_equal(raw[k], regs.raw[k][:, t]), (t, k)


def test_incremental_without_targets_and_scalar_rows(tok, table):
    ids, _ = render_conversation([{"role": "user", "content": "Hi."},
                                  {"role": "assistant", "content": "Hello there. Bye."}], tok)
    regs = R.compute(ids[None], table=table)
    inc = R.Incremental(table=table)
    for t, v in enumerate(ids):
        reg, clock = inc.step(int(v))
        assert reg.shape == (config.N_REG,) and clock.shape == (config.N_CLOCKS,)
        assert np.array_equal(reg, regs.reg[0, t]) and np.array_equal(clock, regs.clock[0, t])
    assert not regs.reg[..., [NAMES.index(n) for n in ("rem_sent", "rem_item", "rem_word")]].any()


def test_prefill_then_step_equals_compute(tok, table):
    tokens, seg, targets = random_batch(2, tok, 4, 300)
    full = R.compute(tokens, seg, targets, table=table)
    ns = new_segments(seg)
    for cut in (1, 57, 299):
        inc, pre = R.Incremental.prefill(tokens[:, :cut], seg[:, :cut], targets[:, :cut], table=table)
        assert np.array_equal(pre.reg, full.reg[:, :cut])
        for t in range(cut, tokens.shape[1]):
            reg, clock = inc.step(tokens[:, t], targets[:, t], new_segment=ns[:, t])
            assert np.array_equal(reg, full.reg[:, t]) and np.array_equal(clock, full.clock[:, t])


def test_select_reorders_streams(tok, table):
    tokens, seg, targets = random_batch(3, tok, 3, 200)
    inc, _ = R.Incremental.prefill(tokens[:, :120], seg[:, :120], targets[:, :120], table=table)
    order = np.array([2, 0, 2, 1])
    picked = inc.select(order)
    copy = inc.copy()
    full = R.compute(tokens[order], seg[order], targets[order], table=table)
    ns = new_segments(seg[order])
    for t in range(120, 200):
        reg, _ = picked.step(tokens[order, t], targets[order, t], new_segment=ns[:, t])
        assert np.array_equal(reg, full.reg[:, t])
    assert np.array_equal(copy.raw()["state"], inc.raw()["state"])   # copies are independent


def test_batch_rows_are_independent(tok, table):
    tokens, seg, targets = random_batch(4, tok, 5, 160)
    regs = R.compute(tokens, seg, targets, table=table)
    for b in range(tokens.shape[0]):
        one = R.compute(tokens[b:b + 1], seg[b:b + 1], targets[b:b + 1], table=table)
        assert np.array_equal(one.reg[0], regs.reg[b]) and np.array_equal(one.clock[0], regs.clock[b])


# ------------------------------------------------------------------------ hand-written
def field(regs, name, b=0):
    return regs.reg[b, :, NAMES.index(name)].tolist()


def ids_of(tok, *parts) -> np.ndarray:
    out = []
    for p in parts:
        out += tok.encode(p) if isinstance(p, str) else [p]
    return np.array(out, np.int32)


def test_role_tokens_reset_turn_state_and_nothing_leaks(tok, table):
    ids = ids_of(tok, config.BOS, config.USER, '```\n"open (bracket and `code', config.END,
                 config.ASSISTANT, "ok", config.END)
    regs = R.compute(ids[None], table=table)
    t_end = int(np.flatnonzero(ids == config.END)[0])
    t_ast = t_end + 1
    assert field(regs, "fence")[t_end] == 1
    for name in ("fence", "quote", "inline", "depth", "word_in_sent", "sent_in_turn",
                 "line_in_turn", "tok_in_turn"):
        assert field(regs, name)[t_ast] == 0, name
    assert field(regs, "role")[t_ast] == 3 and field(regs, "turn_idx")[t_ast] == 2
    assert regs.raw["seg_lines"][0, t_ast] == 1 and regs.clock[0, t_ast].tolist() == [0, 1, 0]
    assert regs.raw["tokens"][0, -1] == 2


def test_bos_resets_everything_and_segments_are_local(tok, table):
    a = ids_of(tok, config.BOS, config.USER, "Hi. There.\n", config.END)
    b = ids_of(tok, config.BOS, "Raw text. More")
    tokens = np.concatenate([a, b, [config.PAD] * 3]).astype(np.int32)[None]
    seg = np.array([[0] * len(a) + [1] * len(b) + [2] * 3], np.int32)
    regs = R.compute(tokens, seg, table=table)
    only_b = R.compute(b[None], table=table)
    assert np.array_equal(regs.reg[0, len(a): len(a) + len(b)], only_b.reg[0])
    assert np.array_equal(regs.clock[0, len(a): len(a) + len(b)], only_b.clock[0])
    pad = regs.reg[0, -3:]
    assert not pad[:, [NAMES.index(n) for n in NAMES if n != "tok_in_turn"]].any()
    assert regs.raw["tokens"][0, -3:].tolist() == [0, 1, 2]
    # Without seg the BOS alone splits the segments; padding then continues the raw text.
    no_seg = R.compute(tokens, table=table)
    assert np.array_equal(no_seg.reg[0, : len(a) + len(b)], regs.reg[0, : len(a) + len(b)])
    assert no_seg.raw["tokens"][0, -1] == len(b) + 2


def test_a_new_segment_does_not_inherit_the_calculator_state(tok, table):
    # The first segment ends inside a calculator expression; the second starts with EQ, which
    # must then add no bytes, so "- a" still starts a list item at the start of the line.
    first = ids_of(tok, config.BOS, config.ASSISTANT, config.CALC, "1+2")
    second = ids_of(tok, config.EQ, "- a")
    tokens = np.concatenate([first, second]).astype(np.int32)[None]
    seg = np.array([[0] * len(first) + [1] * len(second)], np.int32)
    regs = R.compute(tokens, seg, table=table)
    assert field(regs, "calc")[len(first) - 1] == 1 and field(regs, "calc")[len(first)] == 0
    assert regs.raw["items"][0, -1] == 1 and regs.raw["words"][0, -1] == 1
    assert_matches_brute_force(regs, 0, tokens[0], seg[0], None, tok)
    inc = R.Incremental(table=table)
    ns = new_segments(seg)
    for t in range(tokens.shape[1]):
        reg, _ = inc.step(tokens[:, t], new_segment=ns[:, t])
        assert np.array_equal(reg, regs.reg[:, t])


def test_end_flushes_a_pending_sentence(tok, table):
    ids = ids_of(tok, config.BOS, config.ASSISTANT, "Hello there.", config.END)
    regs = R.compute(ids[None], table=table)
    assert field(regs, "sent_in_turn")[-2:] == [0, 1]
    assert field(regs, "word_in_sent")[-2:] == [2, 0]
    assert regs.clock[0, -1, 0] == 1
    ids2 = ids_of(tok, config.BOS, config.ASSISTANT, "Hello there", config.END)
    assert field(R.compute(ids2[None], table=table), "sent_in_turn")[-1] == 0


def test_calculator_register_and_display_counts(tok, table):
    conv = [{"role": "user", "content": "Price?"},
            {"role": "assistant", "content": [{"type": "text", "text": "Total: "},
                                              {"type": "calc", "expr": "3*4", "result": "12"},
                                              {"type": "text", "text": " apples."}]}]
    ids, _ = render_conversation(conv, tok)
    regs = R.compute(ids[None], table=table)
    calc = field(regs, "calc")
    p_calc, p_eq, p_close = (int(np.flatnonzero(ids == x)[0]) for x in (config.CALC, config.EQ, config.CALC_END))
    assert calc[p_calc] == 1 and calc[p_eq - 1] == 1 and calc[p_eq] == 2 and calc[p_close - 1] == 2
    assert calc[p_close] == 0 and calc[p_calc - 1] == 0
    turn = ids[int(np.flatnonzero(ids == config.ASSISTANT)[0]) + 1: -1]
    shown = display_text(turn, tok)
    assert shown == "Total: 3*4 = 12 apples."
    m = tr.measure_counts(shown)
    assert regs.raw["words"][0, -1] == m["words"] == 5
    assert regs.raw["sentences"][0, -1] == m["sentences"] == 1
    # EQ outside a span, or repeated, adds no bytes.
    odd = ids_of(tok, config.BOS, config.ASSISTANT, "a", config.EQ, "b", config.CALC, "c",
                 config.EQ, "d", config.EQ, "e", config.END)
    r2 = R.compute(odd[None], table=table)
    assert r2.raw["words"][0, -1] == 2                  # the bytes read "abc = de"
    assert field(r2, "calc")[-2] == 2


def test_turn_counters_on_a_formatted_answer(tok, table):
    text = "Steps:\n1. Open (the) box.\n2. Read `a.b` then \"quote\".\n- Done!\n```\n- x. y\n```\n"
    ids = ids_of(tok, config.BOS, config.ASSISTANT, text)
    regs = R.compute(ids[None], table=table)
    m = tr.measure_counts(text, final=False)
    raw = {k: int(regs.raw[k][0, -1]) for k in ("words", "sentences", "lines", "items", "depth")}
    assert raw == {k: m[k] for k in raw}
    # words: Steps 1 Open the box 2 Read a b then quote Done x y; the fence hides "x. y" and "- x"
    assert raw == {"words": 14, "sentences": 3, "lines": 7, "items": 3, "depth": 0}
    assert field(regs, "fence")[-1] == 0 and field(regs, "fence")[-2] == 1


def test_remaining_count_encodings():
    cases_small = [(-1, 5, 0), (3, 0, 5), (3, 3, 2), (3, 4, 1), (100, 0, 33), (33, 2, 33), (32, 0, 33)]
    for target, count, value in cases_small:
        assert R.rem_count(target, count) == value
    cases_word = [(-1, 0, 0), (5, 6, 1), (5, 5, 2), (15, 0, 17), (16, 0, 18), (31, 0, 18),
                  (32, 0, 19), (63, 0, 19), (64, 0, 20), (127, 0, 20), (128, 0, 21), (255, 0, 21),
                  (256, 0, 22), (511, 0, 22), (512, 0, 23), (1023, 0, 23), (1024, 0, 24),
                  (2047, 0, 24), (2048, 0, 25), (10 ** 6, 0, 25), (40, 30, 12)]
    for target, count, value in cases_word:
        assert R.rem_word(target, count) == value
    rng = np.random.default_rng(0)
    target = rng.integers(-3, 5000, 20000)
    count = rng.integers(0, 3000, 20000)
    small, word = R.rem_count(target, count), R.rem_word(target, count)
    assert small.max() < CARDS["rem_sent"] and word.max() < CARDS["rem_word"]
    for i in range(0, 20000, 7):
        assert small[i] == ref_rem_count(int(target[i]), int(count[i]))
        assert word[i] == ref_rem_word(int(target[i]), int(count[i]))


def test_countdowns_only_inside_assistant_turns(tok, table):
    ids = ids_of(tok, config.BOS, config.USER, "Say two sentences.", config.END,
                 config.ASSISTANT, "One. Two. Three.", config.END)    # END flushes the third
    targets = np.tile(np.array([2, 0, 4], np.int32), (1, len(ids), 1))
    regs = R.compute(ids[None], targets=targets, table=table)
    ast = int(np.flatnonzero(ids == config.ASSISTANT)[0])
    assert not regs.reg[0, :ast, -3:].any()
    sent = field(regs, "rem_sent")[ast:]
    assert sent[0] == 4 and sent[-1] == 1 and min(sent) == 1
    words = regs.raw["words"][0, ast:]
    assert field(regs, "rem_word")[ast:] == [ref_rem_word(4, int(w)) for w in words]
    assert field(regs, "rem_item")[ast:] == [2] * len(words)


def test_values_are_clipped_to_their_cardinalities(tok, table):
    many = "A. " * 70 + "\n- x" * 70 + " " + "(" * 12
    body = tok.encode(many)
    ids = np.array([config.BOS] + [config.SYSTEM] * 20 + [config.ASSISTANT] + body
                   + tok.encode(" w") * 4200, np.int32)
    regs = R.compute(ids[None], table=table)
    raw = {k: regs.raw[k][0, -1] for k in ("sentences", "lines", "items", "depth", "turn_idx", "tokens")}
    assert raw["sentences"] == 70 and raw["lines"] == 70 and raw["items"] == 70
    assert raw["depth"] == 12 and raw["turn_idx"] == 21 and raw["tokens"] > 4096
    last = dict(zip(NAMES, regs.reg[0, -1].tolist()))
    assert last["sent_in_turn"] == 63 and last["line_in_turn"] == 63 and last["item_in_turn"] == 63
    assert last["depth"] == 7 and last["turn_idx"] == 15 and last["tok_in_turn"] == 11
    assert last["word_in_sent"] == 63 and regs.raw["word_in_sent"][0, -1] > 4000
    for i, (name, card) in enumerate(REG_FIELDS):
        assert 0 <= regs.reg[..., i].min() and regs.reg[..., i].max() < card, name


def test_shapes_dtypes_and_validation(tok, table):
    tokens, seg, targets = random_batch(5, tok, 3, 64)
    regs = R.compute(tokens, seg, targets, table=table)
    assert regs.reg.shape == (3, 64, config.N_REG) and regs.reg.dtype == np.int16
    assert regs.clock.shape == (3, 64, config.N_CLOCKS) and regs.clock.dtype == np.int32
    assert set(regs.raw) == set(R.RAW_FIELDS) and all(v.dtype == np.int32 for v in regs.raw.values())
    empty = R.compute(np.zeros((2, 0), np.int32), table=table)
    assert empty.reg.shape == (2, 0, config.N_REG)
    with pytest.raises(ValueError):
        R.compute(np.array([[config.VOCAB_SIZE]]), table=table)
    with pytest.raises(ValueError):
        R.compute(tokens, seg[:, :10], table=table)
    with pytest.raises(ValueError):
        R.compute(tokens, seg, targets[..., :2], table=table)
    with pytest.raises(ValueError):
        R.Incremental(2, table=table).step(np.array([1, 2, 3]))


# --------------------------------------------------------------------------- real data
def smoltalk_conversations(limit: int):
    from bes.tokenizer import samples

    (f,) = samples.files_for("chat", "heldout")
    if not f.local_path.exists():
        return []
    out = []
    for row in samples._iter_rows(f.local_path, ["messages"]):
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"] or ()]
        out.append(msgs)
        if len(out) >= limit:
            break
    return out


def test_real_conversations_match_text_level_counts_and_chat_roles(tok, table):
    convs = smoltalk_conversations(80)
    if not convs:
        pytest.skip("samples not downloaded (python scripts/build_tokenizer.py)")
    checked = 0
    for msgs in convs:
        try:
            ids, roles = render_conversation(msgs, tok)
        except ValueError:
            continue
        regs = R.compute(ids[None], table=table)
        assert np.array_equal(regs.raw["role"][0], roles.astype(np.int32))
        ends = np.flatnonzero(ids == config.END)
        assert len(ends) == len(msgs)
        seg_s = seg_l = 0
        for msg, e in zip(msgs, ends):
            m = tr.measure_counts(msg["content"])
            got = {k: int(regs.raw[k][0, e]) for k in ("words", "sentences", "lines", "items", "depth")}
            assert got == {k: m[k] for k in got}
            assert dict(zip(NAMES, regs.reg[0, e].tolist()))["fence"] == m["fence"]
            seg_s, seg_l = seg_s + m["sentences"], seg_l + m["lines"]
            assert regs.clock[0, e, :2].tolist() == [seg_s, seg_l]
            checked += 1
    assert checked >= 100


def test_throughput_per_core(tok, table):
    """Sequences of 1024 tokens per second for one process (numpy here is single-threaded)."""
    rng = random.Random(6)
    texts = [m["content"] for msgs in smoltalk_conversations(400) for m in msgs]
    if not texts:
        texts = ["Some text. With (brackets) and `code`.\n- a list\n"] * 50
    stream = [config.BOS]
    while len(stream) < 32 * 1024:
        stream += [rng.choice(config.ROLE_TOKENS)] + tok.encode(rng.choice(texts))[:400] + [config.END]
    tokens = np.array(stream[: 32 * 1024], np.int32).reshape(32, 1024)
    tokens[:, 0] = config.BOS
    seg = np.zeros_like(tokens)
    R.compute(tokens[:2], seg[:2], table=table)
    best = min(_timed(lambda: R.compute(tokens, seg, table=table)) for _ in range(3))
    rate = 32 / best
    print(f"\nregisters.compute: {rate:.0f} sequences of 1024 tokens per second (B=32)")
    assert rate >= 60


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
