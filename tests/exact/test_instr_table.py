"""Tests of the per-token tables (bes.exact.instr_table).

The reference is the byte-level transducer (bes.exact.transducer.step / flush) run over the
concatenated bytes of the tokens; the tables must reproduce it at every token boundary. The
depth map of a token is checked by applying the per-byte rule to every start depth 0..24.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from bes import config
from bes.exact import instr_table as it
from bes.exact import transducer as tr
from bes.tokenizer.tok import Tokenizer, load

ROLE_NAMES = {"<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|library|>": 4}


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return load()


@pytest.fixture(scope="module")
def table(tok) -> it.InstrTable:
    return it.load(tok)


@pytest.fixture(scope="module")
def machine() -> tr.ByteMachine:
    return tr.compile_machine()


def ref_token(state: tr.State, data: bytes) -> tuple[tr.State, list[int], list[int]]:
    """Reference effect of one byte string: final state, the 6 count fields of INC_FIELDS
    (words, sentences, lines, items, ws_reset, ws_after) and the depth after it for start
    depths 0..24."""
    words = sents = lines = items = reset = after = 0
    deltas = []
    for b in data:
        state, ev = tr.step(state, b)
        if ev.sentence:
            reset, after = 1, 0
        after += ev.word
        words += ev.word
        sents += ev.sentence
        lines += ev.newline
        items += ev.item
        deltas.append(ev.bracket)
    depths = []
    for d in range(25):
        for x in deltas:
            d = max(0, d + x)
        depths.append(d)
    return state, [words, sents, lines, items, reset, after], depths


def table_depths(inc_row: np.ndarray) -> list[int]:
    add, lo = int(inc_row[it.INC_DEP_ADD]), int(inc_row[it.INC_DEP_LO])
    return [max(lo, d + add) for d in range(25)]


def check_rows_exhaustively(table: it.InstrTable, machine: tr.ByteMachine, rows: dict[int, bytes]):
    for si, s0 in enumerate(machine.states):
        for r, data in rows.items():
            s, fields, depths = ref_token(s0, data)
            assert table.next_state[r, si] == machine.index(s), (r, si)
            assert table.inc[r, si, :6].tolist() == fields, (r, si, data)
            assert table_depths(table.inc[r, si]) == depths, (r, si, data)


# ---------------------------------------------------------------------- exhaustive checks
def test_every_ordinary_row_equals_the_byte_reference_from_every_state(tok, table, machine):
    assert table.next_state.shape == (tok.vocab_size + 1, machine.n_states)
    assert table.inc.shape == (tok.vocab_size + 1, machine.n_states, it.N_INC)
    rows = {v: tok.token_bytes(v) for v in range(tok.vocab_size) if not tok.is_special(v)}
    rows[table.eq_row] = b" = "
    check_rows_exhaustively(table, machine, rows)
    # The register layer relies on dep_lo >= dep_add (the depth map at d = 0 is dep_lo).
    assert (table.inc[..., it.INC_DEP_LO] >= table.inc[..., it.INC_DEP_ADD]).all()
    assert (table.inc[..., it.INC_DEP_LO] >= 0).all()


def test_special_rows(tok, table, machine):
    n = machine.n_states
    identity = np.arange(n)
    for name, tid in tok.special_tokens.items():
        if name == "<|bos|>" or name in ROLE_NAMES:
            assert (table.next_state[tid] == 0).all()
            assert not table.inc[tid].any()
        elif name == "<|end|>":
            for si, s in enumerate(machine.states):
                s2, ev = tr.flush(s)
                assert table.next_state[tid, si] == machine.index(s2)
                assert table.inc[tid, si].tolist() == [0, ev.sentence, 0, 0, ev.sentence, 0, 0, 0]
        else:
            assert (table.next_state[tid] == identity).all(), name
            assert not table.inc[tid].any(), name


def test_kinds_and_role_codes(tok, table):
    assert table.kind[config.BOS] == it.KIND_BOS and table.kind[config.END] == it.KIND_END
    assert table.kind[config.CALC] == it.KIND_CALC and table.kind[config.EQ] == it.KIND_EQ
    assert table.kind[config.CALC_END] == it.KIND_CALC_END and table.kind[config.PAD] == it.KIND_OTHER
    for tid, code in zip(config.ROLE_TOKENS, (1, 2, 3, 4)):
        assert table.kind[tid] == it.KIND_ROLE and table.role_code[tid] == code
    assert (table.kind[: tok.n_ordinary] == it.KIND_ORDINARY).all()
    assert np.count_nonzero(table.role_code) == 4
    assert table.special(it.KIND_EQ) == config.EQ and table.eq_row == config.VOCAB_SIZE


# ------------------------------------------------------------------ sequences of tokens
def ref_sequence(rows: list[int], tok, table: it.InstrTable):
    """Byte-level reference over a sequence of rows: state and running totals after each row."""
    state = tr.INIT
    totals = dict(words=0, sentences=0, lines=0, items=0, wis=0, depth=0)
    out = []
    for r in rows:
        if r == table.eq_row:
            data, special = b" = ", None
        elif tok.is_special(r):
            data, special = b"", tok.special_name(r)
        else:
            data, special = tok.token_bytes(r), None
        if special == "<|end|>":
            state, ev = tr.flush(state)
            evs = [ev]
        elif special == "<|bos|>" or special in ROLE_NAMES:
            state, evs = tr.INIT, []
        else:
            evs = []
            for b in data:
                state, ev = tr.step(state, b)
                evs.append(ev)
        for ev in evs:
            if ev.sentence:
                totals["wis"] = 0
            totals["wis"] += ev.word
            totals["words"] += ev.word
            totals["sentences"] += ev.sentence
            totals["lines"] += ev.newline
            totals["items"] += ev.item
            totals["depth"] = max(0, totals["depth"] + ev.bracket)
        out.append((state, dict(totals)))
    return out


def table_sequence(rows: list[int], table: it.InstrTable):
    s_in, s_out, inc = it.run_rows(table, rows)
    totals = dict(words=0, sentences=0, lines=0, items=0, wis=0, depth=0)
    out = []
    for i in range(len(rows)):
        x = inc[i]
        totals["words"] += x[it.INC_WORDS]
        totals["sentences"] += x[it.INC_SENT]
        totals["lines"] += x[it.INC_LINES]
        totals["items"] += x[it.INC_ITEMS]
        totals["wis"] = x[it.INC_WS_AFTER] if x[it.INC_WS_RESET] else totals["wis"] + x[it.INC_WORDS]
        totals["depth"] = max(x[it.INC_DEP_LO], totals["depth"] + x[it.INC_DEP_ADD])
        out.append((int(s_out[i]), {k: int(v) for k, v in totals.items()}))
    return out


def compare_sequence(rows, tok, table, machine):
    ref = ref_sequence(rows, tok, table)
    got = table_sequence(rows, table)
    for i, ((s_ref, t_ref), (s_got, t_got)) in enumerate(zip(ref, got)):
        assert s_got == machine.index(s_ref), (i, rows[: i + 1])
        assert t_got == t_ref, (i, rows[: i + 1])


def interesting_ids(tok) -> list[int]:
    marks = set(b"`\n\"'.!?()[]{}*_-+0123456789 \t") | {0xE2}
    return [v for v in range(tok.n_ordinary) if marks & set(tok.token_bytes(v))]


def test_random_token_sequences_equal_the_byte_reference(tok, table, machine):
    rng = random.Random(0)
    pool = interesting_ids(tok)
    specials = [config.END, config.BOS, config.USER, config.ASSISTANT, config.PAD, config.CALC,
                config.EQ, config.CALC_END, table.eq_row]
    for _ in range(300):
        rows = []
        for _ in range(rng.randint(1, 120)):
            u = rng.random()
            if u < 0.55:
                rows.append(rng.choice(pool))
            elif u < 0.93:
                rows.append(rng.randrange(tok.n_ordinary))
            else:
                rows.append(rng.choice(specials))
        compare_sequence(rows, tok, table, machine)


def test_fences_and_markers_split_across_tokens(tok, table, machine):
    bt = tok.encode("`")[0]
    texts = ["```python\nx = 1\n```\n", "Intro.\n1. one\n2. two\n", "• a\n• b",
             'He said "hi." Then (left).', "Wait... what?! ok", "**Note.** Done", "v3.14 ok."]
    for text in texts:
        data = text.encode()
        compare_sequence(list(data), tok, table, machine)          # one byte per token
        compare_sequence(tok.encode(text), tok, table, machine)
    for k in range(1, 6):                                          # a fence written k ways
        compare_sequence([bt] * k + tok.encode("\ncode\n") + [bt] * 3, tok, table, machine)


def _sample_texts(max_bytes: int) -> list[str]:
    from bes.tokenizer import samples

    texts = []
    for domain in samples.DOMAINS:
        files = samples.files_for(domain, "heldout")
        if not all(f.local_path.exists() for f in files):
            continue
        texts += list(samples.iter_texts(domain, "heldout", max_bytes, doc_cap=6000))
    return texts


def test_real_text_equals_the_byte_reference(tok, table, machine):
    texts = _sample_texts(60_000)
    if not texts:
        pytest.skip("samples not downloaded (python scripts/build_tokenizer.py)")
    checked = 0
    for text in texts:
        ids = tok.encode(text)
        s_in, s_out, inc = it.run_rows(table, ids)
        ref = tr.run(text.encode("utf-8"))
        assert machine.index(ref.state) == s_out[-1] if ids else True
        assert inc[:, it.INC_WORDS].sum() == ref.words
        assert inc[:, it.INC_SENT].sum() == ref.sentences
        assert inc[:, it.INC_LINES].sum() == ref.lines
        assert inc[:, it.INC_ITEMS].sum() == ref.items
        checked += len(ids)
    for text in texts[:40]:                                        # every boundary on some texts
        compare_sequence(tok.encode(text)[:600], tok, table, machine)
    assert checked > 10_000


# ------------------------------------------------------------------------------- caching
def test_build_is_deterministic_and_matches_the_cache(tok, table, tmp_path):
    fresh = it.build(tok)
    path = tmp_path / "t.npz"
    it.save(fresh, path)
    back = it.read(path)
    for t in (table, back):
        assert t.key == fresh.key and t.tokenizer_sha256 == tok.sha256
        for name in ("next_state", "inc", "kind", "role_code", "components"):
            assert np.array_equal(getattr(t, name), getattr(fresh, name)), name
            assert getattr(t, name).dtype == getattr(fresh, name).dtype
    assert fresh.key.startswith(tok.sha256[:16])


def test_load_writes_reads_and_repairs_the_cache(tok, tmp_path, monkeypatch):
    it._LOADED.clear()
    t1 = it.load(tok, cache_dir=tmp_path)
    path = it.cache_path(tok.sha256, tmp_path)
    assert path.exists() and path.name == f"instr_{t1.key}.npz"
    assert not list(tmp_path.glob("*.tmp"))

    it._LOADED.clear()
    def no_build(*args, **kwargs):
        raise AssertionError("the cache should have been read")
    monkeypatch.setattr(it, "build", no_build)
    t2 = it.load(tok, cache_dir=tmp_path)
    assert np.array_equal(t2.next_state, t1.next_state)
    monkeypatch.undo()

    good = path.read_bytes()
    for broken in (b"not a zip file", good[: len(good) // 2]):     # foreign and truncated files
        path.write_bytes(broken)
        it._LOADED.clear()
        t3 = it.load(tok, cache_dir=tmp_path)
        assert np.array_equal(t3.inc, t1.inc)
        assert it.read(path).key == t1.key
    it._LOADED.clear()


def test_cache_key_tracks_the_tokenizer_and_the_transducer(tok, machine):
    key = it.table_key(tok.sha256, machine)
    assert it.table_key("0" * 64, machine) != key
    changed = tr.ByteMachine(machine.states, machine.next_state, machine.events,
                             machine.flush_state, machine.flush_sentence, machine.components,
                             "f" * 64)
    assert it.table_key(tok.sha256, changed) != key


# --------------------------------------------------------------------- a toy vocabulary
def test_toy_tokenizer_tables_are_exact(tok, machine):
    merges = [(0x60, 0x60), (256, 0x60), (0x0A, 0x0A), (0x2E, 0x20), (0x31, 0x2E), (260, 0x20),
              (0x20, 0x22), (0xE2, 0x80), (263, 0xA2), (0x27, 0x73)]
    toy = Tokenizer(merges, tok.pattern)
    t = it.build(toy, machine)
    assert t.vocab_size == 256 + len(merges) + config.N_SPECIAL
    rows = {v: toy.token_bytes(v) for v in range(toy.vocab_size) if not toy.is_special(v)}
    rows[t.eq_row] = b" = "
    check_rows_exhaustively(t, machine, rows)
    assert t.special(it.KIND_BOS) == toy.bos and t.role_code[toy.assistant] == 3
    assert t.key != it.table_key(tok.sha256, machine)
