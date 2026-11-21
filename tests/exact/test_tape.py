"""Tests of the precedent candidate tape (bes.exact.tape) against a brute-force reference.

The reference below follows the definition of ``docs/SPEC.md`` section 2.3 (as made precise in
the ``bes.exact.tape`` docstring) literally, in plain quadratic Python: it compares token
tuples directly and shares no code with the vectorised implementation.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from bes import config
from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, PAD, ROLE_TOKENS,
                        SYSTEM, USER)
from bes.exact import tape

FIRST_SPECIAL = config.VOCAB_SIZE - config.N_SPECIAL
ROLES = tuple(ROLE_TOKENS)
NAMES = [name for name, _ in config.CAND_FEATS]
KIND, COUNT, TOTAL, TYPES, SRC, MLEN = (NAMES.index(n) for n in
                                        ("kind", "count", "total", "types", "src", "mlen"))
SPECIALS = [BOS, SYSTEM, USER, ASSISTANT, LIBRARY, END, CALC, EQ, CALC_END, PAD,
            config.special_id("<|r12|>")]
SAMPLES = Path(__file__).resolve().parents[2] / "data_cache" / "samples"


# ---------------------------------------------------------------------------- reference
def ref_bucket(v: int) -> int:
    if v <= 0:
        return 0
    if v == 1:
        return 1
    if v == 2:
        return 2
    if v <= 4:
        return 3
    if v <= 8:
        return 4
    if v <= 16:
        return 5
    if v <= 64:
        return 6
    return 7


def ref_tape_row(x, seg) -> tuple[np.ndarray, np.ndarray]:
    """Literal brute force for one row: ``(cand_tok int32[T, 16], cand_feat int8[T, 16, 6])``."""
    x = [int(v) for v in x]
    seg = [int(v) for v in seg]
    n = len(x)
    lo = []                                   # first position of the segment of t
    for t in range(n):
        lo.append(t if t == 0 or seg[t] != seg[t - 1] or x[t] == BOS else lo[-1])

    # ok[t][L]: the L tokens ending at t lie in t's segment and hold no role token or BOS
    # after their first token (the context does not cross a turn start).
    ok = [[False] * 34 for _ in range(n)]
    for t in range(n):
        for L in range(1, 34):
            s = t - L + 1
            ok[t][L] = s >= lo[t] and all(x[p] != BOS and x[p] not in ROLES
                                          for p in range(s + 1, t + 1))

    def lcs(t: int, j: int) -> int:           # match length, capped at 32
        L = 0
        while L < 32 and ok[t][L + 1] and ok[j][L + 1] and x[t - L] == x[j - L]:
            L += 1
        return L

    def turn_start(p: int) -> int:
        q = p
        while q > lo[p] and x[q] not in ROLES:
            q -= 1
        return q

    def role(p: int) -> int:
        v = x[turn_start(p)]
        return ROLES.index(v) + 1 if v in ROLES else 0

    cand_tok = np.full((n, 16), -1, np.int32)
    cand_feat = np.zeros((n, 16, 6), np.int8)
    for t in range(n):
        earlier = range(lo[t], t)
        match_len = [lcs(t, j) for j in earlier]
        lstar = max(match_len, default=0)
        heads = []
        for o in (1, 2, 3):
            if ok[t][o]:
                ctx = x[t - o + 1:t + 1]
                m = [j for j in earlier if ok[j][o] and x[j - o + 1:j + 1] == ctx]
                heads.append((o, o, m, 4 * (o - 1)))
        if lstar >= 4:
            heads.append((4, lstar, [j for j, ml in zip(earlier, match_len) if ml == lstar], 12))
        for kind, length, m, base in heads:
            succ = [x[j + 1] for j in m]
            picked: list[tuple[int, int]] = []
            for j in sorted(m, reverse=True)[:16]:
                b = x[j + 1]
                if b >= FIRST_SPECIAL or b in [pb for pb, _ in picked]:
                    continue
                picked.append((b, j))
                if len(picked) == 4:
                    break
            for slot, (b, j) in enumerate(picked):
                src = 5 if role(t) == 3 and turn_start(j) == turn_start(t) else role(j)
                cand_tok[t, base + slot] = b
                cand_feat[t, base + slot] = (kind, ref_bucket(succ.count(b)), ref_bucket(len(m)),
                                             ref_bucket(len(set(succ))), src, ref_bucket(length))
    return cand_tok, cand_feat


def ref_tape(tokens, seg):
    rows = [ref_tape_row(tokens[b], seg[b]) for b in range(len(tokens))]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def assert_matches_reference(tokens, seg):
    out = tape.compute(tokens, seg)
    want_tok, want_feat = ref_tape(tokens, seg)
    for b in range(len(tokens)):
        bad = np.flatnonzero((out.cand_tok[b] != want_tok[b]).any(-1)
                             | (out.cand_feat[b] != want_feat[b]).any((-1, -2)))
        assert bad.size == 0, (f"row {b} position {bad[0]}: got {out.cand_tok[b, bad[0]]} "
                               f"{out.cand_feat[b, bad[0]].tolist()}, want {want_tok[b, bad[0]]} "
                               f"{want_feat[b, bad[0]].tolist()}")
    return out


# ---------------------------------------------------------------------------- generators
def random_rows(rng, n_rows, n_t, alphabet, p_special=0.05, p_seg=0.02, base=0):
    toks = base + rng.integers(0, alphabet, (n_rows, n_t))
    mask = rng.random((n_rows, n_t)) < p_special
    toks[mask] = rng.choice(SPECIALS, int(mask.sum()))
    seg = np.cumsum(rng.random((n_rows, n_t)) < p_seg, axis=1)
    return toks.astype(np.int32), seg.astype(np.int32)


def periodic_rows(rng, n_rows, n_t):
    """Rows with long repeats: matches longer than 32 and match sets larger than 16."""
    rows = []
    for _ in range(n_rows):
        period = int(rng.integers(1, 7))
        motif = rng.integers(0, 40, period)
        row = motif[np.arange(n_t) % period]
        noise = rng.random(n_t) < 0.03
        row[noise] = rng.integers(0, 40, int(noise.sum()))
        rows.append(row)
    toks = np.stack(rows).astype(np.int32)
    toks[:, 0] = BOS
    return toks, np.zeros_like(toks)


def chat_rows(rng, n_rows, n_t, alphabet=6):
    """BOS + turns ROLE content END with content copied between turns (all src values)."""
    toks = np.empty((n_rows, n_t), np.int32)
    seg = np.empty((n_rows, n_t), np.int32)
    for b in range(n_rows):
        row, sg, s = [], [], 0
        while len(row) < n_t:
            row.append(BOS)
            sg.append(s)
            phrase = list(rng.integers(0, alphabet, 8))
            for _ in range(int(rng.integers(1, 6))):
                r = ROLES[int(rng.integers(0, 4))]
                body = list(rng.integers(0, alphabet, int(rng.integers(0, 12))))
                if rng.random() < 0.6:
                    body = phrase[:int(rng.integers(3, 9))] + body
                if r == ASSISTANT and rng.random() < 0.4:
                    body = body + [CALC, 1, 2, EQ, 3, CALC_END] + body[:5]
                turn = [r, *body, END]
                row += turn
                sg += [s] * len(turn)
            s += 1
        toks[b], seg[b] = row[:n_t], sg[:n_t]
    return toks, seg


# ---------------------------------------------------------------------- reference tests
@pytest.mark.parametrize("alphabet", [1, 2, 3, 5, 12])
def test_matches_reference_small_alphabets(alphabet):
    rng = np.random.default_rng(alphabet)
    for n_t in (1, 2, 7, 64, 150):
        toks, seg = random_rows(rng, 3, n_t, alphabet, p_special=0.0, p_seg=0.0)
        assert_matches_reference(toks, seg)


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference_roles_specials_segments(seed):
    rng = np.random.default_rng(100 + seed)
    toks, seg = random_rows(rng, 3, 160, int(rng.integers(2, 6)), p_special=[0.02, 0.08, 0.2][seed % 3],
                            p_seg=[0.0, 0.01, 0.05][seed % 3])
    assert_matches_reference(toks, seg)


def test_matches_reference_last_ordinary_ids():
    rng = np.random.default_rng(5)
    toks, seg = random_rows(rng, 2, 120, 3, p_special=0.1, base=FIRST_SPECIAL - 3)
    assert (toks == FIRST_SPECIAL - 1).any()                       # the last ordinary id occurs
    out = assert_matches_reference(toks, seg)
    assert (out.cand_tok == FIRST_SPECIAL - 1).any()


def test_matches_reference_long_repeats():
    rng = np.random.default_rng(7)
    toks, seg = periodic_rows(rng, 4, 260)
    out = assert_matches_reference(toks, seg)
    kinds = out.cand_feat[..., KIND]
    assert (kinds == 4).any() and (out.cand_feat[..., MLEN][kinds == 4] == 6).any()   # L* = 32
    assert (out.cand_feat[..., TOTAL] == 7).any()                                      # n > 64


def test_matches_reference_chat_structure():
    rng = np.random.default_rng(11)
    toks, seg = chat_rows(rng, 4, 220)
    out = assert_matches_reference(toks, seg)
    srcs = set(np.unique(out.cand_feat[..., SRC][out.cand_tok >= 0]).tolist())
    assert {1, 2, 3, 4, 5} <= srcs


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


def rendered_segments(tok) -> list[np.ndarray]:
    from bes.data import chat_format as cf

    convs = [
        [{"role": "system", "content": "Answer in one sentence."},
         {"role": "user", "content": "What is the capital of France? Please answer briefly."},
         {"role": "assistant", "content": "The capital of France is Paris."},
         {"role": "user", "content": "And the capital of Italy? Please answer briefly."},
         {"role": "assistant", "content": "The capital of Italy is Rome."}],
        [{"role": "library", "content": "Mount Everest\n\nMount Everest is Earth's highest mountain "
                                        "above sea level, located in the Himalayas."},
         {"role": "user", "content": "How high is Mount Everest above sea level?"},
         {"role": "assistant", "content": [{"type": "text", "text": "Mount Everest is Earth's highest "
                                                                    "mountain above sea level: "},
                                           {"type": "calc", "expr": "8848 + 1", "result": "8849"},
                                           {"type": "text", "text": " metres."}]}],
    ]
    segs = [cf.render_conversation(c, tok)[0] for c in convs]
    segs.append(cf.render_document("def f(x):\n    return x + 1\n\ndef g(x):\n    return x + 2\n" * 3, tok)[0])
    return segs


def sample_segments(tok, n_web=12, n_chat=12) -> list[np.ndarray]:
    """Real documents and conversations from the held-out samples (empty when absent)."""
    import pyarrow.parquet as pq
    from bes.data import chat_format as cf

    web = SAMPLES / "karpathy__climbmix-400b-shuffle" / "shard_00997.parquet"
    chat = SAMPLES / "HuggingFaceTB__smol-smoltalk" / "data" / "test-00000-of-00001.parquet"
    segs: list[np.ndarray] = []
    if web.exists():
        batch = next(pq.ParquetFile(str(web)).iter_batches(batch_size=n_web, columns=["text"]))
        segs += [cf.render_document(r["text"][:6000], tok)[0] for r in batch.to_pylist()]
    if chat.exists():
        batch = next(pq.ParquetFile(str(chat)).iter_batches(batch_size=n_chat, columns=["messages"]))
        for r in batch.to_pylist():
            try:
                segs.append(cf.render_conversation(
                    [{"role": m["role"], "content": m["content"]} for m in r["messages"]], tok)[0])
            except ValueError:
                pass
    return segs


def pack_rows(segs, n_rows, n_t, rng) -> tuple[np.ndarray, np.ndarray]:
    """Pack segments into rows that start with BOS; cut segments restart with a fresh BOS."""
    toks = np.full((n_rows, n_t), PAD, np.int32)
    seg = np.zeros((n_rows, n_t), np.int32)
    k = int(rng.integers(0, len(segs)))
    for b in range(n_rows):
        t, s = 0, 0
        while t < n_t:
            ids = np.asarray(segs[k % len(segs)], np.int32)
            k += 1
            start = int(rng.integers(0, max(1, len(ids) - 16))) if rng.random() < 0.3 else 0
            piece = ids[start:start + n_t - t].copy()
            piece[0] = BOS
            toks[b, t:t + len(piece)] = piece
            seg[b, t:t + len(piece)] = s
            t += len(piece)
            s += 1
    return toks, seg


def test_matches_reference_rendered_text(tok):
    rng = np.random.default_rng(3)
    toks, seg = pack_rows(rendered_segments(tok), 3, 200, rng)
    assert_matches_reference(toks, seg)


def test_matches_reference_real_streams(tok):
    segs = sample_segments(tok)
    if not segs:
        pytest.skip("data_cache/samples is not present")
    rng = np.random.default_rng(4)
    toks, seg = pack_rows(segs, 4, 384, rng)
    # A padding segment at the end of the last row (its own id, PAD tokens).
    toks[-1, -40:] = PAD
    seg[-1, -40:] = seg[-1].max() + 1
    out = assert_matches_reference(toks, seg)
    assert (out.cand_feat[..., KIND] == 4).any()


# ------------------------------------------------------------------- incremental, forks
def test_incremental_equals_batch():
    rng = np.random.default_rng(21)
    batches = [random_rows(rng, 3, 300, 3, p_special=0.08, p_seg=0.01), chat_rows(rng, 3, 300),
               periodic_rows(rng, 2, 300)]
    for toks, seg in batches:
        out = tape.compute(toks, seg)
        for b in range(len(toks)):
            inc = tape.Incremental().extend(toks[b], seg[b])
            np.testing.assert_array_equal(inc.cand_tok, out.cand_tok[b])
            np.testing.assert_array_equal(inc.cand_feat, out.cand_feat[b])


def test_incremental_equals_batch_real_streams(tok):
    segs = sample_segments(tok) or rendered_segments(tok)
    rng = np.random.default_rng(22)
    toks, seg = pack_rows(segs, 3, 1024, rng)
    out = tape.compute(toks, seg)
    for b in range(len(toks)):
        inc = tape.Incremental()
        for t in range(toks.shape[1]):           # step() one token at a time
            new = t > 0 and seg[b, t] != seg[b, t - 1]
            ct, cf_ = inc.step(int(toks[b, t]), new_segment=new)
            assert np.array_equal(ct, out.cand_tok[b, t]) and np.array_equal(cf_, out.cand_feat[b, t]), (b, t)


def test_incremental_without_seg_uses_bos():
    rng = np.random.default_rng(23)
    toks, seg = chat_rows(rng, 2, 250)
    want = tape.compute(toks, seg)                 # every segment of chat_rows starts with BOS
    np.testing.assert_array_equal(tape.compute(toks, None).cand_tok, want.cand_tok)
    for b in range(2):
        got = tape.Incremental().extend(toks[b])
        np.testing.assert_array_equal(got.cand_feat, want.cand_feat[b])


def test_fork_diverges_independently():
    rng = np.random.default_rng(24)
    toks, _ = chat_rows(rng, 1, 400)
    row = toks[0].tolist()
    prefix, cont_a = row[:250], row[250:330]
    cont_b = [int(v) for v in rng.integers(0, 6, 80)]
    cont_c = row[250:290] + [ASSISTANT] + row[290:320]
    parent = tape.Incremental()
    parent.extend(prefix)
    child = parent.fork()
    grandchild = child.fork()
    got_a = [parent.step(v) for v in cont_a]
    got_b = [child.step(v) for v in cont_b]
    got_c = [grandchild.step(v) for v in cont_c]
    for cont, got in ((cont_a, got_a), (cont_b, got_b), (cont_c, got_c)):
        want = tape.Incremental().extend(prefix + cont)
        for i, (ct, cf_) in enumerate(got):
            np.testing.assert_array_equal(ct, want.cand_tok[len(prefix) + i])
            np.testing.assert_array_equal(cf_, want.cand_feat[len(prefix) + i])


# --------------------------------------------------------------- causality and locality
def test_causal_and_segment_local():
    rng = np.random.default_rng(31)
    for toks, seg in (random_rows(rng, 2, 200, 3, p_special=0.05, p_seg=0.02), chat_rows(rng, 2, 200)):
        base = tape.compute(toks, seg)
        for _ in range(12):
            b, j = int(rng.integers(0, 2)), int(rng.integers(0, toks.shape[1]))
            mod = toks.copy()
            mod[b, j] = (int(mod[b, j]) + 1 + int(rng.integers(0, 4))) % 8      # an ordinary id
            out = tape.compute(mod, seg)
            np.testing.assert_array_equal(out.cand_tok[b, :j], base.cand_tok[b, :j])
            np.testing.assert_array_equal(out.cand_feat[b, :j], base.cand_feat[b, :j])
            # positions of other segments, and other rows, never change
            lay_seg = np.cumsum((np.diff(seg[b], prepend=-1) != 0) | (toks[b] == BOS))
            other = lay_seg != lay_seg[j]
            np.testing.assert_array_equal(out.cand_feat[b, other], base.cand_feat[b, other])
            np.testing.assert_array_equal(out.cand_tok[1 - b], base.cand_tok[1 - b])


def test_rows_are_independent_of_batch_composition():
    rng = np.random.default_rng(32)
    toks, seg = random_rows(rng, 5, 180, 4, p_special=0.05, p_seg=0.02)
    whole = tape.compute(toks, seg)
    for b in range(5):
        single = tape.compute(toks[b:b + 1], seg[b:b + 1])
        np.testing.assert_array_equal(single.cand_feat[0], whole.cand_feat[b])
        np.testing.assert_array_equal(single.cand_tok[0], whole.cand_tok[b])


def test_chunking_does_not_change_results(monkeypatch):
    rng = np.random.default_rng(33)
    toks, seg = random_rows(rng, 7, 90, 3, p_special=0.05, p_seg=0.02)
    want = tape.compute(toks, seg)
    monkeypatch.setattr(tape, "CHUNK_TOKENS", 100)                              # one row per chunk
    got = tape.compute(toks, seg)
    np.testing.assert_array_equal(got.cand_tok, want.cand_tok)
    np.testing.assert_array_equal(got.cand_feat, want.cand_feat)


# -------------------------------------------------------------------------- invariants
def check_invariants(out):
    ct, cf_ = out.cand_tok, out.cand_feat
    assert ct.dtype == np.int32 and cf_.dtype == np.int8
    assert ct.shape[-1] == config.CAND_SLOTS and cf_.shape[-1] == config.N_CAND_FEAT
    empty = ct < 0
    assert (ct[~empty] < FIRST_SPECIAL).all()                      # never a special token
    assert (ct[empty] == -1).all() and (cf_[empty] == 0).all()     # empty slots are all zero
    assert ((cf_[..., KIND] == 0) == empty).all()
    for f, (_, card) in enumerate(config.CAND_FEATS):
        assert cf_[..., f].min() >= 0 and cf_[..., f].max() < card
    heads = ct.reshape(*ct.shape[:-1], 4, 4)
    kinds = cf_[..., KIND].reshape(*ct.shape[:-1], 4, 4)
    filled = heads >= 0
    assert (filled[..., 1:] <= filled[..., :-1]).all()              # left-aligned in each head
    assert ((kinds == np.arange(1, 5)[:, None]) | ~filled).all()    # slot group -> kind
    a, b = heads[..., :, None], heads[..., None, :]
    dup = (a == b) & (a >= 0) & ~np.eye(4, dtype=bool)
    assert not dup.any()                                           # distinct within a head


def test_invariants_on_all_generators(tok):
    rng = np.random.default_rng(41)
    segs = sample_segments(tok) or rendered_segments(tok)
    for toks, seg in (random_rows(rng, 4, 400, 3, p_special=0.1, p_seg=0.02), chat_rows(rng, 4, 400),
                      periodic_rows(rng, 4, 400), pack_rows(segs, 8, 1024, rng)):
        check_invariants(tape.compute(toks, seg))


def test_bucket_boundaries():
    values = np.arange(0, 300)
    np.testing.assert_array_equal(tape.bucket(values), [ref_bucket(int(v)) for v in values])
    edges = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 8: 4, 9: 5, 16: 5, 17: 6, 64: 6, 65: 7, 10**6: 7}
    for v, want in edges.items():
        assert tape.bucket(v) == want and tape.bucket_int(v) == want


# ---------------------------------------------------------------- constructed sequences
def row_of(*tokens):
    return np.asarray([tokens], np.int32)


def test_sixteen_most_recent_rule():
    # 17 earlier occurrences of 5: the oldest is followed by 7, the 16 most recent by 6.
    out = tape.compute(row_of(BOS, 5, 7, *([5, 6] * 16), 5))
    last_tok, last = out.cand_tok[0, -1], out.cand_feat[0, -1]
    assert last_tok[:4].tolist() == [6, -1, -1, -1]                # 7 lies outside the window
    assert last[0].tolist() == [1, ref_bucket(16), ref_bucket(17), ref_bucket(2), 0, 1]
    # With 16 occurrences the oldest successor is inside the window and is offered.
    out = tape.compute(row_of(BOS, 5, 7, *([5, 6] * 15), 5))
    assert out.cand_tok[0, -1, :4].tolist() == [6, 7, -1, -1]
    assert out.cand_feat[0, -1, 1].tolist() == [1, 1, ref_bucket(16), 2, 0, 1]


def test_special_successors_count_but_are_never_offered():
    out = tape.compute(row_of(BOS, 5, 9, *([5, END] * 16), 5))
    assert (out.cand_tok[0, -1, :4] == -1).all()                   # window holds only END
    out = tape.compute(row_of(BOS, 5, 9, 5, END, 5, CALC, 5))
    assert out.cand_tok[0, -1, :4].tolist() == [9, -1, -1, -1]
    assert out.cand_feat[0, -1, 0].tolist() == [1, 1, ref_bucket(3), ref_bucket(3), 0, 1]


def test_many_types():
    seq = [BOS]
    for v in range(1, 71):
        seq += [0, v]
    out = tape.compute(row_of(*seq, 0))
    assert out.cand_tok[0, -1, :4].tolist() == [70, 69, 68, 67]
    assert out.cand_feat[0, -1, 0].tolist() == [1, 1, 7, 7, 0, 1]   # n = K = 70


def test_context_never_crosses_a_turn_start():
    a, b, c, d, e, f = 11, 12, 13, 14, 15, 16
    seq = [BOS, USER, a, b, c, d, e, f, END, ASSISTANT, a, b, c, d, e]
    out = tape.compute(row_of(*seq))
    t = len(seq) - 1
    feat, cand = out.cand_feat[0, t], out.cand_tok[0, t]
    # orders 1-3 and the long match (L* = 5: the role tokens differ) all offer f from the user turn
    assert cand[[0, 4, 8, 12]].tolist() == [f, f, f, f]
    assert feat[12].tolist() == [4, 1, 1, 1, 2, ref_bucket(5)]
    assert feat[0, SRC] == 2
    # At the first content token after ASSISTANT, the 2-gram [ASSISTANT, a] has no match but
    # [a] does; the 3-gram would cross the role token and does not exist.
    t1 = seq.index(ASSISTANT) + 1
    assert out.cand_tok[0, t1, [0, 4, 8, 12]].tolist() == [b, -1, -1, -1]
    # [USER] and [ASSISTANT] are contexts of their own: the second USER sees what followed the first.
    out = tape.compute(row_of(BOS, USER, a, END, ASSISTANT, b, END, USER))
    assert out.cand_tok[0, -1, :4].tolist() == [a, -1, -1, -1]
    assert out.cand_feat[0, -1, 0, SRC] == 2


def test_self_generated_evidence():
    x, y, z = 21, 22, 23
    seq = [BOS, USER, x, y, END, ASSISTANT, x, y, z, x, y]
    out = tape.compute(row_of(*seq))
    # at the last y: [y] was followed by END in the user turn and by z in this assistant turn
    assert out.cand_tok[0, -1, :4].tolist() == [z, -1, -1, -1]
    assert out.cand_feat[0, -1, 0, SRC] == 5
    assert out.cand_feat[0, -1, 0, TYPES] == ref_bucket(2) and out.cand_feat[0, -1, 0, TOTAL] == 2
    # at the second x of the assistant turn: successors y (assistant turn, most recent) and y (user)
    t = len(seq) - 2
    assert out.cand_tok[0, t, :4].tolist() == [y, -1, -1, -1]
    assert out.cand_feat[0, t, 0, SRC] == 5 and out.cand_feat[0, t, 0, COUNT] == 2
    # a later assistant turn sees the earlier one as role 3, not 5
    out = tape.compute(row_of(*seq, END, USER, x, END, ASSISTANT, z))
    assert out.cand_tok[0, -1, :4].tolist() == [x, -1, -1, -1]
    assert out.cand_feat[0, -1, 0, SRC] == 3


def test_long_match_is_capped_at_32():
    phrase = list(range(100, 140))
    out = tape.compute(row_of(BOS, *phrase, 7, *phrase))
    feat = out.cand_feat[0]
    t_first_long = 1 + len(phrase) + 1 + 3                       # 4th token of the repeat
    assert feat[t_first_long, 12, KIND] == 4 and feat[t_first_long, 12, MLEN] == ref_bucket(4)
    assert feat[t_first_long - 1, 12, KIND] == 0                  # L* = 3 there
    assert feat[-1, 12].tolist() == [4, 1, 1, 1, 0, ref_bucket(32)]
    assert out.cand_tok[0, -2, 12] == phrase[-1]


def test_padding_segment_is_empty():
    toks = row_of(BOS, 1, 2, 1, 2, 1, PAD, PAD, PAD, PAD)
    seg = np.asarray([[0] * 6 + [1] * 4])
    out = tape.compute(toks, seg)
    assert (out.cand_tok[0, 6:] == -1).all() and (out.cand_feat[0, 6:] == 0).all()
    assert out.cand_tok[0, 5, 0] == 2


def test_shapes_and_errors():
    assert tape.compute(np.zeros((0, 5), np.int32)).cand_tok.shape == (0, 5, 16)
    assert tape.compute(np.zeros((3, 0), np.int32)).cand_feat.shape == (3, 0, 16, 6)
    one = tape.compute(row_of(BOS))
    assert one.cand_tok.shape == (1, 1, 16) and (one.cand_tok == -1).all()
    with pytest.raises(ValueError):
        tape.compute(np.array([[1, config.VOCAB_SIZE]]))
    with pytest.raises(ValueError):
        tape.compute(np.array([[1, -1]]))
    with pytest.raises(ValueError):
        tape.compute(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        tape.compute(np.array([[1, 2]]), np.array([[0, 0, 0]]))
    with pytest.raises(ValueError):
        tape.Incremental().step(config.VOCAB_SIZE)


# ------------------------------------------------------------------------- throughput
def test_throughput(tok):
    """At least 100 rows of 1024 tokens per second on one core (tape alone)."""
    segs = sample_segments(tok, 40, 40) or rendered_segments(tok)
    rng = np.random.default_rng(51)
    toks, seg = pack_rows(segs, 32, 1024, rng)
    tape.compute(toks, seg)
    best = min(_timed(tape.compute, toks, seg) for _ in range(3))
    rate = toks.shape[0] / best
    print(f"tape throughput: {rate:.0f} rows of 1024 tokens per second ({best * 1e3:.1f} ms per 32 rows)")
    assert rate >= 100


def _timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0
