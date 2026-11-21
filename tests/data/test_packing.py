"""Tests of bes.data.packing against brute-force references written here: row assembly and
the weight rule, base-row filling, best fit, row kinds, deficit scheduling, epoch partitions,
the step layout, and the padding share of chat rows."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, PAD, SYSTEM, USER,
                        VOCAB_SIZE)
from bes.data import packing as P
from bes.data.samples import WeightRule

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROLE_TOKEN = {SYSTEM: 1, USER: 2, ASSISTANT: 3, LIBRARY: 4}


# ------------------------------------------------------------------------ references
def ref_weights(ids, roles, rule: WeightRule) -> list[float]:
    """Per-token weight rule (samples module docstring), one token at a time."""
    out = []
    in_result = False                  # inside a calculator result (after EQ, before CALC_END)
    in_span = False
    prev_tok, prev_role = None, None
    for t, r in zip([int(x) for x in ids], [int(x) for x in roles]):
        if t in (CALC,):
            in_span, in_result = True, False
        elif t == EQ and in_span:
            pass
        elif t >= VOCAB_SIZE - 16 and t not in (EQ, CALC_END) and in_span:
            in_span = in_result = False     # any other special token ends an open span
        if t in (BOS, PAD):
            w = 0.0
        elif r == 0:
            w = 1.0
        elif r == 3:
            w = 1.0
        elif r in (1, 2):
            w = 0.0 if rule.assistant_only else 1.0
        elif r == 4:
            w = 0.0 if rule.assistant_only else rule.passage_w
        else:
            w = 0.0
        if t == ASSISTANT and prev_tok == END and prev_role == 2:
            w = 0.0
        if in_result:
            w = 0.0
        if t == EQ and in_span and not in_result:
            in_result = True                # tokens after this EQ are forced by the host
        if t == CALC_END and in_span:
            in_span = in_result = False
        out.append(w)
        prev_tok, prev_role = t, r
    return out


def random_conversation(rng) -> tuple[list[int], list[int]]:
    """Token ids and roles of a random conversation shape (no tokenizer needed)."""
    ids, roles = [BOS], [0]
    if rng.random() < 0.3:
        n = int(rng.integers(1, 6))
        ids += [SYSTEM, *rng.integers(0, 8000, n).tolist(), END]
        roles += [1] * (n + 2)
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.3:
            n = int(rng.integers(1, 6))
            ids += [LIBRARY, *rng.integers(0, 8000, n).tolist(), END]
            roles += [4] * (n + 2)
        n = int(rng.integers(1, 6))
        ids += [USER, *rng.integers(0, 8000, n).tolist(), END]
        roles += [2] * (n + 2)
        body = rng.integers(0, 8000, int(rng.integers(1, 6))).tolist()
        if rng.random() < 0.5:
            body += [CALC, *rng.integers(0, 8000, 2).tolist(), EQ, *rng.integers(0, 8000, 2).tolist()]
            if rng.random() < 0.7:
                body += [CALC_END]
            body += rng.integers(0, 8000, int(rng.integers(0, 3))).tolist()
        ids += [ASSISTANT, *body, END]
        roles += [3] * (len(body) + 2)
    return ids, roles


def random_document(rng, lo=2, hi=40) -> tuple[list[int], list[int]]:
    n = int(rng.integers(lo, hi))
    ids = [BOS, *rng.integers(0, 8000, n - 1).tolist()]
    return ids, [0] * n


# ------------------------------------------------------------------- row assembly
@pytest.mark.parametrize("assistant_only", [False, True])
def test_assemble_row_matches_a_per_position_reference(assistant_only):
    rng = np.random.default_rng(1)
    rule = WeightRule(assistant_only=assistant_only, passage_w=0.25)
    for trial in range(300):
        T = int(rng.integers(8, 200))
        pieces, used = [], 0
        while True:
            ids, roles = random_conversation(rng) if rng.random() < 0.5 else random_document(rng)
            if used + len(ids) > T + 1:
                break
            pieces.append((np.array(ids, np.int32), np.array(roles, np.int8)))
            used += len(ids)
        if not pieces:
            pieces = [(np.array([BOS, 5], np.int32), np.zeros(2, np.int8))]
            used = 2
        row = P.assemble_row(pieces, T, rule)
        assert row.tokens.shape == (T + 1,) and row.seg.shape == (T,) and row.w.shape == (T,)
        assert row.tokens.dtype == np.int32 and row.seg.dtype == np.int32 and row.w.dtype == np.float32
        assert row.n_pad == T + 1 - used and row.lengths == tuple(len(p[0]) for p in pieces)
        # reference: positions of every piece, then padding
        tok_ref, seg_ref, w_tok = [], [], []
        for k, (ids, roles) in enumerate(pieces):
            tok_ref += ids.tolist()
            seg_ref += [k] * len(ids)
            w_tok += ref_weights(ids, roles, rule)
        tok_ref += [PAD] * (T + 1 - used)
        seg_ref += [len(pieces)] * (T + 1 - used)
        w_tok += [0.0] * (T + 1 - used)
        assert row.tokens.tolist() == tok_ref
        assert row.seg.tolist() == seg_ref[:T]
        np.testing.assert_array_equal(row.w, np.array(w_tok[1:], np.float32))
        assert row.tokens[0] == BOS


def test_assemble_row_rejects_bad_pieces():
    with pytest.raises(ValueError):
        P.assemble_row([(np.array([5, 6]), np.zeros(2, np.int8))], 4)
    with pytest.raises(ValueError):
        P.assemble_row([(np.array([BOS, 6, 7, 8, 9, 10]), np.zeros(6, np.int8))], 4)


# ------------------------------------------------------------------------ base rows
def test_fill_base_cuts_documents_and_continues_them_with_a_fresh_bos():
    rng = np.random.default_rng(2)
    for trial in range(60):
        T1 = int(rng.integers(2, 120))
        docs = [np.array(random_document(rng, 2, 300)[0], np.int32) for _ in range(int(rng.integers(1, 30)))]
        it = iter(range(len(docs)))

        def next_doc():
            k = next(it, None)
            return None if k is None else P.Carry(k, docs[k], 1)

        carry, rows = None, []
        while True:
            placed, carry = P.fill_base(T1, carry, next_doc)
            if not placed:
                break
            rows.append(placed)
        # every row: pieces start with BOS, fill exactly T1 unless one position or no data is left
        got: dict[int, list[int]] = {}
        order = []
        for r, placed in enumerate(rows):
            total = sum(len(ids) for _, ids in placed)
            assert total <= T1
            last = r == len(rows) - 1
            assert total == T1 or last or T1 - total == 1
            for key, ids in placed:
                assert ids[0] == BOS and len(ids) >= 2
                if key not in got:
                    got[key] = []
                    order.append(key)
                got[key] += ids[1:].tolist()
        assert order == list(range(len(docs)))         # in order, none skipped
        for k, d in enumerate(docs):
            assert got[k] == d[1:].tolist()             # every token exactly once, in order
        # continuation: when a row's last piece is not the end of its document, the next row
        # starts with the same document
        seen: dict[int, int] = {}
        for r, placed in enumerate(rows):
            for j, (key, ids) in enumerate(placed):
                seen[key] = seen.get(key, 0) + len(ids) - 1
            key = placed[-1][0]
            if seen[key] < len(docs[key]) - 1 and r + 1 < len(rows):
                assert rows[r + 1][0][0] == key


def test_carry_piece_limits():
    ids = np.array([BOS, 10, 11, 12, 13], np.int32)
    c = P.Carry("d", ids, 3)
    assert c.remaining == 3 and c.piece().tolist() == [BOS, 12, 13]
    assert c.piece(2).tolist() == [BOS, 12]
    assert P.Carry("d", ids, 1).piece(3).tolist() == [BOS, 10, 11]


# ------------------------------------------------------------------------ best fit
def test_best_fit_against_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        lengths = rng.integers(1, 100, int(rng.integers(0, 12))).tolist()
        room = int(rng.integers(0, 120))
        fits = [(n, -i) for i, n in enumerate(lengths) if n <= room]
        expect = None if not fits else -max(fits)[1]
        assert P.best_fit(lengths, room) == expect


def test_pack_best_fit_places_the_best_fit_every_time():
    rng = np.random.default_rng(4)
    for trial in range(200):
        L, room0 = int(rng.integers(1, 10)), int(rng.integers(5, 300))
        future = rng.integers(1, room0 + 1, 400).tolist()
        it_impl, it_ref = iter(future[L:]), iter(future[L:])
        buf, ref = future[:L], list(future[:L])
        for _ in range(20):
            first = int(rng.integers(len(buf))) if buf and rng.random() < 0.3 else None
            placed = P.pack_best_fit(buf, room0, lambda x: x, lambda: next(it_impl, None), first)
            out, room = [], room0           # reference: the longest that fits, oldest first

            def take(i: int) -> int:
                x = ref.pop(i)
                out.append(x)
                nxt = next(it_ref, None)
                if nxt is not None:
                    ref.append(nxt)
                return x

            if first is not None:
                room -= take(first)
            while True:
                cand = [(n, -i) for i, n in enumerate(ref) if n <= room]
                if not cand:
                    break
                room -= take(-max(cand)[1])
            assert placed == out and buf == ref and sum(placed) <= room0


def test_oldest_waiting():
    assert P.oldest_waiting([5, 3, 3, 9], now=10, max_wait=7) == 1
    assert P.oldest_waiting([5, 3, 9], now=10, max_wait=8) is None
    assert P.oldest_waiting([0, 1], now=100, max_wait=0) is None


# ------------------------------------------------------------------------ row kinds
def ref_row_kind(v: int, n: int, V: int, shares) -> int:
    """frac(n * (W / 2^64) + rev(v) / V) against the cumulative shares, in exact rationals."""
    from fractions import Fraction

    bits = V.bit_length() - 1
    rev = int(format(v, f"0{bits}b")[::-1], 2) if bits else 0
    u = (Fraction(n * P.WEYL64, 2 ** 64) + Fraction(rev, V)) % 1
    total = sum(Fraction(s) for s in shares)
    acc = Fraction(0)
    last = max(i for i, s in enumerate(shares) if s > 0)
    for k, s in enumerate(shares):
        acc += Fraction(s) / total
        if u < acc or k == last:
            return k
    return len(shares) - 1


def test_row_kind_matches_exact_rational_reference():
    rng = np.random.default_rng(5)
    for _ in range(40):
        V = 2 ** int(rng.integers(0, 7))
        shares = rng.dirichlet(np.ones(3)).tolist()
        if rng.random() < 0.3:
            shares[int(rng.integers(3))] = 0.0
        cum = P.thresholds(shares)
        for _ in range(50):
            v, n = int(rng.integers(V)), int(rng.integers(0, 10 ** 7))
            assert P.row_kind(v, n, V, cum) == ref_row_kind(v, n, V, shares)


@pytest.mark.parametrize("V", [8, 64])
def test_row_kinds_are_balanced_in_every_aligned_block_of_streams(V):
    """2^k streams of an aligned block at the same row number hold 2^k * share +- 1 rows of
    each kind (brute-force count over many row numbers)."""
    rng = np.random.default_rng(6)
    for _ in range(6):
        shares = rng.dirichlet(np.ones(3))
        cum = P.thresholds(shares.tolist())
        for size in (4, 8, 16, 32):
            if size > V:
                continue
            for n in rng.integers(0, 10 ** 6, 30).tolist():
                for c in range(V // size):
                    kinds = [P.row_kind(v, n, V, cum) for v in range(c * size, (c + 1) * size)]
                    for k in range(3):
                        assert abs(kinds.count(k) - size * shares[k]) < 1 + 1e-9


def test_row_kinds_of_one_stream_follow_the_shares():
    rng = np.random.default_rng(7)
    for _ in range(10):
        shares = rng.dirichlet(np.ones(3))
        cum = P.thresholds(shares.tolist())
        v, V = int(rng.integers(64)), 64
        counts = np.zeros(3)
        for n in range(1, 4001):
            counts[P.row_kind(v, n - 1, V, cum)] += 1
            if n in (10, 100, 1000, 4000):
                assert np.all(np.abs(counts - n * shares) <= 2 + math.log(n))


def test_thresholds_skip_kinds_without_share():
    cum = P.thresholds([0.0, 0.5, 0.0])
    assert all(P.row_kind(v, n, 8, cum) == 1 for v in range(8) for n in range(50))
    with pytest.raises(ValueError):
        P.thresholds([0.0, 0.0])
    with pytest.raises(ValueError):
        P.stream_bits(12)


# ------------------------------------------------------------------------- deficits
def test_deficit_pick_keeps_every_source_near_its_share():
    """Brute-force simulation: over-placement is at most the longest item (+1), under-placement
    at most (K - 1) times it (+1), at every prefix; ties go to the first index."""
    rng = np.random.default_rng(8)
    assert P.deficit_pick([0.5, 0.5], [0, 0]) == 0
    assert P.deficit_pick([0.0, 1.0], [0, 10 ** 6]) == 1
    for _ in range(100):
        K = int(rng.integers(1, 7))
        shares = rng.dirichlet(np.ones(K))
        if K > 1 and rng.random() < 0.3:
            shares[0] = 0.0
            shares /= shares.sum()
        lmax = int(rng.integers(1, 500))
        sh = [float(x) for x in shares]
        placed = [0] * K
        for _ in range(400):
            k = P.deficit_pick(sh, placed)
            total = float(sum(placed))
            d = [s * (total + 1.0) - float(p) for s, p in zip(sh, placed)]
            ok = [i for i in range(K) if sh[i] > 0]
            assert k == min(i for i in ok if d[i] == max(d[j] for j in ok))
            placed[k] += int(rng.integers(1, lmax + 1))
            total = float(sum(placed))
            for s, p in zip(sh, placed):
                assert p - s * total <= lmax + 1 + 1e-6
                assert s * total - p <= (K - 1) * lmax + 1 + 1e-6
            assert sum(p for s, p in zip(sh, placed) if s == 0) == 0


# ---------------------------------------------------------------------------- epochs
def test_epoch_positions_cover_each_position_once():
    for V in (1, 2, 8, 64):
        for N in list(range(1, 140)) + [1000, 4097]:
            got = []
            for v in range(V):
                share = P.epoch_share(v, V, N)
                got += [P.epoch_position(v, j, V, N) for j in range(share)]
            if N >= V:
                assert sorted(got) == list(range(N))
            else:
                assert got == [v % N for v in range(V)]
    assert P.epoch_share(3, 8, 0) == 0
    with pytest.raises(ValueError):
        P.epoch_position(0, 0, 8, 0)


# ----------------------------------------------------------------------- step layout
def test_step_rows_deals_rows_round_robin():
    for V in (1, 4, 8, 64):
        for A, B in ((2, 8), (8, 32), (8, 16), (4, 4)):
            m = 0
            seen_n: dict[int, list[int]] = {}
            for step in range(3):
                lay = P.step_rows(m, A, B, V)
                cells = [(a, b) for rows in lay.values() for _, a, b in rows]
                assert sorted(cells) == [(a, b) for a in range(A) for b in range(B)]
                for v, rows in lay.items():
                    for n, a, b in rows:
                        assert (m + a * B + b) % V == v and (m + a * B + b) // V == n
                        seen_n.setdefault(v, []).append(n)
                m += A * B
            for v, ns in seen_n.items():
                assert ns == list(range(ns[0], ns[0] + len(ns)))   # consecutive rows per stream


# --------------------------------------------------------------------- chat padding
def simulate_padding(lengths, T1: int, L: int, max_wait: int) -> tuple[float, int]:
    it = iter(lengths)
    row = [0]
    buf: list[list[int]] = []

    def refill():
        x = next(it, None)
        return None if x is None else [x, row[0]]

    pad = rows = wait = 0
    while True:
        while len(buf) < L:
            x = refill()
            if x is None:
                break
            buf.append(x)
        if not buf:
            break
        first = P.oldest_waiting([b[1] for b in buf], row[0], max_wait)
        placed = P.pack_best_fit(buf, T1, lambda b: b[0], refill, first)
        wait = max([wait] + [row[0] - b[1] for b in placed])
        pad += T1 - sum(b[0] for b in placed)
        rows += 1
        row[0] += 1
        if rows >= 1500:
            break
    return pad / (rows * T1), wait


def test_chat_padding_of_a_mixed_length_distribution_is_low():
    """Lengths like a chat-row mix (long conversations cut to T+1, short tasks): padding well
    under 6% of the chat-row tokens at T=1024 with the default look-ahead, waits bounded."""
    from bes.config import LoaderCfg

    rng = np.random.default_rng(9)
    lc = LoaderCfg()
    for T1 in (1025, 2049):
        long = np.minimum(rng.lognormal(6.3, 0.8, 4000).astype(int) + 20, T1)
        short = rng.integers(30, 260, 3000)
        lengths = np.concatenate([long, short])
        rng.shuffle(lengths)
        pad, wait = simulate_padding(lengths.tolist(), T1, lc.lookahead, lc.max_wait)
        assert pad < 0.06, (T1, pad)
        assert wait <= lc.max_wait + lc.lookahead + 2


def test_chat_padding_on_real_conversations():
    """smoltalk_constraints conversations (when the shard exists): measured padding share."""
    from bes.config import LoaderCfg
    from bes.data.samples import conversation_cuts
    from bes.data.shards import ShardSet

    d = ROOT / "data_cache" / "tok" / "smoltalk_constraints"
    if not (d / "manifest.jsonl").exists():
        pytest.skip("no real chat shard")
    ss = ShardSet(d)
    rng = np.random.default_rng(10)
    lengths = []
    for j in rng.permutation(len(ss))[:6000].tolist():
        cuts = conversation_cuts(np.asarray(ss.segment(j)))
        cuts = cuts[cuts <= 1025]
        if cuts.size:
            lengths.append(int(cuts[-1]))
    lc = LoaderCfg()
    pad, _ = simulate_padding(lengths, 1025, lc.lookahead, lc.max_wait)
    print(f"padding share at T=1024 on smoltalk_constraints: {pad:.4f}")
    assert pad < 0.06
