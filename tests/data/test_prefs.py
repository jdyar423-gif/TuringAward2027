"""Preference pairs and their batches (``bes/data/prefs.py``).

References are built inside the tests: row contents by list concatenation, weights from the
definition "target t is a response token or the closing END", features row by row, and pair
shards through the real tokenisation pipeline (parquet -> ``tokenize_worker`` -> shard).
"""
from __future__ import annotations

import numpy as np
import pytest

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, PAD, SYSTEM, USER,
                        PrefCfg)
from bes.data import chat_format as cf
from bes.data import download as D
from bes.data import prefs as P
from bes.data import shards as SH
from bes.data import tokenize_worker as W
from bes.exact.features import exact_features


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


def prompt_ids(tok, text: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}]
    return cf.render_conversation(msgs, tok)[0].tolist()


def ref_row(prompt, resp, T, forced=()):
    """Independent reference of one row: list concatenation and the weight definition."""
    content = list(prompt) + [ASSISTANT] + list(resp) + [END]
    L = len(content)
    tokens = content + [PAD] * (T + 1 - L)
    resp_targets = set(range(len(prompt) + 1, L))           # content positions of resp + END
    forced_targets = {len(prompt) + 1 + i for i in forced}
    w = [1.0 if (t + 1) in resp_targets and (t + 1) not in forced_targets else 0.0
         for t in range(T)]
    seg = [0 if t < L else 1 for t in range(T)]
    return np.array(tokens, np.int32), np.array(w, np.float32), np.array(seg, np.int32)


# ------------------------------------------------------------------------------ rows
def test_row_layout_matches_the_definition(tok):
    T = 48
    prompt = prompt_ids(tok, "Name a colour.", system="Be brief.")
    resp = tok.encode("Blue, like the sky.")
    tokens, w, seg = P.response_row(np.array(prompt), np.array(resp), T)
    rt, rw, rs = ref_row(prompt, resp, T)
    np.testing.assert_array_equal(tokens, rt)
    np.testing.assert_array_equal(w, rw)
    np.testing.assert_array_equal(seg, rs)
    assert tokens.dtype == np.int32 and w.dtype == np.float32 and seg.dtype == np.int32
    assert w.sum() == len(resp) + 1                        # response tokens and END
    # the weighted targets are exactly the response followed by END
    np.testing.assert_array_equal(tokens[1:][w > 0], np.array(resp + [END]))


def test_row_boundary_fits_t_plus_one(tok):
    prompt = prompt_ids(tok, "Hi")
    resp = list(range(65, 65 + 7))
    L = len(prompt) + len(resp) + 2
    tokens, w, seg = P.response_row(np.array(prompt), np.array(resp), L - 1)  # L == T + 1
    assert tokens[-1] == END and (seg == 0).all() and w[-1] == 1.0
    with pytest.raises(ValueError):
        P.response_row(np.array(prompt), np.array(resp), L - 2)


def test_forced_calculator_results_get_no_weight(tok):
    T = 64
    prompt = prompt_ids(tok, "What is 2+3?")
    head, expr, result, tail = tok.encode("So "), tok.encode("2+3"), tok.encode("5"), tok.encode(" apples.")
    resp = head + [CALC] + expr + [EQ] + result + [CALC_END] + tail
    start = len(head) + 1 + len(expr) + 1                     # first result token in resp
    forced = list(range(start, start + len(result) + 1))      # result tokens and CALC_END
    tokens, w, seg = P.response_row(np.array(prompt), np.array(resp), T, mask_forced=True)
    np.testing.assert_array_equal(w, ref_row(prompt, resp, T, forced)[1])
    _, w_all, _ = P.response_row(np.array(prompt), np.array(resp), T, mask_forced=False)
    np.testing.assert_array_equal(w_all, ref_row(prompt, resp, T)[1])
    assert P.target_count(np.array(resp)) == len(resp) + 1 - len(forced)
    assert P.target_count(np.array(resp), mask_forced=False) == len(resp) + 1


def test_pair_rows_and_row_tokens(tok):
    prompt = prompt_ids(tok, "Say yes or no.")
    pair = P.make_pair(prompt + [ASSISTANT], tok.encode("Yes.") + [END], tok.encode("No."))
    rows = P.pair_rows(pair, 32)
    assert rows["tokens"].shape == (2, 33) and rows["w"].shape == (2, 32)
    np.testing.assert_array_equal(rows["tokens"][0], ref_row(prompt, tok.encode("Yes."), 32)[0])
    np.testing.assert_array_equal(rows["tokens"][1], ref_row(prompt, tok.encode("No."), 32)[0])
    content = P.row_tokens(prompt + [ASSISTANT], tok.encode("Yes.") + [END])
    assert content.tolist() == prompt + [ASSISTANT] + tok.encode("Yes.") + [END]


# --------------------------------------------------------------------------- filters
def test_filters_count_every_reason(tok):
    T = 40
    p = prompt_ids(tok, "Q")
    words = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
    good = P.make_pair(p, words[:4], words[4:6])
    pairs = [
        good,
        P.make_pair(p[1:], [65], [66]),                                   # no BOS
        P.make_pair([BOS, USER, 65, END, ASSISTANT, 66, END], [67], [68]),  # last turn assistant
        P.make_pair(p, [65, USER, 66], [67]),                             # role token in response
        P.make_pair(p, [], [67]),                                         # empty chosen
        P.make_pair(p, [65] * (T - len(p)), [66]),                        # does not fit
        P.make_pair(p, [65, 66], [65, 66]),                               # identical
        P.make_pair(p, words[:9], words[:2]),                             # 10 > 3 * 3
        P.make_pair(p, words[:8], words[1:3]),                            # 9 <= 3 * 3: kept
        P.make_pair(p, words[:4], words[4:6]),                            # duplicate of good
        P.make_pair(p, [65, CALC, 66, EQ, 67, CALC_END], [68, 69]),       # calculator: valid
    ]
    kept, counts = P.filter_pairs(pairs, T)
    assert counts == {"invalid": 4, "too_long": 1, "identical": 1, "length_ratio": 1,
                      "duplicate": 1, "kept": 3}
    assert [k is q for k, q in zip(kept, (pairs[0], pairs[8], pairs[10]))] == [True] * 3
    kept_off, counts_off = P.filter_pairs(pairs, T, max_len_ratio=0)
    assert counts_off["length_ratio"] == 0 and counts_off["kept"] == 4
    _, counts_dup = P.filter_pairs(pairs, T, dedupe=False)
    assert counts_dup["duplicate"] == 0 and counts_dup["kept"] == 4


def test_prompt_problems():
    assert P.prompt_problem(np.array([BOS, USER, 65, END])) is None
    assert P.prompt_problem(np.array([BOS, SYSTEM, 65, END, USER, 66, END])) is None
    assert P.prompt_problem(np.array([BOS, USER, 65])) is not None           # not closed
    assert P.prompt_problem(np.array([BOS, USER, 65, END, 66, END])) is not None
    assert P.prompt_problem(np.array([BOS, USER, BOS, END])) is not None
    assert P.prompt_problem(np.array([BOS, SYSTEM, 65, END])) is not None    # no user turn last


def test_prepare_dedupes_across_groups_with_per_group_guard(tok):
    p = prompt_ids(tok, "Count")
    long_c = P.make_pair(p, list(range(65, 85)), [66])                 # 21 > 3 * 2
    same = P.make_pair(p, [65, 66], [67])
    split = P.prepare({"judge": [long_c, same], "verifier": [long_c, same]}, 64,
                      heldout_frac=0.0, max_len_ratio={"judge": 3.0, "verifier": 0.0})
    assert split.stats["judge"]["length_ratio"] == 1 and split.stats["judge"]["kept"] == 1
    assert split.stats["verifier"]["duplicate"] == 1 and split.stats["verifier"]["kept"] == 1
    assert len(split.train) == 2 and split.heldout == []


# ------------------------------------------------------------------ split and order
def make_prompt_pairs(n_prompts: int, seed: int = 0) -> list[P.Pair]:
    """1 to 3 pairs for each of ``n_prompts`` distinct prompts."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_prompts):
        prompt = [BOS, USER, *rng.integers(65, 200, size=3).tolist(), i % 256, i // 256, END]
        for j in range(1 + i % 3):
            out.append(P.make_pair(prompt, [65 + j, 66], [67 + j]))
    return out


def test_split_is_by_prompt_deterministic_and_order_free():
    pairs = make_prompt_pairs(900)
    train, held = P.split_pairs(pairs, 0.2, seed=3)
    key = lambda p: p.prompt.tobytes()                                           # noqa: E731
    assert not ({key(p) for p in train} & {key(p) for p in held})
    n_prompts = len({key(p) for p in pairs})
    frac = len({key(p) for p in held}) / n_prompts
    assert 0.15 < frac < 0.25
    assert len(train) + len(held) == len(pairs)
    rev_train, rev_held = P.split_pairs(pairs[::-1], 0.2, seed=3)
    assert {key(p) for p in rev_held} == {key(p) for p in held}
    _, other = P.split_pairs(pairs, 0.2, seed=4)
    assert {key(p) for p in other} != {key(p) for p in held}
    # input order is kept on both sides
    pos = {id(p): i for i, p in enumerate(pairs)}
    assert [pos[id(p)] for p in held] == sorted(pos[id(p)] for p in held)


def test_split_keeps_both_sides_non_empty():
    pairs = [P.make_pair([BOS, USER, 65 + i, END], [65], [66]) for i in range(2)]
    for frac in (1e-9, 0.999999):
        train, held = P.split_pairs(pairs, frac, seed=0)
        assert len(train) == 1 and len(held) == 1
    assert P.split_pairs(pairs, 0.0)[1] == []


def test_shuffled_order_is_a_seeded_permutation():
    a, b, c = P.shuffled_order(50, 1), P.shuffled_order(50, 1), P.shuffled_order(50, 2)
    np.testing.assert_array_equal(a, b)
    assert sorted(a.tolist()) == list(range(50)) and not np.array_equal(a, c)


# --------------------------------------------------------------------------- batches
def toy_pairs(tok, n: int) -> list[P.Pair]:
    out = []
    for i in range(n):
        prompt = prompt_ids(tok, f"Question {i}: say something about {i % 3}.")
        out.append(P.make_pair(prompt, tok.encode(f"Good answer {i}, good answer."),
                               tok.encode(f"Bad {i}")))
    return out


def test_batches_shapes_interleaving_padding_and_features(tok):
    T, ppm, A = 40, 2, 2
    pairs = toy_pairs(tok, 7)
    order = P.shuffled_order(len(pairs), 5)
    data = P.PairBatches(pairs, T, ppm, A, order=order, tok=tok)
    assert len(data) == 2 and data.shape == (A, 2 * ppm, T)
    seen = []
    for k, batch in enumerate(data):
        assert batch["tokens"].shape == (A, 4, T + 1) and batch["tokens"].dtype == np.int32
        assert batch["w"].shape == (A, 4, T) and batch["w"].dtype == np.float32
        assert batch["seg"].shape == (A, 4, T) and batch["reg"].dtype == np.int16
        assert batch["cand_tok"].shape[:3] == (A, 4, T) and batch["cand_feat"].dtype == np.int8
        assert batch["pair_mask"].shape == (A, ppm) and batch["ref"].shape == (A, 4)
        idx = batch["index"]
        np.testing.assert_array_equal(batch["pair_mask"], (idx >= 0).astype(np.float32))
        for a in range(A):
            for p in range(ppm):
                i = int(idx[a, p])
                rows = (2 * p, 2 * p + 1)
                if i < 0:                                      # filler pair: no weight at all
                    assert batch["w"][a, rows[0]].sum() == 0 and batch["w"][a, rows[1]].sum() == 0
                    continue
                seen.append(i)
                for r, resp in zip(rows, (pairs[i].chosen, pairs[i].rejected)):
                    rt, rw, rs = ref_row(pairs[i].prompt.tolist(), resp.tolist(), T)
                    np.testing.assert_array_equal(batch["tokens"][a, r], rt)
                    np.testing.assert_array_equal(batch["w"][a, r], rw)
                    np.testing.assert_array_equal(batch["seg"][a, r], rs)
                    single = exact_features(rt[None, :T], rs[None], tok)     # row on its own
                    for key in P.FEATURE_KEYS:
                        np.testing.assert_array_equal(batch[key][a, r], single[key][0])
    assert seen == order.tolist()                              # every pair once, in order
    assert int((data.indices(1) < 0).sum()) == 1


def test_reference_values_are_attached_per_row(tok):
    pairs = toy_pairs(tok, 5)
    data = P.PairBatches(pairs, 40, 2, 1, features=False)
    ref = np.arange(10, dtype=np.float32).reshape(5, 2) - 20
    data.set_ref(ref)
    for batch in data:
        for p, i in enumerate(batch["index"][0].tolist()):
            want = ref[i] if i >= 0 else np.zeros(2, np.float32)
            np.testing.assert_array_equal(batch["ref"][0, 2 * p:2 * p + 2], want)
    with pytest.raises(ValueError):
        data.set_ref(ref[:4])
    with pytest.raises(ValueError):
        data.set_ref(np.full((5, 2), np.nan))
    np.testing.assert_array_equal(
        data.target_counts(), [[len(p.chosen) + 1, len(p.rejected) + 1] for p in pairs])


def test_batches_are_deterministic_and_cached(tok):
    pairs = toy_pairs(tok, 6)
    kw = dict(order=P.shuffled_order(6, 0), tok=tok)
    a = P.PairBatches(pairs, 40, 2, 2, cache_bytes=1 << 30, **kw)
    b = P.PairBatches(pairs, 40, 2, 2, **kw)
    first = a.get(0)
    assert a.get(0)["tokens"] is first["tokens"]              # served from the cache
    assert b.get(0)["tokens"] is not b.get(0)["tokens"]       # recomputed without a cache
    for k in range(len(a)):
        for key, v in a.get(k).items():
            np.testing.assert_array_equal(v, b.get(k)[key])


def test_batches_reject_pairs_that_do_not_fit_and_bad_orders(tok):
    pairs = toy_pairs(tok, 3)
    with pytest.raises(ValueError):
        P.PairBatches(pairs, 10, 1, 1, features=False)
    for order in ([0, 1], [0, 1, 1], [0, 1, 3]):            # every pair exactly once
        with pytest.raises(ValueError):
            P.PairBatches(pairs, 40, 1, 1, order=order, features=False)
    assert len(P.PairBatches(pairs, 40, 1, 1, order=[2, 0, 1], features=False)) == 3


# ---------------------------------------------------------------------------- sources
def ultrafeedback_rows():
    rows = []
    for k in range(9):
        u = {"role": "user", "content": f"Explain topic number {k} in a sentence."}
        rows.append({"chosen": [u, {"role": "assistant", "content": f"Topic {k} is about care."}],
                     "rejected": [u, {"role": "assistant", "content": f"No idea about {k}."}],
                     "score_chosen": 8.0, "score_rejected": 8.0 if k % 4 == 3 else 2.0})
    return rows


def test_pairs_from_shards_written_by_the_pipeline(tmp_path, tok):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = ultrafeedback_rows()
    path = tmp_path / "raw" / "part-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    files = [D.LocalFile("ultrafeedback", 0, "test/repo", "0" * 40, path.name, str(path),
                         path.stat().st_size, None)]
    out = tmp_path / "tok"
    W.run("ultrafeedback", files=files, out_dir=out, workers=0, tokenizer=tok,
          log=lambda msg: None)
    pairs, counts = P.pairs_from_shards(out)
    expected = [r for r in rows if r["score_chosen"] > r["score_rejected"]]
    assert counts["pairs"] == len(expected) == len(pairs) == 7
    for pair, r in zip(pairs, expected):
        prompt = cf.render_conversation(r["chosen"][:1], tok)[0].tolist()
        assert pair.prompt.tolist() == prompt
        assert pair.chosen.tolist() == tok.encode(r["chosen"][1]["content"])
        assert pair.rejected.tolist() == tok.encode(r["rejected"][1]["content"])
        assert pair.source == "ultrafeedback"
    kept, fc = P.filter_pairs(pairs, 64)
    assert fc["kept"] == 7
    assert len(P.pairs_from_shards(out, limit=3)[0]) == 3


def test_pairs_from_shards_drops_malformed_pairs(tmp_path):
    good_p = [BOS, USER, 65, END]
    segs = [
        (good_p + [ASSISTANT, 66, END], 0), (good_p + [ASSISTANT, 67, END], 0),       # ok
        (good_p + [ASSISTANT, 66, END], 1), ([BOS, USER, 70, END, ASSISTANT, 67, END], 1),
        (good_p + [ASSISTANT, 66, END], 2), (good_p + [ASSISTANT, 67, END], 3),       # recs
        (good_p + [ASSISTANT, 66, END], 4), (good_p + [ASSISTANT, 67, END, USER, 68, END], 4),
    ]
    wr = SH.ShardWriter(tmp_path, 0, with_roles=False)
    for ids, rec in segs:
        wr.add(ids, rec=rec)
    entry = wr.close()
    entry.update(pairs=True, source="unit")
    SH.append_manifest(tmp_path, entry)
    wr = SH.ShardWriter(tmp_path, 1, with_roles=False)
    wr.add([BOS, 65, 66])
    other = wr.close()
    other.update(pairs=False)
    SH.append_manifest(tmp_path, other)
    pairs, counts = P.pairs_from_shards(tmp_path)
    assert counts == {"prompt_mismatch": 1, "rec_mismatch": 1, "unparsable": 1,
                      "not_pair_shards": 1, "pairs": 1}
    assert pairs[0].prompt.tolist() == good_p and pairs[0].chosen.tolist() == [66]
    assert pairs[0].rejected.tolist() == [67] and pairs[0].source == "unit"


def test_pairs_from_tokens_normalises_prompt_and_responses():
    prompt = [BOS, USER, 65, END]
    items = [(prompt + [ASSISTANT], [66, END], [67]),
             {"prompt": prompt, "chosen": [66, 68], "rejected": [69, END], "source": "smc"}]
    a, b = P.pairs_from_tokens(items, source="decoder")
    assert a.prompt.tolist() == prompt and a.chosen.tolist() == [66] and a.source == "decoder"
    assert b.rejected.tolist() == [69] and b.source == "smc"
    assert not a.prompt.flags.writeable
    with pytest.raises(ValueError):
        P.make_pair(prompt, [70000], [65])


def test_pairs_from_scores_pairs_best_with_worst():
    prompt = [BOS, USER, 65, END]
    responses = [[70 + i] for i in range(6)]
    scores = [0.5, 1.0, 0.0, 1.0, 0.0, 0.5]
    pairs = P.pairs_from_scores(prompt, responses, scores)
    got = [(int(p.chosen[0]) - 70, int(p.rejected[0]) - 70) for p in pairs]
    assert got == [(1, 4), (3, 2)]                    # ties stop pairing: 0.5 vs 0.5 is no pair
    assert len(P.pairs_from_scores(prompt, responses, scores, max_pairs=1)) == 1
    assert P.pairs_from_scores(prompt, responses, scores, min_gap=1.0) == []
    verifier = P.pairs_from_scores(prompt, responses[:3], [1, 0, 1])
    assert [(int(p.chosen[0]), int(p.rejected[0])) for p in verifier] == [(70, 71)]


def test_defaults_come_from_the_config():
    assert P.filter_pairs.__kwdefaults__["max_len_ratio"] == PrefCfg().max_len_ratio


def test_warm_computes_the_same_rows_in_other_processes(tok):
    """``PairBatches.warm``: the rows of every batch, computed by a pool, equal the rows this
    process computes; nothing is warmed without a cache, without features, with one worker or
    with a feature function of the caller."""
    pairs = toy_pairs(tok, 11)
    kw = dict(order=P.shuffled_order(len(pairs), 3), tok=tok)
    ref = P.PairBatches(pairs, 40, 2, 2, **kw)
    warm = P.PairBatches(pairs, 40, 2, 2, cache_bytes=1 << 30, **kw)
    assert warm.warm(2) == len(warm) == 3
    assert warm.warm(2) == 0                                    # everything is kept already
    for k in range(len(ref)):
        a, b = ref.get(k), warm.get(k)
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
    assert warm.get(0)["tokens"] is warm.get(0)["tokens"]      # served from the cache
    assert P.PairBatches(pairs, 40, 2, 2, **kw).warm(2) == 0   # no cache
    assert P.PairBatches(pairs, 40, 2, 2, cache_bytes=1 << 30, **kw).warm(1) == 0
    assert P.PairBatches(pairs, 40, 2, 2, cache_bytes=1 << 30, features=False).warm(2) == 0
    own = P.PairBatches(pairs, 40, 2, 2, cache_bytes=1 << 30, tok=tok,
                        feature_fn=lambda t, s, k: exact_features(t, s, k))
    assert own.warm(2) == 0
