"""Tests of bes.data.loader on small synthetic shards written here: the SPEC section 3 batch for
both stage shapes, features equal to exact_features, weights equal to an independent per-token
rule, whole documents and turn-boundary conversations, the low-discrepancy row kinds,
determinism across worker counts, restarts (mid-block, after a rollback skip, into another
shape), shards that appear mid-run, a killed worker, epochs that reshuffle, the library, the
journal checks, IteratorData and a few steps of the real model's loss."""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, PAD, SYSTEM, USER,
                        VOCAB_SIZE, Cfg, LoaderCfg, MixCfg)
from bes.data import chat_format as cf
from bes.data import loader as LD
from bes.data import shards as SH
from bes.data.samples import WeightRule
from bes.exact.features import exact_features
from bes.train.schedule import stage_at
from bes.tokenizer.tok import load

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROLE_CODE = {SYSTEM: 1, USER: 2, ASSISTANT: 3, LIBRARY: 4}
WORDS = ("the cat sat on a mat and then it ran far away to see what is new in town today with "
         "three green apples seven red boxes people often walk near quiet rivers during long "
         "summer evenings while birds sing").split()
#: Small streams and blocks; a token horizon of the size of these tests, so that the epoch caps
#: of mixture.plan do not bind on the tiny sources and the stage decides the mixture.
LC = LoaderCfg(n_streams=8, block_steps=4, lookahead=6, lib_lookahead=6, ring=3, max_wait=16,
               workers=0, horizon_tokens=2e5)


# ------------------------------------------------------------------------- the data
def sentence(rng, lo: int, hi: int) -> str:
    return " ".join(rng.choice(WORDS, int(rng.integers(lo, hi)))).capitalize() + "."


def chat_messages(rng, turns: int) -> list[dict]:
    msgs = []
    if rng.random() < 0.2:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    for _ in range(turns):
        msgs.append({"role": "user", "content": "What about " + " ".join(rng.choice(WORDS, int(rng.integers(2, 6)))) + "?"})
        msgs.append({"role": "assistant",
                     "content": " ".join(sentence(rng, 3, 9) for _ in range(int(rng.integers(1, 4))))})
    return msgs


def math_messages(rng) -> list[dict]:
    a, b = int(rng.integers(2, 50)), int(rng.integers(2, 50))
    return [{"role": "user", "content": f"Tom has {a} boxes with {b} apples each. How many apples?"},
            {"role": "assistant", "content": [{"type": "text", "text": "He has "},
                                              {"type": "calc", "expr": f"{a}*{b}", "result": str(a * b)},
                                              {"type": "text", "text": " apples."}]}]


def rag_messages(rng) -> list[dict]:
    return [{"role": "library", "content": "Rivers\n" + sentence(rng, 5, 12)},
            {"role": "user", "content": "What do people do near rivers?"},
            {"role": "assistant", "content": sentence(rng, 3, 8)}]


def write_docs(root: pathlib.Path, name: str, shard: int, rng, n: int, lo: int, hi: int,
               offset: int = 0, store: dict | None = None, marker: int | None = None,
               input_size: int | None = None) -> None:
    """Documents of random ordinary ids (``offset`` shifts the id range: a recognisable shard;
    ``marker``: document ``k`` has the id ``marker + k`` right after its BOS and nowhere else;
    ``input_size``: parquet bytes recorded in the manifest, as tokenize_worker does)."""
    d = root / "tok" / name
    w = SH.ShardWriter(d, shard, with_roles=False)
    for k in range(n):
        ids = rng.integers(offset, offset + 4000, int(rng.integers(lo, hi))).astype(np.int32)
        ids[0] = BOS
        if marker is not None:
            ids[1] = marker + k
        w.add(ids)
        if store is not None:
            store.setdefault(name, []).append(ids)
    entry = w.close()
    if input_size is not None:
        entry["input"] = {"size": int(input_size)}
    SH.append_manifest(d, entry)


def write_convs(root: pathlib.Path, name: str, shard: int, make, n: int, tok,
                store: dict | None = None) -> None:
    d = root / "tok" / name
    w = SH.ShardWriter(d, shard, with_roles=True)
    for _ in range(n):
        ids, roles = cf.render_conversation(make(), tok)
        w.add(ids, roles)
        if store is not None:
            store.setdefault(name, []).append(ids)
    SH.append_manifest(d, w.close())


def build_root(root: pathlib.Path, tok, seed: int = 0, base_docs: int = 400, doc_hi: int = 160,
               convs: int = 200) -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    store: dict[str, list[np.ndarray]] = {}
    write_docs(root, "climbmix", 0, rng, base_docs, 8, doc_hi, store=store)
    write_docs(root, "simplewiki", 0, rng, base_docs // 4, 8, doc_hi, store=store)
    write_convs(root, "smol_smoltalk", 0, lambda: chat_messages(rng, int(rng.integers(1, 4))), convs, tok, store)
    write_convs(root, "gsm8k", 0, lambda: math_messages(rng), convs // 4, tok, store)
    write_convs(root, "squad_v2", 0, lambda: rag_messages(rng), convs // 4, tok, store)
    return store


def small_cfg(seq_len: int = 64, micro_seqs: int = 8, accum: int = 2, **data) -> Cfg:
    cfg = Cfg()
    return dataclasses.replace(
        cfg, optim=dataclasses.replace(cfg.optim, tokens_per_step=accum * micro_seqs * seq_len,
                                       micro_seqs=micro_seqs),
        data=dataclasses.replace(cfg.data, seq_len=seq_len, seq_len_late=2 * seq_len, **data))


def stage(cfg: Cfg, p: float = 0.1, **kw):
    return dataclasses.replace(stage_at(p, cfg), **kw)


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def tok():
    return load()


@pytest.fixture(scope="module")
def data(tmp_path_factory, tok):
    root = tmp_path_factory.mktemp("data")
    store = build_root(root, tok)
    return root, store


def make(cfg, tok, root, journal, n_workers=0, lc=LC, **kw):
    return LD.Loader(cfg, tok, root, n_workers=n_workers, seed=kw.pop("seed", 7), loader_cfg=lc,
                     journal=journal, **kw)


# ----------------------------------------------------------------------- references
def ref_roles(tokens) -> list[int]:
    """Role code of every token of a packed row: the turn opened by the last role token of its
    segment (``BOS`` and ``PAD`` and raw text: 0)."""
    out, cur = [], 0
    for t in [int(x) for x in tokens]:
        if t in (BOS, PAD):
            cur = 0
        elif t in ROLE_CODE:
            cur = ROLE_CODE[t]
        out.append(cur)
    return out


def ref_weights(ids, roles, rule: WeightRule) -> list[float]:
    """Per-token weight rule of bes.data.samples, one token at a time."""
    out = []
    in_span = in_result = False
    prev_tok = prev_role = None
    for t, r in zip([int(x) for x in ids], [int(x) for x in roles]):
        if t == CALC:
            in_span, in_result = True, False
        elif t >= VOCAB_SIZE - 16 and t not in (EQ, CALC_END) and in_span:
            in_span = in_result = False
        if t in (BOS, PAD):
            w = 0.0
        elif r in (0, 3):
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
            in_result = True
        if t == CALC_END and in_span:
            in_span = in_result = False
        out.append(w)
        prev_tok, prev_role = t, r
    return out


def segments_of(tokens, seg) -> list[tuple[int, int]]:
    """``(start, stop)`` of every segment of a row over its T+1 tokens: runs of equal ``seg``
    (the last run extends to position T, which has no ``seg`` entry), padding split off where
    it starts (a single ``PAD`` at position T belongs to no input position)."""
    T = len(seg)
    bounds = [0] + [t for t in range(1, T) if seg[t] != seg[t - 1]]
    out = []
    for a, b in zip(bounds, bounds[1:] + [T + 1]):
        pad = np.flatnonzero(tokens[a:b] == PAD)
        if pad.size and pad[0] > 0:
            out += [(a, a + int(pad[0])), (a + int(pad[0]), b)]
        else:
            out.append((a, b))
    return out


def ref_turns(conv: list[int]) -> list[tuple[int, int, int]]:
    """``(role token, start, stop)`` of the turns of a rendered conversation (scanning for END)."""
    out, i = [], 1
    while i < len(conv):
        j = conv.index(END, i + 1)
        out.append((conv[i], i, j + 1))
        i = j + 1
    return out


def ref_pieces(conv: list[int]) -> tuple[set[tuple], set[tuple]]:
    """``(prefixes, windows)`` the loader may place for a stored conversation: its prefixes that
    end with the END of an assistant turn, and its windows: BOS, the system turn when it opens
    the conversation, then the turns from a later exchange on (a user turn, or the library turn
    right before it), cut after the END of an assistant turn."""
    turns = ref_turns(conv)
    head = conv[:turns[0][2]] if turns and turns[0][0] == SYSTEM else conv[:1]
    starts = [turns[k - 1][1] if k and turns[k - 1][0] == LIBRARY else s
              for k, (r, s, _) in enumerate(turns) if r == USER]
    prefixes = {tuple(conv[:e]) for r, _, e in turns if r == ASSISTANT}
    windows = set()
    for s in starts[1:]:
        for r, _, e in turns:
            if r == ASSISTANT and e > s:
                windows.add(tuple(head + conv[s:e]))
    return prefixes, windows


def check_row_structure(tokens, seg, w) -> None:
    """SPEC section 1/3 invariants of one row."""
    T = len(seg)
    assert tokens[0] == BOS
    assert seg[0] == 0 and np.all(np.diff(seg) >= 0) and np.all(np.diff(seg) <= 1)
    assert np.all((tokens >= 0) & (tokens < VOCAB_SIZE))
    for t in range(1, T):
        if seg[t] != seg[t - 1]:              # a new segment starts with BOS, or is the padding
            assert tokens[t] == BOS or np.all(tokens[t:] == PAD)
        elif tokens[t] == BOS:
            raise AssertionError("BOS inside a segment")
    pad = np.flatnonzero(tokens == PAD)
    if pad.size:
        assert np.all(tokens[pad[0]:] == PAD)
        assert np.all(w[pad[0] - 1:] == 0)
        if pad[0] < T:
            assert np.all(seg[pad[0]:] == seg[pad[0]]) and (pad[0] == 0 or seg[pad[0]] == seg[pad[0] - 1] + 1)
    assert np.all(w[tokens[1:] == BOS] == 0)


# ------------------------------------------------------------------------ the batch
def test_batch_fields_match_the_trainer():
    from bes.train.train import BATCH_FIELDS, batch_spec

    assert LD.BATCH_FIELDS == BATCH_FIELDS
    for shape in ((8, 32, 1024), (8, 16, 2048), (2, 8, 64)):
        spec = batch_spec(shape)
        assert {k: (np.dtype(d), s) for k, (d, s) in LD.batch_shapes(shape).items()} == \
            {k: (np.dtype(v.dtype), tuple(v.shape)) for k, v in spec.items()}
        buf = np.zeros(LD.batch_nbytes(shape), np.uint8)
        views = LD.batch_views(buf, shape)
        for k, v in views.items():
            v.reshape(-1).view(np.uint8)[:] = 1
        assert int(buf.sum()) == sum(v.nbytes for v in views.values())   # no overlap


def test_spec_shapes_rows_and_the_switch_to_the_late_shape(tmp_path, tok):
    """(8, 32, 1024) then (8, 16, 2048) with the final configuration: keys, dtypes and shapes,
    BOS-led rows, segment ids, padding; the switch re-plans at the requested step."""
    root = tmp_path / "data"
    rng = np.random.default_rng(1)
    write_docs(root, "climbmix", 0, rng, 900, 20, 1600)
    write_convs(root, "smol_smoltalk", 0, lambda: chat_messages(rng, int(rng.integers(1, 5))), 1500, tok)
    cfg = Cfg()
    early, late = stage_at(0.1, cfg), stage_at(0.6, cfg)
    assert early.shape == (8, 32, 1024) and late.shape == (8, 16, 2048)
    lc = LoaderCfg(workers=0, block_steps=4, horizon_tokens=1e6)    # no epoch cap binds
    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
        b1 = loader(0, early)
        b2 = loader(1, late)
        recs = loader.records
    for b, shape in ((b1, early.shape), (b2, late.shape)):
        assert {k: (v.dtype, v.shape) for k, v in b.items()} == \
            {k: (d, s) for k, (d, s) in LD.batch_shapes(shape).items()}
        a_, b_, t_ = shape
        for a in range(a_):
            for r in range(b_):
                check_row_structure(b["tokens"][a, r], b["seg"][a, r], b["w"][a, r])
        assert b["w"].sum() > 0.5 * a_ * b_ * t_
    assert [(r["start"], tuple(r["shape"])) for r in recs] == [(0, early.shape), (1, late.shape)]
    assert recs[0]["rows"][2] == pytest.approx(early.chat_frac, abs=0.02)


def test_features_equal_exact_features(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    with make(cfg, tok, root, tmp_path / "j.jsonl") as loader:
        for i, st in enumerate((stage(cfg, chat_frac=0.6), stage(cfg, chat_frac=0.6))):
            b = loader(i, st)
            for a in range(b["tokens"].shape[0]):
                f = exact_features(b["tokens"][a, :, :-1], b["seg"][a], tok)
                for k in ("reg", "clock", "cand_tok", "cand_feat"):
                    np.testing.assert_array_equal(b[k][a], f[k], err_msg=k)


@pytest.mark.parametrize("assistant_only", [False, True])
def test_weights_follow_the_weight_rule(data, tok, tmp_path, assistant_only):
    root, _ = data
    cfg = small_cfg()
    rule = WeightRule(assistant_only=assistant_only, passage_w=cfg.data.passage_loss_w)
    st = stage(cfg, chat_frac=0.7, assistant_only=assistant_only)
    seen = set()
    with make(cfg, tok, root, tmp_path / "j.jsonl") as loader:
        for i in range(6):
            b = loader(i, st)
            for a in range(b["tokens"].shape[0]):
                for r in range(b["tokens"].shape[1]):
                    tokens, w = b["tokens"][a, r], b["w"][a, r]
                    roles = ref_roles(tokens)
                    seen.update(roles)
                    expect = ref_weights(tokens, roles, rule)
                    np.testing.assert_array_equal(w, np.array(expect[1:], np.float32))
                    check_row_structure(tokens, b["seg"][a, r], w)
    assert {0, 2, 3, 4} <= seen          # raw text, user, assistant and library turns were seen


def test_rows_hold_whole_documents_and_turn_boundary_conversations(data, tok, tmp_path):
    """Without content transforms: every base piece is a contiguous slice of a document (with a
    fresh BOS when it does not start at the document's start), a cut document continues in the
    stream's next base row, and every chat piece is a prefix of a stored conversation that ends
    at an assistant turn (the whole conversation when it fits) or, for a conversation longer
    than the row, one of its later windows (system turn kept, oldest exchanges dropped)."""
    root, store = data
    cfg = small_cfg(hindsight_frac=0.0)
    V = LC.n_streams
    gram: dict[tuple, tuple[str, int, int]] = {}
    for name in ("climbmix", "simplewiki"):
        for k, d in enumerate(store[name]):
            for i in range(1, len(d) - 3):
                gram.setdefault(tuple(d[i:i + 4].tolist()), (name, k, i))
    convs = {tuple(c.tolist()) for name in ("smol_smoltalk", "gsm8k", "squad_v2") for c in store[name]}
    prefixes, windows = set(), set()
    for c in convs:
        p, w = ref_pieces(list(c))
        prefixes |= p
        windows |= {x for x in w if len(c) > cfg.data.seq_len + 1}
    st = stage(cfg, chat_frac=0.4)
    last_cut: dict[int, tuple[str, int, int]] = {}       # stream -> (doc, next offset) after its last base row
    chat_pieces = base_pieces = continued = n_windows = 0
    with make(cfg, tok, root, tmp_path / "j.jsonl") as loader:
        A, B, T = st.shape
        for i in range(12):
            b = loader(i, st)
            for a in range(A):
                for r in range(B):
                    m = i * A * B + a * B + r
                    v = m % V
                    tokens = b["tokens"][a, r]
                    segs = segments_of(tokens, b["seg"][a, r])
                    pieces = [tokens[s:e] for s, e in segs if tokens[s] == BOS]
                    is_chat = any(int(x) in ROLE_CODE for x in tokens)
                    if is_chat:
                        for p in pieces:
                            key = tuple(p.tolist())
                            assert key in prefixes or key in windows
                            n_windows += key not in prefixes
                            chat_pieces += 1
                        continue
                    for j, p in enumerate(pieces):
                        if len(p) < 5:
                            if j == len(pieces) - 1:
                                last_cut[v] = "unknown"
                            continue
                        name, k, off = gram[tuple(p[1:5].tolist())]
                        doc = store[name][k]
                        np.testing.assert_array_equal(p[1:], doc[off:off + len(p) - 1])
                        if j == 0 and last_cut.get(v) not in (None, "unknown"):
                            assert last_cut[v] == (name, k, off)   # the cut document goes on
                            continued += 1
                        if j == 0 and off > 1:
                            assert last_cut.get(v) is not None        # only after a cut
                        if j == len(pieces) - 1:
                            end = off + len(p) - 1
                            last_cut[v] = (name, k, end) if end < len(doc) else None
                        base_pieces += 1
    assert chat_pieces > 20 and base_pieces > 50 and continued > 5 and n_windows > 0


def test_row_kinds_are_balanced_in_every_micro_batch(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.35, library_frac_base=0.0, library_frac_chat=0.0)
    with make(cfg, tok, root, tmp_path / "j.jsonl") as loader:
        batches = [loader(i, st) for i in range(10)]
        planned = dict(zip(LD.ROW_KINDS, loader.records[0]["rows"]))
        stats = loader.stats()
    A, B, _ = st.shape
    assert planned["chat"] == pytest.approx(0.35, abs=0.02) and planned["library"] == 0.0
    n_chat = []
    for b in batches:
        for a in range(A):
            k = sum(any(int(x) in ROLE_CODE for x in b["tokens"][a, r]) for r in range(B))
            n_chat.append(k)
            assert abs(k - B * planned["chat"]) < 1 + 1e-9          # 8 aligned streams: +-1
    assert abs(np.mean(n_chat) / B - planned["chat"]) < 0.05
    blk = stats["blocks"][0]
    assert set(blk["planned_rows"]) == set(LD.ROW_KINDS)
    assert blk["pad_share"]["base"] < 0.01 and 0 <= blk["pad_share"]["chat"] < 0.5
    assert set(blk["epochs"]) >= {"climbmix", "smol_smoltalk"}
    json.dumps(stats)                                                  # loggable as JSON


# ---------------------------------------------------------------------- determinism
def test_determinism_across_worker_counts(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.5)
    out = {}
    for nw in (0, 1, 3):
        with make(cfg, tok, root, tmp_path / f"j{nw}.jsonl", n_workers=nw) as loader:
            out[nw] = [loader(i, st) for i in range(10)]
            if nw:
                assert len(loader.worker_pids()) == nw
    for nw in (1, 3):
        assert all(same(x, y) for x, y in zip(out[0], out[nw]))
    plans = []
    for nw in (0, 1, 3):
        recs = [r for r in SH.scan_jsonl(tmp_path / f"j{nw}.jsonl")[0] if r["type"] == "block"]
        plans.append([{k: v for k, v in r.items() if k not in ("time", "horizon")} for r in recs])
    assert plans[0] == plans[1] == plans[2]


def test_killed_worker_neither_loses_nor_duplicates_rows(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.5)
    with make(cfg, tok, root, tmp_path / "ref.jsonl") as loader:
        ref = [loader(i, st) for i in range(9)]
    with make(cfg, tok, root, tmp_path / "pool.jsonl", n_workers=2) as loader:
        got = []
        for i in range(9):
            got.append(loader(i, st))
            if i in (2, 5):                     # steps i+1.. are in flight: kill mid-work
                os.kill(loader.worker_pids()[i % 2], signal.SIGKILL)
        restarts = loader.stats()["workers"]["restarts"]
    assert restarts == 2
    assert all(same(x, y) for x, y in zip(ref, got))


def _gone(pids, timeout=15.0) -> bool:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().split()[2] != "Z":
                        alive.append(pid)
            except (ProcessLookupError, FileNotFoundError):
                pass
        if not alive:
            return True
        time.sleep(0.2)
    return False


def test_shutdown_leaves_no_workers_and_no_shared_memory(data, tok, tmp_path):
    import subprocess
    import sys
    import time

    root, _ = data
    cfg = small_cfg()
    with make(cfg, tok, root, tmp_path / "a.jsonl", n_workers=2) as loader:
        loader(0, stage(cfg))
        pids, names = loader.worker_pids(), [s.name for s in loader.exec.shms]
    assert _gone(pids) and not any((pathlib.Path("/dev/shm") / n).exists() for n in names)
    # a coordinator killed with SIGKILL takes its workers (and, via the resource tracker, its
    # shared memory) with it
    script = (
        "import sys, json, dataclasses\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from bes.config import Cfg, LoaderCfg\n"
        "from bes.tokenizer.tok import load\n"
        "from bes.data.loader import Loader\n"
        "from bes.train.schedule import stage_at\n"
        "cfg = Cfg()\n"
        "cfg = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, tokens_per_step=1024, micro_seqs=8),"
        " data=dataclasses.replace(cfg.data, seq_len=64, seq_len_late=128))\n"
        "lc = LoaderCfg(n_streams=8, block_steps=4, lookahead=6, ring=3, horizon_tokens=2e5)\n"
        f"L = Loader(cfg, load(), {str(root)!r}, n_workers=2, loader_cfg=lc, journal={str(tmp_path / 'b.jsonl')!r})\n"
        "L(0, stage_at(0.1, cfg))\n"
        "print(json.dumps([L.worker_pids(), [s.name for s in L.exec.shms]]), flush=True)\n"
        "import time; time.sleep(120)\n")
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                            stderr=subprocess.DEVNULL)
    try:
        pids, names = json.loads(proc.stdout.readline())
        assert all(os.kill(p, 0) is None for p in pids)
        proc.kill()
        proc.wait(timeout=30)
        assert _gone(pids)
        end = time.monotonic() + 15
        while time.monotonic() < end and any((pathlib.Path("/dev/shm") / n).exists() for n in names):
            time.sleep(0.2)
        assert not any((pathlib.Path("/dev/shm") / n).exists() for n in names)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_restart_mid_block_and_after_a_rollback_skip(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.5)
    j = tmp_path / "j.jsonl"
    with make(cfg, tok, root, j) as loader:
        ref = {i: loader(i, st) for i in range(10)}
        ref.update({i: loader(i, st) for i in range(30, 37)})     # rollback skip: a gap, then on
        chain = [(b.start, b.end) for b in loader.chain]
    assert chain[:5] == [(0, 4), (4, 8), (8, 12), (30, 34), (34, 38)]
    for start, stop in ((6, 10), (9, 10), (31, 37), (30, 34), (3, 6), (0, 2), (35, 36)):
        with make(cfg, tok, root, j) as loader:
            for i in range(start, stop):
                assert same(loader(i, st), ref[i]), (start, i)
            assert loader.stats()["resets"] == {"reproduce": 1}
    # a stage with other parameters (same shape) changes nothing inside planned blocks
    with make(cfg, tok, root, j) as loader:
        assert same(loader(5, stage(cfg, chat_frac=0.9, assistant_only=True)), ref[5])
    # continuing from the journal: without its record of the block at 34, a reopened loader
    # plans that block again from the end states of the block at 30, identically
    j2 = tmp_path / "j2.jsonl"
    lines = j.read_text().splitlines()
    j2.write_text("\n".join(x for x in lines if json.loads(x).get("start", -1) < 34) + "\n")
    with make(cfg, tok, root, j2) as loader:
        for i in range(34, 37):
            assert same(loader(i, st), ref[i]), i
        assert loader.stats()["resets"] == {"gap": 1}


def test_new_shards_do_not_change_planned_batches(tmp_path, tok):
    """A shard committed during the run joins at the next planned block (its source grew by
    more than ``reepoch_growth``, so the streams start a new epoch although the first one,
    about 60 batches long, is far from over); the batches of the blocks planned before are
    reproduced unchanged with the new shard on disk."""
    root = tmp_path / "data"
    rng = np.random.default_rng(3)
    write_docs(root, "climbmix", 0, rng, 2000, 8, 60)
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.0)
    j = tmp_path / "j.jsonl"
    with make(cfg, tok, root, j) as loader:
        before = [loader(i, st) for i in range(6)]
        write_docs(root, "climbmix", 1, rng, 1000, 8, 60, offset=4000)  # recognisable ids >= 4000
        after = [loader(i, st) for i in range(6, 30)]
        recs = loader.records
    assert [(r["start"], r["n_visible"]["climbmix"]) for r in recs][:3] == [(0, 1), (4, 1), (8, 2)]
    assert recs[-1]["universes"]["climbmix"][0] == 1 and recs[-1]["universes"]["climbmix"][-1] == 2

    def new_ids(b):
        return bool(((b["tokens"] >= 4000) & (b["tokens"] < 8000)).any())

    assert not any(new_ids(b) for b in before + after[:2])           # steps 0-7: planned before
    assert all(new_ids(b) for b in after[2:])                         # from the next block on
    with make(cfg, tok, root, j) as loader:            # reopened with both shards on disk
        for i in range(2, 30):
            ref = before[i] if i < 6 else after[i - 6]
            assert same(loader(i, st), ref), i


def test_epoch_caps_use_the_expected_size_while_a_source_grows(tmp_path, tok):
    """A source whose shards are still arriving is capped by the size the registry expects
    after its download (tokens per parquet byte times min(approx_bytes, download cap)), not by
    its first shards; ``growing_blocks`` blocks without a new shard, the cap uses what is on
    disk."""
    from bes.data import sources

    root = tmp_path / "data"
    rng = np.random.default_rng(5)
    write_docs(root, "climbmix", 0, rng, 60, 8, 60, input_size=10 ** 6)     # 1 MB of a big source
    write_convs(root, "smol_smoltalk", 0, lambda: chat_messages(rng, 1), 400, tok)
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.25)
    lc = dataclasses.replace(LC, growing_blocks=2)
    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
        for i in range(20):
            loader(i, st)
        recs = loader.records
    visible = float(sum(e[2] for e in loader.order["climbmix"]))
    src = sources.get("climbmix")
    expected = visible / 10 ** 6 * min(src.approx_bytes, sources.download_cap(src))
    for r in recs[:3]:                                    # new shards in this or the 2 before
        assert "climbmix" in r["growing"] and r["sizes"]["climbmix"] == pytest.approx(expected)
        assert r["mixture"]["kinds"]["base"] == pytest.approx(0.75)
    for r in recs[3:]:
        assert "climbmix" not in r["growing"] and r["sizes"]["climbmix"] == visible
        assert r["mixture"]["kinds"]["base"] < 0.3 and "climbmix" in r["mixture"]["capped"]


def test_shape_switch_and_restart_into_the_other_shape(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    short, long = stage_at(0.1, cfg), stage_at(0.6, cfg)
    assert short.shape == (2, 8, 64) and long.shape == (2, 4, 128)
    j = tmp_path / "j.jsonl"
    with make(cfg, tok, root, j) as loader:
        ref = {i: loader(i, short) for i in range(6)}
        ref.update({i: loader(i, long) for i in range(6, 11)})
        chain = [(b.start, b.end, b.shape) for b in loader.chain]
    assert chain == [(0, 4, short.shape), (4, 6, short.shape), (6, 10, long.shape), (10, 14, long.shape)]
    for i, b in ref.items():
        assert b["tokens"].shape == ((2, 8, 65) if i < 6 else (2, 4, 129))
    with make(cfg, tok, root, j) as loader:
        for i in (7, 8, 9, 10):
            assert same(loader(i, long), ref[i])
        assert same(loader(4, short), ref[4])
    # a restart that lands before the switch with the late shape plans that shape there
    with make(cfg, tok, root, j) as loader:
        b = loader(5, long)
        assert b["tokens"].shape == (2, 4, 129) and loader.stats()["resets"] == {"shape": 1}
        assert loader.chain[-1].start == 5 and loader.chain[-1].shape == long.shape
    with make(cfg, tok, root, j) as loader:            # the new plan is the effective one
        assert same(loader(5, long), b) and same(loader(3, short), ref[3])


def test_epochs_cover_the_source_and_reshuffle(tmp_path, tok):
    """A tiny source read for three epochs: every epoch reads every document exactly once
    (8 streams, 40 documents) in a new order."""
    root = tmp_path / "data"
    rng = np.random.default_rng(4)
    marker = 7000
    write_docs(root, "climbmix", 0, rng, 40, 4, 40, marker=marker)
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.0)
    V = LC.n_streams
    firsts: dict[int, list[int]] = {v: [] for v in range(V)}
    with make(cfg, tok, root, tmp_path / "j.jsonl") as loader:
        A, B, T = st.shape
        for i in range(40):
            b = loader(i, st)
            for a in range(A):
                for r in range(B):
                    v = (i * A * B + a * B + r) % V
                    tokens = b["tokens"][a, r]
                    for s, e in segments_of(tokens, b["seg"][a, r]):
                        if tokens[s] == BOS and e - s >= 2 and marker <= tokens[s + 1] < marker + 40:
                            firsts[v].append(int(tokens[s + 1]) - marker)   # a document's start
    share = 40 // V
    epochs = [sorted(sum((firsts[v][e * share:(e + 1) * share] for v in range(V)), [])) for e in range(3)]
    assert all(len(firsts[v]) >= 3 * share for v in range(V))
    assert epochs == [list(range(40))] * 3
    order = [[firsts[v][e * share:(e + 1) * share] for v in range(V)] for e in range(3)]
    assert order[0] != order[1] and order[1] != order[2]


# --------------------------------------------------------------------------- library
class FakePassage(SimpleNamespace):
    pass


class FakeLibrary:
    """Wiki articles and qa entries with the extended library contract (term overlap
    ranking, a score gate, kinds); ``fingerprint`` identifies it in the journal."""

    ARTICLES = [("Alan Turing", ["Alan Turing was an English mathematician. He worked on codes.",
                                 "Early life\nTuring was born in London in 1912. He liked science."]),
                ("Paris", ["Paris is the capital and largest city of France. The Seine flows through it.",
                           "History\nParis was founded long ago by a tribe. It grew into a large city."]),
                ("Rivers", ["A river is a natural flow of water. People often walk near quiet rivers."]),
                ("Apples", ["An apple is a round fruit. Apples can be red or green."])]
    QAS = [("What is the capital of France?", "Paris"), ("Who broke codes in the war?", "Alan Turing"),
           ("What colour are apples?", "Red or green")]

    def __init__(self, tag: str = "fake-1"):
        self.rows = [("wiki", t, b, k) for k, (t, bodies) in enumerate(self.ARTICLES) for b in bodies]
        self.n_wiki = len(self.rows)
        self.rows += [("qa", q, a, -1) for q, a in self.QAS]
        self.cfg = SimpleNamespace(gate=SimpleNamespace(allowed_types=("question", "lookup")))
        self.fingerprint = tag

    @property
    def n_passages(self):
        return len(self.rows)

    def kind_range(self, kind):
        return range(0, self.n_wiki) if kind == "wiki" else range(self.n_wiki, len(self.rows))

    def title_id_of(self, pid):
        return self.rows[pid][3]

    def title_range(self, tid):
        pids = [i for i, r in enumerate(self.rows) if r[3] == tid and tid >= 0]
        return range(pids[0], pids[-1] + 1) if pids else range(0)

    def find_title(self, title):
        return next((k for k, (t, _) in enumerate(self.ARTICLES) if t == title), None)

    def passage(self, pid, score=0.0):
        kind, title, body, tid = self.rows[pid]
        text = f"Q: {title}\nA: {body}" if kind == "qa" else f"{title}\n{body}"
        return FakePassage(pid=pid, title_id=tid, title=title, body=body, score=score, text=text, kind=kind)

    def _ranked(self, query, exclude_pids, exclude_titles, kinds):
        q = set(query.lower().replace("?", " ").split())
        out = []
        for pid, (kind, title, body, tid) in enumerate(self.rows):
            if pid in set(exclude_pids) or (tid >= 0 and tid in set(exclude_titles)):
                continue
            if kinds is not None and kind not in kinds:
                continue
            s = len(q & set((title + " " + body).lower().replace(".", " ").split()))
            if s:
                out.append((-s, pid))
        return [(pid, -s) for s, pid in sorted(out)]

    def retrieve(self, query, exclude_pids=(), exclude_titles=(), log=None, kinds=None):
        ranked = self._ranked(query, exclude_pids, exclude_titles, kinds)
        return self.passage(*ranked[0]) if ranked and ranked[0][1] >= 2 else None

    def retrieve_topk(self, query, k, exclude_pids=(), exclude_titles=(), log=None, kinds=None):
        return [self.passage(p, s) for p, s in self._ranked(query, exclude_pids, exclude_titles, kinds)[:k]]


def test_library_rows_retrieval_and_the_library_check(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, chat_frac=0.5, library_frac_base=0.3, library_frac_chat=1.0)
    mix = MixCfg(max_epochs=1000.0)               # the fake library is tiny: no epoch cap
    j = tmp_path / "j.jsonl"
    with make(cfg, tok, root, j, library=FakeLibrary(), mix=mix) as loader:
        batches = [loader(i, st) for i in range(8)]
        stats = loader.stats()
        rec = loader.records[0]
    assert rec["library"]["id"] == "fake-1" and rec["rows"][1] > 0.1 and rec["lib"]
    from bes.data import rag_format as R

    suitable = R.reading_suitable(FakeLibrary(), mix)          # reading tasks draw only these
    assert rec["library"]["tasks"] == {"rule": R.SUITABLE_RULE, "library_read": int(suitable.sum())}
    assert suitable.sum() > 0
    assert not any(k.startswith("library_read/") and k.endswith("declined") for b in stats["blocks"] for k in b["fail"])
    assert rec["probs"]["chat"][1] == 1.0                     # retrieval on every chat draw
    lib_turns = sum(int((b["tokens"] == LIBRARY).sum()) for b in batches)
    assert lib_turns > 0
    tokens = {k: v for b in stats["blocks"] for k, v in b["tokens"].items()}
    assert any(k.startswith("library/library/library_") for k in tokens)
    with make(cfg, tok, root, j, library=FakeLibrary(), mix=mix) as loader:
        assert same(loader(5, st), batches[5])
    with pytest.raises(ValueError, match="library"):
        make(cfg, tok, root, j, library=FakeLibrary("another"), mix=mix)
    # a library added later applies from the next block on
    j2 = tmp_path / "late.jsonl"
    with make(cfg, tok, root, j2, mix=mix) as loader:
        loader(0, st)
        loader.set_library(FakeLibrary())
        for i in range(1, 9):
            loader(i, st)
        used = [r["library"] is not None for r in loader.records]
    assert used[0] is False and used[-1] is True


class ClosedGate(FakeLibrary):
    """The fake library with a gate that never opens (ranked look-ups still work)."""

    def retrieve(self, query, exclude_pids=(), exclude_titles=(), log=None, kinds=None):
        return None


def test_chat_samples_carry_passages_only_where_the_gate_opens(tmp_path, tok):
    """Retrieval on every chat draw, no other source of library turns: with a gate that never
    opens no conversation gets a passage (a distractor only ever replaces a gated passage); with
    the fake gate some do, and Loader.stats() reports the realised shares."""
    root = tmp_path / "data"
    rng = np.random.default_rng(11)
    write_docs(root, "climbmix", 0, rng, 200, 8, 120)
    write_convs(root, "smol_smoltalk", 0, lambda: chat_messages(rng, int(rng.integers(1, 3))), 300, tok)
    cfg = small_cfg(hindsight_frac=0.0)
    st = stage(cfg, chat_frac=0.9, library_frac_base=0.0, library_frac_chat=1.0)
    mix = MixCfg(max_epochs=1000.0, library_weight=0.0)
    seen = {}
    for name, lib in (("closed", ClosedGate()), ("open", FakeLibrary())):
        with make(cfg, tok, root, tmp_path / f"{name}.jsonl", library=lib, mix=mix) as loader:
            batches = [loader(i, st) for i in range(6)]
            blocks = loader.stats()["blocks"]
        if name == "closed":
            assert not any((b["tokens"] == LIBRARY).any() for b in batches)
        n = sum(b["passages"]["chat"]["samples"] for b in blocks)
        m = sum(b["passages"]["chat"]["with_passage"] for b in blocks)
        rag = [b["retrieval"]["retrieval"] for b in blocks]
        seen[name] = (sum(int((b["tokens"] == LIBRARY).sum()) for b in batches), n, m, rag)
        for b in blocks:
            p = b["passages"]["chat"]
            assert p["passage_share"] == pytest.approx(p["with_passage"] / p["samples"])
            assert 0.0 <= p["distractor_share"] <= 1.0 and p["with_passage"] <= p["samples"]
    turns, n, m, rag = seen["closed"]
    assert n > 50 and m == 0 and turns == 0
    assert sum(r["calls"] for r in rag) > 50 and all(r["gate_rate"] == 0.0 for r in rag)
    turns, n, m, rag = seen["open"]
    assert n > 50 and 0 < m < n and 0 < turns <= m


def test_a_planned_share_cut_by_the_caps_is_warned_once(data, tok, tmp_path, caplog):
    """A source whose planned token share the epoch caps cut below two thirds of its target is
    named in one warning per run (with both numbers), however many blocks are planned, and
    listed in stats()["warnings"]; sources the caps do not cut that much are not."""
    import logging

    root, _ = data
    cfg = small_cfg()
    st = stage(cfg, p=0.01, chat_frac=0.5)                  # p < horizon_min_p: the configured horizon
    lc = dataclasses.replace(LC, horizon_tokens=3e5)
    with caplog.at_level(logging.WARNING, logger="bes.data.loader"):
        with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
            for i in range(14):
                loader(i, st)
            warnings = loader.stats()["warnings"]
            recs = loader.records
    assert len(recs) >= 3
    cut = {s for r in recs for s, want in r["mixture"]["target"].items()
           if r["mixture"]["share"].get(s, 0.0) < (2 / 3) * want}
    assert cut and {w["source"] for w in warnings} == cut
    logged = [r.getMessage() for r in caplog.records if r.name == "bes.data.loader"]
    for w in warnings:
        mine = [msg for msg in logged if f" of {w['source']} from " in msg]
        assert len(mine) == 1 and f"{w['target']:.4f}" in mine[0] and f"{w['share']:.4f}" in mine[0]
        rec = next(r for r in recs if r["id"] == w["block"])
        assert w["target"] == rec["mixture"]["target"][w["source"]] and w["share"] < (2 / 3) * w["target"]


def ref_heldout(name: str, shard: int, seg: int, frac: float) -> bool:
    """The held-out rule with Python integers (independent of the loader's numpy code)."""
    mask = (1 << 64) - 1

    def sm(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)

    import hashlib

    key = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8, person=b"bes-heldout").digest(), "little")
    return (sm(sm(sm(key) ^ shard) ^ seg) >> 11) < frac * 2 ** 53


def test_heldout_conversations_never_reach_training(tmp_path, tok):
    """About heldout_frac of the conversations of every chat, rag and math source (a hash of
    source, shard id and segment index) never appear in a batch over several epochs, every other
    one does; heldout_samples / Loader.heldout return exactly those, deterministically, without
    worker processes; the rule is part of the journal header."""
    root = tmp_path / "data"
    rng = np.random.default_rng(13)
    store: dict[str, list[np.ndarray]] = {}
    write_convs(root, "smol_smoltalk", 0, lambda: chat_messages(rng, 1), 120, tok, store)
    write_convs(root, "smol_smoltalk", 1, lambda: chat_messages(rng, 1), 60, tok, store)
    write_convs(root, "gsm8k", 0, lambda: math_messages(rng), 60, tok, store)
    write_convs(root, "squad_v2", 0, lambda: rag_messages(rng), 60, tok, store)
    write_docs(root, "climbmix", 0, rng, 100, 8, 60)
    frac = 0.15
    where = {"smol_smoltalk": [(0, i) for i in range(120)] + [(1, i) for i in range(60)],
             "gsm8k": [(0, i) for i in range(60)], "squad_v2": [(0, i) for i in range(60)]}
    held = {name: {tuple(store[name][k].tolist()) for k, (sh, i) in enumerate(pos) if ref_heldout(name, sh, i, frac)}
            for name, pos in where.items()}
    kept = {name: {tuple(c.tolist()) for c in store[name]} - held[name] for name in where}
    assert all(0 < len(held[n]) < 0.4 * len(store[n]) for n in where)
    # the API needs no loader and no worker
    got = LD.heldout_samples(root, "chat", 10 ** 6, seed=0, frac=frac)
    assert {tuple(s.ids.tolist()) for s in got} == held["smol_smoltalk"]
    assert all(s.source == "smol_smoltalk" and s.kind == "chat" and s.tags == ("heldout",) for s in got)
    some = LD.heldout_samples(root, "math", 3, seed=1, frac=frac)
    assert len(some) == 3 and {tuple(s.ids.tolist()) for s in some} <= held["gsm8k"]
    assert [s.key() for s in some] == [s.key() for s in LD.heldout_samples(root, "math", 3, seed=1, frac=frac)]
    assert LD.heldout_samples(root, "chat", 0, frac=frac) == [] and LD.heldout_samples(root, "rag", 5, frac=0.0) == []
    with pytest.raises(ValueError):
        LD.heldout_samples(root, "climbmix", 3)
    cfg = small_cfg(seq_len=128, micro_seqs=4, accum=2, hindsight_frac=0.0)
    st = stage(cfg, chat_frac=1.0, library_frac_base=0.0, library_frac_chat=0.0)
    lc = dataclasses.replace(LC, heldout_frac=frac)
    pieces: set[tuple] = set()
    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
        for i in range(70):
            b = loader(i, st)
            for a in range(b["tokens"].shape[0]):
                for r in range(b["tokens"].shape[1]):
                    tokens = b["tokens"][a, r]
                    pieces |= {tuple(tokens[s:e].tolist()) for s, e in segments_of(tokens, b["seg"][a, r])
                               if tokens[s] == BOS}
        epochs = loader.stats()["blocks"][-1]["epochs"]
        assert {tuple(s.ids.tolist()) for s in loader.heldout("squad_v2", 1000)} == held["squad_v2"]
        fails = sum(blk["fail"].get(f"{n}/heldout", 0) for blk in loader.stats()["blocks"] for n in where)
    assert all(epochs[n]["min"] >= 2 for n in where)            # every stream is in its third epoch
    for name in where:
        assert not (pieces & held[name]), name                  # never trained
        assert kept[name] <= pieces, name                       # everything else was
    assert fails > 0
    head = json.loads((tmp_path / "j.jsonl").read_text().splitlines()[0])["settings"]
    assert head["loader"]["heldout_frac"] == frac and head["heldout"]["rule"] == LD.HELDOUT_RULE
    with pytest.raises(ValueError, match="other settings"):
        make(cfg, tok, root, tmp_path / "j.jsonl", lc=dataclasses.replace(lc, heldout_frac=0.01))


# --------------------------------------------------------------------------- journal
def test_journal_records_and_settings_check(data, tok, tmp_path):
    root, _ = data
    cfg = small_cfg()
    st = stage(cfg)
    j = tmp_path / "j.jsonl"
    with make(cfg, tok, root, j) as loader:
        for i in range(5):
            loader(i, st)
    lines = j.read_text().splitlines()
    head, recs = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    assert head["type"] == "header" and head["settings"]["seed"] == 7
    for r in recs:
        states = LD.decode_states(r["states"])
        assert len(states) == LC.n_streams and all(s[7] == [] for s in states)   # no deficits
        assert {"start", "shape", "stage", "rows", "base", "chat", "probs", "new_shards",
                "n_visible", "universes", "consumed"} <= set(r)
    with pytest.raises(ValueError, match="other settings"):
        make(cfg, tok, root, j, seed=8)
    with pytest.raises(ValueError, match="other settings"):
        make(cfg, tok, root, j, lc=dataclasses.replace(LC, lookahead=7))
    with open(j, "ab") as f:                      # a torn last line (crash during an append)
        f.write(b'{"type": "block", "id": 99, "sta')
    with make(cfg, tok, root, j) as loader:
        assert len(loader.records) == len(recs)
        loader(5, st)


def test_iterator_data_adapter(data, tok, tmp_path):
    from bes.train.train import IteratorData

    root, _ = data
    cfg = small_cfg()
    short, long = stage_at(0.1, cfg), stage_at(0.6, cfg)
    with make(cfg, tok, root, tmp_path / "a.jsonl") as loader:
        ref = [loader(i, short) for i in range(5)] + [loader(i, long) for i in range(5, 7)]
    with make(cfg, tok, root, tmp_path / "b.jsonl") as loader:
        it = IteratorData(loader.open)
        got = [it(i, short) for i in range(5)] + [it(i, long) for i in range(5, 7)]
    assert all(same(x, y) for x, y in zip(ref, got))


def test_no_data_is_a_clear_error(tmp_path, tok):
    cfg = small_cfg()
    (tmp_path / "data" / "tok").mkdir(parents=True)
    with make(cfg, tok, tmp_path / "data", tmp_path / "j.jsonl") as loader:
        with pytest.raises(RuntimeError, match="no training data"):
            loader(0, stage(cfg))


# ------------------------------------------------------------------------ end to end
def test_trainer_restart_receives_the_same_batches(data, tok, tmp_path):
    """bes.train.train.Trainer with ``data=loader``: a run killed after a checkpoint and
    restarted (new process state, same journal) is fed the batches of an uninterrupted run."""
    import hashlib

    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from bes.config import RunCfg
    from bes.train import schedule as S
    from bes.train.train import Trainer

    root, _ = data
    base = small_cfg(chat_frac=0.0)
    cfg = dataclasses.replace(base, run=RunCfg(deadline_hours=0.1, post_reserve_hours=0.02,
                                               ckpt_every_min=0.25, log_every=10, guard_warmup=5))
    n_tok, dim = 64, 16

    def init_fn(model_cfg, key):
        k1, k2 = jax.random.split(key)
        return {"emb": jax.random.normal(k1, (n_tok, dim)), "out": jax.random.normal(k2, (dim, n_tok)) / 4}

    def loss_fn(params, consts, batch, model_cfg):
        t = batch["tokens"] % n_tok
        z = params["emb"][t[:, :-1]] @ params["out"]
        ce = jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, t[:, 1:, None], -1)[..., 0]
        return jnp.sum(batch["w"] * ce) / jnp.maximum(jnp.sum(batch["w"]), 1e-9), {}

    class Killed(Exception):
        pass

    def run(out, journal, clock, crash_at=None):
        seen: dict[int, bytes] = {}
        with make(cfg, tok, root, journal) as loader:
            def feed(index, st):
                if index == crash_at:
                    raise Killed(index)
                clock.advance(2.0)
                b = loader(index, st)
                seen[index] = hashlib.blake2b(b"".join(b[k].tobytes() for k in sorted(b))).digest()
                return b

            tr = Trainer(cfg, loss_fn=loss_fn, init_fn=init_fn, role_fn={"emb": "embed", "out": "matrix"}.get,
                         data=feed, out_dir=out, clock=clock, max_steps=14, batch_keys=("tokens", "w"))
            try:
                tr.run()
            except Killed:
                pass
        return seen

    whole = run(tmp_path / "a", tmp_path / "a.jsonl", S.FakeClock(1.0e6))
    clock = S.FakeClock(1.0e6)
    before = run(tmp_path / "b", tmp_path / "b.jsonl", clock, crash_at=11)
    clock.advance(30.0)                                           # downtime
    after = run(tmp_path / "b", tmp_path / "b.jsonl", clock)
    assert sorted(whole) == list(range(14)) and sorted(before) == list(range(11))
    assert min(after) < 11 and max(after) == 13                   # restored from a checkpoint
    assert all(before[i] == whole[i] for i in before)
    assert all(after[i] == whole[i] for i in after)


def test_ten_batches_through_the_model_loss(data, tok, tmp_path):
    jax = pytest.importorskip("jax")
    M = pytest.importorskip("bes.model.model")
    from bes.config import load_config

    root, _ = data
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    params = M.init_params(cfg.model, jax.random.key(0))
    consts = M.make_consts(cfg.model)
    loss = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, cfg.model)[0])
    stages = [stage_at(0.1, cfg)] * 5 + [stage_at(0.6, cfg)] * 5
    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=dataclasses.replace(LC, block_steps=3)) as loader:
        for i, st in enumerate(stages):
            batch = loader(i, st)
            for a in range(batch["tokens"].shape[0]):
                value = float(loss(params, consts, {k: v[a] for k, v in batch.items()}))
                assert np.isfinite(value) and 0 < value < 20


# --------------------------------------------------------------------- stock refusals
REFUSAL_TEXT = "I'm sorry, but I can't assist with that. It's important to respect user privacy."


def refusing_messages(rng, k: int) -> list[dict]:
    """Every third conversation refuses (in its first or its second assistant turn)."""
    msgs = chat_messages(rng, 2)
    if k % 3 == 0:
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        msgs[turns[(k // 3) % 2]]["content"] = REFUSAL_TEXT
    return msgs


def test_conversations_with_a_stock_refusal_never_reach_training(tmp_path, tok, monkeypatch):
    """A conversation in which an assistant turn opens with a stock refusal is skipped like a
    held-out one and counted; every other conversation is trained on. A journal written without
    the rule is continued with it (the run of 23-24 Nov got the rule at a tenth of its way)."""
    from bes.data.samples import stock_refusal

    root = tmp_path / "data"
    rng = np.random.default_rng(5)
    store: dict[str, list[np.ndarray]] = {}
    count = iter(range(10 ** 6))
    write_convs(root, "smol_smoltalk", 0, lambda: refusing_messages(rng, next(count)), 150, tok, store)
    write_docs(root, "climbmix", 0, rng, 100, 8, 60)
    convs = store["smol_smoltalk"]
    refused = {tuple(c.tolist()) for k, c in enumerate(convs) if k % 3 == 0}
    assert all(stock_refusal(c, tok) == (k % 3 == 0) for k, c in enumerate(convs))
    kept = {tuple(c.tolist()) for c in convs} - refused
    cfg = small_cfg(seq_len=128, micro_seqs=4, accum=2, hindsight_frac=0.0)
    st = stage(cfg, chat_frac=1.0, library_frac_base=0.0, library_frac_chat=0.0)
    lc = dataclasses.replace(LC, heldout_frac=0.0)

    def pieces_of(loader, steps) -> set[tuple]:
        out: set[tuple] = set()
        for i in steps:
            b = loader(i, st)
            for a in range(b["tokens"].shape[0]):
                for r in range(b["tokens"].shape[1]):
                    tokens = b["tokens"][a, r]
                    out |= {tuple(tokens[s:e].tolist()) for s, e in segments_of(tokens, b["seg"][a, r])
                            if tokens[s] == BOS}
        return out

    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
        pieces = pieces_of(loader, range(60))
        fails = sum(blk["fail"].get("smol_smoltalk/refusal", 0) for blk in loader.stats()["blocks"])
    assert not (pieces & refused) and kept <= pieces and fails > 0
    # a journal written without the rule: the refusals were trained on ... (the later, wider rule
    # for deflections would leave "I'm sorry, but ..." out as well: off too)
    monkeypatch.setattr(LD, "DROP_STOCK_REFUSALS", False)
    monkeypatch.setattr(LD, "DROP_STOCK_DEFLECTIONS", False)
    with make(cfg, tok, root, tmp_path / "old.jsonl", lc=lc) as loader:
        before = pieces_of(loader, range(11))
    assert before & refused
    # ... and the same journal continues with the rule, from the middle of a block
    monkeypatch.setattr(LD, "DROP_STOCK_REFUSALS", True)
    with make(cfg, tok, root, tmp_path / "old.jsonl", lc=lc) as loader:
        after = pieces_of(loader, range(11, 60))
    assert not (after & refused) and len(after) > 20


DEFLECTION_TEXT = "As an AI language model, I don't have access to real-time data. However, here is some general advice."


def test_chat_conversations_that_open_with_a_deflection_never_reach_training(tmp_path, tok, monkeypatch):
    """A chat conversation in which an assistant turn opens by saying what the assistant is or
    lacks ("As an AI language model, I don't have access to ...") is skipped and counted, like a
    stock refusal; the rule is for sources of kind "chat" only, and a journal written without it
    is continued with it (the run of 23-24 Nov got the rule at 0.42 of its way)."""
    from bes.data.samples import stock_deflection, stock_refusal

    def messages(rng, k: int) -> list[dict]:
        msgs = refusing_messages(rng, 1)                           # k = 1: no refusal in it
        if k % 4 == 0:
            turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            msgs[turns[(k // 4) % 2]]["content"] = DEFLECTION_TEXT
        return msgs

    root = tmp_path / "data"
    rng = np.random.default_rng(7)
    store: dict[str, list[np.ndarray]] = {}
    count = iter(range(10 ** 6))
    write_convs(root, "smol_smoltalk", 0, lambda: messages(rng, next(count)), 160, tok, store)
    write_docs(root, "climbmix", 0, rng, 100, 8, 60)
    convs = store["smol_smoltalk"]
    assert all(stock_deflection(c, tok) == (k % 4 == 0) for k, c in enumerate(convs))
    assert not any(stock_refusal(c, tok) for c in convs)
    deflecting = {tuple(c.tolist()) for k, c in enumerate(convs) if k % 4 == 0}
    kept = {tuple(c.tolist()) for c in convs} - deflecting
    cfg = small_cfg(seq_len=128, micro_seqs=4, accum=2, hindsight_frac=0.0)
    st = stage(cfg, chat_frac=1.0, library_frac_base=0.0, library_frac_chat=0.0)
    lc = dataclasses.replace(LC, heldout_frac=0.0)

    def pieces_of(loader, steps) -> set[tuple]:
        out: set[tuple] = set()
        for i in steps:
            b = loader(i, st)
            for a in range(b["tokens"].shape[0]):
                for r in range(b["tokens"].shape[1]):
                    tokens = b["tokens"][a, r]
                    out |= {tuple(tokens[s:e].tolist()) for s, e in segments_of(tokens, b["seg"][a, r])
                            if tokens[s] == BOS}
        return out

    with make(cfg, tok, root, tmp_path / "j.jsonl", lc=lc) as loader:
        pieces = pieces_of(loader, range(60))
        fails = sum(blk["fail"].get("smol_smoltalk/deflection", 0) for blk in loader.stats()["blocks"])
    assert not (pieces & deflecting) and kept <= pieces and fails > 0
    # sources of another kind keep such conversations (there "the passage does not say" is the task)
    monkeypatch.setattr(LD, "DEFLECTION_KINDS", ("none",))
    with make(cfg, tok, root, tmp_path / "other.jsonl", lc=lc) as loader:
        assert pieces_of(loader, range(20)) & deflecting
    monkeypatch.setattr(LD, "DEFLECTION_KINDS", ("chat",))
    # a journal written without the rule continues with it, from the middle of a block
    monkeypatch.setattr(LD, "DROP_STOCK_DEFLECTIONS", False)
    with make(cfg, tok, root, tmp_path / "old.jsonl", lc=lc) as loader:
        before = pieces_of(loader, range(11))
    assert before & deflecting
    monkeypatch.setattr(LD, "DROP_STOCK_DEFLECTIONS", True)
    with make(cfg, tok, root, tmp_path / "old.jsonl", lc=lc) as loader:
        after = pieces_of(loader, range(11, 60))
    assert not (after & deflecting) and len(after) > 20
