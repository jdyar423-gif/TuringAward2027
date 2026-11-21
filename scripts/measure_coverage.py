"""Gate E1: coverage of the written library on open-domain questions.

    python scripts/measure_coverage.py --stage tune             # BM25 weights, dev questions only
    python scripts/measure_coverage.py --stage pool             # exact BM25 over every passage
    python scripts/measure_coverage.py --stage lib --budgets 20 # one library budget (repeat)
    python scripts/measure_coverage.py --stage report           # JSON files -> docs/E1_COVERAGE.md

Each stage stays below ten minutes on 2 threads and below 3 GB; results go to
``data_cache/library/e1_*.json``.

Question sets (Hugging Face, revisions pinned below): NQ-open validation (all 3,610),
TriviaQA rc.nocontext validation (seeded sample), PopQA test (seeded sample, with the
subject popularity ``s_pop``), and for the "everyday prompt" gate check, Dolly-15k
instructions by category. Dev data for every choice made here (BM25 weights, gate
thresholds): an NQ-open *train* sample and a TriviaQA validation sample disjoint from the
evaluation sample.

A question counts as answered at k when one of its normalised answer strings occurs, as a
whole-token sequence, in one of the top-k passages (title line + body).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bes.library.build import build_for_budget, prepare_pool, prepare_signals  # noqa: E402
from bes.library.clean import render_passage  # noqa: E402
from bes.library.index import (ExactBM25, IndexCfg, Postings, TruncatedIndex, bucket_map,  # noqa: E402
                               lead_flags, rekey_postings, term_postings)
from bes.library.pack import PackCfg  # noqa: E402
from bes.library.retrieve import GateCfg, Library, ReadLog  # noqa: E402
from bes.library.select import importance, passage_prior  # noqa: E402
from bes.library.terms import terms, tokens  # noqa: E402

REVISIONS = {
    "nq_open": ("google-research-datasets/nq_open", "5dd9790a83002ad084ddeb7c420dc716852c6f28"),
    "trivia_qa": ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"),
    "popqa": ("akariasai/PopQA", "098765c79ea10a2cb19c828324e33281b8336ec0"),
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"),
}
FILES = {
    "nq_val": ("nq_open", "nq_open/validation-00000-of-00001.parquet"),
    "nq_train": ("nq_open", "nq_open/train-00000-of-00001.parquet"),
    "tqa_val": ("trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet"),
    "popqa": ("popqa", "test.tsv"),
    "dolly": ("dolly", "databricks-dolly-15k.jsonl"),
}


def t_now(t0: float) -> str:
    return f"[{time.time() - t0:7.1f}s]"


# --------------------------------------------------------------------------- data


def fetch(cache_dir: str, key: str) -> str:
    from huggingface_hub import hf_hub_download

    d, fn = FILES[key]
    out = os.path.join(cache_dir, "raw", d)
    path = os.path.join(out, fn)
    if os.path.exists(path):
        return path
    repo, rev = REVISIONS[d]
    os.makedirs(out, exist_ok=True)
    return hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=out)


def norm(s: str) -> str:
    return " ".join(tokens(s))


def _answers(xs) -> list[str]:
    out = []
    for a in xs:
        n = norm(a)
        if n and n not in out:
            out.append(n)
    return out


def load_sets(cache_dir: str, n_tqa: int, n_pop: int, n_dev: int, seed: int) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    sets: dict[str, list[dict]] = {}
    t = pq.read_table(fetch(cache_dir, "nq_val")).to_pylist()
    sets["nq"] = [{"q": r["question"], "ans": _answers(r["answer"])} for r in t]
    t = pq.read_table(fetch(cache_dir, "nq_train")).to_pylist()
    idx = rng.sample(range(len(t)), n_dev)
    sets["dev_nq"] = [{"q": t[i]["question"], "ans": _answers(t[i]["answer"])} for i in idx]
    t = pq.read_table(fetch(cache_dir, "tqa_val"), columns=["question", "answer"]).to_pylist()
    idx = rng.sample(range(len(t)), n_tqa + n_dev)
    rows = [{"q": t[i]["question"], "ans": _answers(t[i]["answer"]["aliases"]
                                                     + t[i]["answer"]["normalized_aliases"]
                                                     + [t[i]["answer"]["value"]])} for i in idx]
    sets["tqa"], sets["dev_tqa"] = rows[:n_tqa], rows[n_tqa:]
    csv.field_size_limit(1 << 30)
    with open(fetch(cache_dir, "popqa")) as f:
        prow = list(csv.DictReader(f, delimiter="\t"))
    idx = rng.sample(range(len(prow)), n_pop)
    sets["popqa"] = [{"q": prow[i]["question"], "ans": _answers(json.loads(prow[i]["possible_answers"])),
                      "pop": int(prow[i]["s_pop"]), "prop": prow[i]["prop"]} for i in idx]
    with open(fetch(cache_dir, "dolly")) as f:
        dolly = [json.loads(line) for line in f]
    by_cat: dict[str, list[dict]] = {}
    for r in dolly:
        by_cat.setdefault(r["category"], []).append(r)
    sets["dolly"] = []
    for c in sorted(by_cat):
        for r in rng.sample(by_cat[c], min(150, len(by_cat[c]))):
            sets["dolly"].append({"q": r["instruction"], "ans": [], "cat": c})
    for k, v in sets.items():
        if k != "dolly":
            sets[k] = [x for x in v if x["ans"]]
    return sets


QA_SETS = ("nq", "tqa", "popqa")
DEV_SETS = ("dev_nq", "dev_tqa")


# --------------------------------------------------------------------------- scoring


class Texts:
    """Normalised passage text by pid (title line + body), cached for small ranks."""

    def __init__(self, passages: list[tuple[str, str]]):
        self.passages = passages
        self.cache: dict[int, str] = {}

    def get(self, pid: int) -> str:
        s = self.cache.get(pid)
        if s is None:
            t, b = self.passages[pid]
            s = " " + norm(render_passage(t, b)) + " "
            if len(self.cache) < 400_000:
                self.cache[pid] = s
        return s


def hit_rank(pids, ans: list[str], texts: Texts) -> int:
    """1-based rank of the first passage containing an answer, 0 if none."""
    pads = [" " + a + " " for a in ans]
    for r, p in enumerate(pids):
        t = texts.get(int(p))
        if any(a in t for a in pads):
            return r + 1
    return 0


def recall(ranks: list[int], k: int) -> float:
    return float(np.mean([1.0 if 0 < r <= k else 0.0 for r in ranks])) if ranks else 0.0


def summarize(ranks: list[int], ks=(1, 3, 20, 100)) -> dict:
    return {f"r@{k}": recall(ranks, k) for k in ks} | {"n": len(ranks)}


# --------------------------------------------------------------------------- tuning


def split_postings(docs: list[tuple[str, str]]):
    """True-term postings with title and body counts kept apart (for weight tuning)."""
    from array import array

    vocab: dict[str, int] = {}
    keys, pids, tt, tb = array("q"), array("i"), array("f"), array("f")
    for pid, (title, body) in enumerate(docs):
        c: dict[int, list[float]] = {}
        for t in terms(title):
            k = vocab.setdefault(t, len(vocab))
            c.setdefault(k, [0.0, 0.0])[0] += 1.0
        for t in terms(body):
            k = vocab.setdefault(t, len(vocab))
            c.setdefault(k, [0.0, 0.0])[1] += 1.0
        keys.extend(c.keys())
        pids.extend([pid] * len(c))
        for a, b in c.values():
            tt.append(a)
            tb.append(b)
    return (np.frombuffer(keys, np.int64).copy(), np.frombuffer(pids, np.int32).copy(),
            np.frombuffer(tt, np.float32).copy(), np.frombuffer(tb, np.float32).copy(), list(vocab))


def weighted(sp, n_docs: int, leads: np.ndarray, tw: float, ltw: float) -> Postings:
    key, pid, tt, tb, _ = sp
    w = np.where(leads[pid], 1.0 + ltw, 1.0 + tw).astype(np.float32)
    tf = tb + w * tt
    dl = np.bincount(pid, weights=tf, minlength=n_docs).astype(np.float32)
    return Postings(key=key, pid=pid, tf=tf, dl=dl)


def stage_tune(a, t0) -> dict:
    pool = prepare_pool(a.cache_dir, log=lambda m: print(t_now(t0), m, flush=True))
    sig = prepare_signals(pool, a.cache_dir, PackCfg(), log=lambda m: print(t_now(t0), m, flush=True))
    sets = load_sets(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dev, a.seed)
    order = sig.order[~sig.dup[sig.order]]
    docs = [(pool.titles[int(pool.p_article[i])], pool.body(int(i))) for i in order]
    leads = np.asarray(lead_flags(docs))
    texts = Texts(docs)
    cache = os.path.join(a.cache_dir, "pool", "split_postings.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        sp = (z["key"], z["pid"], z["tt"], z["tb"], [str(x) for x in z["vocab"]])
    else:
        sp = split_postings(docs)
        np.savez(cache, key=sp[0], pid=sp[1], tt=sp[2], tb=sp[3], vocab=np.asarray(sp[4]))
    print(t_now(t0), "tune: split postings", sp[0].shape[0], flush=True)
    grid = []
    for k1, b in [(0.9, 0.4), (1.2, 0.75), (0.6, 0.3), (0.9, 0.2), (0.9, 0.6), (1.5, 0.4)]:
        for tw in (0.0, 1.0, 2.0):
            for ltw in (tw, tw + 2.0, tw + 5.0):
                grid.append((k1, b, tw, ltw))
    part = [int(x) for x in a.tune_part.split("/")]
    grid = grid[part[0]::part[1]]
    out = []
    for k1, b, tw, ltw in grid:
        ex = ExactBM25(weighted(sp, len(docs), leads, tw, ltw), sp[4], k1, b)
        res = {}
        for s in DEV_SETS:
            ranks = [hit_rank(ex.topk(x["q"], 3)[0], x["ans"], texts) for x in sets[s]]
            res[s] = {"r@1": recall(ranks, 1), "r@3": recall(ranks, 3)}
        score = float(np.mean([res[s]["r@1"] for s in DEV_SETS]))
        out.append({"k1": k1, "b": b, "title_weight": tw, "lead_title_weight": ltw,
                    "dev": res, "mean_r1": score})
        print(t_now(t0), f"tune k1={k1} b={b} tw={tw} ltw={ltw}: "
              + " ".join(f"{s} {res[s]['r@1']:.3f}/{res[s]['r@3']:.3f}" for s in DEV_SETS), flush=True)
        del ex
        gc.collect()
    return {"tune": out}


# --------------------------------------------------------------------------- gate


def gate_grid():
    for ms in (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0):
        for mm in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 2.0):
            for mmt in (0.0, 0.02, 0.05, 0.1, 0.2):
                for mtm in (0.5, 0.75, 1.0):
                    for mc in (0.0, 0.5, 0.7, 0.85):
                        yield GateCfg(min_score=ms, min_margin=mm, min_margin_title=mmt,
                                      min_title_match=mtm, min_coverage=mc)


def gate_eval(feats: list[dict], hit1: np.ndarray, g: GateCfg) -> dict:
    s1 = np.array([f["s1"] for f in feats])
    m = np.array([f["margin"] for f in feats])
    tm = np.array([f["title_match"] for f in feats])
    cv = np.array([f.get("coverage", 1.0) for f in feats])
    ok_t = np.array([f["qtype"] in g.allowed_types for f in feats])
    p = ok_t & (s1 >= g.min_score) & (cv >= g.min_coverage) & (
        (m >= g.min_margin) | ((tm >= g.min_title_match) & (m >= g.min_margin_title)))
    cov = float(p.mean()) if p.size else 0.0
    prec = float(hit1[p].mean()) if p.any() else 0.0
    return {"coverage": cov, "precision": prec, "answered": float((p & hit1).mean()) if p.size else 0.0}


def choose_gate(dev_feats: list[dict], dev_hit: np.ndarray, target: float) -> tuple[GateCfg, list[dict]]:
    rows = []
    for g in gate_grid():
        r = gate_eval(dev_feats, dev_hit, g)
        rows.append((g, r))
    ok = [x for x in rows if x[1]["precision"] >= target]
    best = max(ok, key=lambda x: (x[1]["answered"], -x[0].min_score)) if ok else max(rows, key=lambda x: x[1]["precision"])
    # Pareto frontier on dev (coverage vs precision)
    pts = sorted(rows, key=lambda x: (-x[1]["coverage"], -x[1]["precision"]))
    front, best_p = [], -1.0
    for g, r in pts:
        if r["precision"] > best_p + 1e-9:
            front.append((g, r))
            best_p = r["precision"]
    return best[0], [{"gate": dataclasses.asdict(g), **r} for g, r in front]


# --------------------------------------------------------------------------- main stage


def eval_library(name: str, lib: Library, passages: list[tuple[str, str]], prior: np.ndarray,
                 sets: dict, budgets: list[int], icfg: IndexCfg, t0: float, extra=()) -> dict:
    out: dict = {"accounting": lib.accounting()}
    texts = Texts(passages)
    p, tl = term_postings(passages, icfg.title_weight, icfg.lead_title_weight)
    ex = ExactBM25(p, tl, icfg.k1, icfg.b)
    res = {}
    for s in QA_SETS + DEV_SETS:
        res[s] = [hit_rank(ex.topk(x["q"], 3)[0], x["ans"], texts) for x in sets[s]]
    out["exact_lib"] = {s: summarize(r, (1, 3)) for s, r in res.items()}
    out["exact_lib_ranks_popqa"] = res["popqa"]
    del ex
    gc.collect()
    print(t_now(t0), name, "exact over library done", flush=True)
    variants = [dataclasses.replace(icfg, budget_numbers=b) for b in budgets] + list(extra)
    out["index"] = []
    for bits in sorted({v.n_bits for v in variants}):
        hp = rekey_postings(p, bucket_map(tl, bits))
        full = TruncatedIndex.from_postings(hp, dataclasses.replace(icfg, n_bits=bits, budget_numbers=0),
                                            prior, keep_full=True)
        rf = {s: [hit_rank(full.score(x["q"], full=True)[0][:3], x["ans"], texts) for x in sets[s]]
              for s in QA_SETS + DEV_SETS}
        out["index"].append({"n_bits": bits, "budget": None, "numbers": None, "postings": int(hp.key.shape[0]),
                             "max_list": None, **{s: summarize(r, (1, 3)) for s, r in rf.items()}})
        del full
        for v in variants:
            if v.n_bits != bits:
                continue
            idx = TruncatedIndex.from_postings(hp, v, prior)
            rk = {s: [hit_rank(idx.score(x["q"])[0][:3], x["ans"], texts) for x in sets[s]]
                  for s in QA_SETS + DEV_SETS}
            acc = idx.accounting()
            row = {"n_bits": bits, "budget": v.budget_numbers, "numbers": acc["index_numbers"],
                   "postings": acc["index_postings"], "max_list": v.max_list,
                   **{s: summarize(r, (1, 3)) for s, r in rk.items()}}
            if v == icfg:
                row["popqa_ranks"] = rk["popqa"]
                row["default"] = True
            out["index"].append(row)
            print(t_now(t0), name, f"bits={bits} budget={v.budget_numbers} max_list={v.max_list}:",
                  " ".join(f"{s} {row[s]['r@1']:.3f}/{row[s]['r@3']:.3f}" for s in QA_SETS), flush=True)
            del idx
        del hp
        gc.collect()
    del p
    gc.collect()
    # gate features from the stored library itself (default index)
    feats, hits = {}, {}
    for s in QA_SETS + DEV_SETS + ("dolly",):
        fs, hs = [], []
        for x in sets[s]:
            pids, sc, smax = lib.search_max(x["q"])
            f = lib.gate_features(x["q"], pids, sc, smax=smax)
            fs.append({k: f[k] for k in ("s1", "s2", "margin", "title_match", "coverage", "qtype")})
            hs.append(bool(x["ans"]) and hit_rank(pids[:1], x["ans"], texts) == 1)
        feats[s], hits[s] = fs, hs
    out["gate_feats"] = feats
    out["gate_hits"] = hits
    print(t_now(t0), name, "gate features done", flush=True)
    return out


def string_scan(passages: list[tuple[str, str]], qs: list[dict], n: int, seed: int) -> dict:
    """Share of questions whose answers occur nowhere in the library text (sample)."""
    rng = random.Random(seed)
    sample = rng.sample(qs, min(n, len(qs)))
    big = " " + " | ".join(norm(render_passage(t, b)) for t, b in passages) + " "
    absent = 0
    for x in sample:
        if not any((" " + a + " ") in big for a in x["ans"]):
            absent += 1
    del big
    return {"n": len(sample), "absent": absent, "absent_share": absent / max(1, len(sample))}


def _setup(a, t0):
    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    pool = prepare_pool(a.cache_dir, log=log)
    sig = prepare_signals(pool, a.cache_dir, PackCfg(), log=log)
    sets = load_sets(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dev, a.seed)
    log("sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    return log, pool, sig, sets


def stage_pool(a, t0) -> dict:
    """Exact BM25 over every pool passage, and the answer-string scan."""
    log, pool, sig, sets = _setup(a, t0)
    icfg = IndexCfg()
    R: dict = {"index_cfg": dataclasses.asdict(icfg), "n": {k: len(v) for k, v in sets.items()},
               "pool": {"articles": pool.n_articles, "passages": pool.n_passages,
                        "body_bytes": len(pool.body_blob), "drops": pool.drops,
                        "duplicates": int(sig.dup.sum())}}
    z = np.load(os.path.join(a.cache_dir, "pool", "signals.npz"))
    R["pool"]["packed_numbers"] = int(z["pool_numbers"])
    docs = [(pool.titles[int(pool.p_article[i])], pool.body(int(i))) for i in sig.order]
    texts = Texts(docs)
    ex = ExactBM25.build(docs, icfg.k1, icfg.b, icfg.title_weight, icfg.lead_title_weight)
    log("exact pool index built")
    pool_ranks = {}
    for s in QA_SETS + DEV_SETS:
        pool_ranks[s] = [hit_rank(ex.topk(x["q"], 100)[0], x["ans"], texts) for x in sets[s]]
    R["exact_pool"] = {s: summarize(r) for s, r in pool_ranks.items()}
    R["exact_pool_ranks_popqa"] = pool_ranks["popqa"]
    del ex, texts
    gc.collect()
    log("exact pool done: " + " ".join(f"{s} {R['exact_pool'][s]['r@1']:.3f}" for s in QA_SETS))
    R["string_scan_nq"] = string_scan(docs, sets["nq"], a.scan_sample, a.seed)
    R["string_scan_tqa"] = string_scan(docs, sets["tqa"], a.scan_sample, a.seed)
    log(f"string scan: {R['string_scan_nq']} {R['string_scan_tqa']}")
    return R


def stage_readset(a, t0) -> dict:
    """Read set and gate decisions of the saved library on every question set."""
    sets = load_sets(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dev, a.seed)
    lib = Library.load(a.library)
    out = {"library": a.library, "gate": dataclasses.asdict(lib.cfg.gate),
           "accounting": lib.accounting(), "sets": {}}
    for s in QA_SETS + ("dolly",):
        tots, passed, idx_reads, blocks = [], [], [], []
        for x in sets[s]:
            rl = ReadLog()
            r = lib.retrieve(x["q"], log=rl)
            tots.append(rl.total)
            idx_reads.append(rl.index)
            passed.append(r is not None)
            blocks.append(len(rl.blocks))
        tp = [t for t, p in zip(tots, passed) if p]
        out["sets"][s] = {"n": len(tots), "passed": int(sum(passed)),
                          "mean_all": float(np.mean(tots)), "max_all": int(np.max(tots)),
                          "mean_index": float(np.mean(idx_reads)), "max_index": int(np.max(idx_reads)),
                          "mean_passed": float(np.mean(tp)) if tp else 0.0,
                          "p99_passed": float(np.percentile(tp, 99)) if tp else 0.0,
                          "max_passed": int(np.max(tp)) if tp else 0, "max_text_blocks": int(np.max(blocks))}
        print(t_now(t0), s, out["sets"][s], flush=True)
    return out


def stage_lib(a, t0, budget: int) -> dict:
    """All measurements for the library of one budget."""
    log, pool, sig, sets = _setup(a, t0)
    icfg = IndexCfg()
    name = f"{budget // 1_000_000}M"
    lib, sel = build_for_budget(pool, sig, budget, icfg, PackCfg(), log=log)
    sel_path = os.path.join(a.cache_dir, f"e1_sel_{name}.npy")
    np.save(sel_path, sel.pool_idx)
    for other in ([] if a.force else sorted(os.listdir(a.cache_dir))):
        if other.startswith("e1_sel_") and other != os.path.basename(sel_path):
            prev = np.load(os.path.join(a.cache_dir, other))
            oname = other[len("e1_sel_"):-len(".npy")]
            if np.array_equal(prev, sel.pool_idx) and os.path.exists(
                    os.path.join(a.cache_dir, f"e1_lib_{oname}.json")):
                log(f"{name}: same selection as {oname}, reusing its measurements")
                return {"same_as": oname, "accounting": lib.accounting()}
    rng = random.Random(a.seed)
    for pid in rng.sample(range(lib.n_passages), 200):  # stored text == rendered passage
        assert lib.passage(pid).text == render_passage(*sel.passages[pid]), pid
    extra = ([dataclasses.replace(icfg, n_bits=17), dataclasses.replace(icfg, n_bits=19),
              dataclasses.replace(icfg, max_list=0xFFFF)] if a.extra_bits else [])
    imp = importance(pool, sig.mentions)
    prior = passage_prior(pool, imp)[sel.pool_idx]
    res = eval_library(name, lib, sel.passages, prior, sets, a.index_budgets, icfg, t0, extra)
    res["selection"] = {"passages": int(sel.mask.sum()), "history": sel.history,
                        "articles": lib.packed.n_titles,
                        "lead_passages": int((pool.p_pos[sel.pool_idx] == 0).sum())}
    tots, passed = [], 0
    for x in sets["nq"]:
        rl = ReadLog()
        r = lib.retrieve(x["q"], log=rl)
        tots.append(rl.total)
        passed += r is not None
    res["readset_nq"] = {"mean": float(np.mean(tots)), "p50": float(np.percentile(tots, 50)),
                         "p99": float(np.percentile(tots, 99)), "max": int(np.max(tots)),
                         "passed": passed}
    return res


# --------------------------------------------------------------------------- report


def pop_bins(ranks: list[int], pops: list[int]) -> list[dict]:
    edges = [0, 100, 1_000, 10_000, 100_000, 10**12]
    names = ["<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]
    out = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        rr = [r for r, p in zip(ranks, pops) if lo <= p < hi]
        out.append({"bin": nm, "n": len(rr), "r@1": recall(rr, 1), "r@3": recall(rr, 3)})
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def stage_report(a, R: dict, sets: dict) -> str:
    L: list[str] = []
    w = L.append
    names = {"nq": "NQ-open val", "tqa": "TriviaQA val", "popqa": "PopQA"}
    w("# E1 — library coverage (written library, Simple English Wikipedia)\n")
    w("Generated by `scripts/measure_coverage.py` (stages `tune`, `pool`, `lib`, `readset`, `report`). All numbers below "
      "are measured; nothing is extrapolated. Recall@k = share of questions for which one of the "
      "normalised answer strings occurs as a whole-token sequence in one of the top-k passages "
      "(title line + body). This is a proxy: a passage can contain the answer string by chance, "
      "and can answer a question without containing the exact string.\n")
    n = R["n"]
    w(f"Question samples: NQ-open validation {n['nq']} (all with answers), TriviaQA rc.nocontext "
      f"validation {n['tqa']} (seeded sample), PopQA {n['popqa']} (seeded sample), Dolly-15k "
      f"{n['dolly']} instructions (150 per category). Dev sets used for every choice (BM25 weights, "
      f"gate thresholds): NQ-open train sample {n['dev_nq']}, TriviaQA validation sample "
      f"{n['dev_tqa']} disjoint from the evaluation sample.\n")
    # summary: measured text/index combinations within the total envelope
    combos = []
    for name, b in R["budgets"].items():
        if "same_as" in b:
            continue
        lib_n = b["accounting"]["library_numbers"]
        for r in b["index"]:
            if not r["numbers"]:
                continue
            tot = lib_n + r["numbers"]
            if tot <= a.total_cap:
                combos.append((float(np.mean([r[s]["r@1"] for s in QA_SETS])), name, lib_n, r, tot))
    combos.sort(key=lambda x: (-x[0], x[4]))
    seen_c, uniq = set(), []
    for c in combos:  # identical (library, index) pairs reached from several budgets
        key = (c[2], c[3]["numbers"], c[3]["n_bits"], c[3]["max_list"])
        if key not in seen_c:
            seen_c.add(key)
            uniq.append(c)
    combos = uniq
    dflt = {nm: next(r for r in b["index"] if r.get("default")) for nm, b in R["budgets"].items()}
    w("## Summary\n")
    big_name = [k for k, b in R["budgets"].items() if "same_as" not in b][-1]
    w(f"- All of Simple English Wikipedia (cleaned) packs into {R['pool']['packed_numbers'] / 1e6:.1f}M stored "
      f"numbers: the 40M and 60M budgets hold the whole pool, only the 20M budget forces a selection.")
    e = R["exact_pool"]
    w(f"- Exact BM25 over every passage (the ceiling of this library): R@1 / R@3 = "
      + ", ".join(f"{names[s]} {pct(e[s]['r@1'])} / {pct(e[s]['r@3'])}" for s in QA_SETS) + ".")
    w(f"- With the plan's 4M-number index the stored retriever reaches "
      + ", ".join(f"{names[s]} {pct(dflt[big_name][s]['r@1'])} / {pct(dflt[big_name][s]['r@3'])}" for s in QA_SETS)
      + f" on the whole pool, and practically the same on the 20M library: at this index size the index, "
      f"not the text budget, limits recall.")
    w(f"- Best measured combinations within {a.total_cap / 1e6:.0f}M numbers (library + index), by mean R@1 "
      f"over the three sets:\n")
    w("| library | library numbers | index numbers | total | NQ R@1 / R@3 | TriviaQA R@1 / R@3 | PopQA R@1 / R@3 |")
    w("|---|---|---|---|---|---|---|")
    for m, name, lib_n, r, tot in combos[:5]:
        w(f"| {name} | {lib_n:,} | {r['numbers']:,} | {tot:,} | " + " | ".join(
            f"{pct(r[s]['r@1'])} / {pct(r[s]['r@3'])}" for s in QA_SETS) + " |")
    w("")
    p = R["pool"]
    w("## Candidate pool and cleaning\n")
    w(f"- Source: `wikimedia/wikipedia` 20231101.simple (241,787 articles, 267.5M characters).")
    w(f"- Kept {p['articles']:,} articles, {p['passages']:,} passages, {p['body_bytes']:,} bytes of "
      f"passage text; dropped: " + ", ".join(f"{k} {v:,}" for k, v in sorted(p["drops"].items())) + ".")
    pm_path = os.path.join(a.cache_dir, "pool", "pool.json")
    if os.path.exists(pm_path):
        with open(pm_path) as f:
            pm = json.load(f)["meta"]
        if "tail_cut_lines" in pm:
            w(f"- Rule effects over the whole dump: tail sections cut in {pm['articles_with_tail_cut']:,} "
              f"articles ({pm['tail_cut_lines']:,} lines); a final category group found and removed in "
              f"{pm['articles_with_categories']:,} articles; {pm['dropped_lines']:,} table-row/image lines and "
              f"{pm['dropped_items']:,} list items of fewer than 3 words dropped; {pm['broken_sentences']:,} "
              f"sentences ending in a dangling word (removed template) dropped; {pm['lists_dropped']:,} list "
              f"words removed from list-heavy pages (date, year and month pages).")
    w(f"- Exact duplicate passage bodies: {p['duplicates']:,} (never selected).")
    w(f"- The whole pool packs into **{p['packed_numbers']:,} stored numbers** (text blocks + directory "
      f"+ titles), so every budget of 35M numbers or more holds all of Simple English Wikipedia.\n")
    if "tune" in R:
        w("## BM25 weights (dev only)\n")
        w("| k1 | b | title w | lead title w | dev NQ R@1 | dev NQ R@3 | dev TQA R@1 | dev TQA R@3 |")
        w("|---|---|---|---|---|---|---|---|")
        for r in R["tune"][:8]:
            d = r["dev"]
            w(f"| {r['k1']} | {r['b']} | {r['title_weight']} | {r['lead_title_weight']} | "
              f"{pct(d['dev_nq']['r@1'])} | {pct(d['dev_nq']['r@3'])} | {pct(d['dev_tqa']['r@1'])} | "
              f"{pct(d['dev_tqa']['r@3'])} |")
        w(f"\n(top 8 of {len(R['tune'])} settings by mean dev R@1; the first row is the default "
          f"`IndexCfg`.)\n")
    w("## Recall by budget (stored truncated index vs exact BM25)\n")
    ic = dataclasses.asdict(IndexCfg())
    R_ic = ic
    w(f"Stored index: 2^{ic['n_bits']} buckets, {ic['budget_numbers']:,} stored numbers (variable-length "
      f"posting lists of 8-bit impacts, at most {ic['max_list']} per bucket, + counts, coarse offsets "
      f"and df tables). \"Exact (all passages)\" = BM25 over true terms on every pool passage, no "
      f"hashing, no truncation; \"exact (library)\" = the same restricted to the selected passages.\n")
    w("| budget | library numbers | index numbers | set | exact all R@1 / R@3 | exact library R@1 / R@3 | stored index R@1 / R@3 |")
    w("|---|---|---|---|---|---|---|")
    for name, b in R["budgets"].items():
        acc = b["accounting"]
        row = next(r for r in b["index"] if r.get("default"))
        for s in QA_SETS:
            e = R["exact_pool"][s]
            el = b["exact_lib"][s]
            w(f"| {name} | {acc['library_numbers']:,} | {acc['index_numbers']:,} | {names[s]} | "
              f"{pct(e['r@1'])} / {pct(e['r@3'])} | {pct(el['r@1'])} / {pct(el['r@3'])} | "
              f"{pct(row[s]['r@1'])} / {pct(row[s]['r@3'])} |")
    w("")
    for name, b in R["budgets"].items():
        if "same_as" in b:
            w(f"- {name}: the greedy selection is identical to {b['same_as']} (the whole pool fits); "
              f"its measurements are the same.")
    w("")
    w("## Index size (recall against stored numbers)\n")
    w("Same library, same queries; only the stored index changes. \"untruncated\" keeps every "
      "hashed posting (not storable within these budgets) and isolates the cost of hashing and "
      "8-bit quantisation; the other rows keep the highest-utility postings within the budget.\n")
    for name, b in R["budgets"].items():
        if "same_as" in b:
            continue
        w(f"Library {name}:\n")
        w("| buckets | max list | index numbers | postings kept | NQ R@1 / R@3 | TriviaQA R@1 / R@3 | PopQA R@1 / R@3 | dev NQ R@1 | dev TQA R@1 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in b["index"]:
            num = f"{r['numbers']:,}" if r["numbers"] else "untruncated"
            ml = r["max_list"] if r["max_list"] else "-"
            w(f"| 2^{r['n_bits']} | {ml} | {num} | {r['postings']:,} | " + " | ".join(
                f"{pct(r[s]['r@1'])} / {pct(r[s]['r@3'])}" for s in QA_SETS)
              + f" | {pct(r['dev_nq']['r@1'])} | {pct(r['dev_tqa']['r@1'])} |")
        w("")
    # PopQA by popularity
    pops = [x["pop"] for x in sets["popqa"]]
    w("## PopQA by subject popularity (monthly page views, `s_pop`)\n")
    last = [b for b in R["budgets"].values() if "same_as" not in b][-1]
    row = next(r for r in last["index"] if r.get("default"))
    tables = [("exact (all passages)", pop_bins(R["exact_pool_ranks_popqa"], pops)),
              ("stored index, largest library", pop_bins(row["popqa_ranks"], pops))]
    first = [b for b in R["budgets"].values() if "same_as" not in b][0]
    row0 = next(r for r in first["index"] if r.get("default"))
    tables.append(("stored index, smallest library", pop_bins(row0["popqa_ranks"], pops)))
    w("| popularity | n | " + " | ".join(f"{t} R@1 / R@3" for t, _ in tables) + " |")
    w("|---|---|" + "---|" * len(tables))
    for i in range(len(tables[0][1])):
        w(f"| {tables[0][1][i]['bin']} | {tables[0][1][i]['n']} | " + " | ".join(
            f"{pct(t[i]['r@1'])} / {pct(t[i]['r@3'])}" for _, t in tables) + " |")
    w("")
    # gate
    w("## Gate: precision against coverage\n")
    gb = R["gate"]
    g = gb["chosen"]
    w(f"Gate thresholds were chosen on the dev sets (NQ train + TriviaQA dev), maximising the share "
      f"of questions answered by the single passage subject to dev precision >= {gb['target']:.2f}: "
      f"`min_score={g['min_score']}`, `min_margin={g['min_margin']}`, "
      f"`min_margin_title={g['min_margin_title']}`, `min_title_match={g['min_title_match']}`, "
      f"allowed query types {tuple(g['allowed_types'])}. Precision = share of passed questions whose "
      f"passage contains an answer; coverage = share of questions for which a passage is returned.\n")
    w("| library | set | coverage | precision | answered (coverage x precision) | top-1 recall without gate |")
    w("|---|---|---|---|---|---|")
    for name, rows in gb["eval"].items():
        for s in QA_SETS:
            r = rows[s]
            w(f"| {name} | {names[s]} | {pct(r['coverage'])} | {pct(r['precision'])} | "
              f"{pct(r['answered'])} | {pct(r['r1'])} |")
    w("")
    w("Coverage/precision frontier (dev-optimal gates, evaluated on the evaluation sets, largest "
      "library):\n")
    w("| min_score | min_margin | min_margin_title | min_title_match | min_coverage | dev cov / prec | NQ cov / prec | TriviaQA cov / prec | PopQA cov / prec |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in gb["frontier"]:
        gg = r["gate"]
        w(f"| {gg['min_score']} | {gg['min_margin']} | {gg['min_margin_title']} | {gg['min_title_match']} | "
          f"{gg['min_coverage']} | "
          f"{pct(r['coverage'])} / {pct(r['precision'])} | " + " | ".join(
              f"{pct(r['eval'][s]['coverage'])} / {pct(r['eval'][s]['precision'])}" for s in QA_SETS) + " |")
    w("")
    w("Gate on everyday prompts (Dolly-15k instructions, largest library): share of prompts for "
      "which a passage is returned, with the query-type filter and without it.\n")
    w("| category | n | query type (question / lookup / task / math / other) | fires | fires without type filter |")
    w("|---|---|---|---|---|")
    for c, r in gb["dolly"].items():
        qt = r["types"]
        w(f"| {c} | {r['n']} | {qt.get('question', 0)} / {qt.get('lookup', 0)} / {qt.get('task', 0)} / "
          f"{qt.get('math', 0)} / {qt.get('other', 0)} | {pct(r['fires'])} | {pct(r['fires_no_type'])} |")
    w("")
    # read set
    w("## Read set per query (stored numbers)\n")
    if "readset" in R:
        rs = R["readset"]
        w(f"Measured on the saved library `{rs['library']}` with its default gate. A query reads one df "
          "entry per distinct query bucket, the counts and coarse offset needed to locate at most "
          f"{R_ic['max_query_terms']} posting lists, the lists themselves (2 numbers per posting, at "
          f"most {R_ic['max_list']} postings each), a few directory entries, one title block when a "
          "candidate exists, and exactly one text block when the gate passes.\n")
        w("| set | questions | passage returned | mean, all queries | max, all queries | mean index part | max index part | mean when returned | p99 when returned | max when returned |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        nm = {**names, "dolly": "Dolly prompts"}
        for s, r in rs["sets"].items():
            w(f"| {nm[s]} | {r['n']} | {r['passed']} | {r['mean_all']:,.0f} | {r['max_all']:,} | "
              f"{r['mean_index']:,.0f} | {r['max_index']:,} | {r['mean_passed']:,.0f} | {r['p99_passed']:,.0f} | {r['max_passed']:,} |")
        w("")
    # unanswerable
    w("## What the library cannot answer\n")
    e = R["exact_pool"]
    sn, st = R["string_scan_nq"], R["string_scan_tqa"]
    w(f"- NQ-open (real search-engine questions, the closest of these sets to everyday "
      f"questions): with exact BM25 over all of Simple English Wikipedia, **{pct(1 - e['nq']['r@3'])}%** "
      f"have no answer string in the top 3 passages, {pct(1 - e['nq']['r@20'])}% none in the top 20, "
      f"{pct(1 - e['nq']['r@100'])}% none in the top 100; for **{pct(sn['absent_share'])}%** "
      f"(sample of {sn['n']}) no answer string occurs anywhere in the library text.")
    w(f"- TriviaQA: {pct(1 - e['tqa']['r@3'])}% no answer in the top 3, {pct(1 - e['tqa']['r@100'])}% "
      f"none in the top 100, {pct(st['absent_share'])}% (sample of {st['n']}) answer absent from "
      f"the whole library.")
    w(f"- PopQA: {pct(1 - e['popqa']['r@3'])}% no answer in the top 3, {pct(1 - e['popqa']['r@100'])}% none in the top 100.")
    d = gb["dolly"]
    nonfact = [c for c in d if c in ("creative_writing", "brainstorming", "classification",
                                     "summarization", "information_extraction")]
    if nonfact:
        w(f"- Everyday prompts that are not factual look-ups (Dolly {', '.join(nonfact)}): the gate "
          f"returns a passage for " + ", ".join(f"{c} {pct(d[c]['fires'])}%" for c in nonfact) + ".")
    w("")
    if "notes" in R:
        w("## Notes\n")
        for x in R["notes"]:
            w(f"- {x}")
        w("")
    return "\n".join(L) + "\n"


def compute_gate(R: dict, sets: dict, target: float) -> dict:
    """Choose the gate on dev features of the largest library and evaluate everywhere."""
    names = [k for k, b in R["budgets"].items() if "same_as" not in b]
    big = R["budgets"][names[-1]]
    dev_f = big["gate_feats"]["dev_nq"] + big["gate_feats"]["dev_tqa"]
    dev_h = np.array(big["gate_hits"]["dev_nq"] + big["gate_hits"]["dev_tqa"])
    g, front = choose_gate(dev_f, dev_h, target)
    out = {"target": target, "chosen": dataclasses.asdict(g), "eval": {}, "frontier": []}
    for name in names:
        b = R["budgets"][name]
        out["eval"][name] = {}
        for s in QA_SETS:
            h = np.array(b["gate_hits"][s])
            r = gate_eval(b["gate_feats"][s], h, g)
            r["r1"] = float(h.mean())
            out["eval"][name][s] = r
    # thin the frontier to about 10 points
    step = max(1, len(front) // 10)
    for r in front[::step]:
        gg = GateCfg(**{k: v for k, v in r["gate"].items() if k != "allowed_types"})
        ev = {s: gate_eval(big["gate_feats"][s], np.array(big["gate_hits"][s]), gg) for s in QA_SETS}
        out["frontier"].append({**r, "eval": ev})
    dolly = {}
    f_all = big["gate_feats"]["dolly"]
    cats = [x["cat"] for x in sets["dolly"]]
    no_type = dataclasses.replace(g, allowed_types=("question", "lookup", "task", "math", "other"))
    for c in sorted(set(cats)):
        fs = [f for f, cc in zip(f_all, cats) if cc == c]
        h = np.zeros(len(fs), dtype=bool)
        types: dict[str, int] = {}
        for f in fs:
            types[f["qtype"]] = types.get(f["qtype"], 0) + 1
        dolly[c] = {"n": len(fs), "fires": gate_eval(fs, h, g)["coverage"],
                    "fires_no_type": gate_eval(fs, h, no_type)["coverage"], "types": types}
    out["dolly"] = dolly
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("tune", "pool", "lib", "readset", "report"), default="report")
    ap.add_argument("--cache-dir", default="data_cache/library")
    ap.add_argument("--budgets", default="20,40,60", help="millions of stored numbers")
    ap.add_argument("--index-budgets", default="1,2,4,8,16,32", help="millions of stored numbers")
    ap.add_argument("--force", action="store_true", help="measure even if the selection equals another budget's")
    ap.add_argument("--extra-bits", action="store_true",
                    help="also 2^17 and 2^19 buckets and an uncapped list length at the default budget")
    ap.add_argument("--n-tqa", type=int, default=3000)
    ap.add_argument("--n-popqa", type=int, default=3000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--scan-sample", type=int, default=200)
    ap.add_argument("--precision-target", type=float, default=0.35,
                    help="dev precision floor used to choose the default gate")
    ap.add_argument("--library", default="data_cache/library/lib_60M", help="saved library (readset stage)")
    ap.add_argument("--total-cap", type=int, default=64_000_000,
                    help="library + index envelope for the summary (plan: 60M + 4M)")
    ap.add_argument("--tune-part", default="0/1", help="i/n: run every n-th grid point from i")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default="docs/E1_COVERAGE.md")
    a = ap.parse_args(argv)
    a.budgets = [int(float(x) * 1_000_000) for x in a.budgets.split(",")]
    a.index_budgets = [int(float(x) * 1_000_000) for x in a.index_budgets.split(",")]
    t0 = time.time()
    J = lambda nm: os.path.join(a.cache_dir, f"e1_{nm}.json")  # noqa: E731
    if a.stage == "tune":
        R = stage_tune(a, t0)
        path = J("tune_" + a.tune_part.replace("/", "of"))
        with open(path, "w") as f:
            json.dump(R, f, indent=1)
        return 0
    if a.stage == "pool":
        with open(J("pool"), "w") as f:
            json.dump(stage_pool(a, t0), f)
        return 0
    if a.stage == "readset":
        with open(J("readset"), "w") as f:
            json.dump(stage_readset(a, t0), f)
        return 0
    if a.stage == "lib":
        for bud in a.budgets:
            res = stage_lib(a, t0, bud)
            with open(J(f"lib_{bud // 1_000_000}M"), "w") as f:
                json.dump(res, f)
        return 0
    with open(J("pool")) as f:
        R = json.load(f)
    R["budgets"] = {}
    for bud in a.budgets:
        nm = f"{bud // 1_000_000}M"
        if os.path.exists(J(f"lib_{nm}")):
            with open(J(f"lib_{nm}")) as f:
                R["budgets"][nm] = json.load(f)
    for nm, b in list(R["budgets"].items()):
        if "same_as" in b:
            R["budgets"][nm] = {**R["budgets"][b["same_as"]], "same_as": b["same_as"],
                                "accounting": b["accounting"]}
    tunes = []
    for fn in sorted(os.listdir(a.cache_dir)):
        if fn.startswith("e1_tune_") and fn.endswith(".json"):
            with open(os.path.join(a.cache_dir, fn)) as f:
                tunes.extend(json.load(f)["tune"])
    if tunes:
        tunes.sort(key=lambda r: -r["mean_r1"])
        R["tune"] = tunes
    sets = load_sets(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dev, a.seed)
    R["gate"] = compute_gate(R, sets, a.precision_target)
    if os.path.exists(J("notes")):
        with open(J("notes")) as f:
            R["notes"] = json.load(f)
    if os.path.exists(J("readset")):
        with open(J("readset")) as f:
            R["readset"] = json.load(f)
    md = stage_report(a, R, sets)
    with open(a.md, "w") as f:
        f.write(md)
    with open(J("report"), "w") as f:
        json.dump({k: v for k, v in R.items() if k != "budgets"} |
                  {"gate": R["gate"]}, f)
    print(t_now(t0), "wrote", a.md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
