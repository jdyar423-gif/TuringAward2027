"""Gate E1 (library v2): coverage and precision of the written library on factual questions.

    python scripts/measure_coverage.py --stage pool                 # pool statistics, exact BM25 ceiling
    python scripts/measure_coverage.py --stage sweep --library D    # wiki index size sweep (dev recall)
    python scripts/measure_coverage.py --stage lib --library D      # all E1 tables for a built library
    python scripts/measure_coverage.py --stage adv-sets --library D # the review's adversarial sets (once)
    python scripts/measure_coverage.py --stage adv --library D [--tag before]   # evaluate them on D
    python scripts/measure_coverage.py --stage report               # JSON files -> docs/E1_COVERAGE.md

Every stage stays below 3 GB and uses one thread; results go to ``data_cache/library/e1v2_*.json``.

Dev data (every choice: index size, gate) - ``bes.library.devsets``: seeded samples of the
stored NQ-open train and TriviaQA train questions asked with their own qa entry excluded
(leave-one-out), and head-entity probes ("What is X?") for the dev half of the important
titles. Evaluation data (measured once, after the choices): NQ-open validation (all),
TriviaQA rc.nocontext validation (seeded sample of the unique questions), PopQA test (seeded
sample), probes for the evaluation half of the titles, Dolly-15k instructions by category and
the dev half of the project's proxy prompts (``bes.eval.proxy.prompts``).

A wiki passage counts as a hit when a normalised answer string occurs as a whole-token
sequence in its text (title line + body); a qa entry when an answer string occurs in its
stored answer (``qa EM``: the stored answer equals an answer after SQuAD normalisation); a
probe when the passage belongs to the probed article.
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

from bes.config import LibraryCfg  # noqa: E402
from bes.library import devsets as D  # noqa: E402
from bes.library.build import (candidate_rows, decisions, dev_queries, load_qa_pairs,  # noqa: E402
                               prepare_pool, prepare_signals)
from bes.library.clean import bpe_count_tokens, render_passage  # noqa: E402
from bes.library.gate import pr_curve  # noqa: E402
from bes.library.index import ExactBM25, IndexCfg, TruncatedIndex, bucket_map, rekey_postings, term_postings  # noqa: E402
from bes.library.pack import PackCfg  # noqa: E402
from bes.library.retrieve import Library, ReadLog, query_type  # noqa: E402
from bes.library.select import importance, passage_prior  # noqa: E402
from bes.library.sources import QA_SOURCES, Pool, download_qa, iter_qa  # noqa: E402
from bes.library.terms import terms  # noqa: E402

L = LibraryCfg()
EVAL_SETS = ("nq", "tqa", "popqa", "eval_probe", "eval_names")
ALL_TYPES = ("question", "lookup", "task", "math", "other", "followup")
DEV_SETS = ("dev_nq", "dev_tqa", "dev_probe", "dev_names")
NEG_DEV_SETS = ("dev_contrast", "dev_chat")
NAMES = {"nq": "NQ-open val", "tqa": "TriviaQA val", "popqa": "PopQA", "eval_probe": "head-entity probes (eval)",
         "eval_names": "common-name probes (eval)", "dev_nq": "dev NQ train (LOO)",
         "dev_tqa": "dev TriviaQA train (LOO)", "dev_probe": "head-entity probes (dev)",
         "dev_names": "common-name probes (dev)", "dev_contrast": "contrastive questions (dev)",
         "dev_chat": "conversation negatives (dev)"}


def t_now(t0: float) -> str:
    return f"[{time.time() - t0:7.1f}s]"


# --------------------------------------------------------------------------- data


def fetch_dolly(cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    out = os.path.join(cache_dir, "raw", "dolly")
    p = os.path.join(out, "databricks-dolly-15k.jsonl")
    if os.path.exists(p):
        return p
    return hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset",
                           revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", local_dir=out)


def load_eval(cache_dir: str, n_tqa: int, n_pop: int, n_dolly: int, seed: int) -> dict[str, list[dict]]:
    """Evaluation questions (never stored in the library; see ``bes.library.build.eval_questions``)."""
    rng = random.Random(seed)
    sets: dict[str, list[dict]] = {}
    s = QA_SOURCES["nq_open_validation"]
    sets["nq"] = [{"q": q, "ans": D.answers(a), "set": "nq"} for _, q, a in iter_qa(s, download_qa(s, cache_dir))]
    s = QA_SOURCES["trivia_qa_validation"]
    uniq: dict[str, list[str]] = {}
    for _, q, a in iter_qa(s, download_qa(s, cache_dir)):
        uniq.setdefault(q, a)                 # rc has one row per evidence document
    qs = sorted(uniq)
    sets["tqa"] = [{"q": q, "ans": D.answers(uniq[q]), "set": "tqa"} for q in rng.sample(qs, min(n_tqa, len(qs)))]
    s = QA_SOURCES["popqa_test"]
    csv.field_size_limit(1 << 30)
    with open(download_qa(s, cache_dir), newline="", encoding="utf-8") as f:
        prow = list(csv.DictReader(f, delimiter="\t"))
    sets["popqa"] = [{"q": prow[i]["question"], "ans": D.answers(json.loads(prow[i]["possible_answers"])),
                      "pop": int(prow[i]["s_pop"]), "prop": prow[i]["prop"], "set": "popqa"}
                     for i in rng.sample(range(len(prow)), min(n_pop, len(prow)))]
    for k in ("nq", "tqa", "popqa"):
        sets[k] = [x for x in sets[k] if x["ans"]]
    with open(fetch_dolly(cache_dir)) as f:
        dolly = [json.loads(line) for line in f]
    by_cat: dict[str, list[dict]] = {}
    for r in dolly:
        by_cat.setdefault(r["category"], []).append(r)
    sets["dolly"] = [{"q": r["instruction"], "ans": [], "cat": c, "set": "dolly"}
                     for c in sorted(by_cat) for r in rng.sample(by_cat[c], min(n_dolly, len(by_cat[c])))]
    from bes.eval.proxy import prompts as P

    sets["proxy"] = [{"q": p.turns[0], "ans": [], "cat": p.category, "set": "proxy"}
                     for p in P.select(P.load(), "dev")]
    return sets


def load_lib_and_dev(a, t0, lib_path: str):
    """The library, its title analysis and importance, the stored pairs and the dev sets as
    the build made them (``bes.library.build.dev_queries``)."""
    from bes.library.build import drop_long_renders, load_provenance
    from bes.library.titles import analyse_library

    lib = Library.load(lib_path)
    imp = load_provenance(lib_path, "title_importance")
    nimp = load_provenance(lib_path, "name_importance")
    infos = analyse_library(lib)
    pairs, stats, keys, raw = load_qa_pairs(a.cache_dir)
    pairs, _ = drop_long_renders(pairs, L.max_passage_tokens, bpe_count_tokens)
    b = lib.meta.get("build", {})
    ds = b.get("dev_sets", {})
    dq = dev_queries(lib, pairs, raw, a.n_dev, a.n_probes, b.get("seed", a.seed), imp, infos=infos,
                     n_names=ds.get("dev_names", a.n_probes), n_contrast=ds.get("dev_contrast", a.n_probes),
                     chat="dev_chat" in ds, name_imp=nimp)
    dev = {s: [x for x in dq if x["set"] == s] for s in DEV_SETS + NEG_DEV_SETS}
    print(t_now(t0), "dev:", {k: len(v) for k, v in dev.items()}, flush=True)
    return lib, (imp, nimp, infos), pairs, dev


# --------------------------------------------------------------------------- stage pool


def pool_stats(pool: Pool) -> dict:
    ntok = np.array([bpe_count_tokens(render_passage(pool.titles[int(pool.p_article[i])], pool.body(i)))
                     for i in range(pool.n_passages)], dtype=np.int64)
    words = pool.p_words.astype(np.int64)
    by = pool.body_bytes()
    per_art = np.bincount(pool.p_article, minlength=pool.n_articles)
    return {"articles": pool.n_articles, "passages": pool.n_passages, "body_bytes": int(by.sum()),
            "tokens_mean": float(ntok.mean()), "tokens_p50": float(np.median(ntok)),
            "tokens_p90": float(np.percentile(ntok, 90)), "tokens_max": int(ntok.max()),
            "at_cap_150": float((ntok >= 150).mean()), "over_cap": int((ntok > L.max_passage_tokens).sum()),
            "words_mean": float(words.mean()), "words_max": int(words.max()),
            "bytes_mean": float(by.mean()), "passages_per_article": float(per_art.mean()),
            "single_passage_articles": float((per_art == 1).mean()), "drops": pool.drops,
            "hole_sentences": int(pool.meta.get("hole_sentences", 0)),
            "hole_repairs": int(pool.meta.get("hole_repairs", 0)),
            "broken_sentences": int(pool.meta.get("broken_sentences", 0))}


HOLE_PROBES = {   # independent detectors used for the before/after count (broader than the filter)
    "approx_measure": r"\b(?:about|approximately|around|nearly|almost|roughly)\s+(?:long|high|tall|wide|deep|across|in length|in diameter|in area|above sea level)\b",
    "is_measure_end": r"\b(?:is|was|are|were)\s+(?:long|tall|wide|deep|high)\s*[.;]",
    "measure_of_hole": r"\b(?:area|elevation|height|length|altitude|depth|distance|width|diameter) of\s*(?:[.,;)]|and\b|with\b|making\b)",
    "from_to_between_and": r"\bfrom\s+to\b|\bbetween\s+and\b",
    "approx_direction": r"\b(?:about|approximately|around|roughly)\s+(?:north|south|east|west|northeast|northwest|southeast|southwest)\b",
}


def hole_counts(pool: Pool, idx) -> dict:
    import re

    rx = {k: re.compile(v) for k, v in HOLE_PROBES.items()}
    out = {k: 0 for k in rx}
    for i in idx:
        b = pool.body(int(i))
        for k, r in rx.items():
            out[k] += len(r.findall(b))
    return out


def stage_pool(a, t0) -> dict:
    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    R: dict = {}
    pool = prepare_pool(a.cache_dir, log=log)
    sig = prepare_signals(pool, a.cache_dir, PackCfg(), log=log)
    R["pool_bpe"] = pool_stats(pool)
    z = np.load(os.path.join(a.cache_dir, "pool_bpe", "signals.npz"))
    R["pool_bpe"]["packed_numbers"] = int(z["pool_numbers"])
    R["pool_bpe"]["duplicates"] = int(sig.dup.sum())
    log(f"pool stats {R['pool_bpe']}")
    old_dir = os.path.join(a.cache_dir, "pool")
    rng = random.Random(a.seed)
    if os.path.exists(os.path.join(old_dir, "pool.json")):
        old = Pool.load(old_dir)
        R["pool_proxy"] = pool_stats(old)
        zo = os.path.join(old_dir, "signals.npz")
        if os.path.exists(zo):
            R["pool_proxy"]["packed_numbers"] = int(np.load(zo)["pool_numbers"])
        # holes: same sample of article titles in both pools
        titles = sorted(set(old.titles) & set(pool.titles))
        pick = set(rng.sample(titles, min(20000, len(titles))))
        R["holes"] = {"articles": len(pick),
                      "before": hole_counts(old, [i for i in range(old.n_passages) if old.titles[int(old.p_article[i])] in pick]),
                      "after": hole_counts(pool, [i for i in range(pool.n_passages) if pool.titles[int(pool.p_article[i])] in pick])}
        log(f"holes {R['holes']}")
        del old
        gc.collect()
    sets = load_eval(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dolly, a.seed)
    docs = [(pool.titles[int(pool.p_article[i])], pool.body(int(i))) for i in sig.order]
    icfg = IndexCfg()
    ex = ExactBM25.build(docs, icfg.k1, icfg.b, icfg.title_weight, icfg.lead_title_weight)
    log("exact pool index built")
    texts = {}

    def text(pid):
        t = texts.get(pid)
        if t is None:
            t = render_passage(*docs[pid])
            if len(texts) < 300_000:
                texts[pid] = t
        return t

    R["exact_pool"] = {}
    for s in ("nq", "tqa", "popqa"):
        ranks = []
        for x in sets[s]:
            pids, _ = ex.topk(x["q"], 100)
            r = 0
            for k, p in enumerate(pids):
                if D.contains(text(int(p)), x["ans"]):
                    r = k + 1
                    break
            ranks.append(r)
        R["exact_pool"][s] = {f"r@{k}": float(np.mean([0 < r <= k for r in ranks])) for k in (1, 3, 20, 100)}
        if s == "popqa":
            R["exact_pool_ranks_popqa"] = ranks
        log(f"exact {s} {R['exact_pool'][s]}")
    return R


# --------------------------------------------------------------------------- stage sweep


def wiki_hit(lib: Library, x: dict) -> tuple[bool, bool]:
    """(hit of the wiki candidate the gate would see, hit within the BM25 top 3)."""
    cands = lib._wiki_candidates(x["q"], [], [], None, 1)
    if not cands:
        return False, False
    c = cands[0]
    if "tid" in x:
        h1 = lib.title_id_of(c.pid) == x["tid"]
    else:
        h1 = D.contains(lib.passage(c.pid).text, x["ans"])
    pids, _ = lib.search(x["q"])
    h3 = False
    for p in pids[:3]:
        if ("tid" in x and lib.title_id_of(int(p)) == x["tid"]) or (
                "tid" not in x and D.contains(lib.passage(int(p)).text, x["ans"])):
            h3 = True
            break
    return h1, h3


def stage_sweep(a, t0) -> dict:
    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    from bes.library.build import load_provenance

    base, imp, pairs, dev = load_lib_and_dev(a, t0, a.library)
    del pairs
    dev = {s: dev[s] for s in DEV_SETS if s != "dev_names"}
    pool = prepare_pool(a.cache_dir, log=log)
    sig = prepare_signals(pool, a.cache_dir, PackCfg(), log=log)
    pool_idx = load_provenance(a.library, "pool_idx")
    prior = passage_prior(pool, importance(pool, sig.mentions))[pool_idx]
    passages = [(pool.titles[int(pool.p_article[i])], pool.body(int(i))) for i in pool_idx]
    del pool, sig
    gc.collect()
    icfg0 = base.index.cfg
    p, tl = term_postings(passages, icfg0.title_weight, icfg0.lead_title_weight)
    hp = rekey_postings(p, bucket_map(tl, icfg0.n_bits))
    del p, tl, passages
    gc.collect()
    log(f"sweep: {hp.key.shape[0]} hashed postings")
    fixed = base.accounting()["total_numbers"] - base.accounting()["index_numbers"]
    out = {"fixed_numbers": fixed, "rows": []}
    budgets = [int(float(x) * 1e6) for x in a.sweep_budgets.split(",")]
    lists = [int(x) for x in a.sweep_lists.split(",")]
    qsets = dict(dev)      # dev only: the evaluation sets are measured once, on the final library
    for bud in budgets:
        for ml in lists:
            v = dataclasses.replace(icfg0, budget_numbers=bud, max_list=ml)
            idx = TruncatedIndex.from_postings(hp, v, prior)
            lib = Library(base.packed, idx, base.cfg, base.count_tokens, base.meta, qa=None, titles=base.titles)
            acc = idx.accounting()
            row = {"budget": bud, "max_list": ml, "index_numbers": acc["index_numbers"],
                   "postings": acc["index_postings"], "total_numbers": fixed + acc["index_numbers"]}
            for s, xs in qsets.items():
                hs = [wiki_hit(lib, x) for x in xs]
                row[s] = {"r@1": float(np.mean([h[0] for h in hs])), "bm25_r@3": float(np.mean([h[1] for h in hs]))}
            reads = []
            for x in qsets["dev_nq"][:1000] + qsets["dev_tqa"][:1000]:
                rl = ReadLog()
                lib.search_max(x["q"], log=rl)
                reads.append(rl.index)
            row["index_read_max"] = int(np.max(reads))
            row["index_read_mean"] = float(np.mean(reads))
            row["dev_mean_r1"] = float(np.mean([row[s]["r@1"] for s in qsets]))
            out["rows"].append(row)
            log(f"sweep {bud} {ml}: total {row['total_numbers']} dev {row['dev_mean_r1']:.4f} "
                + " ".join(f"{s} {row[s]['r@1']:.3f}" for s in qsets))
            del idx, lib
            gc.collect()
    ok = [r for r in out["rows"] if r["total_numbers"] <= a.total_cap]
    best = max(ok, key=lambda r: (round(r["dev_mean_r1"], 6), -r["total_numbers"]))
    out["chosen"] = {"budget": best["budget"], "max_list": best["max_list"]}
    log(f"sweep: chosen {out['chosen']}")
    return out


# --------------------------------------------------------------------------- stage lib


def near_dup_flags(questions: list[str], stored: list[str], thr: float) -> tuple[np.ndarray, np.ndarray]:
    """(max Jaccard of the content-term sets against every stored question, flag >= thr):
    brute force over true terms (no hashing), vectorised through an inverted list."""
    vocab: dict[str, int] = {}
    sets = []
    for q in stored:
        sets.append(sorted({vocab.setdefault(t, len(vocab)) for t in terms(q)}))
    lens = np.array([len(s) for s in sets], dtype=np.int64)
    flat_t = np.fromiter((t for s in sets for t in s), dtype=np.int64, count=int(lens.sum()))
    flat_e = np.repeat(np.arange(len(sets), dtype=np.int64), lens)
    o = np.argsort(flat_t, kind="stable")
    ft, fe = flat_t[o], flat_e[o]
    starts = np.searchsorted(ft, np.arange(len(vocab) + 1))
    best = np.zeros(len(questions))
    for i, q in enumerate(questions):
        qs = {vocab[t] for t in terms(q) if t in vocab}
        nq = len(set(terms(q)))
        if not qs or nq == 0:
            continue
        cand = np.concatenate([fe[starts[t]:starts[t + 1]] for t in qs])
        u, inter = np.unique(cand, return_counts=True)
        jac = inter / (nq + lens[u] - inter)
        best[i] = float(jac.max())
    return best, best >= thr


def summarize_rows(rows, lib, kinds_list=(("wiki",), ("qa",), ("wiki", "qa"))) -> dict:
    gate = lib.gate
    out = {}
    for kinds in kinds_list:
        name = "+".join(kinds)
        p, y, kd = decisions(rows, gate, kinds=kinds)
        fired = p >= gate.threshold
        # R@1 without the gate and without the query-type filter: label of the best candidate
        _, y_all, _ = decisions(rows, gate, allowed=ALL_TYPES, kinds=kinds)
        top = np.array([np.isfinite(v) for v in p])
        r1 = float(np.mean(y_all)) if len(y_all) else 0.0
        em = None
        if kinds == ("qa",):
            em = float(np.mean([bool(r["qa"] and r["qa"].get("em")) for r in rows]))
        out[name] = {"n": len(rows), "r@1": r1, "em": em, "candidate": float(top.mean()) if len(top) else 0.0,
                     "coverage": float(fired.mean()) if len(rows) else 0.0,
                     "precision": float(y[fired].mean()) if fired.any() else 0.0,
                     "answered": float((fired & y).mean()) if len(rows) else 0.0,
                     "kind_share": {k: float(np.mean([kd[i] == k for i in np.flatnonzero(fired)])) if fired.any() else 0.0
                                    for k in ("wiki", "qa")}}
        if kinds == ("wiki", "qa"):
            qa_fired = [i for i in np.flatnonzero(fired) if kd[i] == "qa"]
            out[name]["qa_em_when_fired"] = float(np.mean([bool(rows[i]["qa"].get("em")) for i in qa_fired])) if qa_fired else 0.0
        pts = pr_curve(p, y)
        grid = sorted({0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, float(gate.threshold)})
        curve = []
        for t in grid:
            sel = p >= t
            curve.append({"t": t, "coverage": float(sel.mean()) if len(rows) else 0.0,
                          "precision": float(y[sel].mean()) if sel.any() else 0.0})
        out[name]["curve"] = curve
        out[name]["n_points"] = len(pts)
    return out


def stage_lib(a, t0) -> dict:
    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    lib, (imp, nimp, infos), pairs, dev = load_lib_and_dev(a, t0, a.library)
    sets = load_eval(a.cache_dir, a.n_tqa, a.n_popqa, a.n_dolly, a.seed)
    titles = lib.packed.all_titles()
    sets["eval_probe"] = D.probes(titles, imp, a.n_probes, "eval", a.seed, persons=[i.person for i in infos])
    sets["eval_names"] = D.name_probes(titles, nimp, infos, a.n_probes, "eval", a.seed)
    R: dict = {"library": a.library, "accounting": lib.accounting(), "gate_threshold": lib.gate.threshold,
               "read_bound": lib.read_bound(),
               "gate_coef": {k: [float(x) for x in v] for k, v in lib.gate.coef.items()},
               "n": {k: len(v) for k, v in {**sets, **dev}.items()}, "meta_build": lib.meta.get("build", {})}
    stored = [p.question for p in pairs]
    rows_all = {}
    for s in DEV_SETS + NEG_DEV_SETS + EVAL_SETS + ("dolly", "proxy"):
        xs = dev[s] if s in dev else sets[s]
        rows = candidate_rows(lib, xs)
        for r, x in zip(rows, xs):
            if r["qa"] is not None and x["ans"]:
                r["qa"]["em"] = D.qa_em(lib.passage(r["qa"]["pid"]).body, x["ans"])
        rows_all[s] = rows
        log(f"rows {s}: {len(rows)}")
    R["sets"] = {s: summarize_rows(rows_all[s], lib) for s in DEV_SETS + EVAL_SETS}
    R["qtypes"] = {s: float(np.mean([r["qtype"] in ("question", "lookup") for r in rows_all[s]]))
                   for s in DEV_SETS + EVAL_SETS}
    R["negative_dev"] = {}
    for s in NEG_DEV_SETS:                      # no right answers: the share of known-wrong returns
        p, y, kd = decisions(rows_all[s], lib.gate, labelled_only=True)
        fired = p >= lib.gate.threshold
        p_all, _, _ = decisions(rows_all[s], lib.gate)
        R["negative_dev"][s] = {"n": len(rows_all[s]), "known_wrong": float(fired.mean()) if len(rows_all[s]) else 0.0,
                                "any_entry": float((p_all >= lib.gate.threshold).mean()) if len(rows_all[s]) else 0.0,
                                "followup_or_chat_type": float(np.mean([r["qtype"] not in ("question", "lookup")
                                                                        for r in rows_all[s]]))}
    # train/test question overlap (near-duplicate stored question, Jaccard of content terms)
    R["overlap"] = {}
    for s in ("nq", "tqa", "popqa"):
        mx, flag = near_dup_flags([x["q"] for x in sets[s]], stored, a.near_dup)
        R["overlap"][s] = {"share": float(flag.mean()), "threshold": a.near_dup}
        for val, name in ((True, "with"), (False, "without")):
            sub = [r for r, f in zip(rows_all[s], flag) if bool(f) == val]
            R["overlap"][s][name] = summarize_rows(sub, lib) if sub else {}
        log(f"overlap {s}: {flag.mean():.3f}")
    # PopQA by popularity
    pops = np.array([x["pop"] for x in sets["popqa"]])
    edges = [0, 100, 1000, 10_000, 100_000, 10**12]
    names = ["<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]
    R["popqa_bins"] = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        sub = [r for r, pp in zip(rows_all["popqa"], pops) if lo <= pp < hi]
        R["popqa_bins"].append({"bin": nm, "n": len(sub), **({"s": summarize_rows(sub, lib)} if sub else {})})
    # firing on non-factual prompts
    R["firing"] = {}
    for s in ("dolly", "proxy"):
        cats = sorted({x["cat"] for x in sets[s]})
        R["firing"][s] = {}
        for c in cats:
            sub = [r for r, x in zip(rows_all[s], sets[s]) if x["cat"] == c]
            p, _, kd = decisions(sub, lib.gate)
            fired = p >= lib.gate.threshold
            types: dict[str, int] = {}
            for r in sub:
                types[r["qtype"]] = types.get(r["qtype"], 0) + 1
            pn, _, _ = decisions(sub, lib.gate, allowed=("question", "lookup", "task", "math", "other"))
            R["firing"][s][c] = {"n": len(sub), "fires": float(fired.mean()), "types": types,
                                 "fires_no_type_filter": float((pn >= lib.gate.threshold).mean()),
                                 "qa_share": float(np.mean([kd[i] == "qa" for i in np.flatnonzero(fired)])) if fired.any() else 0.0,
                                 "examples": [x["q"][:160] for r, x, f in zip(sub, [y for y in sets[s] if y["cat"] == c], fired) if f][:5]}
        log(f"firing {s}: " + ", ".join(f"{c} {v['fires']:.3f}" for c, v in R["firing"][s].items()))
    # read set and throughput of retrieve() on the saved library
    R["readset"] = {}
    for s in ("nq", "tqa", "popqa", "eval_probe", "dolly", "proxy"):
        tots, passed, parts = [], [], []
        tstart = time.time()
        for x in sets[s]:
            rl = ReadLog()
            r = lib.retrieve(x["q"], log=rl)
            tots.append(rl.total)
            passed.append(r is not None)
            parts.append(rl.as_dict())
        dt = time.time() - tstart
        tp = [t for t, p_ in zip(tots, passed) if p_]
        R["readset"][s] = {"n": len(tots), "passed": int(sum(passed)), "mean_all": float(np.mean(tots)),
                           "max_all": int(np.max(tots)), "p99_all": float(np.percentile(tots, 99)),
                           "mean_passed": float(np.mean(tp)) if tp else 0.0, "max_passed": int(np.max(tp)) if tp else 0,
                           "max_index": int(max(d["index_df"] + d["index_offsets"] + d["index_postings"] for d in parts)),
                           "max_qa_index": int(max(d["qa_index"] for d in parts)),
                           "max_blocks": int(max(d["n_text_blocks"] + d["n_qa_blocks"] for d in parts)),
                           "queries_per_s_with_log": len(tots) / dt}
        log(f"readset {s}: {R['readset'][s]}")
    # throughput without logging (fresh library: cold block cache)
    lib2 = Library.load(a.library)
    qs = [x["q"] for x in sets["nq"][:1500]]
    tstart = time.time()
    for q in qs:
        lib2.retrieve(q)
    R["throughput"] = {"retrieve_per_s": len(qs) / (time.time() - tstart)}
    rng = random.Random(a.seed)
    pids = [rng.randrange(lib2.n_passages) for _ in range(3000)]
    lib3 = Library.load(a.library)
    tstart = time.time()
    for p_ in pids:
        lib3.passage(p_)
    R["throughput"]["passage_random_per_s"] = len(pids) / (time.time() - tstart)
    tstart = time.time()
    for p_ in range(0, 20000):
        lib3.passage(p_)
    R["throughput"]["passage_sequential_per_s"] = 20000 / (time.time() - tstart)
    tstart = time.time()
    qa0 = lib3.kind_range("qa").start
    for p_ in range(qa0, qa0 + 20000):
        lib3.passage(p_)
    R["throughput"]["qa_passage_sequential_per_s"] = 20000 / (time.time() - tstart)
    log(f"throughput {R['throughput']}")
    return R


# --------------------------------------------------------------------------- adversarial sets
#
# The review's measurements (docs/E1_COVERAGE.md, "Review fixes"): perturbed stored questions,
# common names, messages that are not factual look-ups and real second user turns. ``adv-sets``
# writes the sets once (``e1v2_advsets.json``, from the library given); ``adv`` evaluates them on
# any library through ``Library.retrieve`` with only the calls an older ``bes.library`` also has,
# so the same file measures the library before and after the fixes.

PERTURB_COUNTRIES = ("france germany italy spain england scotland ireland wales canada mexico brazil argentina "
                     "china japan india russia australia egypt kenya nigeria sweden norway denmark finland poland "
                     "austria hungary greece turkey iran iraq israel portugal belgium netherlands switzerland").split()


def review_variants(q: str, keys: set) -> list[tuple[str, str]]:
    """The review's perturbation classes (its qa_perturb.py): before<->after, one ordinal edit,
    dropping 'not', another year, another country (capitalised); plus the same country swap in
    lower case (no case information, so no name check can see it)."""
    import re

    from bes.library.qa import question_key

    def swap(q, a, b):
        rx = re.compile(r"\b" + a + r"\b", re.I)
        if not rx.search(q):
            return None
        return rx.sub(lambda m: b if m.group(0).islower() else b.capitalize(), q, count=1)

    out = []
    for a, b in (("before", "after"), ("after", "before")):
        v = swap(q, a, b)
        if v:
            out.append(("before<->after", v))
    for a, b in (("first", "second"), ("first", "last"), ("last", "first")):
        v = swap(q, a, b)
        if v:
            out.append((f"{a}->{b}", v))
            break
    if re.search(r"\bnot\b", q, re.I):
        out.append(("drop 'not'", re.sub(r"\s*\bnot\b", "", q, count=1, flags=re.I)))
    m = re.search(r"\b(19[0-9]{2}|20[0-2][0-9])\b", q)
    if m:
        y = int(m.group(1))
        for d in (4, 1, -4, -1):
            v = q[:m.start(1)] + str(y + d) + q[m.end(1):]
            if question_key(v) not in keys:
                out.append(("other year", v))
                break
    low = q.lower()
    for c in PERTURB_COUNTRIES:
        if re.search(r"\b" + c + r"\b", low):
            others = [x for x in PERTURB_COUNTRIES if x != c and not re.search(r"\b" + x + r"\b", low)]
            o = others[(len(q) * 7 + len(c)) % len(others)]
            out.append(("other country", re.sub(r"\b" + c + r"\b", o.capitalize(), q, count=1, flags=re.I)))
            out.append(("other country, lower case", re.sub(r"\b" + c + r"\b", o, low, count=1)))
            break
    return out


def stored_questions(lib) -> list[str]:
    """Every stored qa question in entry order (block by block)."""
    from bes.library.qa import parse_entry

    b = lib.qa.blocks
    out = []
    for k in range(b.n_blocks):
        raw = b.read_block(k)
        lo, hi = int(b.block_first[k]), int(b.block_first[k + 1])
        for e in range(lo, hi):
            s0 = int(b.entry_off[e])
            s1 = int(b.entry_off[e + 1]) if e + 1 < hi else len(raw)
            out.append(parse_entry(raw[s0:s1])[0])
    return out


REVIEW_GROUPS = {
    "follow-up turns": [
        "What about the second one?", "And the third?", "What does that mean?", "Is that true?",
        "Which one is better?", "How long does it take?", "Who invented it?", "When did he die?",
        "What year was that?", "Can you explain it in simpler terms?", "What happened next?",
        "Why is that?", "What about Mars?", "And in France?", "What is the first one?",
        "Which is bigger?", "Who won?", "What is its capital?", "How old was he?",
        "What was his name?", "Where was she born?", "What about the other one?", "Now in Spanish.",
        "Now translate it to French.", "Shorter, please.", "Make it funnier", "In one sentence?",
        "What about dogs?", "And cats?", "What is the last step?", "What is the next step?",
        "What does the second line do?", "Is it safe?", "How much does it cost?", "Why not?",
        "What else?", "Any other examples?", "What is the opposite?", "What color is it?",
        "Who is the author?", "What is the title?", "What is the answer?", "Is it a mammal?"],
    "acknowledgements": [
        "thanks, that was helpful", "ok great", "that makes sense", "cool, thanks!", "nice one", "good job",
        "yes please", "no thanks", "sounds good", "perfect", "awesome", "interesting", "great answer",
        "that is wrong", "you are wrong", "try again", "go on", "continue", "more details", "keep going",
        "in French", "shorter", "simpler", "bullet points", "with examples", "the second option",
        "option B", "the first one", "both", "neither", "never mind", "good night", "see you later",
        "happy birthday", "merry christmas", "I love you", "hello there", "what's up"],
    "chat / opinion": [
        "How are you doing today?", "Good morning! How's it going?", "What is the meaning of life?",
        "What do you think about cats?", "Is coffee bad for you?", "What should I eat for dinner?",
        "Can you help me with my homework?", "I feel sad today. What should I do?",
        "What is the best way to learn guitar?", "Why is the sky blue?", "What is love?",
        "Tell me a joke.", "What's your favorite color?", "Who are you?", "Are you a robot?",
        "What time is it?", "What is today's date?", "What's the weather like?",
        "Recommend a good book.", "What's a good name for a dog?", "Is it better to rent or buy a house?",
        "What are some tips for a job interview?", "How do I make friends?", "What is happiness?",
        "Should I learn Python or JavaScript first?", "What is the best phone?"],
    "code": [
        "How do I reverse a list in Python?", "What is a for loop?", "What does len do?",
        "What is recursion?", "Explain bubble sort.", "What is a linked list?", "What is Python?",
        "Fix this: print 'hello'", "What is the difference between a list and a tuple?",
        "How to sort a dictionary by value", "What is a variable?", "What is an array?",
        "What is a hash table?", "Binary search tree", "What is a pointer?", "What is HTML?"],
    "math": [
        "What is 15% of 80?", "What is 7 times 8?", "What is 12 plus 30?", "Is 17 a prime number?",
        "What is the square root of 144?", "How many seconds are in an hour?", "What is half of 90?",
        "What is 100 divided by 4?", "What is pi?", "What is the derivative of x^2?",
        "How many centimeters are in a meter?", "What is two plus two?", "What is a prime number?",
        "If I have 3 apples and eat one, how many are left?", "What is 3/4 as a percentage?"],
    "stale answers": [
        "Who is the most subscribed YouTuber?", "What is the most viewed video on YouTube?",
        "Who has the most followers on Twitter?", "Who is the oldest person in the world?",
        "How many seasons of Grey's Anatomy are there?", "How many states have the death penalty?",
        "How many stores does Toys R Us have?", "Who has the most followers on Instagram?",
        "Who is the most followed person on Instagram?", "How many seasons of Ray Donovan are there?",
        "How many seasons of The Simpsons are there?", "How many episodes of One Piece are there?",
        "What is the tallest building in the world?", "Who is the richest man in the world?",
        "Who is the richest person in the world?", "How many countries are in NATO?",
        "Who is the oldest living person?", "How many members does the European Union have?"],
    "other question, same words": [
        "Who wrote Romeo and Juliet?", "Who won the Super Bowl in 2020?", "Who won the Super Bowl in 2017?",
        "Which countries do have a bill of rights?", "What year did the pistons win their first championship?",
        "Who was president after Lincoln?", "Who was the second man on the moon?"],
}
REVIEW_NAMES = {   # the review's head_names.py, with the article each name means (None: unlabelled)
    "Mozart": "Wolfgang Amadeus Mozart", "Beethoven": "Ludwig van Beethoven", "Bach": "Johann Sebastian Bach",
    "Gandhi": "Mahatma Gandhi", "Darwin": "Charles Darwin", "Newton": "Isaac Newton", "Picasso": "Pablo Picasso",
    "Columbus": "Christopher Columbus", "Churchill": "Winston Churchill", "Stalin": "Joseph Stalin",
    "Hitler": "Adolf Hitler", "Lenin": "Vladimir Lenin", "Galileo": "Galileo Galilei",
    "Copernicus": "Nicolaus Copernicus", "Aristotle": "Aristotle", "Plato": "Plato", "Socrates": "Socrates",
    "Cleopatra": "Cleopatra VII", "Elvis": "Elvis Presley", "Shakespeare": "William Shakespeare",
    "Einstein": "Albert Einstein", "Napoleon": "Napoleon", "Lincoln": "Abraham Lincoln",
    "Washington": "George Washington", "Edison": "Thomas Edison", "Tesla": "Nikola Tesla",
    "Freud": "Sigmund Freud", "Marx": "Karl Marx", "Chopin": "Frédéric Chopin", "Rembrandt": "Rembrandt",
    "Michelangelo": "Michelangelo", "Dante": "Dante Alighieri", "Homer": "Homer", "Confucius": "Confucius",
    "Buddha": None, "Moses": "Moses", "Jesus": "Jesus", "Muhammad": "Muhammad", "Caesar": "Julius Caesar",
    "Beyonce": "Beyoncé", "Rihanna": "Rihanna", "Madonna": "Madonna (entertainer)", "Obama": "Barack Obama",
    "Trump": "Donald Trump", "Putin": "Vladimir Putin", "Mandela": "Nelson Mandela",
    "Kennedy": "John F. Kennedy", "Nixon": "Richard Nixon", "Reagan": "Ronald Reagan",
    "Thatcher": "Margaret Thatcher"}
LIVING = ("Beyonce", "Rihanna", "Madonna", "Obama", "Trump", "Putin")
STALE_PATTERNS = {   # the review's stale_scan.py and the verifier's patterns
    "most followed / subscribed / viewed": r"\bmost (?:followed|subscribed|subscribers|followers|viewed|liked|watched|"
                                           r"streamed|downloaded)\b",
    "how many seasons / episodes (no season number)": r"^how many (?:seasons|episodes|series)\b(?!.*\bseason \d)",
    "how many ... does / do X have": r"^how many \w+(?: \w+)? (?:does|do) .* have\b",
    "still have": r"\bstill have\b",
    "oldest person / man / woman": r"\boldest (?:person|man|woman|living)\b",
    "got the most": r"\bgot the most\b",
    "member count of a union / league": r"how many (?:countries|members|member states|teams|states) (?:are )?(?:in|of|make up|belong)",
}


def stage_adv_sets(a, t0) -> dict:
    """Write the adversarial evaluation sets (from the library given, with this code)."""
    import pyarrow.parquet as pq

    from bes.eval.proxy import prompts as P
    from bes.library.build import load_provenance
    from bes.library.devsets import entry_half
    from bes.library.qa import question_key
    from bes.library.titles import analyse_library

    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    lib = Library.load(a.library)
    qs = stored_questions(lib)
    keys = {question_key(q) for q in qs}
    rng = random.Random(a.seed)
    order = [i for i in range(len(qs)) if entry_half(qs[i]) == "eval"]
    rng.shuffle(order)
    count: dict[str, int] = {}
    perturb = []
    for e in order:
        for cls, v in review_variants(qs[e], keys):
            if count.get(cls, 0) >= a.n_perturb or question_key(v) in keys or question_key(v) == question_key(qs[e]):
                continue
            count[cls] = count.get(cls, 0) + 1
            perturb.append({"cls": cls, "q": v, "orig_q": qs[e]})
    log(f"perturbation variants {count}")
    titles = lib.packed.all_titles()
    imp = load_provenance(a.library, "title_importance")
    nimp = load_provenance(a.library, "name_importance")
    infos = analyse_library(lib)
    # the review's 300 most important person articles (its surname_probe2.py), asked by surname
    import re

    people = []
    for tid in np.argsort(-imp, kind="stable")[:20000]:
        t = titles[int(tid)]
        if "(" in t or not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}", t):
            continue
        lead = lib.passage(lib.title_range(int(tid)).start).body[:160]
        m = re.match(r"([^\n(]{0,80})\(([^)]*)\)", lead)
        life = m is not None and (re.search(r"\bborn\b[^)]*\d{4}", m.group(2)) or
                                  re.search(r"\d{3,4}[^)]*[\u2013\u2014-][^)]*\d{3,4}", m.group(2)))
        if life and m.group(1).split()[:1] == t.split()[:1]:
            people.append({"title": t, "q": f"Who was {t.split()[-1]}?"})
        if len(people) >= 300:
            break
    R = {"library": a.library, "perturb": perturb, "surnames": people,
         "names50": [{"name": k, "want": v, "q": ("Who is {}?" if k in LIVING else "Who was {}?").format(k)}
                     for k, v in REVIEW_NAMES.items()],
         "eval_probe": D.probes(titles, imp, a.n_probes, "eval", a.seed, persons=[i.person for i in infos]),
         "eval_names": D.name_probes(titles, nimp, infos, a.n_probes, "eval", a.seed),
         "groups": REVIEW_GROUPS}
    for x in R["eval_probe"] + R["eval_names"]:
        x["title"] = titles[x["tid"]]
    # real second user turns: the review's seeded sample of the smol-smoltalk test split
    f = pq.ParquetFile(os.path.join("data_cache", "samples", "HuggingFaceTB__smol-smoltalk", "data",
                                    "test-00000-of-00001.parquet"))
    convs = []
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["messages", "source"])
        for msgs, src in zip(t.column("messages").to_pylist(), t.column("source").to_pylist()):
            users = [m_["content"] for m_ in msgs if m_["role"] == "user"]
            if len(users) >= 2:
                convs.append((src, users[0], users[1]))
        del t
    R["smoltalk"] = [{"src": s_, "first": u1, "second": u2}
                     for s_, u1, u2 in random.Random(0).sample(convs, min(3000, len(convs)))]
    R["proxy_second"] = [{"id": p_.id, "first": p_.turns[0], "second": p_.turns[1]}
                         for p_ in P.select(P.load(), "dev") if len(p_.turns) >= 2]
    log(f"sets: perturb {len(perturb)}, surnames {len(people)}, smoltalk {len(R['smoltalk'])}")
    return R


def stage_adv(a, t0) -> dict:
    """Evaluate the adversarial sets on ``--library`` with the ``bes`` on ``sys.path``."""
    import inspect
    import re

    log = lambda m: print(t_now(t0), m, flush=True)  # noqa: E731
    S = json.load(open(os.path.join(a.cache_dir, "e1v2_advsets.json")))
    lib = Library.load(a.library)
    ctx_ok = "context" in inspect.signature(lib.retrieve).parameters

    def ret(q, context=None):
        return lib.retrieve(q, context=context) if (ctx_ok and context is not None) else lib.retrieve(q)

    def show(p):
        if p is None:
            return None
        return {"kind": p.kind, "prob": round(float(p.prob), 3), "title": p.title,
                "answer": p.body if p.kind == "qa" else None}

    qs = stored_questions(lib)
    pid_of = {q: lib.n_wiki + e for e, q in enumerate(qs)}
    R: dict = {"library": a.library, "code": os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["bes"].__file__))),
               "threshold": float(lib.gate.threshold), "context_argument": ctx_ok}
    per: dict = {}
    ex: dict = {}
    for x in S["perturb"]:
        d = per.setdefault(x["cls"], {"tested": 0, "fired": 0, "original": 0, "skipped": 0})
        orig = pid_of.get(x["orig_q"])
        if orig is None:                     # the original is not stored in this library
            d["skipped"] += 1
            continue
        d["tested"] += 1
        p = ret(x["q"])
        if p is None:
            continue
        d["fired"] += 1
        if p.kind == "qa" and p.pid == orig:
            d["original"] += 1
            if len(ex.setdefault(x["cls"], [])) < 3:
                ex[x["cls"]].append({"asked": x["q"], "stored": x["orig_q"], "answer": p.body, "p": round(p.prob, 2)})
    R["perturb"], R["perturb_examples"] = per, ex
    log(f"perturb {per}")
    titles = lib.packed.all_titles()
    tid_of = {t: i for i, t in enumerate(titles)}
    c = {"own": 0, "another article": 0, "qa entry": 0, "nothing": 0}
    rows = []
    for x in S["surnames"]:
        p = ret(x["q"])
        lab = "nothing" if p is None else "qa entry" if p.kind == "qa" else \
            "own" if p.title == x["title"] else "another article"
        c[lab] += 1
        rows.append({"q": x["q"], "person": x["title"], "label": lab, "got": show(p)})
    R["surnames"] = {"counts": c, "rows": rows}
    c = {"own": 0, "another article": 0, "qa entry": 0, "nothing": 0, "unlabelled": 0}
    rows = []
    for x in S["names50"]:
        p = ret(x["q"])
        lab = "nothing" if p is None else "unlabelled" if x["want"] is None else "qa entry" if p.kind == "qa" \
            else "own" if p.title == x["want"] else "another article"
        c[lab] += 1
        rows.append({"q": x["q"], "want": x["want"], "label": lab, "got": show(p)})
    R["names50"] = {"counts": c, "rows": rows}
    for s in ("eval_probe", "eval_names"):
        n = fired = hit = 0
        for x in S[s]:
            if titles[x["tid"]] != x["title"]:
                raise ValueError("the probe sets were made for a library with other titles")
            n += 1
            p = ret(x["q"])
            if p is not None:
                fired += 1
                hit += p.kind == "wiki" and p.title_id == x["tid"]
        R[s] = {"n": n, "coverage": fired / max(1, n), "precision": hit / max(1, fired)}
    log(f"probes {R['eval_probe']} names {R['eval_names']}")
    R["groups"] = {}
    for g, qs_ in S["groups"].items():
        rows = [{"q": q, "qtype": query_type(q), "got": show(ret(q)),
                 "got_later_turn": show(ret(q, ["Tell me about the history of Rome."]))} for q in qs_]
        R["groups"][g] = {"n": len(rows), "fired": sum(r["got"] is not None for r in rows),
                          "fired_later_turn": sum(r["got_later_turn"] is not None for r in rows), "rows": rows}
    sm = {"n": len(S["smoltalk"]), "first": 0, "second_alone": 0, "second_with_context": 0}
    fired_ctx, fired_alone = [], []
    for i, x in enumerate(S["smoltalk"]):
        sm["first"] += ret(x["first"]) is not None
        p = ret(x["second"])
        if p is not None:
            sm["second_alone"] += 1
            fired_alone.append(i)
        p2 = ret(x["second"], [x["first"]]) if ctx_ok else p
        if p2 is not None:
            sm["second_with_context"] += 1
            fired_ctx.append({"i": i, "got": show(p2)})
    sm["fired_with_context"] = fired_ctx
    sm["fired_alone"] = fired_alone
    R["smoltalk"] = sm
    pr = {"n": len(S["proxy_second"]), "second_alone": 0, "second_with_context": 0, "rows": []}
    for x in S["proxy_second"]:
        p, p2 = ret(x["second"]), (ret(x["second"], [x["first"]]) if ctx_ok else None)
        pr["second_alone"] += p is not None
        pr["second_with_context"] += (p2 if ctx_ok else p) is not None
        if p is not None or p2 is not None:
            pr["rows"].append({"id": x["id"], "second": x["second"], "alone": show(p), "context": show(p2)})
    R["proxy_second"] = pr
    log(f"smoltalk {({k: v for k, v in sm.items() if not isinstance(v, list)})}, proxy second turns "
        f"{pr['second_alone']}/{pr['second_with_context']}")
    R["stale_scan"] = {}
    for name, rx in STALE_PATTERNS.items():
        r_ = re.compile(rx, re.I)
        hits = [q for q in qs if r_.search(q)]
        R["stale_scan"][name] = {"stored": len(hits), "examples": hits[:: max(1, len(hits) // 6)][:6]}
    R["accounting"] = lib.accounting()
    R["read_bound"] = lib.read_bound()
    return R


# --------------------------------------------------------------------------- report


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def review_section(a, R: dict, add) -> None:
    """Before / after tables of the review fixes: the library and code at the commit before
    them (``e1v2_lib_before.json``, ``e1v2_adv_before.json``) against this library."""
    J = lambda nm: os.path.join(a.cache_dir, f"e1v2_{nm}.json")  # noqa: E731
    if not (os.path.exists(J("adv")) and os.path.exists(J("adv_before")) and os.path.exists(J("lib_before"))):
        return
    A, AB, RB = json.load(open(J("adv"))), json.load(open(J("adv_before"))), json.load(open(J("lib_before")))
    lab = json.load(open(J("smoltalk_labels"))) if os.path.exists(J("smoltalk_labels")) else None
    add("## Review fixes: before / after\n")
    add("A review of this library showed that a wrong entry placed before a question produces a confident wrong "
        "answer: stored answers of a different question (another year, ordinal, polarity or entity), a person's "
        "common name answered with a film or a unit, follow-up and chat messages answered with unrelated articles "
        "through the exact-title rule, stale answers. What changed (`bes/library`):\n")
    add("- **Contrast check** (`qa.contrast_mismatch`): a stored question is dropped as a candidate when it differs "
        "from the user's in its numbers (digit tokens, ordinals, roman numerals, compared as numbers), order words "
        "(last, next, ...), degree words (most, least, superlatives, comparatives), relation words (before, after, "
        "since, until) or number of negations, or when a capitalised name of either side is missing from the other. "
        "Up to three candidates are read. Decided by measurement (old library and gate, top candidates returned at "
        "p >= 0.328 on dev NQ, dev TriviaQA, NQ-open val and TriviaQA val): the candidates each class removes were "
        "22-70% correct (relation words 50-83% on groups of 8 to 23 questions), below the 0.8 target, so a veto, "
        "not a gate feature (details in the notes).")
    add("- **Primary topics** (`titles.TitleTable`): a plain title owns its key; a variant (disambiguated, 'The X', "
        "plural, stopword-dropped) or a person article (surname or the single name its article uses) owns a key "
        f"only when it leads every other article containing it by {L.title_margin} in importance; who-questions "
        f"look among persons (lead by {L.person_margin}, importance without the bare-name mentions a disambiguated "
        "title shares with its namesakes); ambiguous keys are not stored, so they give no exact-title credit; a "
        "title word the article never writes in lower case is not plural-stemmed, nor is a capitalised query word. "
        "Exact-title credit only for identity requests (who/what is X, tell me about X, define X, a bare name); a "
        "one-term key needs such a frame or a capitalised name.")
    add("- **Conversation turns** (`retrieve.turn_type`, `Library.retrieve(..., context=...)`): a later turn that "
        "leans on an earlier one (it, its, he, she, they, that, 'the second one', 'what about X?', fewer than two "
        "content terms without a capitalised name) retrieves nothing; chat and acknowledgements retrieve nothing in "
        "any turn.")
    add("- **Time filter**: rules for most followed / viewed / subscribed, 'got the most', counts of seasons, "
        "episodes, stores, members ... 'are there' / 'does X have', 'still have', the oldest / richest / tallest "
        "living, office holders without 'current' ('who is france's president', 'who runs X').")
    add(f"- **Gate**: refitted on dev data with the contrastive questions, common-name probes and conversation "
        f"negatives; a negative set's rows weigh like rows of the largest dev set; operating point: the smallest "
        f"threshold at which every dev set with right answers reaches precision {L.gate_precision} (was 0.5) and "
        f"the negative sets return a known-wrong entry for at most 5% of their queries.")
    add("- **Provenance** (pool index, title importance) is written next to the library "
        "(`data_cache/library/provenance/<hash of library.json>/`), no longer inside the shipped directory.\n")
    acc_b, acc_a = RB["accounting"], R["accounting"]
    add(f"Stored numbers {acc_b['total_numbers']:,} -> {acc_a['total_numbers']:,} (title look-up "
        f"{acc_b['title_table_numbers']:,} -> {acc_a['title_table_numbers']:,}, qa layer {acc_b['qa_numbers']:,} -> "
        f"{acc_a['qa_numbers']:,}, gate {acc_b['gate_numbers']} -> {acc_a['gate_numbers']}); worst-case read of one "
        f"retrieval {RB['read_bound']['total']:,} -> {R['read_bound']['total']:,} (three qa entries read instead of "
        f"two, four title look-ups instead of one); gate threshold {RB['gate_threshold']:.3f} -> "
        f"{R['gate_threshold']:.3f}.\n")
    # ---- evaluation, both kinds
    add("Evaluation, both kinds, coverage / precision (before: the code and library before the fixes; the probe "
        "rows of 'before' are the old probe definition, the head-entity and common-name rows below use the new "
        "probe sets for both):\n")
    add("| set | before: coverage / precision | after: coverage / precision |")
    add("|---|---|---|")
    for s in ("dev_nq", "dev_tqa", "nq", "tqa", "popqa"):
        b, x = RB["sets"][s]["wiki+qa"], R["sets"][s]["wiki+qa"]
        add(f"| {NAMES[s]} | {pct(b['coverage'])} / {pct(b['precision'])} | {pct(x['coverage'])} / {pct(x['precision'])} |")
    for s, nm in (("eval_probe", "head-entity probes (eval, new definition)"), ("eval_names", "common-name probes (eval)")):
        add(f"| {nm} | {pct(AB[s]['coverage'])} / {pct(AB[s]['precision'])} | {pct(A[s]['coverage'])} / "
            f"{pct(A[s]['precision'])} |")
    ob, oa = RB["overlap"]["nq"]["without"]["wiki+qa"], R["overlap"]["nq"]["without"]["wiki+qa"]
    add(f"| NQ-open val without a near-duplicate stored question | {pct(ob['coverage'])} / {pct(ob['precision'])} | "
        f"{pct(oa['coverage'])} / {pct(oa['precision'])} |")
    add("")
    # ---- curves before
    add("Precision / coverage of the gate before the fixes (both kinds; the curves after are in the section below):\n")
    grid = [c["t"] for c in RB["sets"]["dev_nq"]["wiki+qa"]["curve"]]
    add("| set | " + " | ".join(f"t={t:.3f}" for t in grid) + " |")
    add("|---|" + "---|" * len(grid))
    for s in ("dev_nq", "dev_tqa", "dev_probe", "nq", "tqa", "popqa"):
        cs = RB["sets"][s]["wiki+qa"]["curve"]
        add(f"| {NAMES[s]} | " + " | ".join(f"{pct(c['coverage'])} / {pct(c['precision'])}" for c in cs) + " |")
    add("")
    # ---- perturbation
    add("**Perturbed questions** (the review's test, `stage adv`): stored questions of the evaluation half of the store "
        "edited so that the stored answer no longer fits (skipped when the edit is itself stored); share for which "
        "the library returns the original entry, with its answer:\n")
    add("| edit | variants | before | after |")
    add("|---|---|---|---|")
    for cls, x in A["perturb"].items():
        b = AB["perturb"].get(cls, {})
        add(f"| {cls} | {x['tested']} | {b.get('original', 0)} ({pct(b.get('original', 0) / max(1, b.get('tested', 1)))}%) | "
            f"{x['original']} ({pct(x['original'] / max(1, x['tested']))}%) |")
    add("\nThe lower-case country swap is an extra row: with no capital letters on either side no name can be "
        "compared, and a rule for rare substituted terms would have removed dev returns that are 75-83% correct "
        "at the operating point, so it was not added; long questions that differ in one lower-case place name are "
        "the remaining known gap.\n")
    # ---- common names
    sb, sa = AB["surnames"]["counts"], A["surnames"]["counts"]
    add("**Common names.** The review's 300 most important person articles asked 'Who was <surname>?', the review's "
        "50 famous names, and the common-name probes of the evaluation half (persons asked by the name their "
        "article uses most, when no more important person bears it):\n")
    add("| set | before | after |")
    add("|---|---|---|")
    add(f"| 300 persons by surname: own article / another article / qa entry / nothing | {sb['own']} / "
        f"{sb['another article']} / {sb['qa entry']} / {sb['nothing']} | {sa['own']} / {sa['another article']} / "
        f"{sa['qa entry']} / {sa['nothing']} |")
    nb, na = AB["names50"]["counts"], A["names50"]["counts"]
    add(f"| 50 famous names: own / another / qa / nothing | {nb['own']} / {nb['another article']} / {nb['qa entry']} / "
        f"{nb['nothing']} | {na['own']} / {na['another article']} / {na['qa entry']} / {na['nothing']} |")
    add(f"| common-name probes (eval): coverage / precision | {pct(AB['eval_names']['coverage'])} / "
        f"{pct(AB['eval_names']['precision'])} | {pct(A['eval_names']['coverage'])} / {pct(A['eval_names']['precision'])} |")
    other = [r for r in A["surnames"]["rows"] if r["label"] == "another article"]
    if other:
        add("\nAfter, the 300-person set's 'another article' answers: " + "; ".join(
            f"'{r['q']}' ({r['person']}) -> {r['got']['title']}" for r in other) + ".")
    add("")
    # ---- firing on messages that are not look-ups
    add("**Messages that are not factual look-ups** (the review's lists; 'later turn': asked after 'Tell me about the "
        "history of Rome.'): messages that get an entry:\n")
    add("| messages | n | before | after, first turn | after, later turn |")
    add("|---|---|---|---|---|")
    for g, x in A["groups"].items():
        add(f"| {g} | {x['n']} | {AB['groups'][g]['fired']} | {x['fired']} | {x['fired_later_turn']} |")
    fired = [f"'{r['q']}' -> {r['got']['title'] if r['got']['kind'] == 'wiki' else 'qa: ' + str(r['got']['answer'])}"
             for g, x in A["groups"].items() for r in x["rows"] if r["got"] is not None]
    if fired:
        add("\nAfter, first turns that get an entry: " + "; ".join(fired) + ".")
    fb, fa = RB["firing"]["proxy"], R["firing"]["proxy"]
    add("\nDev-half proxy prompts (first turns), share that gets an entry, before -> after: " + ", ".join(
        f"{c} {pct(fb[c]['fires'])} -> {pct(fa[c]['fires'])}%" for c in fa) + "; Dolly-15k: " + ", ".join(
        f"{c} {pct(RB['firing']['dolly'][c]['fires'])} -> {pct(R['firing']['dolly'][c]['fires'])}%"
        for c in R["firing"]["dolly"]) + ".\n")
    smb, sma = AB["smoltalk"], A["smoltalk"]
    add(f"**Real second user turns** (the review's seeded sample of {sma['n']:,} smol-smoltalk test conversations): "
        f"second turns that get an entry, retrieving on the turn alone as the responder does today: before "
        f"{smb['second_alone']} ({pct(smb['second_alone'] / sma['n'])}%), after {sma['second_alone']} "
        f"({pct(sma['second_alone'] / sma['n'])}%); after, with the first turn passed as context: "
        f"{sma['second_with_context']} ({pct(sma['second_with_context'] / sma['n'])}%). First turns: "
        f"{smb['first']} -> {sma['first']}. Dev-half proxy second turns: {AB['proxy_second']['second_alone']} -> "
        f"{A['proxy_second']['second_with_context']} of {A['proxy_second']['n']}.")
    if lab:
        n_ok = sum(1 for v in lab["labels"].values() if v == "related")
        add(f" Every second turn that gets an entry after the fixes (with context) was read by hand: {n_ok} of "
            f"{len(lab['labels'])} entries are related to the turn ({lab.get('note', '')}).")
    add("")
    # ---- stale scan
    add("**Stale answers**: stored questions matching the review's scan patterns (they also match stable "
        "questions, e.g. 'how many toes does a llama have?'):\n")
    add("| pattern | before | after |")
    add("|---|---|---|")
    for k, x in A["stale_scan"].items():
        add(f"| {k} | {AB['stale_scan'][k]['stored']} | {x['stored']} |")
    stale = A["groups"].get("stale answers", {}).get("rows", [])
    add(f"\nThe review's natural stale questions that get an entry: before {AB['groups']['stale answers']['fired']} of "
        f"{len(stale)}, after {sum(r['got'] is not None for r in stale)}.\n")


def stage_report(a) -> str:
    J = lambda nm: os.path.join(a.cache_dir, f"e1v2_{nm}.json")  # noqa: E731
    P = json.load(open(J("pool")))
    S = json.load(open(J("sweep"))) if os.path.exists(J("sweep")) else None
    R = json.load(open(J("lib")))
    B = json.load(open(J("build"))) if os.path.exists(J("build")) else {}
    N = json.load(open(J("notes"))) if os.path.exists(J("notes")) else []
    w = []
    add = w.append
    acc = R["accounting"]
    add("# E1 — library coverage (written library v2)\n")
    add("Generated by `scripts/measure_coverage.py` (stages `pool`, `sweep`, `lib`, `report`) from the saved library "
        f"`{R['library']}`. Every number is measured; nothing is extrapolated. Hits: a wiki passage is a hit when a "
        "normalised answer string occurs as a whole-token sequence in its text (title line + body); a qa entry when "
        "an answer string occurs in its stored answer (qa EM: stored answer equals an answer after SQuAD "
        "normalisation); a head-entity probe when the passage belongs to the probed article. These are proxies: a "
        "passage can contain the answer string by chance, and can answer a question without containing it.\n")
    n = R["n"]
    add(f"Evaluation questions: NQ-open validation {n['nq']} (all with answers), TriviaQA rc.nocontext validation "
        f"{n['tqa']} (seeded sample of the 9,961 unique questions), PopQA {n['popqa']} (seeded sample), "
        f"head-entity probes {n['eval_probe']} and common-name probes {n.get('eval_names', 0)} (evaluation half of "
        f"the titles), Dolly-15k {n['dolly']} instructions and the {n['proxy']} dev-half proxy prompts (first "
        f"turns). Dev data (every choice: index size, gate): {n['dev_nq']} stored NQ-open train and {n['dev_tqa']} "
        f"stored TriviaQA train questions asked with their own qa entry excluded (leave-one-out), "
        f"{n['dev_probe']} head-entity and {n.get('dev_names', 0)} common-name probes for the dev half of the "
        f"titles, {n.get('dev_contrast', 0)} contrastive questions (perturbed stored questions of the dev half "
        f"of the store) and {n.get('dev_chat', 0)} conversation turns that must retrieve nothing.\n")
    add("**Fair measurements.** NQ-open and TriviaQA have a known overlap between their train and test questions "
        "(paraphrases of the same question), and the qa layer stores the train questions, so their qa numbers "
        "are reported separately for test questions with and without a near-duplicate stored question. PopQA "
        "(templated questions about long-tail entities, no train split stored) and the proxy prompts are the fair "
        "out-of-distribution measurements.\n")
    review_section(a, R, add)
    # ---- summary
    sb = R["sets"]
    add("## Summary\n")
    add(f"- Everything the library stores: **{acc['total_numbers']:,} numbers** (cap {L.total_numbers_cap:,}): "
        f"wiki text blocks + directory + titles {acc['library_numbers']:,}, BM25 index {acc['index_numbers']:,}, "
        f"title look-up {acc['title_table_numbers']:,}, qa layer {acc['qa_numbers']:,}, gate {acc['gate_numbers']:,}.")
    th = R["gate_threshold"]
    add(f"- Calibrated gate: returns the candidate with the larger probability when p >= {th:.3f}, the smallest "
        f"threshold at which every dev set with right answers separately reaches precision >= {L.gate_precision} "
        f"and the contrastive and conversation dev sets return a known-wrong entry for at most 5% of their "
        f"queries (measured: " + ", ".join(f"{NAMES[k]} {pct(v['known_wrong'])}%" for k, v in
                                           R.get("negative_dev", {}).items()) + ").")
    bo = {s: sb[s]["wiki+qa"] for s in EVAL_SETS}
    ovn = R["overlap"]
    add(f"- Evaluation, both kinds: NQ-open val {pct(bo['nq']['coverage'])}% of questions answered at "
        f"{pct(bo['nq']['precision'])}% precision, TriviaQA val {pct(bo['tqa']['coverage'])}% at "
        f"{pct(bo['tqa']['precision'])}%, PopQA {pct(bo['popqa']['coverage'])}% at {pct(bo['popqa']['precision'])}%, "
        f"head-entity probes {pct(bo['eval_probe']['coverage'])}% at {pct(bo['eval_probe']['precision'])}%, "
        f"common-name probes {pct(bo['eval_names']['coverage'])}% at {pct(bo['eval_names']['precision'])}%.")
    add(f"- Fair numbers: NQ-open questions without a near-duplicate stored question "
        f"({pct(1 - ovn['nq']['share'])}% of them) {pct(ovn['nq']['without']['wiki+qa']['coverage'])}% at "
        f"{pct(ovn['nq']['without']['wiki+qa']['precision'])}% (with one: {pct(ovn['nq']['with']['wiki+qa']['coverage'])}% at "
        f"{pct(ovn['nq']['with']['wiki+qa']['precision'])}%); PopQA, out of distribution: "
        f"{pct(bo['popqa']['coverage'])}% at {pct(bo['popqa']['precision'])}% (the floor is fitted on "
        f"NQ/TriviaQA-style questions and head entities, not on long-tail entities).")
    fp = R["firing"]["proxy"]
    add("- Firing on the dev-half proxy prompts: " + ", ".join(f"{c} {pct(v['fires'])}%" for c, v in fp.items())
        + f"; Dolly creative_writing {pct(R['firing']['dolly']['creative_writing']['fires'])}%, brainstorming "
        f"{pct(R['firing']['dolly']['brainstorming']['fires'])}%.")
    rs = R["readset"]
    add(f"- Largest read of one retrieve(): {max(r['max_all'] for r in rs.values()):,} numbers (bound "
        f"{R['read_bound']['total']:,}, limit {L.max_read_numbers:,}); {R['throughput']['retrieve_per_s']:.0f} "
        f"retrieve() calls per second on one core.")
    add("")
    add("| set | wiki only: R@1 / coverage / precision | qa only: R@1 / EM / coverage / precision | both: R@1 / coverage / precision / answered |")
    add("|---|---|---|---|")
    for s in DEV_SETS + EVAL_SETS:
        x = sb[s]
        wk, qa, bo = x["wiki"], x["qa"], x["wiki+qa"]
        add(f"| {NAMES[s]} | {pct(wk['r@1'])} / {pct(wk['coverage'])} / {pct(wk['precision'])} | "
            f"{pct(qa['r@1'])} / {pct(qa['em'])} / {pct(qa['coverage'])} / {pct(qa['precision'])} | "
            f"{pct(bo['r@1'])} / {pct(bo['coverage'])} / {pct(bo['precision'])} / {pct(bo['answered'])} |")
    add("\nR@1 = hit of the single candidate without the gate and without the query-type filter (qa: 0 when no "
        "stored question reaches the minimum overlap); coverage = share of questions for which an entry is returned; precision = share of returned "
        "entries that are hits; answered = coverage x precision. Probes have no answer string, so their qa "
        "columns are empty and a qa entry returned for a probe counts as a miss in the \"both\" column.\n")
    # ---- gate coefficients
    from bes.library.gate import QA_FEATURES, WIKI_FEATURES

    add("## Gate model\n")
    add(f"Two logistic regressions fitted on the dev rows (each dev set with right answers weighted equally, a row "
        f"of the contrastive or conversation sets like a row of the largest dev set; L2 1e-3), float32, "
        f"{acc['gate_numbers']} stored numbers with the threshold. The threshold is the smallest at which every "
        f"dev set with right answers separately reaches precision >= {L.gate_precision} and the contrastive and "
        f"conversation sets return a known-wrong entry for at most 5% of their queries.\n")
    for kd, names in (("wiki", WIKI_FEATURES), ("qa", QA_FEATURES)):
        c = R["gate_coef"][kd]
        add(f"- {kd}: " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(list(names) + ["intercept"], c)))
    qt = R.get("qtypes", {})
    if qt:
        add("\nShare of questions the query-type filter lets through (question or look-up): "
            + ", ".join(f"{NAMES.get(k, k)} {pct(v)}%" for k, v in qt.items()) + ".")
    add("")
    # ---- overlap
    add("## Train/test question overlap (NQ-open, TriviaQA) and PopQA\n")
    ov = R["overlap"]
    add(f"A test question has a near-duplicate stored question when the Jaccard similarity of their content-term "
        f"sets (stopwords removed, plural-stemmed, no hashing) reaches {ov['nq']['threshold']} for some stored "
        f"question (brute force over all {acc.get('qa_entries', 0):,} stored questions).\n")
    add("| set | subset | n share | wiki R@1 / cov / prec | qa R@1 / EM / cov / prec | both R@1 / cov / prec |")
    add("|---|---|---|---|---|---|")
    for s in ("nq", "tqa", "popqa"):
        for name in ("with", "without"):
            x = ov[s].get(name) or {}
            if not x:
                add(f"| {NAMES[s]} | {name} near-duplicate | 0 | - | - | - |")
                continue
            sh = ov[s]["share"] if name == "with" else 1 - ov[s]["share"]
            wk, qa, bo = x["wiki"], x["qa"], x["wiki+qa"]
            add(f"| {NAMES[s]} | {name} near-duplicate | {wk['n']} ({pct(sh)}%) | {pct(wk['r@1'])} / {pct(wk['coverage'])} / "
                f"{pct(wk['precision'])} | {pct(qa['r@1'])} / {pct(qa['em'])} / {pct(qa['coverage'])} / {pct(qa['precision'])} | "
                f"{pct(bo['r@1'])} / {pct(bo['coverage'])} / {pct(bo['precision'])} |")
    add("")
    # ---- curves
    add("## Precision / coverage curves (calibrated probability threshold)\n")
    add("Each cell is coverage / precision of `p >= t` (one entry at most per question; \"both\" returns the "
        "kind with the larger probability). The chosen threshold is marked.\n")
    for kind in ("wiki", "qa", "wiki+qa"):
        add(f"Kind: **{kind}**\n")
        grid = [c["t"] for c in sb["dev_nq"][kind]["curve"]]
        add("| set | " + " | ".join(("**" if abs(t - th) < 1e-12 else "") + f"t={t:.3f}" + ("**" if abs(t - th) < 1e-12 else "")
                                     for t in grid) + " |")
        add("|---|" + "---|" * len(grid))
        for s in DEV_SETS + EVAL_SETS:
            cs = sb[s][kind]["curve"]
            add(f"| {NAMES[s]} | " + " | ".join(f"{pct(c['coverage'])} / {pct(c['precision'])}" for c in cs) + " |")
        add("")
    # ---- popqa bins
    add("## PopQA by subject popularity (monthly page views, `s_pop`)\n")
    add("| popularity | n | wiki R@1 | both coverage / precision |")
    add("|---|---|---|---|")
    for b in R["popqa_bins"]:
        if "s" not in b:
            continue
        add(f"| {b['bin']} | {b['n']} | {pct(b['s']['wiki']['r@1'])} | {pct(b['s']['wiki+qa']['coverage'])} / "
            f"{pct(b['s']['wiki+qa']['precision'])} |")
    add("")
    # ---- firing
    add("## Firing on prompts that are not factual look-ups\n")
    add("Share of prompts for which the library returns an entry (with the query-type filter, and without it).\n")
    for s, title in (("proxy", "Proxy prompts (dev half)"), ("dolly", "Dolly-15k")):
        add(f"{title}:\n")
        add("| category | n | query types | fires | of which qa | fires without type filter |")
        add("|---|---|---|---|---|---|")
        for c, v in R["firing"][s].items():
            ty = ", ".join(f"{k} {n_}" for k, n_ in sorted(v["types"].items()))
            add(f"| {c} | {v['n']} | {ty} | {pct(v['fires'])} | {pct(v['qa_share'])} | {pct(v['fires_no_type_filter'])} |")
        add("")
    # ---- read set
    add("## Read set per retrieve() (stored numbers)\n")
    add(f"Hard limit {L.max_read_numbers:,} numbers per retrieval (index lists + df + offsets + title look-up + "
        "directory + title block + text blocks + qa index + qa blocks).\n")
    add("| set | n | returned | mean | p99 | max | max index part | max qa index part | max blocks | queries/s (with log) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for s, r in R["readset"].items():
        add(f"| {NAMES.get(s, s)} | {r['n']} | {r['passed']} | {r['mean_all']:,.0f} | {r['p99_all']:,.0f} | {r['max_all']:,} | "
            f"{r['max_index']:,} | {r['max_qa_index']:,} | {r['max_blocks']} | {r['queries_per_s_with_log']:.0f} |")
    rb = R.get("read_bound")
    if rb:
        add("\nWorst case from the configuration and the largest stored blocks (`Library.read_bound`): "
            + ", ".join(f"{k} {v:,}" for k, v in rb.items() if k != "total") + f"; **total {rb['total']:,}**.")
    tp = R["throughput"]
    add(f"\nThroughput on one core (no read log, cold block cache): retrieve() {tp['retrieve_per_s']:.0f} queries/s "
        f"(NQ-open validation questions); passage() {tp['passage_random_per_s']:.0f} random passages/s, "
        f"{tp['passage_sequential_per_s']:.0f} sequential wiki passages/s, {tp['qa_passage_sequential_per_s']:.0f} "
        f"sequential qa entries/s.\n")
    # ---- budget sweep
    if S:
        add("## Index size and list cap (dev recall of the wiki candidate)\n")
        add(f"Everything else stored: {S['fixed_numbers']:,} numbers. The wiki candidate is the lead passage of the "
            "exact title match when there is one, else the best BM25 passage. Chosen by mean dev R@1 under the cap.\n")
        add("| index budget | max list | index numbers | total | dev NQ R@1 / BM25 R@3 | dev TQA R@1 / BM25 R@3 | dev probes R@1 | mean dev R@1 | max index read (dev) |")
        add("|---|---|---|---|---|---|---|---|---|")
        for r in S["rows"]:
            mark = "**" if (r["budget"], r["max_list"]) == (S["chosen"]["budget"], S["chosen"]["max_list"]) else ""
            add(f"| {mark}{r['budget'] / 1e6:.1f}M{mark} | {r['max_list']} | {r['index_numbers']:,} | {r['total_numbers']:,} | "
                f"{pct(r['dev_nq']['r@1'])} / {pct(r['dev_nq']['bm25_r@3'])} | {pct(r['dev_tqa']['r@1'])} / "
                f"{pct(r['dev_tqa']['bm25_r@3'])} | {pct(r['dev_probe']['r@1'])} | {pct(r['dev_mean_r1'])} | {r['index_read_max']:,} |")
        add("")
    # ---- pool
    add("## Candidate pool (BPE token counts) and text repairs\n")
    pb, pp = P["pool_bpe"], P.get("pool_proxy")
    add("| pool | articles | passages | mean tokens | median | p90 | max | >= 150 tokens | mean words | mean bytes | passages per article | packed numbers |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, x in (("v1, utf8 bytes / 3.3", pp), ("v2, BPE tokens", pb)):
        if x:
            add(f"| {name} | {x['articles']:,} | {x['passages']:,} | {x['tokens_mean']:.1f} | {x['tokens_p50']:.0f} | "
                f"{x['tokens_p90']:.0f} | {x['tokens_max']} | {pct(x['at_cap_150'])}% | {x['words_mean']:.1f} | "
                f"{x['bytes_mean']:.0f} | {x['passages_per_article']:.2f} | {x.get('packed_numbers', 0):,} |")
    add("\n(token columns measured with the BPE tokenizer in both pools.)\n")
    if "holes" in P:
        h = P["holes"]
        add(f"Removed-template holes, counted with independent detectors on the passages of the same "
            f"{h['articles']:,} sampled articles before (v1 pool) and after (v2 pool) the repair rules:\n")
        add("| detector | before | after |")
        add("|---|---|---|")
        for k in h["before"]:
            add(f"| `{k}` | {h['before'][k]:,} | {h['after'][k]:,} |")
        add(f"\nOver the whole dump the v2 rules dropped {pb['hole_sentences']:,} sentences/items with a hole and "
            f"repaired {pb['hole_repairs']:,} stranded approximators.\n")
    e = P["exact_pool"]
    add("Exact BM25 over every v2 pool passage (true terms, no hashing, no truncation; the ceiling of passage "
        "retrieval): " + ", ".join(f"{NAMES[s]} R@1 / R@3 / R@100 = {pct(e[s]['r@1'])} / {pct(e[s]['r@3'])} / "
                                    f"{pct(e[s]['r@100'])}" for s in ("nq", "tqa", "popqa")) + ".\n")
    # ---- qa layer
    if B.get("qa_clean"):
        qc = B["qa_clean"]
        add("## Question-answer layer: cleaning\n")
        add("| source | rows | kept | time-sensitive | evaluation overlap | duplicates (exact / normalised) | ambiguous | other drops |")
        add("|---|---|---|---|---|---|---|---|")
        for src, st in sorted(qc["stats"].items()):
            ts = sum(v for k, v in st.items() if k.startswith("time:"))
            other = sum(st.get(k, 0) for k in ("empty_question", "no_terms", "empty_answer", "long_answer", "long_question"))
            add(f"| {src} | {st['in']:,} | {st.get('kept', 0):,} | {ts:,} | {st.get('eval_overlap', 0):,} | "
                f"{st.get('duplicate_exact', 0):,} / {st.get('duplicate_norm', 0):,} | {st.get('ambiguous', 0):,} | {other:,} |")
        rules: dict[str, int] = {}
        for st in qc["stats"].values():
            for k, v in st.items():
                if k.startswith("time:"):
                    rules[k[5:]] = rules.get(k[5:], 0) + v
        add("\nTime-sensitive rules (all sources): " + ", ".join(f"{k} {v:,}" for k, v in sorted(rules.items(), key=lambda kv: -kv[1]))
            + f". Rendered entries over {L.max_passage_tokens} tokens dropped: {qc['long_render']}. Stored: "
            f"{qc['kept']:,} entries.\n")
        if "orders" in B:
            add("Entry order and compression (raw LZMA2 preset 6, 64 kB blocks; measured on the 156,800-entry store "
                "before the review's time rules):\n")
            add("| order | stored text numbers | compression |")
            add("|---|---|---|")
            for o, v in B["orders"].items():
                add(f"| {o} | {v['qa_text_numbers']:,} | {v['ratio']:.3f}x |")
            add("")
    if B.get("build_runs"):
        add("## Build\n")
        for r in B["build_runs"]:
            add(f"- {r}")
        add("")
    add("## Sources and licences\n")
    add("| source | Hugging Face dataset (config / split) | revision | licence (dataset card) | use |")
    add("|---|---|---|---|---|")
    add("| Simple English Wikipedia | wikimedia/wikipedia (20231101.simple / train) | b04c8d1ceb2f5cd4588862100d08de323dccfbaa | "
        "cc-by-sa-3.0 and gfdl | library text (`ATTRIBUTION.tsv`: one row per article with page id and url) |")
    for s in QA_SOURCES.values():
        use = ("qa layer" + ("" if s.permissive else "; left out by the permissive-only build")) if s.role == "store" \
            else "evaluation only; its questions are excluded from the store"
        add(f"| {s.name} | {s.repo} ({s.config} / {s.split}) | {s.revision} | {s.license} | {use} |")
    add("| Dolly-15k | databricks/databricks-dolly-15k | bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a | cc-by-sa-3.0 | "
        "firing measurement only |")
    add("\nThe built library carries `ATTRIBUTION.tsv` (wiki articles) and `ATTRIBUTION_QA.tsv` (qa sources with "
        "licences, then one row per stored entry: source and row). The licence of TriviaQA and WebQuestions is not "
        "stated on their cards; `--permissive-only` (`LibraryCfg.qa_permissive_only`) builds the qa layer from "
        "NQ-open alone.\n")
    if N:
        add("## Notes\n")
        for x in N:
            add(f"- {x}")
        add("")
    return "\n".join(w) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("pool", "sweep", "lib", "adv-sets", "adv", "report"), default="report")
    ap.add_argument("--tag", default="", help="stage adv: suffix of the output file (e.g. 'before')")
    ap.add_argument("--n-perturb", type=int, default=400, help="variants per perturbation class")
    ap.add_argument("--cache-dir", default="data_cache/library")
    ap.add_argument("--library", default=L.library_dir)
    ap.add_argument("--n-tqa", type=int, default=3000)
    ap.add_argument("--n-popqa", type=int, default=3000)
    ap.add_argument("--n-dolly", type=int, default=200)
    ap.add_argument("--n-dev", type=int, default=3000)
    ap.add_argument("--n-probes", type=int, default=2000)
    ap.add_argument("--near-dup", type=float, default=0.6)
    ap.add_argument("--sweep-budgets", default="8,16,23.5")
    ap.add_argument("--sweep-lists", default="1024,2048,4096")
    ap.add_argument("--total-cap", type=int, default=L.total_numbers_cap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default="docs/E1_COVERAGE.md")
    a = ap.parse_args(argv)
    t0 = time.time()
    J = lambda nm: os.path.join(a.cache_dir, f"e1v2_{nm}.json")  # noqa: E731
    if a.stage == "report":
        md = stage_report(a)
        with open(a.md, "w") as f:
            f.write(md)
        print(t_now(t0), "wrote", a.md, flush=True)
        return 0
    fn = {"pool": stage_pool, "sweep": stage_sweep, "lib": stage_lib, "adv-sets": stage_adv_sets,
          "adv": stage_adv}[a.stage]
    res = fn(a, t0)
    name = {"adv-sets": "advsets"}.get(a.stage, a.stage) + (f"_{a.tag}" if a.tag else "")
    with open(J(name), "w") as f:
        json.dump(res, f)
    print(t_now(t0), "wrote", J(name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
