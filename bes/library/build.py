"""Library build pipeline: source -> cleaned pool -> signals -> selection -> pack -> index,
plus the title look-up table, the question-answer layer and the calibrated gate.

Intermediate results are cached under ``cache_dir`` (``pool_bpe/`` with ``signals.npz``)
so that libraries for several budgets can be built from one cleaning pass. ``build_v2`` is
the complete build used by ``scripts/build_library.py``; with the same inputs it writes
byte-identical files.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from bes.config import LibraryCfg
from bes.library.clean import CleanCfg, CountTokens, count_tokens_name, default_count_tokens, iter_articles
from bes.library.index import IndexCfg, TruncatedIndex
from bes.library.pack import PackCfg, _atomic_write
from bes.library.retrieve import Library, RetrieveCfg, TitleTable, query_type
from bes.library.select import (SelectCfg, Selection, article_order, benefits, duplicate_mask,
                                estimate_costs, importance, mention_counts, passage_order,
                                passage_prior, select_for_budget)
from bes.library.sources import (DEFAULT_QA_STORE, QA_SOURCES, SOURCES, Pool, build_pool, download,
                                 download_qa, iter_qa)

Log = Callable[[str], None]
_L = LibraryCfg()


def _say(log: Log | None, msg: str) -> None:
    if log is not None:
        log(msg)


def pool_dir_for(cache_dir: str, count_tokens: CountTokens) -> str:
    """Pool cache directory of one token counter (``pool`` for the old byte proxy)."""
    name = count_tokens_name(count_tokens)
    return os.path.join(cache_dir, "pool" if name == "utf8_bytes/3.3" else f"pool_{name}")


def prepare_pool(cache_dir: str, source: str = "simplewiki", rebuild: bool = False,
                 clean_cfg: CleanCfg = CleanCfg(), max_words: int = _L.max_passage_words,
                 max_tokens: float = _L.max_passage_tokens,
                 count_tokens: CountTokens = default_count_tokens, log: Log | None = None,
                 pool_dir: str | None = None) -> Pool:
    """Load the cached pool or build it from the (downloaded) source."""
    pool_dir = pool_dir or pool_dir_for(cache_dir, count_tokens)
    name = count_tokens_name(count_tokens)
    if not rebuild and os.path.exists(os.path.join(pool_dir, "pool.json")):
        pool = Pool.load(pool_dir)
        if pool.meta.get("count_tokens", "utf8_bytes/3.3") == name:
            _say(log, f"pool: loaded {pool_dir}")
            return pool
        _say(log, f"pool: cached pool counts tokens with {pool.meta.get('count_tokens')}, rebuilding")
    src = SOURCES[source]
    path = download(src, cache_dir)
    t0 = time.time()
    pool = build_pool(path, clean_cfg, max_words, max_tokens, count_tokens,
                      progress=lambda k: _say(log, f"pool: {k} articles, {time.time() - t0:.0f}s"),
                      meta={"source": dataclasses.asdict(src), "clean_cfg": dataclasses.asdict(clean_cfg),
                            "count_tokens": name})
    pool.save(pool_dir)
    _say(log, f"pool: {pool.n_articles} articles, {pool.n_passages} passages, "
              f"{len(pool.body_blob)} body bytes, drops {pool.drops} ({time.time() - t0:.0f}s)")
    return pool


@dataclass
class Signals:
    mentions: np.ndarray  # float64 [A]
    dup: np.ndarray       # bool [P]
    order: np.ndarray     # int64 [P] pack order of pool passages
    cost: np.ndarray      # float64 [P] estimated stored numbers


def prepare_signals(pool: Pool, cache_dir: str, pack_cfg: PackCfg = PackCfg(),
                    rebuild: bool = False, log: Log | None = None, pool_dir: str | None = None) -> Signals:
    pool_dir = pool_dir or os.path.join(cache_dir, "pool" if pool.meta.get("count_tokens", "utf8_bytes/3.3")
                                        == "utf8_bytes/3.3" else f"pool_{pool.meta['count_tokens']}")
    path = os.path.join(pool_dir, "signals.npz")
    tag = json.dumps(dataclasses.asdict(pack_cfg), sort_keys=True)
    if not rebuild and os.path.exists(path):
        z = np.load(path)
        if str(z["tag"]) == tag and z["dup"].shape[0] == pool.n_passages:
            _say(log, "signals: cached")
            return Signals(z["mentions"], z["dup"], z["order"], z["cost"])
    t0 = time.time()
    src_path = download(SOURCES[pool.meta["source"]["name"]], cache_dir)
    row_to_art = {int(r): a for a, r in enumerate(pool.article_src)}

    def texts():
        for k, row in enumerate(iter_articles(src_path, columns=("text",))):
            yield row_to_art.get(k, -1), row["text"]

    mentions = mention_counts(pool.titles, texts())
    _say(log, f"signals: mentions done ({time.time() - t0:.0f}s), "
              f"articles with >=1 mention {(mentions > 0).mean():.3f}")
    dup = duplicate_mask(pool)
    _say(log, f"signals: {int(dup.sum())} duplicate bodies")
    order = passage_order(pool, article_order(pool))
    cost, packed = estimate_costs(pool, order, pack_cfg)
    acc = packed.accounting()
    _say(log, f"signals: whole pool packs to {acc['library_numbers']} numbers "
              f"(text {acc['text_numbers']}, directory {acc['directory_numbers']}, "
              f"titles {acc['title_numbers']}) ({time.time() - t0:.0f}s)")
    tmp = path + ".tmp.npz"
    np.savez(tmp, mentions=mentions, dup=dup, order=order, cost=cost, tag=np.asarray(tag),
             pool_numbers=np.asarray(acc["library_numbers"]))
    os.replace(tmp, path)
    return Signals(mentions, dup, order, cost)


def write_attribution(pool: Pool, sel: Selection, out_dir: str, cache_dir: str | None = None) -> str:
    """Write ``ATTRIBUTION.tsv``: one row per stored title (title id, first passage id, number
    of passages, title, source page id, url) under the source licence. Metadata for the
    licence (CC BY-SA for Wikipedia text); not read by the model, not counted."""
    import pyarrow.parquet as pq

    src = pool.meta.get("source") or {}
    ids: list[str] = []
    urls: list[str] = []
    path = None
    if cache_dir is not None and src:
        p = os.path.join(cache_dir, "raw", src["name"], src["filename"])
        path = p if os.path.exists(p) else None
    if path is not None:
        t = pq.read_table(path, columns=["id", "url"])
        ids, urls = t.column("id").to_pylist(), t.column("url").to_pylist()
    rows = [f"# {src.get('repo', '?')}@{src.get('revision', '?')} {src.get('filename', '')} "
            f"licence: {src.get('license', '?')}",
            "title_id\tfirst_pid\tn_passages\ttitle\tsource_id\turl"]
    tf = np.asarray(sel.packed.title_first, dtype=np.int64)
    arts = pool.p_article[sel.pool_idx]
    for tid in range(sel.packed.n_titles):
        a = int(arts[tf[tid]])
        r = int(pool.article_src[a])
        sid = ids[r] if r < len(ids) else ""
        url = urls[r] if r < len(urls) else ""
        rows.append(f"{tid}\t{tf[tid]}\t{tf[tid + 1] - tf[tid]}\t{pool.titles[a]}\t{sid}\t{url}")
    out = os.path.join(out_dir, "ATTRIBUTION.tsv")
    with open(out + ".tmp", "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(out + ".tmp", out)
    return out


def title_importance(pool: Pool, sig: Signals, sel: Selection, select_cfg: SelectCfg = SelectCfg()) -> np.ndarray:
    """float64 [titles of the packed library]: article importance (``select.importance``)."""
    imp = importance(pool, sig.mentions, select_cfg)
    tf = np.asarray(sel.packed.title_first[:-1], dtype=np.int64)
    return imp[pool.p_article[sel.pool_idx[tf]]]


def build_for_budget(pool: Pool, sig: Signals, budget_numbers: int,
                     index_cfg: IndexCfg = IndexCfg(), pack_cfg: PackCfg = PackCfg(),
                     select_cfg: SelectCfg = SelectCfg(), retrieve_cfg: RetrieveCfg = RetrieveCfg(),
                     count_tokens: CountTokens = default_count_tokens, out_dir: str | None = None,
                     log: Log | None = None, cache_dir: str | None = None,
                     count_tokens_name_: str | None = None, qa=None) -> tuple[Library, Selection]:
    """Select under ``budget_numbers`` (text + directory + titles), pack, index, build the
    title table, attach the optional qa store and save (no gate: ``fit_gate`` adds it)."""
    t0 = time.time()
    imp = importance(pool, sig.mentions, select_cfg)
    ben = benefits(pool, imp, sig.dup, select_cfg)
    sel = select_for_budget(pool, budget_numbers, ben, sig.cost, sig.order, pack_cfg, select_cfg, log)
    _say(log, f"build: selected {int(sel.mask.sum())} passages ({time.time() - t0:.0f}s)")
    prior = passage_prior(pool, imp, select_cfg)[sel.pool_idx]
    index = TruncatedIndex.build(sel.passages, index_cfg, prior)
    _say(log, f"build: index {index.accounting()['index_numbers']} numbers ({time.time() - t0:.0f}s)")
    tf = np.asarray(sel.packed.title_first[:-1], dtype=np.int64)
    titles = [pool.titles[int(pool.p_article[sel.pool_idx[t]])] for t in tf]
    table = TitleTable.build(titles, title_importance(pool, sig, sel, select_cfg))
    meta = {
        "source": pool.meta.get("source"), "budget_numbers": int(budget_numbers),
        "select_cfg": dataclasses.asdict(select_cfg), "selection_history": sel.history,
        "pool_passages": pool.n_passages, "pool_articles": pool.n_articles,
        "selected_articles": int(len(set(pool.p_article[sel.pool_idx].tolist()))),
        "max_tokens": pool.meta.get("max_tokens"), "max_words": pool.meta.get("max_words"),
        "count_tokens": count_tokens_name_ or count_tokens_name(count_tokens),
    }
    lib = Library(sel.packed, index, retrieve_cfg, count_tokens, meta, qa=qa, titles=table)
    if out_dir is not None:
        lib.save(out_dir)
        prov = os.path.join(out_dir, "provenance")
        os.makedirs(prov, exist_ok=True)
        np.save(os.path.join(prov, "pool_idx.npy"), sel.pool_idx)  # provenance (not shipped)
        write_attribution(pool, sel, out_dir, cache_dir)
    _say(log, f"build: done ({time.time() - t0:.0f}s) {lib.accounting()['total_numbers']} numbers")
    return lib, sel


# ------------------------------------------------------------------------------ qa layer


def eval_questions(cache_dir: str, include_proxy: bool = True) -> list[str]:
    """Every question of every evaluation split (``QA_SOURCES`` with role ``eval``) and, by
    default, every turn of the proxy prompt set: the store must contain none of them."""
    out: list[str] = []
    for s in QA_SOURCES.values():
        if s.role != "eval":
            continue
        path = download_qa(s, cache_dir)
        out.extend(q for _, q, _ in iter_qa(s, path))
    if include_proxy:
        from bes.eval.proxy import prompts as proxy_prompts

        out.extend(t for p in proxy_prompts.load() for t in p.turns)
    return out


def load_qa_pairs(cache_dir: str, sources: Sequence[str] = DEFAULT_QA_STORE, permissive_only: bool = False,
                  clean_cfg=None, log: Log | None = None):
    """(pairs, stats, eval_keys, raw) for the store sources: cleaned pairs (``qa.clean_pairs``)
    with every evaluation question excluded; ``raw[(source, row)]`` = (original question,
    normalised answers) for every kept pair (the dev labels)."""
    from bes.library import qa as Q

    names = [n for n in sources if QA_SOURCES[n].role == "store" and (QA_SOURCES[n].permissive or not permissive_only)]
    keys = Q.eval_keys(eval_questions(cache_dir))
    from bes.library.devsets import answers as norm_answers

    raw: dict[tuple[str, int], tuple[str, tuple[str, ...]]] = {}

    def rows():
        for n in names:
            s = QA_SOURCES[n]
            for i, q, a in iter_qa(s, download_qa(s, cache_dir)):
                raw[(n, i)] = (q, tuple(norm_answers(a)))     # normalised answers (dev labels)
                yield n, i, q, a[:1]

    pairs, stats = Q.clean_pairs(rows(), keys, clean_cfg or Q.QACleanCfg())
    Q.assert_no_eval_overlap(pairs, keys)
    kept = {(p.source, p.row) for p in pairs}
    raw = {k: v for k, v in raw.items() if k in kept}
    _say(log, f"qa: {len(pairs)} pairs from {names} ({len(keys)} evaluation questions excluded)")
    return pairs, stats, keys, raw


def drop_long_renders(pairs, max_tokens: float, count_tokens: CountTokens):
    """Pairs whose rendered text fits ``max_tokens`` tokens, and the number dropped."""
    from bes.library.qa import render_qa

    keep = [p for p in pairs if count_tokens(render_qa(p.question, p.answer)) <= max_tokens]
    return keep, len(pairs) - len(keep)


# ------------------------------------------------------------------------------ gate


def dev_queries(lib: Library, pairs, raw, n_per_source: int, n_probes: int, seed: int,
                title_imp: np.ndarray | None) -> list[dict]:
    """Dev queries (``bes.library.devsets``): stored NQ / TriviaQA train questions with their
    own entry excluded, and head-entity probes for the dev half of the titles."""
    from bes.library import devsets as D
    from bes.library.qa import entry_order

    out: list[dict] = []
    if lib.qa is not None and pairs:
        order = entry_order(pairs, lib.qa.cfg.order)
        ent_of = np.empty(len(pairs), np.int64)
        ent_of[order] = np.arange(len(pairs))
        rng = random.Random(seed)
        for src, name in (("nq_open_train", "dev_nq"), ("trivia_qa_train", "dev_tqa")):
            idx = [i for i, p in enumerate(pairs) if p.source == src]
            for i in rng.sample(idx, min(n_per_source, len(idx))):
                p = pairs[i]
                if lib.qa.entry(int(ent_of[i])) != (p.question, p.answer):
                    raise ValueError("the pairs do not match the stored qa entries (rebuild the library)")
                q_raw, ans = raw[(p.source, p.row)]
                out.append({"q": q_raw, "ans": D.answers(ans), "set": name,
                            "exclude_pids": [lib.n_wiki + int(ent_of[i])]})
    if n_probes and title_imp is not None:
        titles = lib.packed.all_titles()
        for x in D.probes(titles, title_imp, n_probes, "dev", seed):
            x["set"] = "dev_probe"
            x["exclude_pids"] = []
            out.append(x)
    return out


def candidate_rows(lib: Library, queries: Sequence[dict], kinds=None) -> list[dict]:
    """For every query: the wiki and qa candidates (features + labels) as the gate sees them."""
    from bes.library import devsets as D

    rows = []
    for x in queries:
        r = {"set": x["set"], "qtype": query_type(x["q"]), "wiki": None, "qa": None}
        for c in lib.candidates(x["q"], 1, x.get("exclude_pids", ()), (), None, kinds, cross=True):
            if r[c.kind] is not None:
                continue
            if c.kind == "wiki":
                if "tid" in x:
                    y = lib.title_id_of(c.pid) == x["tid"]
                else:
                    y = D.contains(lib.passage(c.pid).text, x["ans"])
            else:
                y = bool(x["ans"]) and D.qa_hit(lib.passage(c.pid).body, x["ans"])
            r[c.kind] = {"pid": c.pid, "feats": {k: v for k, v in c.feats.items() if k != "answer"},
                         "y": bool(y)}
        rows.append(r)
    return rows


def _set_weights(rows: Sequence[dict]) -> np.ndarray:
    sets = sorted({r["set"] for r in rows})
    n = {s: sum(1 for r in rows if r["set"] == s) for s in sets}
    return np.array([1.0 / n[r["set"]] for r in rows])


def labelled_kinds(row: dict, kinds=("wiki", "qa")) -> tuple[str, ...]:
    """Kinds whose label is defined for a row: probes have no answer string, so only their
    wiki candidate (the probed article or not) is labelled."""
    return tuple(k for k in kinds if k == "wiki") if row["set"].endswith("probe") else tuple(kinds)


def decisions(rows: Sequence[dict], gate, allowed=("question", "lookup"), kinds=("wiki", "qa"),
              labelled_only: bool = False):
    """(p of the returned candidate or -inf, label of that candidate, its kind) per row.
    ``labelled_only`` restricts every row to its ``labelled_kinds``."""
    p = np.full(len(rows), -np.inf)
    y = np.zeros(len(rows), dtype=bool)
    kind = [""] * len(rows)
    for i, r in enumerate(rows):
        if r["qtype"] not in allowed:
            continue
        best = None
        for kd in (labelled_kinds(r, kinds) if labelled_only else kinds):
            c = r[kd]
            if c is None:
                continue
            pr = gate.prob(kd, c["feats"])
            if best is None or pr > best[0]:
                best = (pr, kd)
        if best is not None:
            p[i], kind[i] = best
            y[i] = r[best[1]]["y"]
    return p, y, kind


def fit_gate(rows: Sequence[dict], target: float = _L.gate_precision, l2: float = 1e-3,
             allowed=("question", "lookup")):
    """Fit both kinds' logistic models on dev rows (each dev set weighted equally) and choose
    the threshold: the largest dev coverage at which every dev set separately has precision
    >= ``target`` (``gate.choose_threshold_sets``)."""
    from bes.library import gate as G

    coef = {}
    for kd in G.KINDS:
        sub = [r for r in rows if r[kd] is not None and r["qtype"] in allowed and kd in labelled_kinds(r)]
        if not sub:
            coef[kd] = np.zeros(len(G.FEATURES[kd]) + 1)
            coef[kd][-1] = -20.0
            continue
        x = np.stack([G.VECTOR[kd](r[kd]["feats"]) for r in sub])
        y = np.array([r[kd]["y"] for r in sub], dtype=np.float64)
        coef[kd] = G.fit_logistic(x, y, _set_weights(sub), l2=l2)
    model = G.GateModel.stored(coef, 1.0)
    p, y, _ = decisions(rows, model, allowed, labelled_only=True)
    t, point = G.choose_threshold_sets(p, y, [r["set"] for r in rows], target)
    model = G.GateModel.stored(coef, t if np.isfinite(t) else 1.0 + 1e-6)
    return model, {"threshold": float(model.threshold), "dev_point": point}


# ------------------------------------------------------------------------------ v2


@dataclass(frozen=True)
class BuildCfg:
    budget_numbers: int = _L.budget_numbers       # wiki text + directory + titles at most
    index: IndexCfg = dataclasses.field(default_factory=IndexCfg)
    qa: bool = True
    permissive_only: bool = _L.qa_permissive_only
    dev_per_source: int = 3000
    dev_probes: int = 2000
    seed: int = 0
    total_cap: int = _L.total_numbers_cap


def build_v2(cache_dir: str, out_dir: str | None, cfg: BuildCfg = BuildCfg(),
             count_tokens: CountTokens = default_count_tokens, log: Log | None = None,
             rebuild_pool: bool = False, rebuild_signals: bool = False) -> tuple[Library, dict]:
    """The complete build: pool, selection, index, title table, qa layer, gate fit, save."""
    from bes.library import qa as Q

    t0 = time.time()
    report: dict = {}
    pool = prepare_pool(cache_dir, rebuild=rebuild_pool, count_tokens=count_tokens, log=log)
    pack_cfg = PackCfg()
    sig = prepare_signals(pool, cache_dir, pack_cfg, rebuild=rebuild_signals or rebuild_pool, log=log)
    # wiki selection and index first: the index build is the memory peak, so the qa data
    # is loaded only afterwards (the output does not depend on the order)
    lib, sel = build_for_budget(pool, sig, cfg.budget_numbers, cfg.index, pack_cfg, log=log,
                                cache_dir=cache_dir, count_tokens=count_tokens)
    sel.passages = []                     # not needed any more (packed into lib)
    imp = title_importance(pool, sig, sel)
    store = None
    pairs, raw = [], {}
    if cfg.qa:
        pairs, stats, _, raw = load_qa_pairs(cache_dir, permissive_only=cfg.permissive_only, log=log)
        pairs, n_long = drop_long_renders(pairs, _L.max_passage_tokens, count_tokens)
        report["qa_clean"] = {"stats": stats, "long_render": n_long, "kept": len(pairs)}
        store = Q.QAStore.build(pairs, Q.QACfg(), meta={"permissive_only": cfg.permissive_only})
        lib.qa = store
        _say(log, f"qa: store {store.accounting()['qa_numbers']} numbers ({time.time() - t0:.0f}s)")
    dq = dev_queries(lib, pairs, raw, cfg.dev_per_source, cfg.dev_probes, cfg.seed, imp)
    rows = candidate_rows(lib, dq)
    gate, info = fit_gate(rows)
    lib.gate = gate
    report["gate"] = info
    report["dev_sets"] = {s: sum(1 for x in dq if x["set"] == s) for s in sorted({x["set"] for x in dq})}
    _say(log, f"gate: threshold {gate.threshold:.4f} dev {info['dev_point']} ({time.time() - t0:.0f}s)")
    acc = lib.accounting()
    if acc["total_numbers"] > cfg.total_cap:
        raise ValueError(f"library stores {acc['total_numbers']} numbers > cap {cfg.total_cap}")
    lib.meta["build"] = {"budget_numbers": cfg.budget_numbers, "index_cfg": dataclasses.asdict(cfg.index),
                         "qa": cfg.qa, "permissive_only": cfg.permissive_only, "seed": cfg.seed,
                         "dev_sets": report["dev_sets"], "gate_dev_point": info["dev_point"]}
    if out_dir is not None:
        lib.save(out_dir)
        write_attribution(pool, sel, out_dir, cache_dir)
        prov = os.path.join(out_dir, "provenance")         # not shipped, not counted
        os.makedirs(prov, exist_ok=True)
        np.save(os.path.join(prov, "pool_idx.npy"), sel.pool_idx)
        np.save(os.path.join(prov, "title_importance.npy"), imp)
        write_attribution_md(out_dir, sorted({p.source for p in pairs}))
        if store is not None:
            write_qa_attribution(store, pairs, out_dir)
            _atomic_write(os.path.join(out_dir, "qa_clean.json"),
                          json.dumps(report["qa_clean"], indent=1, sort_keys=True).encode())
    report["accounting"] = acc
    _say(log, f"build_v2: {acc['total_numbers']} numbers ({time.time() - t0:.0f}s)")
    return lib, report


def write_attribution_md(out_dir: str, qa_sources: Sequence[str] = ()) -> str:
    """``ATTRIBUTION.md``: the sources of the built library with their licences (metadata,
    not read by the model, not counted)."""
    w = SOURCES["simplewiki"]
    lines = ["# Attribution", "",
             "This directory holds a library built from the sources below. Texts are reproduced in "
             "cleaned form (markup, references and template debris removed; split into passages).", "",
             f"- Simple English Wikipedia: Hugging Face `{w.repo}`, file `{w.filename}`, revision "
             f"`{w.revision}`; licence: {w.license}. `ATTRIBUTION.tsv` lists every stored article "
             "(title, Wikipedia page id, url) and its passage ids."]
    for n in qa_sources:
        s = QA_SOURCES[n]
        lines.append(f"- {n}: Hugging Face `{s.repo}` (config `{s.config}`, split `{s.split}`), revision "
                     f"`{s.revision}`; licence: {s.license}.")
    if qa_sources:
        lines.append("")
        lines.append("`ATTRIBUTION_QA.tsv` lists, for every stored question-answer entry, its source and "
                     "row. Sources whose card states no licence are left out by a `--permissive-only` build.")
    path = os.path.join(out_dir, "ATTRIBUTION.md")
    _atomic_write(path, ("\n".join(lines) + "\n").encode())
    return path


def write_qa_attribution(store, pairs, out_dir: str) -> str:
    """``ATTRIBUTION_QA.tsv``: the qa sources with licences, then one row per stored entry
    (entry id, passage id offset, source, row in the source file). Metadata only."""
    from bes.library.qa import entry_order

    order = entry_order(pairs, store.cfg.order)
    used = sorted({p.source for p in pairs})
    rows = ["# question-answer layer: sources (Hugging Face dataset, config, split, revision, licence)"]
    for n in used:
        s = QA_SOURCES[n]
        rows.append(f"# {n}\t{s.repo}\t{s.config}\t{s.split}\t{s.revision}\t{s.license}")
    rows.append("entry\tsource\trow")
    for e, i in enumerate(order):
        p = pairs[int(i)]
        rows.append(f"{e}\t{p.source}\t{p.row}")
    out = os.path.join(out_dir, "ATTRIBUTION_QA.tsv")
    with open(out + ".tmp", "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(out + ".tmp", out)
    return out


def dir_digest(path: str) -> dict[str, str]:
    """SHA-256 of every file under ``path`` (relative name -> hex), for determinism checks."""
    out = {}
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, path)] = hashlib.sha256(f.read()).hexdigest()
    return dict(sorted(out.items()))
