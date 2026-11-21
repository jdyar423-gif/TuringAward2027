"""Library build pipeline: source -> cleaned pool -> signals -> selection -> pack -> index.

Intermediate results are cached under ``cache_dir`` (``pool/`` and ``pool/signals.npz``) so
that libraries for several budgets can be built from one cleaning pass.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bes.config import LibraryCfg
from bes.library.clean import CleanCfg, CountTokens, default_count_tokens, iter_articles
from bes.library.index import IndexCfg, TruncatedIndex
from bes.library.pack import PackCfg
from bes.library.retrieve import Library, RetrieveCfg
from bes.library.select import (SelectCfg, Selection, article_order, benefits, duplicate_mask,
                                estimate_costs, importance, mention_counts, passage_order,
                                passage_prior, select_for_budget)
from bes.library.sources import SOURCES, Pool, build_pool, download

Log = Callable[[str], None]


def _say(log: Log | None, msg: str) -> None:
    if log is not None:
        log(msg)


def prepare_pool(cache_dir: str, source: str = "simplewiki", rebuild: bool = False,
                 clean_cfg: CleanCfg = CleanCfg(), max_words: int = LibraryCfg().max_passage_words,
                 max_tokens: float = LibraryCfg().max_passage_tokens,
                 count_tokens: CountTokens = default_count_tokens, log: Log | None = None) -> Pool:
    """Load the cached pool or build it from the (downloaded) source."""
    pool_dir = os.path.join(cache_dir, "pool")
    if not rebuild and os.path.exists(os.path.join(pool_dir, "pool.json")):
        _say(log, f"pool: loading {pool_dir}")
        return Pool.load(pool_dir)
    src = SOURCES[source]
    path = download(src, cache_dir)
    t0 = time.time()
    name = "utf8_bytes/3.3" if count_tokens is default_count_tokens else "bpe"
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
                    rebuild: bool = False, log: Log | None = None) -> Signals:
    path = os.path.join(cache_dir, "pool", "signals.npz")
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
    licence (CC BY-SA 4.0 for Wikipedia text); not read by the model, not counted."""
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


def build_for_budget(pool: Pool, sig: Signals, budget_numbers: int,
                     index_cfg: IndexCfg = IndexCfg(), pack_cfg: PackCfg = PackCfg(),
                     select_cfg: SelectCfg = SelectCfg(), retrieve_cfg: RetrieveCfg = RetrieveCfg(),
                     count_tokens: CountTokens = default_count_tokens, out_dir: str | None = None,
                     log: Log | None = None, cache_dir: str | None = None,
                     count_tokens_name: str = "utf8_bytes/3.3") -> tuple[Library, Selection]:
    """Select under ``budget_numbers`` (text + directory + titles), pack, index, save."""
    t0 = time.time()
    imp = importance(pool, sig.mentions, select_cfg)
    ben = benefits(pool, imp, sig.dup, select_cfg)
    sel = select_for_budget(pool, budget_numbers, ben, sig.cost, sig.order, pack_cfg, select_cfg, log)
    _say(log, f"build: selected {int(sel.mask.sum())} passages ({time.time() - t0:.0f}s)")
    prior = passage_prior(pool, imp, select_cfg)[sel.pool_idx]
    index = TruncatedIndex.build(sel.passages, index_cfg, prior)
    meta = {
        "source": pool.meta.get("source"), "budget_numbers": int(budget_numbers),
        "select_cfg": dataclasses.asdict(select_cfg), "selection_history": sel.history,
        "pool_passages": pool.n_passages, "pool_articles": pool.n_articles,
        "selected_articles": int(len(set(pool.p_article[sel.pool_idx].tolist()))),
        "max_tokens": pool.meta.get("max_tokens"), "max_words": pool.meta.get("max_words"),
        "count_tokens": count_tokens_name,
    }
    lib = Library(sel.packed, index, retrieve_cfg, count_tokens, meta)
    if out_dir is not None:
        lib.save(out_dir)
        np.save(os.path.join(out_dir, "pool_idx.npy"), sel.pool_idx)  # provenance (not shipped)
        write_attribution(pool, sel, out_dir, cache_dir)
    _say(log, f"build: done ({time.time() - t0:.0f}s) {lib.accounting()['total_numbers']} numbers")
    return lib, sel
