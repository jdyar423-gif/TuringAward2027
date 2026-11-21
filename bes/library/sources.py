"""Library sources and the cleaned candidate pool.

``SOURCES`` lists the datasets the library is built from, each verified on the Hugging Face
Hub (repository, revision and file). ``build_pool`` streams one source through
``bes.library.clean`` row group by row group and keeps, for every passage, the body plus
the small metadata the selector needs; ``Pool`` stores it compactly on disk (one UTF-8
blob + offset arrays) so that libraries for several budgets can be built without cleaning
again.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bes.config import LibraryCfg
from bes.library.clean import CleanCfg, CountTokens, clean_and_split, default_count_tokens, iter_articles


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    revision: str          # exact commit, verified on the Hub
    filename: str
    license: str
    columns: tuple[str, ...] = ("id", "title", "text")


SOURCES: dict[str, Source] = {
    "simplewiki": Source(
        name="simplewiki", repo="wikimedia/wikipedia",
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        filename="20231101.simple/train-00000-of-00001.parquet",
        license="CC BY-SA 4.0 (text), GFDL",
    ),
}


def download(source: Source, cache_dir: str) -> str:
    """Download one source file into its own directory under ``cache_dir`` (idempotent)."""
    from huggingface_hub import hf_hub_download

    out = os.path.join(cache_dir, "raw", source.name)
    path = os.path.join(out, source.filename)
    if os.path.exists(path):
        return path
    os.makedirs(out, exist_ok=True)
    return hf_hub_download(source.repo, source.filename, repo_type="dataset",
                           revision=source.revision, local_dir=out)


# ------------------------------------------------------------------------------ pool


@dataclass
class Pool:
    """Cleaned candidate passages of one or more sources (all arrays length P or A)."""

    titles: list[str]          # [A] article titles
    category: list[str]        # [A] first category name ("" when none)
    article_words: np.ndarray  # int32 [A] words kept after cleaning
    article_src: np.ndarray    # int32 [A] row index in the source file
    p_article: np.ndarray      # int32 [P] article of each passage
    p_pos: np.ndarray          # int16 [P] position of the passage in its article
    p_words: np.ndarray        # int32 [P]
    p_list: np.ndarray         # float32 [P] fraction of body words in list items
    body_blob: bytes           # concatenated UTF-8 bodies
    body_off: np.ndarray       # int64 [P + 1]
    drops: dict                # drop reason -> count
    meta: dict

    @property
    def n_passages(self) -> int:
        return int(self.p_article.shape[0])

    @property
    def n_articles(self) -> int:
        return len(self.titles)

    def body(self, i: int) -> str:
        return self.body_blob[self.body_off[i]:self.body_off[i + 1]].decode("utf-8")

    def body_bytes(self) -> np.ndarray:
        return np.diff(self.body_off).astype(np.int64)

    def save(self, out_dir: str) -> None:
        from bes.library.pack import _atomic_save_npy, _atomic_write

        os.makedirs(out_dir, exist_ok=True)
        for k in ("article_words", "article_src", "p_article", "p_pos", "p_words", "p_list",
                  "body_off"):
            _atomic_save_npy(os.path.join(out_dir, f"{k}.npy"), getattr(self, k))
        _atomic_write(os.path.join(out_dir, "bodies.bin"), self.body_blob)
        _atomic_write(os.path.join(out_dir, "articles.json"),
                      json.dumps({"titles": self.titles, "category": self.category}).encode())
        _atomic_write(os.path.join(out_dir, "pool.json"),
                      json.dumps({"drops": self.drops, "meta": self.meta}, indent=1).encode())

    @classmethod
    def load(cls, out_dir: str) -> "Pool":
        arr = {k: np.load(os.path.join(out_dir, f"{k}.npy"))
               for k in ("article_words", "article_src", "p_article", "p_pos", "p_words", "p_list",
                         "body_off")}
        art = json.loads(open(os.path.join(out_dir, "articles.json")).read())
        meta = json.loads(open(os.path.join(out_dir, "pool.json")).read())
        with open(os.path.join(out_dir, "bodies.bin"), "rb") as f:
            blob = f.read()
        return cls(titles=art["titles"], category=art["category"], body_blob=blob,
                   drops=meta["drops"], meta=meta["meta"], **arr)


def _list_fraction(body: str) -> float:
    words = 0
    lw = 0
    for line in body.split("\n"):
        w = len(line.split())
        words += w
        if line.startswith("- "):
            lw += w
    return lw / words if words else 0.0


def build_pool(parquet_path: str, clean_cfg: CleanCfg = CleanCfg(),
               max_words: int = LibraryCfg().max_passage_words,
               max_tokens: float = LibraryCfg().max_passage_tokens,
               count_tokens: CountTokens = default_count_tokens,
               row_groups=None, progress: Callable[[int], None] | None = None,
               meta: dict | None = None) -> Pool:
    """Stream the parquet file, clean every article, split into passages (bounded memory:
    one row group of raw text at a time plus the kept passages)."""
    titles: list[str] = []
    cats: list[str] = []
    aw: list[int] = []
    asrc: list[int] = []
    pa: list[int] = []
    pp: list[int] = []
    pw: list[int] = []
    pl: list[float] = []
    chunks: list[bytes] = []
    offs: list[int] = [0]
    drops: dict[str, int] = {}
    stats = {"articles_in": 0, "raw_chars": 0}
    for k, row in enumerate(iter_articles(parquet_path, row_groups=row_groups)):
        stats["articles_in"] += 1
        stats["raw_chars"] += len(row["text"])
        art, ps = clean_and_split(row["title"], row["text"], clean_cfg, max_words, max_tokens,
                                  count_tokens)
        for key in ("tail_cut_lines", "dropped_items", "dropped_lines", "broken_sentences",
                    "lists_dropped"):
            stats[key] = stats.get(key, 0) + int(art.stats.get(key, 0))
        stats["articles_with_tail_cut"] = stats.get("articles_with_tail_cut", 0) + ("tail_cut_lines" in art.stats)
        stats["articles_with_categories"] = stats.get("articles_with_categories", 0) + bool(art.categories)
        reason = art.drop_reason or ("no_passage" if not ps else None)
        if reason:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        a = len(titles)
        titles.append(art.title)
        cats.append(art.categories[0] if art.categories else "")
        aw.append(sum(p.n_words for p in ps))
        asrc.append(k)
        for p in ps:
            b = p.body.encode("utf-8")
            chunks.append(b)
            offs.append(offs[-1] + len(b))
            pa.append(a)
            pp.append(p.pos)
            pw.append(p.n_words)
            pl.append(_list_fraction(p.body))
        if progress is not None and k % 20000 == 0:
            progress(k)
    m = dict(meta or {})
    m.update(stats, max_words=max_words, max_tokens=max_tokens)
    return Pool(
        titles=titles, category=cats, article_words=np.asarray(aw, np.int32),
        article_src=np.asarray(asrc, np.int32), p_article=np.asarray(pa, np.int32),
        p_pos=np.asarray(pp, np.int16), p_words=np.asarray(pw, np.int32),
        p_list=np.asarray(pl, np.float32), body_blob=b"".join(chunks),
        body_off=np.asarray(offs, np.int64), drops=drops, meta=m,
    )
