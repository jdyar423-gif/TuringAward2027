"""Library sources and the cleaned candidate pool.

``SOURCES`` lists the text datasets the library is built from and ``QA_SOURCES`` the
question-answer splits (stored ones and evaluation ones whose questions are excluded), each
verified on the Hugging Face Hub on 2026-11-21 (repository, config, split, pinned revision,
file with its LFS sha256, row count, licence as stated on the dataset card). ``build_pool``
streams one source through
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
        license="cc-by-sa-3.0 and gfdl (dataset card; Wikipedia text is CC BY-SA, attribution "
                "per article in ATTRIBUTION.tsv)",
    ),
}


@dataclass(frozen=True)
class QASource:
    """One split of a question-answer dataset on the Hugging Face Hub (verified 2026-11-21:
    repository, config, split, file, LFS sha256 at the pinned revision, card licence).

    ``role`` is ``"store"`` (questions written into the qa layer) or ``"eval"`` (evaluation
    questions: never stored; their normalised questions are excluded from the store).
    ``permissive`` is False when the card gives no clear licence; a ``permissive_only`` build
    leaves such store sources out. ``answer`` names the answer field: ``"list"`` (a list of
    strings, first = canonical), ``"trivia"`` (TriviaQA struct: value + aliases), ``"popqa"``
    (JSON list in ``possible_answers``) or ``"none"``.
    """

    name: str
    repo: str
    config: str
    split: str
    revision: str
    filename: str
    sha256: str
    license: str
    permissive: bool
    role: str
    dir: str                 # directory under raw/ (shared with the E1 measurement script)
    question: str = "question"
    answer: str = "list"
    answer_field: str = "answer"
    fmt: str = "parquet"     # "parquet" | "tsv"
    rows: int = 0            # rows in the file (verified)


_NQ_REV = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
_TQA_REV = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
_WQ_REV = "0e473cbe21d1e91ec18da343644498be6a3f5454"
_POPQA_REV = "098765c79ea10a2cb19c828324e33281b8336ec0"
_NQ_LIC = "cc-by-sa-3.0 (card: 'All of the Natural Questions data is released under CC BY-SA 3.0')"
_TQA_LIC = ("unknown (card YAML 'unknown'; card text: 'The University of Washington does not own the "
            "copyright of the questions and documents included in TriviaQA')")
_WQ_LIC = "unknown (card YAML 'unknown'; licensing section 'More Information Needed')"

QA_SOURCES: dict[str, QASource] = {s.name: s for s in (
    QASource("nq_open_train", "google-research-datasets/nq_open", "nq_open", "train", _NQ_REV,
             "nq_open/train-00000-of-00001.parquet",
             "25d3a544324f900b31ebc05a3a5686bdd5b9b42738133500675c3f20eee7d83a", _NQ_LIC, True,
             "store", "nq_open", rows=87_925),
    QASource("trivia_qa_train", "mandarjoshi/trivia_qa", "rc.nocontext", "train", _TQA_REV,
             "rc.nocontext/train-00000-of-00001.parquet",
             "455223e4a3ba6977611914d96b1b812fe19a6cd4ea3aef4df381f093b297670c", _TQA_LIC, False,
             "store", "trivia_qa", answer="trivia", rows=138_384),
    QASource("web_questions_train", "stanfordnlp/web_questions", "default", "train", _WQ_REV,
             "data/train-00000-of-00001.parquet",
             "7b59a32059d001919540ec617afb30d5c06829cbd3159e3da5f915bbd1d973bc", _WQ_LIC, False,
             "store", "web_questions", answer_field="answers", rows=3_778),
    QASource("nq_open_validation", "google-research-datasets/nq_open", "nq_open", "validation",
             _NQ_REV, "nq_open/validation-00000-of-00001.parquet",
             "b074bed0bccb56fa1551a8ac1c9c51ce89bc11c7fbb6a9c713b2c33a98531e12", _NQ_LIC, True,
             "eval", "nq_open", rows=3_610),
    QASource("trivia_qa_validation", "mandarjoshi/trivia_qa", "rc.nocontext", "validation", _TQA_REV,
             "rc.nocontext/validation-00000-of-00001.parquet",
             "48a5005c0eb4f8a4ae5b9868644297fd5bf1e694aeb3dc9c8ab958cac0d5b201", _TQA_LIC, False,
             "eval", "trivia_qa", answer="trivia", rows=17_944),
    QASource("trivia_qa_test", "mandarjoshi/trivia_qa", "rc.nocontext", "test", _TQA_REV,
             "rc.nocontext/test-00000-of-00001.parquet",
             "35718c1b37e786c823a5b1e810bf97cd085a1536396143a153b4cb30ba41e4c5", _TQA_LIC, False,
             "eval", "trivia_qa", answer="none", rows=17_210),
    QASource("web_questions_test", "stanfordnlp/web_questions", "default", "test", _WQ_REV,
             "data/test-00000-of-00001.parquet",
             "3c81463f22162f645bdbdc0d501cfee9ad7d31bee819b3226f734523741c9faa", _WQ_LIC, False,
             "eval", "web_questions", answer_field="answers", rows=2_032),
    # not an LFS file: sha256 of the downloaded file (git blob 0ad1a9bbe0c198f511131c011d71dacefee41001)
    QASource("popqa_test", "akariasai/PopQA", "default", "test", _POPQA_REV, "test.tsv",
             "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
             "none on the card (YAML has no licence field)", False, "eval", "popqa",
             answer="popqa", answer_field="possible_answers", fmt="tsv", rows=14_267),
)}

#: Store sources used by default (the permissive-only build keeps those with ``permissive``).
DEFAULT_QA_STORE = ("nq_open_train", "trivia_qa_train", "web_questions_train")


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


def download_qa(source: QASource, cache_dir: str) -> str:
    """Download one question-answer file into ``cache_dir/raw/<dir>/`` (idempotent)."""
    from huggingface_hub import hf_hub_download

    out = os.path.join(cache_dir, "raw", source.dir)
    path = os.path.join(out, source.filename)
    if os.path.exists(path):
        return path
    os.makedirs(out, exist_ok=True)
    return hf_hub_download(source.repo, source.filename, repo_type="dataset",
                           revision=source.revision, local_dir=out)


def iter_qa(source: QASource, path: str):
    """Yield ``(row index, question, answers)`` of one file; ``answers[0]`` is the canonical
    answer (NQ: the first annotation, TriviaQA: ``answer.value``), the rest are aliases."""
    if source.fmt == "tsv":
        import csv

        csv.field_size_limit(1 << 30)
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                ans = json.loads(row[source.answer_field]) if source.answer == "popqa" else []
                yield i, row[source.question], [str(a) for a in ans]
        return
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = [source.question] + ([] if source.answer == "none" and source.answer_field not in
                                pf.schema_arrow.names else [source.answer_field])
    i = 0
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols)
        qs = t.column(source.question).to_pylist()
        an = t.column(source.answer_field).to_pylist() if source.answer_field in cols else [None] * len(qs)
        for q, a in zip(qs, an):
            if source.answer == "trivia":
                ans = [] if not a or a.get("value") in (None, "", "<unk>") else (
                    [a["value"]] + list(a.get("aliases") or []) + list(a.get("normalized_aliases") or []))
            elif source.answer == "none" or a is None:
                ans = []
            else:
                ans = [str(x) for x in a]
            yield i, q or "", ans
            i += 1
        del t


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
                    "lists_dropped", "hole_sentences", "hole_repairs"):
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
