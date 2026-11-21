"""Text samples used to train and evaluate the tokenizer.

The sample is a deterministic function of pinned dataset revisions: documents are read in row
order from fixed parquet files, so two runs see the same text. Three domains mirror what the
model reads:

- ``web``  : ``karpathy/climbmix-400b-shuffle`` (column ``text``), the pre-training backbone;
- ``chat`` : ``HuggingFaceTB/smol-smoltalk`` (column ``messages``), every message content is
  one text, exactly as ``bes.data.chat_format`` encodes them;
- ``wiki`` : ``wikimedia/wikipedia`` config ``20231101.simple`` (``title`` + ``text``), the
  source of the library passages.

Held-out text never overlaps the training text: ``web`` uses a different shard, ``chat`` the
test split, and ``wiki`` every row whose index is a multiple of ``WIKI_HELDOUT_EVERY``.

Files are downloaded with ``huggingface_hub.hf_hub_download`` into ``data_cache/samples/``
(git-ignored) and read with pyarrow in record batches, so memory stays at a few batches.
Nothing in this module is needed to *load* the tokenizer.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data_cache" / "samples"

CLIMBMIX = ("karpathy/climbmix-400b-shuffle", "915333b4f8b8684f39aeaafea600fea6f43fb703")
SMOLTALK = ("HuggingFaceTB/smol-smoltalk", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc")
WIKIPEDIA = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")

#: Rows of the Simple English Wikipedia file with ``index % WIKI_HELDOUT_EVERY == 0`` are held out.
WIKI_HELDOUT_EVERY = 50

DOMAINS = ("web", "chat", "wiki")


@dataclass(frozen=True)
class SampleFile:
    """One parquet file of a Hugging Face dataset at a pinned revision."""

    repo_id: str
    revision: str
    filename: str

    @property
    def local_path(self) -> Path:
        return SAMPLE_DIR / self.repo_id.replace("/", "__") / self.filename


WEB_TRAIN = (
    SampleFile(*CLIMBMIX, "shard_00000.parquet"),
    SampleFile(*CLIMBMIX, "shard_00001.parquet"),
)
WEB_HELDOUT = (SampleFile(*CLIMBMIX, "shard_00997.parquet"),)
CHAT_TRAIN = (SampleFile(*SMOLTALK, "data/train-00000-of-00004.parquet"),)
CHAT_HELDOUT = (SampleFile(*SMOLTALK, "data/test-00000-of-00001.parquet"),)
WIKI_FILE = SampleFile(*WIKIPEDIA, "20231101.simple/train-00000-of-00001.parquet")

FILES: dict[tuple[str, str], tuple[SampleFile, ...]] = {
    ("web", "train"): WEB_TRAIN,
    ("web", "heldout"): WEB_HELDOUT,
    ("chat", "train"): CHAT_TRAIN,
    ("chat", "heldout"): CHAT_HELDOUT,
    ("wiki", "train"): (WIKI_FILE,),
    ("wiki", "heldout"): (WIKI_FILE,),
}


def files_for(domain: str, split: str) -> tuple[SampleFile, ...]:
    if (domain, split) not in FILES:
        raise KeyError(f"unknown sample {domain!r}/{split!r}")
    return FILES[(domain, split)]


def download(files: tuple[SampleFile, ...] | list[SampleFile]) -> list[Path]:
    """Download ``files`` (skipping those already present) and return their local paths."""
    from huggingface_hub import hf_hub_download

    paths = []
    for f in files:
        if not f.local_path.exists():
            hf_hub_download(
                repo_id=f.repo_id,
                filename=f.filename,
                repo_type="dataset",
                revision=f.revision,
                local_dir=str(SAMPLE_DIR / f.repo_id.replace("/", "__")),
            )
        if not f.local_path.exists():
            raise FileNotFoundError(f.local_path)
        paths.append(f.local_path)
    return paths


def download_all() -> list[Path]:
    unique: dict[Path, SampleFile] = {}
    for files in FILES.values():
        for f in files:
            unique[f.local_path] = f
    return download(list(unique.values()))


def _iter_rows(path: Path, columns: list[str], batch_size: int = 512) -> Iterator[dict]:
    """Rows of a parquet file as dicts, read in record batches of ``batch_size`` rows."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()
    finally:
        pf.close()


def _domain_texts(domain: str, split: str) -> Iterator[str]:
    """Unbounded stream of the texts of one domain/split, in a fixed order."""
    if domain == "web":
        for f in files_for(domain, split):
            for row in _iter_rows(f.local_path, ["text"]):
                if row["text"]:
                    yield row["text"]
    elif domain == "chat":
        for f in files_for(domain, split):
            for row in _iter_rows(f.local_path, ["messages"]):
                for message in row["messages"] or ():
                    if message.get("content"):
                        yield message["content"]
    elif domain == "wiki":
        (f,) = files_for(domain, split)
        keep_heldout = split == "heldout"
        for i, row in enumerate(_iter_rows(f.local_path, ["title", "text"])):
            if (i % WIKI_HELDOUT_EVERY == 0) != keep_heldout:
                continue
            if row["text"]:
                yield f"{row['title']}\n\n{row['text']}"
    else:
        raise KeyError(f"unknown domain {domain!r}")


def iter_texts(domain: str, split: str, max_bytes: int, doc_cap: int | None = None) -> Iterator[str]:
    """Texts of ``domain``/``split`` until about ``max_bytes`` UTF-8 bytes have been yielded.

    Each text is first cut to ``doc_cap`` characters when ``doc_cap`` is given. The stream stops
    after the text that crosses ``max_bytes``; it ends earlier if the files run out.
    """
    total = 0
    for text in _domain_texts(domain, split):
        if doc_cap is not None and len(text) > doc_cap:
            text = text[:doc_cap]
        yield text
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            return
