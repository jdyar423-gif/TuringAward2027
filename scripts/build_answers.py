"""Build the answers library (``bes/library/answers.py``, ``docs/PLAN_M6.md``).

    python scripts/build_answers.py --out data_cache/library/answers_v1
    python scripts/build_answers.py --out ... --pack release/m6/library     # also write tar parts
    python scripts/build_answers.py --unpack release/m6/library --out data_cache/library/answers_v1

Pairs are the first exchange (first user turn, the assistant turn that answers it) of public
instruction datasets whose cards state a licence that allows redistribution with attribution
(``SOURCES``; the stored text is redistributed with the model). A pair is kept when

* the question stands alone: one line of 10 to ``--max-question`` characters in ASCII, no
  reference to material that is not there ("the following", "this text", ...), no URL, at
  least two index terms;
* the answer is 20 to ``--max-answer`` characters, at most 2% non-ASCII, without a URL or an
  image link and without the stock phrases of a model speaking about itself;
* its normalised question (``bes.library.qa.question_key``) is not a prompt, in any turn, of
  the evaluation set ``bes/eval/proxy/prompts.jsonl`` (never stored) and no better-ranked
  source has the same normalised question (``SOURCES`` order: one answer per question).

Entries are stored in the order of their normalised questions (similar questions share a
block: better compression, fewer blocks per query). When the pairs do not fit
``--budget-numbers`` the longest questions go first. The output directory holds the library
arrays, ``answers.json`` (settings, sources with the commit read, counts per filter,
accounting) and ``ATTRIBUTION.md``.

Every source is read at the commit the Hub resolves when the build starts; the commit is
recorded. ``--rows N`` reads at most N rows per source (tests, rehearsals).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bes.library import qa                                    # noqa: E402
from bes.library.answers import ANSWERS_JSON, AnswerCfg, AnswerLibrary  # noqa: E402
from bes.library.terms import terms                           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_CONVERT = "refs/convert/parquet"


def _first_exchange(messages: Any) -> tuple[str | None, str | None]:
    """First user content and the assistant content right after it."""
    u = a = None
    for m in messages or ():
        r = m.get("role") or m.get("from")
        c = m.get("content") if "content" in m else m.get("value")
        if r in ("user", "human") and u is None:
            u = c
        elif r in ("assistant", "gpt") and u is not None:
            a = c
            break
        elif u is not None:
            break
    return u, a


def _messages(r: dict) -> tuple[str | None, str | None]:
    return _first_exchange(r.get("messages"))


def _conversations(r: dict) -> tuple[str | None, str | None]:
    return _first_exchange(r.get("conversations"))


def _alpaca(r: dict) -> tuple[str | None, str | None]:
    if r.get("input"):
        return None, None                      # the instruction leans on an input text
    return r.get("instruction"), r.get("output")


def _dolly(r: dict) -> tuple[str | None, str | None]:
    if r.get("context"):
        return None, None
    return r.get("instruction"), r.get("response")


#: Tulu 3 subsets left out, as in ``bes.data.sources.TULU_EXCLUDED``.
_TULU_EXCLUDED = frozenset({"ai2-adapt-dev/no_robots_converted", "ai2-adapt-dev/tulu_v3.9_aya_100k",
                            "ai2-adapt-dev/tulu_hard_coded_repeated_10"})


def _tulu(r: dict) -> tuple[str | None, str | None]:
    if r.get("source") in _TULU_EXCLUDED:
        return None, None
    return _first_exchange(r.get("messages"))


def _magpie(r: dict) -> tuple[str | None, str | None]:
    if r.get("language") not in (None, "EN") or r.get("input_quality") not in ("good", "excellent", "average"):
        return None, None
    return r.get("instruction"), r.get("response")


def _gpt4all(r: dict) -> tuple[str | None, str | None]:
    if "stackoverflow" in (r.get("source") or ""):
        return None, None                      # pasted HTML questions
    return r.get("prompt"), r.get("response")


#: name -> (repo, revision or convert ref, glob of parquet files, columns, row -> (q, a), licence of the card)
#: in the order of preference when two sources hold the same question.
SOURCES: dict[str, tuple[str, str, str, tuple[str, ...], Callable[[dict], tuple], str]] = {
    "magpie_llama31": ("Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1", "main", "data/train-*.parquet",
                       ("instruction", "response", "language", "input_quality"), _magpie,
                       "llama3.1 (Llama 3.1 Community License: outputs of Llama 3.1 70B)"),
    "magpie_ultra": ("HuggingFaceTB/smoltalk", "main", "data/smol-magpie-ultra/train-*.parquet",
                     ("messages",), _messages, "apache-2.0"),
    "smol_smoltalk": ("HuggingFaceTB/smol-smoltalk", "main", "data/train-*.parquet",
                      ("messages",), _messages, "apache-2.0"),
    "wizardlm": ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", _CONVERT, "default/train/*.parquet",
                 ("conversations",), _conversations, "mit"),
    "tulu3": ("allenai/tulu-3-sft-mixture", "main", "data/train-*.parquet",
              ("messages", "source"), _tulu, "odc-by-1.0 (subsets carry their own licences; No Robots, Aya and the hard-coded identity prompts are left out)"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "main", "data/train_sft-*.parquet",
                  ("messages",), _messages, "mit"),
    "glaive_code": ("glaiveai/glaive-code-assistant", _CONVERT, "default/train/*.parquet",
                    ("question", "answer"), lambda r: (r.get("question"), r.get("answer")), "apache-2.0"),
    "gpt4all": ("nomic-ai/gpt4all-j-prompt-generations", "main", "data/train-*.parquet",
                ("prompt", "response", "source"), _gpt4all, "apache-2.0"),
    "alpaca": ("yahma/alpaca-cleaned", _CONVERT, "default/train/*.parquet",
               ("instruction", "input", "output"), _alpaca, "cc-by-4.0"),
    "codealpaca": ("sahil2801/CodeAlpaca-20k", _CONVERT, "default/train/*.parquet",
                   ("instruction", "input", "output"), _alpaca, "cc-by-4.0"),
    "dolly": ("databricks/databricks-dolly-15k", _CONVERT, "default/train/*.parquet",
              ("instruction", "context", "response"), _dolly, "cc-by-sa-3.0"),
}

_BAD_A = re.compile(r"https?://|!\[|as an ai\b|as a language model|i'm sorry, but|i cannot|i can't assist|"
                    r"openai|knowledge cutoff|as of my last|\bllama\b|\bmeta ai\b", re.I)
_BAD_Q = re.compile(r"https?://|\bthe following\b|\bbelow\b|\babove\b|\bgiven (?:the|this|a)\b|"
                    r"\bthis (?:text|passage|article|paragraph|sentence|code|function|snippet|email|essay|"
                    r"story|poem|review|list|table|data)\b", re.I)


def usable(q: Any, a: Any, max_q: int, max_a: int) -> str | None:
    """Why the pair is not stored (None: it may be)."""
    if not isinstance(q, str) or not isinstance(a, str):
        return "type"
    q1, a1 = q.strip(), a.strip()
    if "\n" in q1 or not 10 <= len(q1) <= max_q:
        return "q_shape"
    if not q1.isascii():
        return "q_non_ascii"
    if _BAD_Q.search(q1):
        return "q_refers"
    if not 20 <= len(a1) <= max_a:
        return "a_len"
    if _BAD_A.search(a1):
        return "a_bad"
    if sum(ord(c) > 127 for c in a1) > 0.02 * len(a1):
        return "a_non_ascii"
    return None


def eval_keys() -> set[str]:
    from bes.eval.proxy import prompts as P

    keys = set()
    for p in P.load():
        for t in p.turns:
            k = qa.question_key(" ".join(t.split()))
            if k:
                keys.add(k)
    return keys


def iter_rows(name: str, max_rows: int | None, log: Callable[[str], None]) -> Iterator[tuple[dict, str]]:
    """Rows of one source as dicts of its columns, with the commit they were read at."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    repo, rev, pattern, cols, _, _ = SOURCES[name]
    sha = HfApi().dataset_info(repo, revision=rev).sha
    fs = HfFileSystem()
    quoted = rev.replace("/", "%2F")
    files = sorted(fs.glob(f"datasets/{repo}@{quoted}/{pattern}"))
    if not files:
        raise RuntimeError(f"{name}: no file matches {pattern} at {rev}")
    n = 0
    for f in files:
        f = f.replace(f"@{quoted}/", f"@{sha}/") if rev == "main" else f
        with fs.open(f, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=2000, columns=list(cols)):
                for r in batch.to_pylist():
                    yield r, sha
                    n += 1
                    if max_rows is not None and n >= max_rows:
                        return
        log(f"  {name}: {f.rsplit('/', 1)[-1]} read, {n} rows")


def collect(names: list[str], max_q: int, max_a: int, max_rows: int | None,
            log: Callable[[str], None]) -> tuple[list[tuple[str, str, str]], dict]:
    """Kept pairs ``(question, answer, source)`` and the build statistics."""
    excluded = eval_keys()
    best: dict[str, tuple[int, str, str, str]] = {}
    dups: Counter = Counter()
    stats: dict[str, Any] = {"sources": {}, "eval_keys": len(excluded)}
    for rank, name in enumerate(names):
        st: Counter = Counter()
        sha = ""
        fn = SOURCES[name][4]
        for r, sha in iter_rows(name, max_rows, log):
            st["rows"] += 1
            try:
                q, a = fn(r)
            except Exception:
                q = a = None
            if not q or not a:
                st["no_pair"] += 1
                continue
            why = usable(q, a, max_q, max_a)
            if why:
                st[why] += 1
                continue
            q1 = " ".join(q.split())
            key = qa.question_key(q1)
            if not key or len(terms(q1)) < 2:
                st["no_terms"] += 1
                continue
            if key in excluded:
                st["eval_overlap"] += 1
                continue
            dups[key] += 1
            old = best.get(key)
            if old is not None and old[0] <= rank:
                st["duplicate"] += 1
                continue
            best[key] = (rank, q1, a.strip().replace("\r\n", "\n"), name)
            st["kept"] += 1
        repo, rev, pattern, _, _, lic = SOURCES[name]
        stats["sources"][name] = {"repo": repo, "revision": rev, "commit": sha, "files": pattern,
                                  "licence": lic, "filters": dict(st)}
        log(f"{name}: {dict(st)}")
    pairs = [(q, a, src) for _, (_, q, a, src) in sorted(best.items())]
    final = Counter(src for _, _, src in pairs)
    for name in names:
        stats["sources"][name]["stored"] = int(final.get(name, 0))
    stats["asked_more_than_once"] = int(sum(1 for v in dups.values() if v > 1))
    return pairs, stats


def fit_budget(pairs: list[tuple[str, str, str]], budget_numbers: int, cfg: AnswerCfg,
               log: Callable[[str], None]) -> tuple[AnswerLibrary, int]:
    """The library of ``pairs``; when it is over ``budget_numbers`` the pairs with the longest
    questions are dropped (in steps) until it fits. Returns it with the number dropped."""
    dropped = 0
    while True:
        lib = AnswerLibrary.build(pairs, cfg)
        total = lib.accounting()["total_numbers"]
        if total <= budget_numbers or len(pairs) < 2:
            return lib, dropped
        keep = max(1, int(len(pairs) * min(0.97, budget_numbers / total * 0.995)))
        order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), i))[:keep]
        dropped += len(pairs) - keep
        pairs = [pairs[i] for i in sorted(order)]
        log(f"over the budget ({total} > {budget_numbers}): keeping the {keep} shortest questions")


def attribution(stats: dict) -> str:
    lines = ["# Sources of the answers library", "",
             "The library stores question-answer pairs taken verbatim from the public datasets below",
             "(first exchange of a conversation). Each entry records its source (`ans_src`). The texts",
             "remain under the licences of their datasets; this file and `answers.json` are the",
             "attribution. The Dolly entries are CC BY-SA 3.0: the library as a whole is offered under",
             "CC BY-SA 4.0 where that licence can apply, and never under terms more permissive than a",
             "source allows.", "",
             "| source | dataset | commit | licence stated by the card | entries |", "|---|---|---|---|---|"]
    for name, s in stats["sources"].items():
        lines.append(f"| {name} | https://huggingface.co/datasets/{s['repo']} | `{s['commit'][:12]}` | {s['licence']} | {s['stored']:,} |")
    lines += ["", "Answers were generated by third-party models or written by people as the dataset cards",
              "describe; no evaluation question of this repository is stored (see `answers.json`)."]
    return "\n".join(lines) + "\n"


def pack(src: Path, out: Path, part_mb: int = 16) -> dict:
    """Tar ``src`` (no compression) into ``out/library.tar.NNN`` parts with ``parts.json``."""
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("library.tar.*"):
        f.unlink()
    tar_path = out / "library.tar"
    members = 0
    with tarfile.open(tar_path, "w") as tar:
        for p in sorted(src.iterdir()):
            if p.is_file():
                tar.add(p, arcname=f"{src.name}/{p.name}")
                members += 1
    whole, parts, i = hashlib.sha256(), [], 0
    with open(tar_path, "rb") as f:
        while chunk := f.read(part_mb * 2 ** 20):
            whole.update(chunk)
            pn = f"library.tar.{i:03d}"
            (out / pn).write_bytes(chunk)
            parts.append({"name": pn, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
            i += 1
    rec = {"name": "library", "directory": src.name, "bytes": tar_path.stat().st_size, "sha256": whole.hexdigest(),
           "parts": parts, "members": members}
    tar_path.unlink()
    (out / "parts.json").write_text(json.dumps(rec, indent=1))
    return rec


def unpack(parts_dir: Path, out: Path) -> Path:
    """Join and verify the parts in ``parts_dir`` and unpack the library to ``out`` (returned)."""
    rec = json.loads((parts_dir / "parts.json").read_text())
    whole = hashlib.sha256()
    tmp = out.parent / (out.name + ".tar.tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as g:
        for p in rec["parts"]:
            data = (parts_dir / p["name"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != p["sha256"]:
                raise RuntimeError(f"{p['name']}: checksum mismatch")
            whole.update(data)
            g.write(data)
    if whole.hexdigest() != rec["sha256"]:
        raise RuntimeError("the joined archive does not match parts.json")
    stage = out.parent / (out.name + ".unpack")
    with tarfile.open(tmp) as tar:
        tar.extractall(stage, filter="data")
    tmp.unlink()
    got = stage / rec["directory"]
    if out.exists():
        for f in out.iterdir():
            f.unlink()
        out.rmdir()
    got.rename(out)
    stage.rmdir()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="library directory to write (or to unpack into)")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated names, best first")
    ap.add_argument("--max-question", type=int, default=120)
    ap.add_argument("--max-answer", type=int, default=1500)
    ap.add_argument("--budget-numbers", type=int, default=60_000_000)
    ap.add_argument("--rows", type=int, default=None, help="rows read per source at most")
    ap.add_argument("--pack", default=None, help="also write tar parts of the library here")
    ap.add_argument("--unpack", default=None, help="unpack the parts in this directory into --out and stop")
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    out = Path(args.out)
    if args.unpack:
        unpack(Path(args.unpack), out)
        lib = AnswerLibrary.load(str(out))
        log(f"unpacked {lib.n_entries} entries to {out}")
        return 0
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    for n in names:
        if n not in SOURCES:
            raise SystemExit(f"unknown source {n!r}; known: {', '.join(SOURCES)}")
    pairs, stats = collect(names, args.max_question, args.max_answer, args.rows, log)
    log(f"{len(pairs)} pairs before the budget")
    cfg = AnswerCfg()
    lib, dropped = fit_budget(pairs, args.budget_numbers, cfg, log)
    stats.update({"max_question": args.max_question, "max_answer": args.max_answer,
                  "budget_numbers": args.budget_numbers, "dropped_for_budget": dropped,
                  "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows_cap": args.rows})
    lib.meta.update({"build": stats})
    lib.save(str(out))
    (out / "ATTRIBUTION.md").write_text(attribution(stats))
    acc = lib.accounting()
    log(f"library: {acc['entries']} entries, {acc['total_numbers']} stored numbers "
        f"(text {acc['text_numbers']}, per entry {acc['entry_numbers']}, index {acc['index_numbers']}), "
        f"raw {acc['raw_bytes']} bytes")
    if args.pack:
        rec = pack(out, Path(args.pack))
        log(f"packed into {len(rec['parts'])} parts under {args.pack} ({rec['bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
