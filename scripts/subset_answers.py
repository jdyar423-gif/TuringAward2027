"""A subset of an answers library that fits a number budget (``docs/PLAN_M6V.md``, proposition 5).

    BES_PROFILE=m6 python scripts/subset_answers.py --src data_cache/library/answers_v1 \\
        --out data_cache/library/answers_s --budget-numbers 25200000 --read-cap 360000

The shipped model shares its dormant numbers between the page rows of rare tokens and written
entries; the library that ships is the part of the full one (the one the run trained with) that
is worth most per number. Which entries stay:

* every entry is used once as a question with itself left out, the way training and the
  development measurements use the library; the ``--top`` best candidates of the index
  (``AnswerLibrary.partial``) are credited with the square of their partial score. An entry's
  credit is how much of the questions around it it serves: an entry no question comes near
  serves nobody but an exact repeat of itself;
* entries are taken in the order of their credit (ties: the shorter entry first) until the
  built library holds ``--budget-numbers`` stored numbers; they keep the order they have in
  the source (similar questions share a block).

The result is an answers library like the source (same format, sources and attribution, the
entry counts of the subset), with ``--read-cap`` as the most one reply may read, and a record
of the selection in ``answers.json`` (``build.subset``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bes.library.answers import ANSWERS_JSON, AnswerCfg, AnswerLibrary  # noqa: E402


def _credit_range(args: tuple[str, int, int, int]) -> np.ndarray:
    """Credit given by the entries ``lo .. hi`` of the library at ``path`` (module docstring)."""
    path, lo, hi, top = args
    lib = AnswerLibrary.load(path)
    out = np.zeros(lib.n_entries, np.float64)
    for e in range(lo, hi):
        ents, part = lib.partial(lib.question(e), exclude=[e])
        for x, s in zip(ents[:top].tolist(), part[:top].tolist()):
            out[x] += s * s
    return out


def credit(path: str, workers: int = 4, top: int = 3, chunk: int = 4000, log=print) -> np.ndarray:
    """``float64[n]``: the credit of every entry of the library at ``path``."""
    n = AnswerLibrary.load(path).n_entries
    tasks = [(path, lo, min(n, lo + chunk), top) for lo in range(0, n, chunk)]
    out = np.zeros(n, np.float64)
    t0 = time.time()
    if workers <= 1:
        for i, t in enumerate(tasks):
            out += _credit_range(t)
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            for i, part in enumerate(pool.map(_credit_range, tasks)):
                out += part
                if i % 10 == 9:
                    log(f"credit: {min(n, (i + 1) * chunk):,} of {n:,} entries ({time.time() - t0:.0f} s)")
    return out


def subset(src: str, budget_numbers: int, *, read_cap: int | None = None, cr: np.ndarray | None = None,
           workers: int = 4, top: int = 3, log=print) -> tuple[AnswerLibrary, dict]:
    """The subset library of ``src`` within ``budget_numbers`` and the record of the selection."""
    lib = AnswerLibrary.load(src)
    n = lib.n_entries
    if cr is None:
        cr = credit(src, workers, top, log=log)
    if cr.shape != (n,):
        raise ValueError(f"credit of {cr.shape[0]} entries for a library of {n}")
    sources = list(lib.meta.get("sources") or [])
    pairs = []
    for e in range(n):
        q, a = lib.entry(e)
        pairs.append((q, a, sources[int(lib.src[e])] if sources else ""))
    size = np.fromiter((len(q.encode()) + len(a.encode()) for q, a, _ in pairs), dtype=np.int64, count=n)
    order = np.lexsort((np.arange(n), size, -cr))               # credit first, then the shorter entry
    acc0 = lib.accounting()
    cfg = lib.acfg if read_cap is None else AnswerCfg(**{**_asdict(lib.acfg), "read_cap": int(read_cap)})
    fixed = int(acc0["ans_array_counts"]) + int(acc0["ans_array_coarse"])
    per = (int(acc0["total_numbers"]) - fixed) / max(1, n)
    keep = max(1, min(n, int((budget_numbers - fixed) / per)))
    built, tries = None, []
    for _ in range(8):
        sel = np.sort(order[:keep])
        cand = AnswerLibrary.build([pairs[i] for i in sel.tolist()], cfg)
        total = int(cand.accounting()["total_numbers"])
        tries.append((keep, total))
        log(f"subset: {keep:,} entries -> {total:,} numbers (budget {budget_numbers:,})")
        if total <= budget_numbers:
            built = (cand, sel, total)
            more = int(keep * min(1.02, budget_numbers / total))
            if more <= keep or keep == n or budget_numbers - total < 0.003 * budget_numbers:
                break
            keep = min(n, more)
        else:
            keep = max(1, int(keep * budget_numbers / total * 0.997))
            if built is not None and keep <= len(built[1]):
                break
    if built is None:
        raise ValueError(f"no subset of {src} fits {budget_numbers} numbers")
    cand, sel, total = built
    parent = json.loads((Path(src) / ANSWERS_JSON).read_text())
    build = dict(parent.get("build") or {})
    per_src = np.bincount(np.asarray(lib.src)[sel], minlength=len(sources)) if sources else []
    srcs = {}
    for name, s in (build.get("sources") or {}).items():
        s = dict(s)
        if name in sources:
            s["stored_parent"], s["stored"] = s.get("stored"), int(per_src[sources.index(name)])
        srcs[name] = s
    build["sources"] = srcs
    # nothing of the machine or the moment goes into the record: the same source, credit and budget
    # give the same files anywhere (the library's digest is part of a generator's cache key)
    rec = {"parent": Path(src).name, "parent_entries": n, "parent_numbers": int(acc0["total_numbers"]),
           "parent_answers_json_sha256": hashlib.sha256((Path(src) / ANSWERS_JSON).read_bytes()).hexdigest(),
           "kept": int(sel.size), "numbers": total, "budget_numbers": int(budget_numbers),
           "read_cap": int(cfg.read_cap), "rule": f"leave-one-out credit of the {top} best index candidates "
           "(squared partial score), highest first; ties: the shorter entry",
           "credit": {"entries_with_credit": int((cr > 0).sum()), "kept_with_credit": int((cr[sel] > 0).sum()),
                      "credit_kept_share": float(cr[sel].sum() / max(cr.sum(), 1e-12)),
                      "lowest_credit_kept": float(cr[sel].min())},
           "tries": tries}
    build["subset"] = rec
    cand.meta.update({k: v for k, v in parent.items()
                      if k not in ("format", "cfg", "pack_cfg", "accounting", "sources", "build")})
    cand.meta["build"] = build
    return cand, rec


def _asdict(cfg: AnswerCfg) -> dict:
    import dataclasses

    return dataclasses.asdict(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-numbers", type=int, required=True)
    ap.add_argument("--read-cap", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--credit", default=None, help="a .npy file: read the credit from it, or write it there")
    a = ap.parse_args(argv)
    cr = None
    if a.credit and Path(a.credit).is_file():
        cr = np.load(a.credit)
    elif a.credit:
        cr = credit(a.src, a.workers, a.top)
        np.save(a.credit, cr)
    lib, rec = subset(a.src, a.budget_numbers, read_cap=a.read_cap, cr=cr, workers=a.workers, top=a.top)
    out = Path(a.out)
    lib.save(str(out))
    from build_answers import attribution

    stats = {"sources": {k: v for k, v in (lib.meta.get("build") or {}).get("sources", {}).items()
                         if v.get("stored")}}
    (out / "ATTRIBUTION.md").write_text(
        attribution(stats) + f"\nThis library is a subset of a larger build ({rec['parent_entries']:,} entries): "
        f"the {rec['kept']:,} entries kept were chosen by the rule recorded in `answers.json` (`build.subset`).\n")
    print(json.dumps({k: rec[k] for k in ("kept", "numbers", "budget_numbers", "read_cap", "credit")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
