"""Token rows evicted from an export (``docs/PLAN_M6V.md``, proposition 5).

The stored numbers a token reads only through its own row have two uses that compete for the
same budget: the page and gain rows of that token, and written entries of the library. A token
that is rare enough is worth less in rows than the entries its numbers would hold; its rows are
*evicted*: the three tables of :data:`ROW_TABLES` (the output gains ``gain/o`` and the keys and
values of the token neurons ``page/key``, ``page/val``) keep the rows of the other tokens only,
and one stored map ``page/rows`` (``uint16[V]``) gives every token the row it has, or
:data:`NO_ROW`. The input embedding and the per-loop value embedding stay whole.

An evicted row is a row of zeros: a zero gain row multiplies the attention output by exactly
one, a zero value row adds exactly zero. The model therefore loads as the model of the
configuration with those rows at zero (``bes.export.load_weights`` expands the tables), every
program that runs it is unchanged, and the ledger counts what is stored: the kept rows and the
map (``V`` numbers, of which one is read per token).

    keep = keep_mask(counts, 5120, always=special_ids)
    evict_export("runs/x/final", "runs/x/shipped", keep, library="data_cache/library/answers_s")

:func:`evict_export` writes a new export directory from an existing one (weights compacted, the
library attached, the accounting recomputed, the model card extended, ``SHA256SUMS`` rewritten)
and checks that the result loads to exactly the masked weights of the source.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: tables read one row per input token whose rows of rare tokens are evicted
ROW_TABLES = ("gain/o", "page/key", "page/val")
#: stored map ``uint16[V]``: the row of a token in the compact tables, or NO_ROW
MAP_LEAF = "page/rows"
NO_ROW = 65535


def ranks(counts: Any) -> np.ndarray:
    """``int64[V]``: the frequency rank of every token (0 = most frequent; ties by id)."""
    c = np.asarray(counts)
    order = np.lexsort((np.arange(c.size), -c.astype(np.float64)))
    r = np.empty(c.size, np.int64)
    r[order] = np.arange(c.size)
    return r


def keep_mask(counts: Any, keep: int, always: Sequence[int] = ()) -> np.ndarray:
    """``bool[V]``: the ``keep`` most frequent tokens, and every id of ``always`` (the role and
    end tokens of the chat format are in every conversation whatever a corpus count says)."""
    m = ranks(counts) < int(keep)
    for i in always:
        m[int(i)] = True
    return m


def row_map(keep: Any) -> np.ndarray:
    keep = np.asarray(keep, bool)
    ids = np.flatnonzero(keep)
    if ids.size >= NO_ROW:
        raise ValueError(f"{ids.size} kept rows do not fit a uint16 map with {NO_ROW} as 'no row'")
    m = np.full(keep.size, NO_ROW, np.uint16)
    m[ids] = np.arange(ids.size, dtype=np.uint16)
    return m


def compact(arrays: Mapping[str, np.ndarray], keep: Any) -> dict[str, np.ndarray]:
    """Stored arrays with the tables of :data:`ROW_TABLES` cut to the kept rows, plus the map."""
    keep = np.asarray(keep, bool)
    ids = np.flatnonzero(keep)
    out = dict(arrays)
    if MAP_LEAF in out:
        raise ValueError(f"the arrays hold {MAP_LEAF} already: rows were evicted before")
    for p in ROW_TABLES:
        if p not in out:
            raise ValueError(f"{p} is not among the stored leaves: no token rows to evict")
        a = np.asarray(out[p])
        if a.shape[0] != keep.size:
            raise ValueError(f"{p} has {a.shape[0]} rows, the mask {keep.size}")
        out[p] = np.ascontiguousarray(a[ids])
    out[MAP_LEAF] = row_map(keep)
    return out


def expand(arrays: Mapping[str, np.ndarray], tables: Sequence[str] = ROW_TABLES,
           map_leaf: str = MAP_LEAF) -> dict[str, np.ndarray]:
    """The inverse of :func:`compact`: whole tables with rows of zeros for the evicted tokens
    (a zero of the stored 16-bit formats is ``+0.0``), without the map."""
    out = dict(arrays)
    m = np.asarray(out.pop(map_leaf))
    if m.dtype != np.uint16 or m.ndim != 1:
        raise ValueError(f"{map_leaf} must be uint16[V], got {m.dtype}{list(m.shape)}")
    ids = np.flatnonzero(m != NO_ROW)
    if not np.array_equal(m[ids], np.arange(ids.size, dtype=np.uint16)):
        raise ValueError(f"{map_leaf} does not number the kept rows 0, 1, 2, ... in token order")
    for p in tables:
        a = np.asarray(out[p])
        if a.shape[0] != ids.size:
            raise ValueError(f"{p} holds {a.shape[0]} rows, {map_leaf} keeps {ids.size}")
        full = np.zeros((m.size,) + a.shape[1:], a.dtype)
        full[ids] = a
        out[p] = full
    return out


def kept_of(arrays: Mapping[str, np.ndarray], map_leaf: str = MAP_LEAF) -> np.ndarray:
    """``bool[V]`` from the stored map."""
    return np.asarray(arrays[map_leaf]) != NO_ROW


def mask_params(params: Any, keep: Any) -> Any:
    """A parameter tree with the rows of the evicted tokens set to zero in :data:`ROW_TABLES`
    (what an evicted export loads to; also the model whose loss prices an eviction)."""
    import jax
    import jax.numpy as jnp

    from bes.train.checkpoint import flatten_with_paths

    keep = np.asarray(keep, bool)
    paths, leaves, treedef = flatten_with_paths(params)
    out = []
    for p, a in zip(paths, leaves):
        if p in ROW_TABLES:
            a = jnp.where(jnp.asarray(keep).reshape((-1,) + (1,) * (np.ndim(a) - 1)), a, 0.0)
        out.append(a)
    return jax.tree_util.tree_unflatten(treedef, out)


def _shape_tree(model_cfg, kept: int) -> dict:
    """Shape structs of the parameter tree with ``kept`` rows in :data:`ROW_TABLES`."""
    import jax

    from bes import export as E
    from bes.train.checkpoint import flatten_with_paths

    paths, leaves, treedef = flatten_with_paths(E.params_template(model_cfg))
    out = [jax.ShapeDtypeStruct((kept,) + tuple(x.shape[1:]), x.dtype) if p in ROW_TABLES else x
           for p, x in zip(paths, leaves)]
    return jax.tree_util.tree_unflatten(treedef, out)


def accounting(model_cfg, kept: int, lib: Any = None) -> dict[str, Any]:
    """``bes.export.accounting_report`` of the compact tree, with the map added to the page row:
    ``V`` stored numbers, one read per token."""
    from bes import budget
    from bes import export as E

    acc = E.accounting_report(_shape_tree(model_cfg, kept), model_cfg, lib)
    v = int(model_cfg.vocab_size)
    acc["total"] += v
    acc["active_step"] += 1
    for row in acc["rows"]:
        if row["row"] == "page":
            row["total"] += v
            row["step"] += 1
    acc["within_caps"] = bool(acc["total"] <= budget.TOTAL_CAP and acc["active_step"] <= budget.ACTIVE_CAP)
    acc["rows_kept"] = int(kept)
    acc["row_map"] = {"leaf": MAP_LEAF, "numbers": v, "read_per_token": 1}
    return acc


def card_section(ev: Mapping[str, Any]) -> str:
    """The model card's paragraph on the evicted rows, from the export record's ``eviction``."""
    lines = ["## Evicted token rows", "",
             f"The page and gain rows (`{'`, `'.join(ev['tables'])}`) of {ev['evicted']:,} of the "
             f"{ev['vocab']:,} vocabulary ids are not stored: the least frequent ones "
             f"({ev.get('rule', 'see export.json')}). `{ev['map']}` (uint16, one entry per id) gives a "
             "token its row or none; a token without a row reads zeros, i.e. no gain and no "
             "private neurons, and the loader restores the tables of the configuration with those "
             f"rows at zero. Stored numbers freed: {ev['numbers_freed']:,} (the map costs {ev['vocab']:,}). "
             "The input embedding and the per-loop value embedding are whole."]
    h = ev.get("heldout") or {}
    if h.get("before") is not None and h.get("after") is not None:
        lines.append("")
        lines.append(f"Held-out loss of the 16-bit weights before the eviction {h['before']:.5f}, after "
                     f"{h['after']:.5f} nats per token ({h.get('tokens', '?')} tokens; by domain: "
                     f"{h.get('by_domain')}).")
    lines.append("")
    lines.append(f"Check made when the rows were evicted: the weights of this directory load to exactly "
                 f"the source's weights with those rows zeroed ({ev.get('leaves_equal')} leaves "
                 f"compared bit for bit). Source export: weights16.npz SHA-256 `{ev.get('source_sha256')}`.")
    return "\n".join(lines) + "\n"


def evict_export(src: str | os.PathLike, dst: str | os.PathLike, keep: Any, *,
                 library: str | os.PathLike | None = None, rule: str = "", heldout: Sequence[Any] | None = None,
                 log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write ``dst``: the export ``src`` with the rows of the tokens outside ``keep`` evicted and
    ``library`` (a built library directory, or None) attached. Returns the new export record.

    keep: ``bool[V]``. rule: one line for the model card on how the rows were chosen. heldout:
    ``bes.export.HeldoutBatch`` rows; when given, the loss before and after is measured and
    recorded. The result is checked: it verifies, and it loads to exactly the masked source."""
    import jax

    from bes import export as E
    from bes.train.checkpoint import flatten_with_paths

    say = log or (lambda m: None)
    src, dst = Path(src), Path(dst)
    rec = json.loads((src / E.EXPORT_JSON).read_text())
    meta = json.loads((src / E.WEIGHTS_JSON).read_text())
    if meta.get("rows"):
        raise ValueError(f"{src} has evicted rows already")
    cfg = E.load_export_config(src / "config.py")
    v = int(cfg.model.vocab_size)
    keep = np.asarray(keep, bool)
    if keep.shape != (v,):
        raise ValueError(f"the mask has shape {keep.shape}, the vocabulary {v} ids")
    kept = int(keep.sum())
    arrays = E.read_npz(src / E.WEIGHTS16)
    new = compact(arrays, keep)
    leaves, freed = [], 0
    for r in meta["leaves"]:
        if r["path"] in ROW_TABLES:
            shape = [kept] + [int(s) for s in r["shape"][1:]]
            n = int(np.prod(shape))
            freed += int(r["numbers"]) - n
            r = {**r, "shape": shape, "numbers": n, "rows_of": v}
        leaves.append(r)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".tmp-{dst.name}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir()
    try:
        E.write_npz(tmp / E.WEIGHTS16, new)
        rows_rec = {"map": MAP_LEAF, "tables": list(ROW_TABLES), "kept": kept, "vocab": v, "no_row": NO_ROW}
        E._write_json(tmp / E.WEIGHTS_JSON, {**{k: x for k, x in meta.items() if k != "leaves"},
                                             "leaves": leaves, "rows": rows_rec})
        for name in ("config.json", "config.py"):
            shutil.copy2(src / name, tmp / name)
        for sub in ("tokenizer", "reports"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, tmp / sub)
        # ---------------------------------------------------------------- library
        lib, lib_rec = None, {"mode": "none"}
        if library is not None:
            from bes.library.retrieve import Library

            library = Path(library)
            lib = Library.load(str(library))
            ship = E.library_files(library)
            files = E.library_file_list(ship)
            numbers = E.library_numbers(library, ship, lib.accounting())
            if not numbers["ok"]:
                raise E.ExportError(f"the library's tables do not match its accounting: {numbers['problems']}")
            for rel in files:
                (tmp / "library" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(library / rel, tmp / "library" / rel)
            dig = E.library_digest(library, files)
            if Library.load(str(tmp / "library")).accounting() != lib.accounting() or \
                    E.library_digest(tmp / "library")["digest"] != dig["digest"]:
                raise E.ExportError("the copied library differs from the built one")
            every = {str(p.relative_to(library)) for p in library.rglob("*") if p.is_file()}
            lib_rec = {"mode": "copy", "source_path": str(library.resolve()), "path": "library",
                       "digest": dig["digest"], "bytes": dig["bytes"], "files": len(dig["files"]),
                       "meta_sha256": E.sha256_file(E._library_marker(library)),
                       "kind": "answers" if (library / "answers.json").is_file() else "passages",
                       "shipped": ship, "numbers": numbers, "left_out": sorted(every - set(files)),
                       "qa_sources": E.library_qa_sources(library),
                       "build_settings": E._read_json(library / E.LIBRARY_SETTINGS),
                       "decision": {"ship": True, "rule": "bes.evict.evict_export",
                                    "reason": "attached with the eviction of token rows "
                                              "(the water line of docs/PLAN_M6V.md)"}}
            attr = library / "ATTRIBUTION.md"
            if attr.exists():
                shutil.copy2(attr, tmp / "LIBRARY_ATTRIBUTION.md")
                lib_rec["attribution"] = "LIBRARY_ATTRIBUTION.md"
        acc = accounting(cfg.model, kept, lib)
        # ---------------------------------------------------------------- the check
        got = E.load_weights(tmp, cfg.model)
        want = mask_params(E.load_weights(src, cfg.model), keep)
        pa, la, _ = flatten_with_paths(got)
        pb, lb, _ = flatten_with_paths(want)
        bad = [p for p, a, b in zip(pa, jax.device_get(la), jax.device_get(lb))
               if not np.array_equal(np.asarray(a), np.asarray(b))]
        if pa != pb or bad:
            raise E.ExportError(f"the evicted weights do not load to the masked source: {bad[:5]}")
        held = None
        if heldout:
            from bes.model import model as M

            consts = M.make_consts(E.check_model_cfg(cfg.model), **((rec.get("consts") or {}).get("enable") or {}))
            before = E.heldout_loss(E.load_weights(src, cfg.model), consts, heldout, cfg.model)
            after = E.heldout_loss(got, consts, heldout, cfg.model)
            held = {"before": before["loss"], "after": after["loss"], "tokens": after["weight"],
                    "by_domain": {d: [round(before["by_domain"][d]["loss"], 5), round(x["loss"], 5)]
                                  for d, x in after["by_domain"].items()}}
            say(f"evict: held-out loss {before['loss']:.5f} -> {after['loss']:.5f}")
        # ---------------------------------------------------------------- the record
        ev = {"tables": list(ROW_TABLES), "map": MAP_LEAF, "vocab": v, "kept": kept, "evicted": v - kept,
              "numbers_freed": freed, "rule": rule, "heldout": held, "leaves_equal": len(pa),
              "source_sha256": (rec.get("weights") or {}).get("sha256"),
              "source_accounting": {k: (rec.get("accounting") or {}).get(k) for k in ("total", "active_step")},
              "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        choices = [E.LeafChoice.from_dict(r) for r in leaves]
        weights = {**E.dtype_summary(choices), "sha256": E.sha256_file(tmp / E.WEIGHTS16),
                   "float32_kept": False, "sha256_float32": None}
        weights["numbers"] += v
        weights["row_map_numbers"] = v
        new_rec = {**rec, "weights": weights, "library": lib_rec, "accounting": acc, "eviction": ev}
        main = [E.WEIGHTS16, E.WEIGHTS_JSON, "config.json", "tokenizer/tokenizer.json"]
        new_rec["main_files"] = [{"path": f, "bytes": (tmp / f).stat().st_size, "sha256": E.sha256_file(tmp / f)}
                                 for f in main if (tmp / f).exists()]
        E._write_json(tmp / E.EXPORT_JSON, new_rec)
        (tmp / E.CARD).write_text(E.model_card(json.loads((tmp / E.EXPORT_JSON).read_text())))
        E.write_sums(tmp)
        problems = E.verify_export(tmp, library=False)
        if problems:
            raise E.ExportError(f"the written export does not verify: {problems[:5]}")
        if dst.exists():
            old = dst.with_name(f".old-{dst.name}-{uuid.uuid4().hex[:8]}")
            os.replace(dst, old)
            shutil.rmtree(old, ignore_errors=True)
        os.replace(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    say(f"evict: {dst}: {kept} of {v} token rows kept, {freed:,} numbers freed; total {acc['total']:,}, "
        f"read per token {acc['active_step']:,}; within the caps: {acc['within_caps']}")
    return new_rec


__all__ = ["ROW_TABLES", "MAP_LEAF", "NO_ROW", "ranks", "keep_mask", "row_map", "compact", "expand",
           "kept_of", "mask_params", "accounting", "card_section", "evict_export"]
