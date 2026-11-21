"""Blind A/B pairs of two runs, ``x`` (ours) against ``y`` (an opponent).

For every prompt answered by both runs (same prompt content), one pair shows the two
conversations as "A" and "B". Which run is "A" is decided by a seeded, balanced assignment:
pairs are ordered by category and then by a seeded shuffle, and the sides alternate along that
order from a seeded start, so the two runs are on the left equally often overall and within
every category (difference at most one). The order of pairs in the file is a second seeded
shuffle, and pair ids are salted hashes, so neither reveals the assignment.

Files (``root`` defaults to ``data_cache/proxy``):

- ``pairs/<name>/pairs.jsonl``: what raters and the judge see; per pair ``pair_id``,
  ``prompt_id``, ``category``, ``multi_turn`` and ``turns = [{"user", "A", "B"}]``. Nothing in
  this directory names a run or a model.
- ``pairs/<name>/meta.json``: counts and the SHA-256 *commitment* of the key.
- ``keys/<name>.json``: the hidden key (seed, salt, run ids, ``assignment[pair_id] = "x" | "y"``
  meaning which run is shown as A). Scoring needs it and checks it against the commitment, so
  a key edited after the ratings is detected. Keep ``keys/`` away from raters.
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bes.eval.proxy.common import (default_root, read_json, read_jsonl, safe_name, stable_hash,
                                   stable_json, write_json_atomic, write_jsonl_atomic)
from bes.eval.proxy.prompts import Prompt


def pairs_dir(root: Path, name: str) -> Path:
    return Path(root) / "pairs" / safe_name(name)


def key_path(root: Path, name: str) -> Path:
    return Path(root) / "keys" / f"{safe_name(name)}.json"


def commitment(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(key).encode("utf-8")).hexdigest()


def assign_sides(items: Sequence[tuple[str, str]], seed: int) -> dict[str, str]:
    """``item_id -> "x" | "y"`` (the run shown as A) for ``items = [(item_id, category)]``.

    Balanced overall and within each category (difference at most one), seeded."""
    rng = random.Random(stable_hash(["sides", int(seed)]))
    by_cat: dict[str, list[str]] = {}
    for item_id, cat in items:
        by_cat.setdefault(cat, []).append(item_id)
    order: list[str] = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        order.extend(ids)
    first = rng.choice(("x", "y"))
    other = "y" if first == "x" else "x"
    return {item_id: (first if i % 2 == 0 else other) for i, item_id in enumerate(order)}


def _turns(x: Mapping[str, Any], y: Mapping[str, Any], x_is_a: bool) -> list[dict[str, str]]:
    out = []
    for tx, ty in zip(x["turns"], y["turns"]):
        a, b = (tx, ty) if x_is_a else (ty, tx)
        out.append({"user": tx["user"], "A": a["reply"], "B": b["reply"]})
    return out


def make_pairs(prompts: Sequence[Prompt], x_records: Mapping[str, Mapping[str, Any]],
               y_records: Mapping[str, Mapping[str, Any]], x_run: str, y_run: str, name: str,
               seed: int = 0, root: Path | None = None) -> dict[str, Any]:
    """Write the blinded pairs and the hidden key; returns the public meta (no run names)."""
    root = Path(root) if root is not None else default_root()
    usable = [p for p in prompts
              if p.id in x_records and p.id in y_records
              and x_records[p.id].get("prompt_hash") == p.content_hash
              and y_records[p.id].get("prompt_hash") == p.content_hash
              and len(x_records[p.id]["turns"]) == len(y_records[p.id]["turns"]) == len(p.turns)]
    if not usable:
        raise ValueError("the two runs have no prompt in common (same prompt content)")
    salt = stable_hash(["salt", int(seed), name], 32)
    pid_of = {p.id: "p" + stable_hash([salt, p.id], 10) for p in usable}
    sides = assign_sides([(pid_of[p.id], p.category) for p in usable], seed)
    pairs = []
    for p in usable:
        pair_id = pid_of[p.id]
        pairs.append({"pair_id": pair_id, "prompt_id": p.id, "category": p.category,
                      "multi_turn": p.multi_turn,
                      "turns": _turns(x_records[p.id], y_records[p.id], sides[pair_id] == "x")})
    random.Random(stable_hash(["order", int(seed), name])).shuffle(pairs)
    key = {"name": name, "seed": int(seed), "salt": salt, "x": x_run, "y": y_run,
           "assignment": dict(sorted(sides.items())),
           "prompt_of": {pid_of[p.id]: p.id for p in usable}}
    meta = {"name": name, "n_pairs": len(pairs), "commitment": commitment(key),
            "by_category": dict(Counter(p.category for p in usable)),
            "multi_turn": sum(p.multi_turn for p in usable),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = pairs_dir(root, name)
    write_jsonl_atomic(out / "pairs.jsonl", pairs)
    write_json_atomic(out / "meta.json", meta)
    write_json_atomic(key_path(root, name), key)
    return meta


def load_pairs(root: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Blinded pairs and public meta of a pair set (what raters and the judge may read)."""
    d = pairs_dir(root, name)
    if not (d / "pairs.jsonl").exists():
        raise FileNotFoundError(f"no pair set {name!r} under {d}")
    return read_jsonl(d / "pairs.jsonl"), read_json(d / "meta.json")


def load_key(root: Path, name: str) -> dict[str, Any]:
    """The hidden key, checked against the commitment published in ``meta.json``."""
    key = read_json(key_path(root, name))
    meta = read_json(pairs_dir(root, name) / "meta.json")
    if commitment(key) != meta.get("commitment"):
        raise ValueError(f"key of pair set {name!r} does not match its commitment")
    return key


def unblind(verdict: str, pair_id: str, key: Mapping[str, Any]) -> str:
    """Map a blinded verdict (``"A" | "B" | "tie" | "both_bad"``) to the outcome for run x:
    ``"win" | "loss" | "tie" | "both_bad"``."""
    if verdict in ("tie", "both_bad"):
        return verdict
    if verdict not in ("A", "B"):
        raise ValueError(f"unknown verdict {verdict!r}")
    x_side = "A" if key["assignment"][pair_id] == "x" else "B"
    return "win" if verdict == x_side else "loss"


def list_pairsets(root: Path | None = None) -> list[str]:
    root = Path(root) if root is not None else default_root()
    base = root / "pairs"
    return sorted(p.parent.name for p in base.glob("*/pairs.jsonl")) if base.exists() else []
