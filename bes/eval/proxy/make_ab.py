"""Blind A/B pairs of two runs, ``x`` (ours) against ``y`` (an opponent).

For every prompt answered by both runs (same prompt content), one pair shows the two
conversations as "A" and "B". Which run is "A" is decided by a seeded, balanced assignment:
pairs are ordered by category and then by a seeded shuffle, and the sides alternate along that
order from a seeded start, so the two runs are on the left equally often overall and within
every category (difference at most one). The order of pairs in the file is a second seeded
shuffle, and pair ids are salted hashes, so neither reveals the assignment.

**A pair set only grows.** Rebuilding a pair set under the same name with the same runs and
seed (a rerun of ``pairs`` or ``all``, the 24-hour run's judge stage after a resume) keeps every
pair already published exactly as it is, side and replies included, and adds the prompts that
became usable as a new *version*: their sides are assigned so that every category stays
balanced given the pairs already there (difference at most one within every category; overall
the difference stays small, at most the number of categories). A published pair whose replies
would change, or a rebuild with other runs, another seed or a fresh assignment while verdicts
are stored, is refused (:class:`PairSetConflict`); nothing is written then.

**Commitments.** The key commits to its versions as a hash chain: ``c_0`` hashes the base
(name, seed, salt, run ids), ``c_k = sha256(c_{k-1} + version_k)`` where a version lists the
side, prompt id and content digest of every pair it added. ``meta.json`` publishes the chain;
its last entry is the current commitment. Every judge and human rating record carries the
commitment it was made under and the digest of the pair it rated. Scoring recomputes the chain
from the key and accepts a record only when its commitment is in the chain, its pair was
already published under that commitment and its digest matches; any other record makes the
score refuse loudly (a key or pair file edited, or a pair set rebuilt behind the verdicts' back).

Files (``root`` defaults to ``data_cache/proxy``):

- ``pairs/<name>/pairs.jsonl``: what raters and the judge see; per pair ``pair_id``,
  ``prompt_id``, ``category``, ``multi_turn`` and ``turns = [{"user", "A", "B"}]``. Nothing in
  this directory names a run or a model.
- ``pairs/<name>/meta.json``: counts, the commitment chain (``commitments``, the last one is
  ``commitment``) and the number of pairs each version ``added``.
- ``keys/<name>.json``: the hidden key (seed, salt, run ids, ``versions`` and the merged
  ``assignment[pair_id] = "x" | "y"``, meaning which run is shown as A). Scoring needs it and
  checks it against the published chain. Keep ``keys/`` away from raters.
- ``ratings/<name>/<source>.jsonl``: verdicts (``bes.eval.proxy.judge``, ``bes.eval.proxy.rate``).
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bes.eval.proxy.common import (default_root, read_json, read_jsonl, safe_name, stable_hash,
                                   stable_json, write_json_atomic, write_jsonl_atomic)
from bes.eval.proxy.prompts import Prompt

#: Key format with versions and a commitment chain (format 1: one hash of the whole key).
KEY_FORMAT = 2
#: Verdict values of rating records (``retracted`` undoes an earlier record of the same pair).
VERDICTS = ("A", "B", "tie", "both_bad")
_BASE_FIELDS = ("name", "seed", "salt", "x", "y")


class PairSetConflict(ValueError):
    """Writing the pair set would change what stored verdicts were made under, or a published
    pair; nothing was written."""


def pairs_dir(root: Path, name: str) -> Path:
    return Path(root) / "pairs" / safe_name(name)


def key_path(root: Path, name: str) -> Path:
    return Path(root) / "keys" / f"{safe_name(name)}.json"


def ratings_dir(root: Path, name: str) -> Path:
    """Where the rating sources of a pair set live (``bes.eval.proxy.judge.ratings_path``)."""
    return Path(root) / "ratings" / safe_name(name)


def commitment(key: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a whole key (the commitment of format-1 keys)."""
    return hashlib.sha256(stable_json(key).encode("utf-8")).hexdigest()


def pair_digest(pair: Mapping[str, Any]) -> str:
    """Digest of what a rater sees of one pair (prompt id and the turns as shown)."""
    return stable_hash({"prompt_id": pair["prompt_id"], "turns": list(pair["turns"])}, 16)


def chain(base: Mapping[str, Any], versions: Sequence[Mapping[str, Any]]) -> list[str]:
    """The commitment after each version: ``c_k = sha256(c_{k-1} + stable_json(version_k))``
    with ``c_0 = sha256(stable_json({"format", "base"}))``."""
    c = hashlib.sha256(stable_json({"format": KEY_FORMAT, "base": dict(base)}).encode("utf-8")).hexdigest()
    out = []
    for v in versions:
        c = hashlib.sha256((c + stable_json(v)).encode("utf-8")).hexdigest()
        out.append(c)
    return out


def _base(key: Mapping[str, Any]) -> dict[str, Any]:
    return {k: key[k] for k in _BASE_FIELDS}


def assign_sides(items: Sequence[tuple[str, str]], seed: int,
                 existing: Sequence[tuple[str, str, str]] = ()) -> dict[str, str]:
    """``item_id -> "x" | "y"`` (the run shown as A) for ``items = [(item_id, category)]``.

    Without ``existing``: balanced overall and within each category (difference at most one),
    seeded. With ``existing = [(item_id, category, side)]`` (pairs already published, whose
    sides are fixed) only the new items are assigned: per category, in a seeded order,
    alternating from the side that has fewer pairs in that category (when the category is even,
    from the side with fewer pairs overall, seeded on a tie), so every category stays within
    one; the overall difference stays at most the number of categories."""
    by_cat: dict[str, list[str]] = {}
    for item_id, cat in items:
        by_cat.setdefault(cat, []).append(item_id)
    if not existing:
        rng = random.Random(stable_hash(["sides", int(seed)]))
        order: list[str] = []
        for cat in sorted(by_cat):
            ids = sorted(by_cat[cat])
            rng.shuffle(ids)
            order.extend(ids)
        first = rng.choice(("x", "y"))
        other = "y" if first == "x" else "x"
        return {item_id: (first if i % 2 == 0 else other) for i, item_id in enumerate(order)}
    rng = random.Random(stable_hash(["sides", int(seed), "added", len(existing)]))
    per = Counter((cat, side) for _, cat, side in existing)
    total = Counter(side for _, _, side in existing)
    out: dict[str, str] = {}
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        d = per[(cat, "x")] - per[(cat, "y")]
        if d:
            first = "y" if d > 0 else "x"
        else:
            lead = total["x"] - total["y"]
            first = "y" if lead > 0 else "x" if lead < 0 else rng.choice(("x", "y"))
        other = "y" if first == "x" else "x"
        for i, item_id in enumerate(ids):
            side = first if i % 2 == 0 else other
            out[item_id] = side
            total[side] += 1
    return out


def _turns(x: Mapping[str, Any], y: Mapping[str, Any], x_is_a: bool) -> list[dict[str, str]]:
    out = []
    for tx, ty in zip(x["turns"], y["turns"]):
        a, b = (tx, ty) if x_is_a else (ty, tx)
        out.append({"user": tx["user"], "A": a["reply"], "B": b["reply"]})
    return out


def _pair(p: Prompt, x_records: Mapping[str, Mapping[str, Any]],
          y_records: Mapping[str, Mapping[str, Any]], pair_id: str, x_is_a: bool) -> dict[str, Any]:
    return {"pair_id": pair_id, "prompt_id": p.id, "category": p.category,
            "multi_turn": p.multi_turn, "turns": _turns(x_records[p.id], y_records[p.id], x_is_a)}


# ------------------------------------------------------------------------- key and records
def verified_commitments(key: Mapping[str, Any], meta: Mapping[str, Any], name: str = "") -> list[str]:
    """The commitment chain recomputed from ``key`` after checking it against the published
    ``meta`` (versions disjoint, merged assignment consistent, chain equal to
    ``meta["commitments"]``). Raises ``ValueError`` when anything differs. A format-1 key is
    checked the old way and has no chain (``[]``): its verdicts cannot be checked."""
    if key.get("format") != KEY_FORMAT:
        if commitment(key) != meta.get("commitment"):
            raise ValueError(f"key of pair set {name!r} does not match its commitment")
        return []
    versions = list(key.get("versions") or [])
    assignment: dict[str, str] = {}
    prompt_of: dict[str, str] = {}
    for v in versions:
        if set(v.get("assignment", {})) != set(v.get("content", {})) or \
                set(v.get("assignment", {})) != set(v.get("prompt_of", {})):
            raise ValueError(f"key of pair set {name!r} is malformed (a version lacks fields)")
        for pid, side in v["assignment"].items():
            if pid in assignment or side not in ("x", "y"):
                raise ValueError(f"key of pair set {name!r} is malformed (pair {pid} repeated or bad side)")
            assignment[pid] = side
            prompt_of[pid] = v["prompt_of"][pid]
    ch = chain(_base(key), versions)
    if (not ch or ch != list(meta.get("commitments") or []) or ch[-1] != meta.get("commitment")
            or assignment != dict(key.get("assignment", {})) or prompt_of != dict(key.get("prompt_of", {}))):
        raise ValueError(f"key of pair set {name!r} does not match its commitment")
    return ch


def load_pairs(root: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Blinded pairs and public meta of a pair set (what raters and the judge may read)."""
    d = pairs_dir(root, name)
    if not (d / "pairs.jsonl").exists():
        raise FileNotFoundError(f"no pair set {name!r} under {d}")
    return read_jsonl(d / "pairs.jsonl"), read_json(d / "meta.json")


def load_key(root: Path, name: str) -> dict[str, Any]:
    """The hidden key, checked against the commitment chain published in ``meta.json``.
    The returned dict also holds ``commitments`` (the verified chain; empty for a format-1 key)
    and ``legacy`` (True for a format-1 key)."""
    key = read_json(key_path(root, name))
    meta = read_json(pairs_dir(root, name) / "meta.json")
    ch = verified_commitments(key, meta, name)
    return {**key, "commitments": ch, "legacy": key.get("format") != KEY_FORMAT}


def published(root: Path, name: str) -> dict[str, Any]:
    """The public view raters and judges check their records against: ``{"commitment",
    "commitments", "digests": {pair_id: digest}}`` from ``meta.json`` and ``pairs.jsonl``.
    Raises ``ValueError`` for a pair set written before commitments were chained (rebuild it
    under a new name)."""
    pairs, meta = load_pairs(root, name)
    if meta.get("format") != KEY_FORMAT or not meta.get("commitments"):
        raise ValueError(f"pair set {name!r} was written by an earlier version of the harness and "
                         "carries no commitment chain; build it again with `pairs` under a new name")
    return {"commitment": meta["commitment"], "commitments": list(meta["commitments"]),
            "digests": {p["pair_id"]: pair_digest(p) for p in pairs}}


def _verdict_records(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in records if r.get("verdict") in (*VERDICTS, "retracted")]


def invalid_records(records: Iterable[Mapping[str, Any]], key: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Rating records that were not made under this key (from :func:`load_key`): commitment
    missing or not in the chain, the pair published only by a later version, or a pair digest
    that differs from the committed one."""
    pos = {c: i for i, c in enumerate(key.get("commitments") or [])}
    intro: dict[str, int] = {}
    digest: dict[str, str] = {}
    for k, v in enumerate(key.get("versions") or []):
        for pid, dg in v["content"].items():
            intro[pid], digest[pid] = k, dg
    bad = []
    for r in _verdict_records(records):
        k = pos.get(r.get("commitment"))
        pid = r.get("pair_id")
        if k is None or pid not in intro or intro[pid] > k or r.get("pair_digest") != digest[pid]:
            bad.append(r)
    return bad


def foreign_records(records: Iterable[Mapping[str, Any]], pub: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Rating records that do not belong to the published pair set ``pub`` (:func:`published`):
    the check raters and judges can make without the key."""
    known = set(pub["commitments"])
    return [r for r in _verdict_records(records)
            if r.get("commitment") not in known or r.get("pair_digest") != pub["digests"].get(r.get("pair_id"))]


def refusal(name: str, source: str, bad: Sequence[Mapping[str, Any]], total: int) -> str:
    return (f"pair set {name!r}, ratings {source!r}: {len(bad)} of {total} records were not made under "
            "this pair set's key (no commitment, another commitment or other replies): the pair set "
            "was rebuilt or edited after they were written, and unblinding them could credit the "
            "wrong model. Refused. Rate into a new pair set (another --name), or move that file "
            "away if you know its records are void.")


def stored_records(root: Path, name: str) -> dict[str, list[dict[str, Any]]]:
    """Every rating source of a pair set: ``source -> records``."""
    d = ratings_dir(root, name)
    return {p.stem: read_jsonl(p) for p in sorted(d.glob("*.jsonl"))} if d.exists() else {}


def check_pairs(pairs: Sequence[Mapping[str, Any]], key: Mapping[str, Any], name: str = "") -> None:
    """``pairs.jsonl`` holds exactly the pairs the key commits to, with the committed replies
    (raises ``ValueError`` otherwise; nothing to check for a format-1 key)."""
    if key.get("legacy") or key.get("format") != KEY_FORMAT:
        return
    digest = {pid: dg for v in key["versions"] for pid, dg in v["content"].items()}
    got = {p["pair_id"]: pair_digest(p) for p in pairs}
    if got != digest or len(got) != len(pairs):
        raise ValueError(f"pairs of pair set {name!r} differ from what its key commits to")


def _refuse_foreign_records(root: Path, name: str, key: Mapping[str, Any]) -> None:
    """:class:`PairSetConflict` when any stored rating record of the pair set was not made
    under ``key`` (with its ``commitments``)."""
    for source, recs in stored_records(root, name).items():
        bad = invalid_records(recs, key)
        if bad:
            raise PairSetConflict(refusal(name, source, bad, len(_verdict_records(recs))))


# ------------------------------------------------------------------------------- building
def make_pairs(prompts: Sequence[Prompt], x_records: Mapping[str, Mapping[str, Any]],
               y_records: Mapping[str, Mapping[str, Any]], x_run: str, y_run: str, name: str,
               seed: int = 0, root: Path | None = None) -> dict[str, Any]:
    """Write (or grow) the blinded pairs and the hidden key; returns the public meta (no run
    names). See the module docstring: published pairs never change; new usable prompts are
    added as a new version; :class:`PairSetConflict` when the stored verdicts or a published
    pair would not survive the write."""
    root = Path(root) if root is not None else default_root()
    usable = [p for p in prompts
              if p.id in x_records and p.id in y_records
              and x_records[p.id].get("prompt_hash") == p.content_hash
              and y_records[p.id].get("prompt_hash") == p.content_hash
              and len(x_records[p.id]["turns"]) == len(y_records[p.id]["turns"]) == len(p.turns)]
    if not usable:
        raise ValueError("the two runs have no prompt in common (same prompt content)")
    salt = stable_hash(["salt", int(seed), name], 32)
    base = {"name": name, "seed": int(seed), "salt": salt, "x": x_run, "y": y_run}
    pid_of = {p.id: "p" + stable_hash([salt, p.id], 10) for p in usable}
    out_dir, kpath = pairs_dir(root, name), key_path(root, name)
    versions: list[dict[str, Any]] = []
    old_pairs: list[dict[str, Any]] = []
    old: dict[str, Any] | None = None
    created = None
    if kpath.exists() and (out_dir / "meta.json").exists() and (out_dir / "pairs.jsonl").exists():
        found = load_key(root, name)            # raises when the key does not match its commitment
        if not found["legacy"] and _base(found) == base:
            old = found
            old_pairs = read_jsonl(out_dir / "pairs.jsonl")
            check_pairs(old_pairs, old, name)
            versions = list(old["versions"])
            created = read_json(out_dir / "meta.json").get("created")
    side = {pid: s for v in versions for pid, s in v["assignment"].items()}
    digest = {pid: dg for v in versions for pid, dg in v["content"].items()}
    new = []
    for p in usable:
        pid = pid_of[p.id]
        if pid not in side:
            new.append(p)
        elif pair_digest(_pair(p, x_records, y_records, pid, side[pid] == "x")) != digest[pid]:
            raise PairSetConflict(f"pair set {name!r}: the replies to prompt {p.id} differ from its "
                                  "published pair; a published pair never changes (use another --name)")
    if old is not None and not new:
        _refuse_foreign_records(root, name, old)        # nothing to add; the stored verdicts must fit
        return read_json(out_dir / "meta.json")
    category = {p["pair_id"]: p["category"] for p in old_pairs}
    sides = assign_sides([(pid_of[p.id], p.category) for p in new], seed,
                         [(pid, category[pid], s) for pid, s in side.items()])
    new_pairs = [_pair(p, x_records, y_records, pid_of[p.id], sides[pid_of[p.id]] == "x") for p in new]
    order_seed = ["order", int(seed), name] + ([len(versions)] if versions else [])
    random.Random(stable_hash(order_seed)).shuffle(new_pairs)
    versions.append({"assignment": dict(sorted((q["pair_id"], sides[q["pair_id"]]) for q in new_pairs)),
                     "prompt_of": dict(sorted((q["pair_id"], q["prompt_id"]) for q in new_pairs)),
                     "content": dict(sorted((q["pair_id"], pair_digest(q)) for q in new_pairs))})
    key = {"format": KEY_FORMAT, **base, "versions": versions,
           "assignment": dict(sorted((pid, s) for v in versions for pid, s in v["assignment"].items())),
           "prompt_of": dict(sorted((pid, q) for v in versions for pid, q in v["prompt_of"].items()))}
    ch = chain(base, versions)
    _refuse_foreign_records(root, name, {**key, "commitments": ch})
    pairs = old_pairs + new_pairs
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {"name": name, "format": KEY_FORMAT, "n_pairs": len(pairs), "commitment": ch[-1],
            "commitments": ch, "added": [len(v["assignment"]) for v in versions],
            "by_category": dict(Counter(q["category"] for q in pairs)),
            "multi_turn": sum(bool(q["multi_turn"]) for q in pairs),
            "created": created or now, "updated": now}
    write_json_atomic(kpath, key)
    write_jsonl_atomic(out_dir / "pairs.jsonl", pairs)
    write_json_atomic(out_dir / "meta.json", meta)
    return meta


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
