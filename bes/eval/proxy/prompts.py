"""The proxy prompt set (``prompts.jsonl``): loading, validation and the dev / held-out split.

One JSON object per line:

- ``id``: unique string, ``<category>-<nnn>``;
- ``category``: ``chat | math | code | writing | instruct``;
- ``turns``: list of 1 or 2 user messages; a second message is a follow-up that depends on the
  model's own first answer (the harness feeds every model its own previous reply);
- ``answer`` (math only): the exact numeric answer of the last turn as a string, or a list with
  one entry per turn (``null`` where a turn has no numeric answer); ``tol`` (optional): absolute
  tolerance for rounded answers;
- ``constraints``: list of machine-checkable constraints (see
  ``bes.eval.exact_score.CONSTRAINT_TYPES``); each applies to the last turn unless it has a
  ``"turn"`` index (0-based);
- ``notes``: what the item exercises.

The split is not stored: within every stratum (category x number of turns) prompts are ordered
by a SHA-256 of their id and alternately assigned to ``dev`` and ``test`` (held out), so the
halves are balanced and stable under additions to other strata.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bes.eval.proxy.common import stable_hash

PROMPTS_FILE = Path(__file__).with_name("prompts.jsonl")
CATEGORIES: tuple[str, ...] = ("chat", "math", "code", "writing", "instruct")
TARGET_COUNTS: dict[str, int] = {"chat": 100, "math": 60, "code": 60, "writing": 100, "instruct": 80}
SPLITS: tuple[str, ...] = ("dev", "test")
SPLIT_SALT = "bes-proxy-split-v1"
_FIELDS = {"id", "category", "turns", "answer", "tol", "constraints", "notes"}


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    turns: tuple[str, ...]
    answer: Any = None
    tol: float | None = None
    constraints: tuple[dict[str, Any], ...] = ()
    notes: str = ""
    split: str = "dev"

    @property
    def multi_turn(self) -> bool:
        return len(self.turns) > 1

    @property
    def content_hash(self) -> str:
        """Changes whenever anything that defines the item changes (not the split)."""
        return stable_hash({"id": self.id, "category": self.category, "turns": list(self.turns),
                            "answer": self.answer, "tol": self.tol,
                            "constraints": list(self.constraints)}, 12)

    def answer_for_turn(self, turn: int) -> str | None:
        if self.answer is None:
            return None
        if isinstance(self.answer, (list, tuple)):
            return self.answer[turn]
        return self.answer if turn == len(self.turns) - 1 else None

    def constraints_for_turn(self, turn: int) -> list[dict[str, Any]]:
        last = len(self.turns) - 1
        return [c for c in self.constraints if int(c.get("turn", last)) == turn]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "category": self.category, "turns": list(self.turns)}
        if self.answer is not None:
            out["answer"] = list(self.answer) if isinstance(self.answer, tuple) else self.answer
        if self.tol is not None:
            out["tol"] = self.tol
        out["constraints"] = list(self.constraints)
        out["notes"] = self.notes
        return out


def _split_rank(prompt_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{prompt_id}".encode("utf-8")).hexdigest()


def assign_splits(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """``id -> "dev" | "test"``: per stratum (category, number of turns), ids sorted by hash
    and assigned alternately; the i-th stratum in sorted order starts with ``SPLITS[i % 2]``,
    so odd-sized strata do not all favour the same half."""
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in records:
        strata[(r["category"], len(r["turns"]))].append(r["id"])
    out: dict[str, str] = {}
    for i, stratum in enumerate(sorted(strata)):
        for rank, pid in enumerate(sorted(strata[stratum], key=_split_rank)):
            out[pid] = SPLITS[(rank + i) % 2]
    return out


def validate(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Problems of a raw record list (empty when valid). Checks fields, ids, categories, turns,
    math answers and constraint shapes (types and parameters are checked by
    ``bes.eval.exact_score.validate_constraint``)."""
    from bes.eval.exact_score import parse_number, validate_constraint

    errors: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(records):
        where = f"record {i} ({r.get('id', '?')})"
        extra = set(r) - _FIELDS
        if extra:
            errors.append(f"{where}: unknown fields {sorted(extra)}")
        pid = r.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"{where}: missing id")
            continue
        if pid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(pid)
        cat = r.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{where}: unknown category {cat!r}")
        turns = r.get("turns")
        if (not isinstance(turns, list) or not 1 <= len(turns) <= 2
                or not all(isinstance(t, str) and t.strip() for t in turns)):
            errors.append(f"{where}: turns must be a list of 1 or 2 non-empty strings")
            continue
        answer = r.get("answer")
        if cat == "math":
            answers = answer if isinstance(answer, list) else [answer]
            if isinstance(answer, list) and len(answer) != len(turns):
                errors.append(f"{where}: answer list needs one entry per turn")
            if answers[-1] is None:
                errors.append(f"{where}: math item without an answer for the last turn")
            for a in answers:
                if a is not None and (not isinstance(a, str) or parse_number(a) is None):
                    errors.append(f"{where}: answer {a!r} is not a number string")
        elif answer is not None:
            errors.append(f"{where}: only math items carry an answer")
        tol = r.get("tol")
        if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
            errors.append(f"{where}: tol must be a non-negative number")
        constraints = r.get("constraints", [])
        if not isinstance(constraints, list):
            errors.append(f"{where}: constraints must be a list")
            continue
        for c in constraints:
            for problem in validate_constraint(c, n_turns=len(turns)):
                errors.append(f"{where}: {problem}")
        if cat == "instruct" and not constraints:
            errors.append(f"{where}: instruct items need at least one constraint")
        if not isinstance(r.get("notes", ""), str):
            errors.append(f"{where}: notes must be a string")
    return errors


def load_records(path: str | Path = PROMPTS_FILE) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise ValueError(f"{path}:{n}: {err}") from err
    return out


def load(path: str | Path = PROMPTS_FILE) -> list[Prompt]:
    """All prompts, validated, with their split. Raises ``ValueError`` listing every problem."""
    records = load_records(path)
    errors = validate(records)
    if errors:
        raise ValueError(f"{path}: {len(errors)} problems:\n" + "\n".join(errors[:50]))
    splits = assign_splits(records)
    return [Prompt(id=r["id"], category=r["category"], turns=tuple(r["turns"]),
                   answer=tuple(r["answer"]) if isinstance(r.get("answer"), list) else r.get("answer"),
                   tol=r.get("tol"), constraints=tuple(r.get("constraints", [])),
                   notes=r.get("notes", ""), split=splits[r["id"]]) for r in records]


def select(prompts: Sequence[Prompt], split: str = "dev", categories: Sequence[str] | None = None,
           limit: int | None = None) -> list[Prompt]:
    """Prompts of one split (``"dev"``, ``"test"`` or ``"all"``), optionally of some categories;
    ``limit`` keeps the first ``limit`` in file order."""
    if split not in (*SPLITS, "all"):
        raise ValueError(f"unknown split {split!r}")
    out = [p for p in prompts if split == "all" or p.split == split]
    if categories:
        out = [p for p in out if p.category in set(categories)]
    return out[:limit] if limit is not None else out


def describe(prompts: Sequence[Prompt]) -> dict[str, Any]:
    """Counts by category, split and number of turns (for ``python -m bes.eval.proxy prompts``)."""
    by = Counter((p.category, p.split, len(p.turns)) for p in prompts)
    out: dict[str, Any] = {"total": len(prompts),
                           "multi_turn": sum(p.multi_turn for p in prompts),
                           "with_constraints": sum(bool(p.constraints) for p in prompts),
                           "by_category": {}}
    for cat in CATEGORIES:
        row = {s: sum(v for (c, sp, _), v in by.items() if c == cat and sp == s) for s in SPLITS}
        row["two_turn"] = sum(v for (c, _, n), v in by.items() if c == cat and n == 2)
        row["total"] = row["dev"] + row["test"]
        out["by_category"][cat] = row
    return out
