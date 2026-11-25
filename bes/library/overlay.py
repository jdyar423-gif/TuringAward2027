"""Entries written after training: the episodic half of the lifelong writes (``docs/PLAN_M6V.md``).

An :class:`Overlay` is a small, appendable list of ``(question, answer)`` pairs that a user
teaches the shipped model ("remember that ..."). Writing one is exact, takes effect at once,
never changes a stored weight and does not depend on the order of the writes; it survives a
restart when the overlay has a file. :class:`LifelongAnswers` puts an overlay on top of an
answers library (``bes.library.answers.AnswerLibrary``): its entries take part in the verbatim
rule and in the exemplars of a reply exactly like stored entries, scored by the same exact
overlap of index terms.

Two things differ from stored entries, both because a taught entry is something this user said
on purpose and there are few of them:

* matching is more generous: a taught entry that shares a term rarer than ``share_idf`` with
  the message, or reaches ``min_sym``, is offered as an exemplar even when its overlap is below
  the library's exemplar threshold (its ranked score is lifted to that threshold plus half of
  its overlap). The verbatim rule is the library's, unchanged;
* the overlay is read whole for every reply: its capacity (``Overlay.capacity_numbers``, two
  bytes of UTF-8 per number) is taken off the read cap of the stored library once and for all
  and is part of the ledger whether the overlay is full or empty.

Because of the last point a write cannot disturb an unrelated reply: for a message that shares
no index term with any taught entry the candidates, their scores, the numbers read and
therefore the reply are exactly what they were before the write (the stored library's document
frequencies and entry count do not move either).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Iterable

from bes.library.answers import NUMBERS_ONLY, AnswerLibrary, AnswerRead, Candidate
from bes.library.qa import contrast, contrast_mismatch, qtype, question_key
from bes.library.terms import terms

FORMAT = "bes-overlay-1"


class OverlayFull(ValueError):
    """The entry does not fit the overlay's capacity."""


def numbers_of_text(question: str, answer: str) -> int:
    """Stored numbers of one entry: its UTF-8 bytes (two line ends included), two per number."""
    return (len(question.encode("utf-8")) + len(answer.encode("utf-8")) + 2 + 1) // 2


class Overlay:
    """Appendable entries with a capacity in stored numbers and an optional file.

    ``path``: a JSON-lines file (one ``{"q", "a", "t"}`` per entry, rewritten atomically after
    every change); ``None`` keeps the entries in memory only."""

    def __init__(self, path: str | os.PathLike | None = None, capacity_numbers: int = 60_000):
        self.path = None if path is None else str(path)
        self.capacity_numbers = int(capacity_numbers)
        self.entries: list[tuple[str, str, float]] = []
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                head = f.readline()
                meta = json.loads(head) if head.strip() else {}
                if meta.get("format") != FORMAT:
                    raise ValueError(f"{self.path} is not an overlay file")
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.entries.append((str(r["q"]), str(r["a"]), float(r.get("t", 0.0))))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def numbers(self) -> int:
        return sum(numbers_of_text(q, a) for q, a, _ in self.entries)

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = f"{self.path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": FORMAT, "capacity_numbers": self.capacity_numbers}) + "\n")
            for q, a, t in self.entries:
                f.write(json.dumps({"q": q, "a": a, "t": t}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def find(self, question: str) -> int | None:
        key = question_key(question)
        for i, (q, _, _) in enumerate(self.entries):
            if question_key(q) == key:
                return i
        return None

    def add(self, question: str, answer: str) -> dict[str, Any]:
        """Write one entry. A question that is already there (same normalised question) is
        replaced: the newest answer is the one that holds. Raises :class:`OverlayFull` when the
        entry does not fit, ``ValueError`` for a question without index terms or an empty answer."""
        q = " ".join(str(question).split())
        a = str(answer).strip()
        if not a or not set(terms(q)):
            raise ValueError("an entry needs a question with at least one index term and an answer")
        i = self.find(q)
        rest = self.numbers - (numbers_of_text(*self.entries[i][:2]) if i is not None else 0)
        if rest + numbers_of_text(q, a) > self.capacity_numbers:
            raise OverlayFull(f"the overlay holds {rest} of {self.capacity_numbers} numbers; "
                              f"this entry needs {numbers_of_text(q, a)}")
        rec = (q, a, time.time())
        if i is None:
            self.entries.append(rec)
        else:
            self.entries[i] = rec
        self._save()
        return {"entry": len(self.entries) - 1 if i is None else i, "replaced": i is not None,
                "numbers": self.numbers, "capacity_numbers": self.capacity_numbers}

    def forget(self, question: str) -> bool:
        i = self.find(question)
        if i is None:
            return False
        del self.entries[i]
        self._save()
        return True

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for q, a, _ in self.entries:
            h.update(q.encode("utf-8") + b"\n" + a.encode("utf-8") + b"\n\0")
        return h.hexdigest()[:16]


class LifelongAnswers(AnswerLibrary):
    """An answers library with an overlay of taught entries (module docstring). Entry ids from
    ``n_stored`` on are the overlay's, in the order they were written."""

    min_sym: float = 0.3          # a taught entry with this much overlap is offered as an exemplar
    share_idf: float = 3.0        # ... or one that shares a term at least this rare with the message

    @classmethod
    def wrap(cls, base: AnswerLibrary, overlay: Overlay) -> "LifelongAnswers":
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.base = base
        self.overlay = overlay
        return self

    # the overlay is not pickled into loader workers: training never sees taught entries
    def __getstate__(self) -> dict:
        return self.base.__getstate__()

    def __setstate__(self, state: dict) -> None:
        AnswerLibrary.__setstate__(self, state)
        self.base, self.overlay = self, Overlay(None, 0)

    @property
    def n_stored(self) -> int:
        """Entries of the stored library (``n_entries`` stays this count: the document
        frequencies of the index refer to it)."""
        return int(self.mass.shape[0])

    @property
    def n_total(self) -> int:
        return self.n_stored + len(self.overlay)

    def new_read(self) -> AnswerRead:
        """The read log of one reply: the stored library may read what its cap leaves after the
        overlay's capacity, which every reply reads whole."""
        log = AnswerLibrary.new_read(self)
        log.cap = max(1, log.cap - int(self.overlay.capacity_numbers))
        return log

    # ------------------------------------------------------------------ writes
    def teach(self, question: str, answer: str) -> dict[str, Any]:
        return self.overlay.add(question, answer)

    def forget(self, question: str) -> bool:
        return self.overlay.forget(question)

    # ------------------------------------------------------------------ text access
    def _taught(self, e: int) -> tuple[str, str] | None:
        i = int(e) - self.n_stored
        if 0 <= i < len(self.overlay):
            return self.overlay.entries[i][:2]
        return None

    def entry(self, e: int, log: AnswerRead | None = None) -> tuple[str, str]:
        t = self._taught(e)
        return t if t is not None else AnswerLibrary.entry(self, e, log)

    def question(self, e: int, log: AnswerRead | None = None) -> str:
        t = self._taught(e)
        return t[0] if t is not None else AnswerLibrary.question(self, e, log)

    # ------------------------------------------------------------------ matching
    def _overlay_candidates(self, query: str, log: AnswerRead | None) -> list[Candidate]:
        if not len(self.overlay):
            return []
        cfg = self.acfg
        q_terms = set(terms(query))
        if not q_terms:
            return []
        wq = sum(self.idf(t) for t in q_terms)
        qt, qkey = qtype(query), question_key(query)
        need = min(cfg.exemplar_min, cfg.second_min, cfg.verbatim_cov_q, cfg.verbatim_cov_e) - 1e-9
        user_contrast = None
        out = []
        for i, (question, answer, _) in enumerate(self.overlay.entries):
            s_terms = set(terms(question))
            shared = q_terms & s_terms
            if not shared:
                continue
            m = sum(self.idf(t) for t in shared)
            we = sum(self.idf(t) for t in s_terms)
            cov_q, cov_e = m / wq, m / we
            sym = min(cov_q, cov_e)
            tm = qtype(question) == qt
            ranked = sym - (0.0 if tm else cfg.type_penalty)
            if ranked < cfg.exemplar_min and (sym >= self.min_sym or
                                              max(self.idf(t) for t in shared) >= self.share_idf):
                ranked = cfg.exemplar_min + 0.5 * sym               # taught on purpose: offer it
            mm = None
            if ranked >= need:
                if user_contrast is None:
                    user_contrast = contrast(query)
                mm = contrast_mismatch(user_contrast, contrast(question))
            miss_q = max((self.idf(t) for t in q_terms - s_terms), default=0.0)
            miss_e = max((self.idf(t) for t in s_terms - q_terms), default=0.0)
            out.append(Candidate(self.n_stored + i, question, answer, ranked, cov_q, cov_e, tm, mm,
                                 question_key(question) == qkey, miss_q, miss_e))
        return out

    def candidates(self, query: str, k: int | None = None, exclude: Iterable[int] = (),
                   log: AnswerRead | None = None, floor: float | None = None) -> list[Candidate]:
        out = AnswerLibrary.candidates(self, query, None, exclude, log, floor) + self._overlay_candidates(query, log)
        out.sort(key=lambda c: (c.contrast not in (None, NUMBERS_ONLY), -round(c.sym, 9),
                                -c.entry if c.entry >= self.n_stored else c.entry))
        return out[:k] if k is not None else out

    # ------------------------------------------------------------------ sizes
    def accounting(self) -> dict[str, int]:
        out = AnswerLibrary.accounting(self)
        cap = int(self.overlay.capacity_numbers)
        out["overlay_capacity_numbers"] = cap
        out["overlay_numbers"] = int(self.overlay.numbers)
        out["overlay_entries"] = len(self.overlay)
        out["text_numbers"] += cap
        out["total_numbers"] += cap
        return out

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((AnswerLibrary.fingerprint.fget(self) + self.overlay.digest).encode()).hexdigest()[:16]


def with_overlay(library: Any, path: str | os.PathLike | None, capacity_numbers: int = 60_000) -> Any:
    """``library`` with an overlay at ``path`` when it is an answers library; any other library
    object (or ``None``) is returned as it is."""
    if isinstance(library, LifelongAnswers) or not isinstance(library, AnswerLibrary):
        return library
    return LifelongAnswers.wrap(library, Overlay(path, capacity_numbers))


__all__ = ["Overlay", "OverlayFull", "LifelongAnswers", "with_overlay", "numbers_of_text", "FORMAT"]
