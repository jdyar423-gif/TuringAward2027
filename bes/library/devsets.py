"""Question sets for fitting the gate (dev) and for gate E1 (evaluation), and their labels.

Dev data (every choice of the library is made on it):

* ``dev_nq`` / ``dev_tqa``: seeded samples of the *stored* NQ-open train and TriviaQA train
  questions, asked in their original wording with their own qa entry excluded
  (leave-one-out through ``exclude_pids``). Because the store and the evaluation splits come
  from the same distribution, this mimics the train/test question overlap of the evaluation
  sets without touching them.
* ``dev_probe``: head-entity probes "What is X?" / "Who was X?" / "Tell me about X." / "X"
  for important article titles X (importance = mention count), the titles split into a dev
  and an evaluation half by a hash of the title.

Labels: a wiki passage is correct when a normalised answer string occurs as a whole-token
sequence in its text (title line + body); for a probe when it belongs to the probed
article. A qa entry is correct when an answer string occurs in its stored answer
(``qa_hit``); ``qa_em`` is exact match after SQuAD normalisation (``bes.library.qa.answer_key``).
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence

from bes.library.terms import strip_disambiguator, tokens

PROBE_TEMPLATES = ("What is {}?", "Who was {}?", "Who is {}?", "Tell me about {}.",
                   "What do you know about {}?", "{}", "Explain {}.", "What are {}?")


def norm(s: str) -> str:
    """Lower-case word sequence used for answer-string matching."""
    return " ".join(tokens(s))


def answers(xs: Sequence[str]) -> list[str]:
    out: list[str] = []
    for a in xs:
        n = norm(a)
        if n and n not in out:
            out.append(n)
    return out


def contains(text: str, ans: Sequence[str]) -> bool:
    """Whether one of the normalised answers occurs as a whole-token sequence in ``text``."""
    t = " " + norm(text) + " "
    return any((" " + a + " ") in t for a in ans if a)


def qa_hit(stored_answer: str, ans: Sequence[str]) -> bool:
    return contains(stored_answer, ans)


def qa_em(stored_answer: str, ans: Sequence[str]) -> bool:
    from bes.library.qa import answer_key

    k = answer_key(stored_answer)
    return bool(k) and any(k == answer_key(a) for a in ans)


def title_half(title: str) -> str:
    """``"dev"`` or ``"eval"`` by a stable hash of the title."""
    h = hashlib.blake2b(("probe:" + title).encode("utf-8"), digest_size=8).digest()
    return "dev" if h[0] % 2 == 0 else "eval"


def probes(titles: Sequence[str], importance: Sequence[float], n: int, half: str, seed: int,
           top: int = 20000) -> list[dict]:
    """``n`` probes for the ``top`` most important titles of one half (seeded templates).

    Each probe is ``{"q", "ans": [], "tid", "title", "set"}``; titles whose disambiguator-free
    form has no letters or more than six words are skipped."""
    order = sorted(range(len(titles)), key=lambda i: (-float(importance[i]), i))
    pool = []
    for tid in order:
        t = titles[tid]
        base = strip_disambiguator(t)
        if title_half(t) != half or not any(c.isalpha() for c in base) or len(base.split()) > 6:
            continue
        pool.append(tid)
        if len(pool) >= top // 2:
            break
    rng = random.Random(seed)
    pick = rng.sample(pool, min(n, len(pool)))
    out = []
    for tid in pick:
        base = strip_disambiguator(titles[tid])
        tmpl = rng.choice(PROBE_TEMPLATES)
        out.append({"q": tmpl.format(base), "ans": [], "tid": int(tid), "title": titles[tid],
                    "set": f"{half}_probe"})
    return out
