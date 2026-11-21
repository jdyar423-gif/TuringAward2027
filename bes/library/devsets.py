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
import re
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


WHO_TEMPLATES = ("Who was {}?", "Who is {}?")
WHAT_TEMPLATES = ("What is {}?", "What are {}?", "Explain {}.")
NEUTRAL_TEMPLATES = ("Tell me about {}.", "What do you know about {}?", "{}")


def probes(titles: Sequence[str], importance: Sequence[float], n: int, half: str, seed: int,
           top: int = 20000, persons: Sequence[bool] | None = None) -> list[dict]:
    """``n`` probes for the ``top`` most important titles of one half (seeded templates).

    Each probe is ``{"q", "ans": [], "tid", "title", "set"}``; titles whose disambiguator-free
    form has no letters or more than six words are skipped. A disambiguated title is asked
    with its disambiguator ("What is Olympia (Manet)?"): its bare name may well mean another
    article, so a bare-name probe would be labelled by construction. With ``persons`` (one
    flag per title) person articles get who-templates and the others what-templates (a
    who-question about a country with a famous namesake has no single right answer);
    without it every template is drawn for every title."""
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
        t = titles[tid]                    # with its disambiguator, if it has one
        if persons is None:
            tmpl = rng.choice(PROBE_TEMPLATES)
        else:
            tmpl = rng.choice((WHO_TEMPLATES if persons[tid] else WHAT_TEMPLATES) + NEUTRAL_TEMPLATES)
        out.append({"q": tmpl.format(t), "ans": [], "tid": int(tid), "title": t, "set": f"{half}_probe"})
    return out


def name_probes(titles: Sequence[str], importance: Sequence[float], infos, n: int, half: str, seed: int,
                top: int = 4000) -> list[dict]:
    """Common-name probes: "Who was <name>?" / "Who is <name>?" for the ``top`` most important
    person articles of one half (``title_half``), asked by the name their article uses most
    (the surname, or a single name such as "Galileo"; ``bes.library.titles.TitleInfo.names``).
    A person is probed only when no more important person article bears that name (otherwise
    the question means the other one); ``importance`` should not credit a disambiguated
    title with its namesakes' mentions (``bes.library.build.name_importance``). Label: the
    person's article (``tid``). Only who-questions: "Tell me about Madison." rightly means
    the city."""
    imp = [float(x) for x in importance]
    best: dict[str, float] = {}
    for tid, info in enumerate(infos):
        if info.person:
            for w in info.names:
                best[w] = max(best.get(w, float("-inf")), imp[tid])
    order = sorted((t for t, info in enumerate(infos) if info.person and info.names),
                   key=lambda t: (-imp[t], t))
    pool = []
    for tid in order:
        if title_half(titles[tid]) != half:
            continue
        name = infos[tid].names[0]
        if best.get(name, float("-inf")) > imp[tid] or not any(c.isalpha() for c in name):
            continue
        pool.append((tid, name))
        if len(pool) >= top:
            break
    rng = random.Random(seed)
    out = []
    for tid, name in rng.sample(pool, min(n, len(pool))):
        tmpl = rng.choice(WHO_TEMPLATES)
        out.append({"q": tmpl.format(name[:1].upper() + name[1:]), "ans": [], "tid": int(tid),
                    "title": titles[tid], "name": name, "set": f"{half}_names"})
    return out


def entry_half(question: str) -> str:
    """``"dev"`` or ``"eval"`` for a stored question (contrastive questions are made from the
    dev half, the E1 perturbation table uses the evaluation half)."""
    h = hashlib.blake2b(("contrast:" + question).encode("utf-8"), digest_size=8).digest()
    return "dev" if h[0] % 2 == 0 else "eval"


_SWAPS = {
    "relation": (("before", "after"), ("after", "before")),
    "order": (("first", "second"), ("second", "third"), ("third", "first"), ("last", "first"), ("first", "last")),
    "degree": (("most", "least"), ("least", "most"), ("largest", "smallest"), ("smallest", "largest"),
               ("biggest", "smallest"), ("highest", "lowest"), ("lowest", "highest"), ("longest", "shortest"),
               ("shortest", "longest"), ("oldest", "youngest"), ("youngest", "oldest"), ("more", "less"),
               ("best", "worst"), ("worst", "best")),
}
_COUNTRIES = ("France Germany Italy Spain England Scotland Ireland Wales Canada Mexico Brazil Argentina China "
              "Japan India Russia Australia Egypt Kenya Nigeria Sweden Norway Denmark Finland Poland Austria "
              "Hungary Greece Turkey Iran Iraq Israel Portugal Belgium Netherlands Switzerland Chile Peru "
              "Cuba Jamaica Vietnam Thailand Korea Pakistan").split()
_AUX_RX = re.compile(r"\b(is|was|are|were|does|did|do|can|could|has|have|had)\b", re.I)


def _swap(q: str, a: str, b: str) -> str | None:
    rx = re.compile(r"\b" + a + r"\b", re.I)
    if not rx.search(q):
        return None
    return rx.sub(lambda m: b if m.group(0).islower() else b.capitalize(), q, count=1)


def perturb(question: str, rng: random.Random) -> list[tuple[str, str]]:
    """Seeded ``(class, variant)`` edits of a question that change the fact it asks for:
    relation (before/after), order (first/second/last), degree (most/least, largest/smallest,
    ...), negation (a "not" removed or added), number (a year or number changed) and name (a
    country replaced by another, case kept)."""
    out = []
    for cls, pairs in _SWAPS.items():
        for a, b in pairs:
            v = _swap(question, a, b)
            if v is not None:
                out.append((cls, v))
                break
    if re.search(r"\bnot\b", question, re.I):
        out.append(("negation", re.sub(r"\s*\bnot\b", "", question, count=1, flags=re.I)))
    else:
        m = _AUX_RX.search(question)
        if m:
            out.append(("negation", question[:m.end()] + " not" + question[m.end():]))
    m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", question) or re.search(r"\b(\d{1,3})\b", question)
    if m:
        x = int(m.group(1))
        y = x + rng.choice((-3, -2, 2, 3)) if x >= 1500 else x + rng.choice((1, 2, 3))
        out.append(("number", question[:m.start(1)] + str(y) + question[m.end(1):]))
    for c in _COUNTRIES:
        if re.search(r"\b" + c + r"\b", question):
            others = [x for x in _COUNTRIES if x != c and not re.search(r"\b" + x + r"\b", question, re.I)]
            out.append(("name", re.sub(r"\b" + c + r"\b", rng.choice(others), question, count=1)))
            break
    return out


def contrast_questions(questions: Sequence[str], n: int, seed: int, half: str = "dev") -> list[dict]:
    """Up to ``n`` perturbed stored questions (``perturb``) of one half (``entry_half``), at most
    one per question, classes balanced by round robin; ``{"q", "entry", "orig_q", "cls", "set"}``
    (``entry``: index into ``questions``). A variant whose normalised form is itself one of the
    ``questions`` is skipped (returning that entry would be right)."""
    from bes.library.qa import question_key

    keys = {question_key(q) for q in questions}
    rng = random.Random(seed)
    idx = [i for i, q in enumerate(questions) if entry_half(q) == half]
    rng.shuffle(idx)
    by_cls: dict[str, list[dict]] = {}
    for i in idx:
        for cls, v in perturb(questions[i], rng):
            k = question_key(v)
            if k in keys or k == question_key(questions[i]):
                continue
            by_cls.setdefault(cls, []).append({"q": v, "entry": i, "orig_q": questions[i], "cls": cls,
                                               "set": f"{half}_contrast", "ans": []})
    out, k = [], 0
    names = sorted(by_cls)
    while len(out) < n and any(by_cls.values()):
        lst = by_cls[names[k % len(names)]]
        if lst:
            out.append(lst.pop(0))
        k += 1
    return out


#: Later turns that lean on the earlier one, and chat: nothing should be looked up for them.
FOLLOWUP_TURNS = (
    "When was it built?", "Who invented it?", "How big is it?", "What is its population?", "Where is it located?",
    "Why is that so?", "How does it work?", "What happened after that?", "Is that correct?", "Who was he married to?",
    "When did she die?", "What did they do next?", "How old was she then?", "What are its main uses?",
    "Can you give an example?", "Tell me more about it.", "What does it mean?", "Which one is bigger?",
    "And the third one?", "What about the second?", "How about Spain?", "And in Japan?", "What about horses?",
    "The first one, please.", "Option B.", "Explain that again.", "Summarize it.", "What is the opposite of that?",
    "Is it dangerous?", "Is he still alive?", "What was her real name?", "What year was it?",
    "How long did it last?", "What are the benefits?", "Who wrote it?", "When was it released?", "What colour is it?",
    "How much does it cost?", "What else?", "Any more?", "Why did they do that?", "Where did he grow up?",
    "What is it made of?", "Who owns it now?", "Can you say that more simply?", "What is the next one?",
    "Is the second one better?", "What did she win?", "How tall is he?", "When did it end?", "Who won?",
    "How many are there?", "What is the capital?", "What is the population?", "Who is the author?",
    "Where is that?", "Was it a success?", "Did he win?", "What language do they speak?")
CHAT_TURNS = (
    "thanks", "thank you!", "ok", "great, thanks", "that helps", "got it, thanks", "nice", "cool", "awesome",
    "perfect, thank you", "lol", "good morning", "happy new year", "you're the best", "try again please", "go on",
    "keep going", "next", "never mind", "wrong", "that is not right", "ok cool", "hello", "hi there", "bye",
    "see you", "good night", "congratulations", "merry christmas!", "thank you very much", "yes", "no",
    "sure", "interesting", "wow", "really?", "makes sense", "sounds great", "well done", "good answer")


def conversation_negatives(first_turns: Sequence[str], seed: int) -> list[dict]:
    """Every ``FOLLOWUP_TURNS`` and ``CHAT_TURNS`` message as the second turn after a seeded
    real first turn (``context``), and every chat message also as a first turn;
    ``{"q", "context", "ans": [], "set": "dev_chat"}``. Any entry returned for them is wrong."""
    rng = random.Random(seed)
    out = []
    for q in FOLLOWUP_TURNS + CHAT_TURNS:
        out.append({"q": q, "context": [rng.choice(list(first_turns))], "ans": [], "set": "dev_chat"})
    for q in CHAT_TURNS:
        out.append({"q": q, "context": [], "ans": [], "set": "dev_chat"})
    return out
