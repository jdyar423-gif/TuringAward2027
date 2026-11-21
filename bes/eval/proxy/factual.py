"""Factual questions answered with the library on and off: the measurement behind the library
decision (``docs/EVAL_PROTOCOL.md`` section 10).

    python -m bes.eval.proxy factual --generator bes.export:export_generator \\
        --generator-arg export=RUN_DIR/final [--per-set 100] [--probes 100] [--minutes M]

The 400 proxy prompts hardly ever open the library's gate (8 of the 200 dev first turns,
``docs/E1_COVERAGE.md``), so they cannot decide whether the library ships. This command asks
our generator a fixed, seeded set of factual questions twice, with the library on (preset
``default``) and off (preset ``nolib``), with the same seeds, and compares the two replies to
every question.

Questions (:func:`build_questions`; built once and kept in ``<root>/factual/``):

- ``nq``, ``tqa``, ``popqa``, ``wq``: seeded samples of the evaluation splits the library track
  downloads under ``data_cache/library/raw`` (``bes.library.sources.QA_SOURCES`` with role
  ``eval``: NQ-open validation, TriviaQA rc.nocontext validation (unique questions), PopQA test,
  WebQuestions test). The library's build excludes every one of these questions from its qa
  store, and no training source holds them. Their answers are normalised as the library track
  does (``bes.library.devsets.answers``). Caveat from E1: many NQ-open and TriviaQA evaluation
  questions have a paraphrase among the stored train questions; PopQA and WebQuestions are the
  fairer slices.
- ``probe``: head-entity probes ("What is X?", "Tell me about X.") for the evaluation half of
  the important titles (``bes.library.devsets.probes``; the gate was fitted on the dev half),
  titles without a disambiguator only. A probe has no answer string of its own; its answers are
  the key facts of the first sentence of the probed article's lead passage (:func:`key_facts`:
  names, places, nationalities and years that the question does not contain), and a probe
  without any is skipped. **Probes are not held out from training**: the data mixture's
  reading tasks (``bes.data.rag_format.reading_task``) ask the same templates about lead
  passages of both title halves and answer with their first sentences, also without a library
  turn; read the probe slice as "does the library help the model state what it was trained
  to read", not as an out-of-distribution test.
- Every question whose normalised form equals a stored qa question of the library (checked
  through the library's own qa search, top 5) is skipped.

Scoring: a reply is right when one of the question's normalised answers occurs in it as a
whole-token sequence (``bes.library.devsets.contains``, the E1 rule). Per question the two arms
are paired: ``helped`` (right only with the library), ``hurt`` (right only without), ``same``
(both right or both wrong). Reported: those counts, the exact sign test of helped against hurt,
95% Wilson intervals of both accuracies and of helped / (helped + hurt), overall and per set,
split by whether the gate fired in the library-on arm (the reply's ``extra["library"]``). Where
the gate did not fire the two replies should be identical (same prompt, same seed); the report
counts the exceptions.

Resumable: generations go through the harness cache (``bes.eval.proxy.runner``), the question
file is fixed once written, and the arms alternate in chunks so a run stopped by ``--minutes``
has both replies for a prefix of the (set-interleaved) questions.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bes.eval.proxy.common import (data_root, read_json, read_jsonl, stable_hash, write_json_atomic,
                                   write_jsonl_atomic, write_text_atomic)
from bes.eval.proxy.prompts import Prompt, interleave

FORMAT = 1
#: question sets in report order; the QA sets name their ``QA_SOURCES`` entry
QA_SETS = {"nq": "nq_open_validation", "tqa": "trivia_qa_validation", "popqa": "popqa_test",
           "wq": "web_questions_test"}
SETS = (*QA_SETS, "probe")
SET_NAMES = {"nq": "NQ-open validation", "tqa": "TriviaQA validation", "popqa": "PopQA test",
             "wq": "WebQuestions test", "probe": "head-entity probes (evaluation half; not held out from training)"}
ARMS = ("lib", "nolib")          # presets "default" (library on) and "nolib" of the harness


@dataclass(frozen=True)
class Question:
    """One factual question: ``id`` (``factual-<set>-<hash>``), its ``set``, the question text
    ``q``, normalised ``answers`` (any one right) and, for a probe, the probed ``title``."""

    id: str
    set: str
    q: str
    answers: tuple[str, ...]
    title: str = ""

    def prompt(self) -> Prompt:
        return Prompt(id=self.id, category="factual", turns=(self.q,), split="factual")


def question_id(set_name: str, q: str) -> str:
    return f"factual-{set_name}-{stable_hash([set_name, q], 12)}"


def correct(reply: str, answers: Sequence[str]) -> bool:
    """One of the normalised ``answers`` occurs in ``reply`` as a whole-token sequence."""
    from bes.library.devsets import contains

    return bool(answers) and contains(reply, answers)


# ------------------------------------------------------------------------------ questions
def stored_question(library: Any, q: str, k: int = 5) -> bool:
    """``q`` normalises to a question of the library's qa store (among its top ``k`` qa
    candidates; an exact duplicate is always the first)."""
    if library is None:
        return False
    from bes.library.qa import clean_question, question_key

    key = question_key(clean_question(q))
    return any(question_key(clean_question(p.title)) == key
               for p in library.retrieve_topk(q, k, kinds=("qa",)))


def qa_questions(cache_dir: str | Path, set_name: str, n: int, seed: int,
                 skip: Callable[[str], bool] = lambda q: False) -> list[Question]:
    """``n`` questions of one evaluation split (seeded sample of its unique questions that have
    an answer), ``skip(q)`` ones left out (and replaced by the next ones of the sample)."""
    from bes.library import devsets as D
    from bes.library.sources import QA_SOURCES, download_qa, iter_qa

    src = QA_SOURCES[QA_SETS[set_name]]
    if src.role != "eval":
        raise ValueError(f"{src.name} is not an evaluation split")
    uniq: dict[str, list[str]] = {}
    for _, q, a in iter_qa(src, download_qa(src, str(cache_dir))):
        q = " ".join(str(q).split())
        if q and q not in uniq:
            uniq[q] = a                     # TriviaQA rc: one row per evidence document
    pool = sorted(q for q, a in uniq.items() if D.answers(a))
    random.Random(stable_hash(["factual", int(seed), set_name])).shuffle(pool)
    out = []
    for q in pool:
        if len(out) >= n:
            break
        if not skip(q):
            out.append(Question(question_id(set_name, q), set_name, q, tuple(D.answers(uniq[q]))))
    return out


_MONTHS = ("january february march april may june july august september october november december "
           "monday tuesday wednesday thursday friday saturday sunday").split()
_GENERIC = set(_MONTHS) | set("the a an it he she they this these those there his her its their in on at "
                              "for from by with as of and or but i we you".split())
_ABBREV = re.compile(r"(?:^|[\s(])(?:[A-Za-z]{1,2}|(?:[A-Za-z]\.)+[A-Za-z]|Mrs|Gen|Col|Capt|Gov|Sen|Rev|"
                     r"Prof|Inc|Ltd|Corp|No|vs|etc|approx|ca|Ave|Bros)$")


def first_sentence(text: str, max_chars: int = 400) -> str:
    """The first sentence of a passage body: up to the first ``.``, ``!`` or ``?`` followed by
    whitespace that does not end an abbreviation or an initial (``Jr.``, ``U.S.``, ``F.``)."""
    t = " ".join(text.split())[: max_chars * 2]
    for m in re.finditer(r"[.!?](?=\s|$)", t):
        if t[m.start()] == "." and _ABBREV.search(t[:m.start()]):
            continue
        return t[: m.end()][:max_chars]
    return t[:max_chars]


def key_facts(lead: str, title: str, question: str, max_facts: int = 8) -> list[str]:
    """Normalised key facts of a probed article (its answer strings): in the first sentence of
    its lead passage, the runs of capitalised words (names, places, nationalities) and the
    years 1000-2099, leaving out words of the title or the question, month and weekday names,
    capitalised function words and runs without a word of three or more characters."""
    from bes.library.devsets import norm

    banned = set(norm(title).split()) | set(norm(question).split())
    sent = first_sentence(lead)
    facts: list[str] = []

    def add(words: list[str]) -> None:
        s = norm(" ".join(words))
        if s and any(len(w) >= 3 for w in s.split()) and s not in facts:
            facts.append(s)

    run: list[str] = []
    prev_end = 0
    for i, m in enumerate(re.finditer(r"[^\s,;:()\"\[\]]+", sent)):
        if run and (sent[prev_end:m.start()].strip() or sent[m.start() - 1:m.start()] in ".!?"):
            add(run)                        # a run of names never crosses punctuation
            run = []
        prev_end = m.end()
        w = m.group(0).strip(".!?'’")
        low = norm(w)
        if re.fullmatch(r"(?:1\d|20)\d\d", w) and w not in banned:
            if run:
                add(run)
                run = []
            add([w])
            continue
        cap = bool(w) and w[0].isupper() and i > 0
        if cap and low and low not in _GENERIC and not set(low.split()) <= banned:
            run.append(w)
            if m.group(0)[-1:] in ",.;:!?":
                add(run)
                run = []
            continue
        if run:
            add(run)
            run = []
    if run:
        add(run)
    return facts[:max_facts]


def probe_questions(titles: Sequence[str], importance: Sequence[float], lead_of: Callable[[int], str],
                    n: int, seed: int, skip: Callable[[str], bool] = lambda q: False) -> list[Question]:
    """``n`` head-entity probes of the evaluation half of the titles (``devsets.probes``):
    titles without a disambiguator, with at least one key fact (:func:`key_facts`)."""
    from bes.library import devsets as D

    out = []
    for x in D.probes(titles, importance, 4 * n + 50, "eval", seed):
        if len(out) >= n:
            break
        title = x["title"]
        if "(" in title or skip(x["q"]):
            continue
        facts = key_facts(lead_of(int(x["tid"])), title, x["q"])
        if facts:
            out.append(Question(question_id("probe", x["q"]), "probe", x["q"], tuple(facts), title))
    return out


def build_questions(cache_dir: str | Path, library: Any, per_set: int, n_probes: int, seed: int,
                    sets: Sequence[str] = SETS, importance: Sequence[float] | None = None) -> list[Question]:
    """Every question set (module docstring). ``library``: the built library (stored-question
    check, titles and leads of the probes); without it no probe is built and the stored check
    is skipped (the evaluation splits are excluded from the store at build time anyway)."""
    skip = (lambda q: stored_question(library, q)) if library is not None else (lambda q: False)
    out: list[Question] = []
    for s in sets:
        if s in QA_SETS and per_set > 0:
            out += qa_questions(cache_dir, s, per_set, seed, skip)
        elif s == "probe" and n_probes > 0 and library is not None and importance is not None:
            def lead_of(tid: int) -> str:
                rng = library.title_range(tid)
                return library.passage(rng[0]).body if len(rng) else ""

            out += probe_questions(library.packed.all_titles(), importance, lead_of, n_probes, seed, skip)
    ids = [q.id for q in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate factual question ids")
    return out


def questions_path(root: Path, seed: int, per_set: int, n_probes: int, sets: Sequence[str],
                   library_id: str) -> Path:
    tag = stable_hash({"format": FORMAT, "seed": int(seed), "per_set": int(per_set), "probes": int(n_probes),
                       "sets": list(sets), "library": library_id}, 10)
    return Path(root) / "factual" / f"questions-{tag}.jsonl"


def save_questions(path: Path, questions: Sequence[Question], meta: Mapping[str, Any]) -> None:
    write_jsonl_atomic(path, [{**asdict(q), "answers": list(q.answers)} for q in questions])
    write_json_atomic(path.with_suffix(".meta.json"), {**meta, "n": len(questions),
                                                       "by_set": {s: sum(q.set == s for q in questions)
                                                                  for s in SETS}})


def load_questions(path: str | Path) -> list[Question]:
    return [Question(r["id"], r["set"], r["q"], tuple(r["answers"]), r.get("title", ""))
            for r in read_jsonl(path)]


def as_prompts(questions: Sequence[Question]) -> list[Prompt]:
    """The questions as harness prompts, interleaved across sets in proportion (a run stopped
    early covers every set)."""
    groups = [[q.prompt() for q in questions if q.set == s] for s in SETS]
    groups += [[q.prompt() for q in questions if q.set not in SETS]]
    return interleave([g for g in groups if g])


# ------------------------------------------------------------------------------- scoring
def pair_rows(questions: Sequence[Question], lib_records: Mapping[str, Mapping[str, Any]],
              nolib_records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per question both arms answered: right or wrong with and without the library,
    whether the gate fired (``None`` when the generator does not say) and the passage title."""
    rows = []
    for q in questions:
        a, b = lib_records.get(q.id), nolib_records.get(q.id)
        if a is None or b is None:
            continue
        ta, tb = a["turns"][0], b["turns"][0]
        extra = ta.get("extra") or {}
        fired = (extra.get("library") is not None) if "library" in extra else None
        rows.append({"id": q.id, "set": q.set, "lib": correct(ta["reply"], q.answers),
                     "nolib": correct(tb["reply"], q.answers), "fired": fired,
                     "passage": extra.get("library"), "identical": ta["reply"] == tb["reply"]})
    return rows


def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, accuracies with 95% Wilson intervals, paired counts and the sign test."""
    from bes.eval.proxy.score import sign_test, wilson

    n = len(rows)
    on = sum(bool(r["lib"]) for r in rows)
    off = sum(bool(r["nolib"]) for r in rows)
    helped = sum(bool(r["lib"]) and not r["nolib"] for r in rows)
    hurt = sum(bool(r["nolib"]) and not r["lib"] for r in rows)
    both = sum(bool(r["lib"]) and bool(r["nolib"]) for r in rows)

    def ci(k: int, m: int) -> list[float] | None:
        return [round(v, 4) for v in wilson(k, m)] if m else None

    return {"n": n, "lib_right": on, "lib_acc": round(on / n, 4) if n else None, "lib_ci95": ci(on, n),
            "nolib_right": off, "nolib_acc": round(off / n, 4) if n else None, "nolib_ci95": ci(off, n),
            "helped": helped, "hurt": hurt, "same": n - helped - hurt, "same_right": both,
            "same_wrong": n - helped - hurt - both,
            "helped_share": round(helped / (helped + hurt), 4) if helped + hurt else None,
            "helped_share_ci95": ci(helped, helped + hurt),
            "sign_test_p": round(sign_test(helped, hurt), 6)}


def summarize(rows: Sequence[Mapping[str, Any]], n_questions: int | None = None) -> dict[str, Any]:
    fired = [r for r in rows if r["fired"]]
    quiet = [r for r in rows if r["fired"] is False]
    out: dict[str, Any] = {
        "n_questions": n_questions if n_questions is not None else len(rows), "n_paired": len(rows),
        "gate_known": all(r["fired"] is not None for r in rows),
        "overall": block(rows), "fired": block(fired), "not_fired": block(quiet),
        "not_fired_identical": sum(bool(r["identical"]) for r in quiet),
        "by_set": {}}
    for s in SETS:
        sub = [r for r in rows if r["set"] == s]
        if sub:
            out["by_set"][s] = {"all": block(sub), "fired": block([r for r in sub if r["fired"]])}
    return out


def _row(name: str, b: Mapping[str, Any]) -> str:
    if not b["n"]:
        return f"| {name} | 0 | | | | | |"
    def acc(k: str) -> str:
        ci = b[f"{k}_ci95"]
        return f"{b[f'{k}_right']} = {b[f'{k}_acc']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
    share = "" if b["helped_share"] is None else \
        f"{b['helped_share']:.2f} [{b['helped_share_ci95'][0]:.2f}, {b['helped_share_ci95'][1]:.2f}]"
    return (f"| {name} | {b['n']} | {acc('lib')} | {acc('nolib')} | {b['helped']} / {b['hurt']} / {b['same']} | "
            f"{share} | {b['sign_test_p']:.4f} |")


def to_markdown(summary: Mapping[str, Any], title: str = "") -> str:
    s = summary
    lines = [f"# Factual questions with the library on and off{(' ' + title) if title else ''}", "",
             f"{s['n_paired']} of {s['n_questions']} questions answered by both arms (same seeds). Right = a "
             "normalised answer occurs in the reply as a whole-token sequence. helped = right only with the "
             "library, hurt = right only without; p = exact two-sided sign test of helped against hurt; "
             "intervals are 95% Wilson.", "",
             "| slice | n | library on: right | library off: right | helped / hurt / same | "
             "helped / (helped + hurt) | p |", "|---|---|---|---|---|---|---|",
             _row("all questions", s["overall"]), _row("gate fired", s["fired"]),
             _row("gate did not fire", s["not_fired"])]
    for k, v in s["by_set"].items():
        lines.append(_row(f"{SET_NAMES.get(k, k)}", v["all"]))
        lines.append(_row(f"{k}: gate fired", v["fired"]))
    q = s["not_fired"]["n"]
    lines += ["", f"Where the gate did not fire the replies were identical in {s['not_fired_identical']} of {q} "
              "questions (they should all be: same prompt, same seed)."]
    if not s["gate_known"]:
        lines.append("The generator did not report whether the gate fired for every reply "
                     "(no `extra[\"library\"]`); those questions are in neither gate slice.")
    lines += ["", "Read the 'gate fired' rows: elsewhere the library changes nothing. The probe set is not "
              "held out from training (reading tasks), and NQ-open / TriviaQA questions often have a stored "
              "paraphrase (docs/E1_COVERAGE.md); PopQA and WebQuestions are the fairer slices."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------- running
def run_factual(questions: Sequence[Question], lib_spec: Any, nolib_spec: Any, root: Path, *,
                backend: Any = None, minutes: float | None = None, chunk: int = 25, name: str = "factual",
                log: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Answer ``questions`` with both arms (``runner.RunSpec`` of presets ``default`` and
    ``nolib`` of the same generator), in chunks of ``chunk`` questions, the library-on arm
    first in each chunk, until ``minutes`` run out; then pair, score and write
    ``<root>/reports/factual/<name>.{json,md}``. Resumable: rerunning generates only what is
    missing. ``backend``: one loaded backend for both arms (built from ``lib_spec`` when
    omitted)."""
    from bes.eval.proxy import runner
    from bes.eval.proxy.common import make_backend

    root = Path(root)
    log = log or (lambda _m: None)
    if lib_spec.key != nolib_spec.key:
        raise ValueError("both arms must use the same generator (same cache key)")
    prompts = as_prompts(questions)
    backend = backend if backend is not None else make_backend(lib_spec.backend)
    deadline = time.time() + 60.0 * minutes if minutes else None
    stopped = False
    for start in range(0, len(prompts), max(1, int(chunk))):
        part = prompts[start:start + max(1, int(chunk))]
        for spec in (lib_spec, nolib_spec):
            left = None if deadline is None else (deadline - time.time()) / 60.0
            if left is not None and left <= 0:
                stopped = True
                break
            runner.run(part, spec, root, workers=0, minutes=left, all_prompts=prompts, log=None, backend=backend)
        if stopped:
            break
    for spec in (lib_spec, nolib_spec):          # outputs exist even when nothing ran
        runner.assemble(spec, root, prompts)
    lib_out, _ = runner.load_outputs(root, lib_spec.run_id)
    nolib_out, _ = runner.load_outputs(root, nolib_spec.run_id)
    summary = summarize(pair_rows(questions, lib_out, nolib_out), len(questions))
    summary.update({"lib_run": lib_spec.run_id, "nolib_run": nolib_spec.run_id, "stopped_early": stopped})
    base = root / "reports" / "factual" / name
    write_json_atomic(base.with_suffix(".json"), summary)
    write_text_atomic(base.with_suffix(".md"), to_markdown(summary, f"(`{name}`)"))
    log(f"[factual] {summary['n_paired']} of {len(questions)} questions paired; gate fired on "
        f"{summary['fired']['n']}; helped {summary['overall']['helped']}, hurt {summary['overall']['hurt']}")
    return summary


def default_cache_dir() -> Path:
    """Where the library track keeps ``raw/`` (``data_cache/library``)."""
    return data_root() / "library"


def default_library_dir() -> Path:
    from bes.config import LibraryCfg
    from bes.eval.proxy.common import REPO_ROOT

    return REPO_ROOT / LibraryCfg().library_dir


def prepare(root: Path, *, per_set: int, n_probes: int, seed: int, sets: Sequence[str] = SETS,
            cache_dir: str | Path | None = None, library_dir: str | Path | None = None,
            questions: str | Path | None = None, log: Callable[[str], None] | None = print) -> tuple[Path, list[Question]]:
    """The fixed question file for these parameters: read when it exists, built (and saved) once
    otherwise. ``questions``: use that file instead. The library (``library_dir``, default
    ``LibraryCfg.library_dir``) gives the probes and the stored-question check: when probes are
    asked for it must load (``ValueError`` otherwise); without probes a library that is missing
    or does not load only skips the check (recorded in the question file's meta)."""
    log = log or (lambda _m: None)
    if questions:
        return Path(questions), load_questions(questions)
    lib, importance, library_id = None, None, "none"
    ldir = Path(library_dir) if library_dir else default_library_dir()
    want_probes = "probe" in sets and n_probes > 0
    try:
        import numpy as np

        from bes.library.retrieve import Library

        if not (ldir / "library.json").exists():
            raise FileNotFoundError(f"no built library at {ldir}")
        lib = Library.load(str(ldir))
        library_id = stable_hash((ldir / "library.json").read_text(), 12)
        imp = ldir / "provenance" / "title_importance.npy"      # layout of older builds
        if imp.exists():
            importance = np.load(imp)
        else:
            from bes.library.build import load_provenance
            try:
                importance = load_provenance(str(ldir), "title_importance")
            except OSError:
                importance = None
        if want_probes and importance is None:
            raise FileNotFoundError("the title importance of the probes is missing "
                                    f"(provenance of the library at {ldir})")
    except (OSError, ValueError) as err:
        if want_probes:
            raise ValueError(f"the head-entity probes need the built library: {err} (pass --library DIR, "
                             "or leave the probes out with --sets nq,tqa,popqa,wq)") from err
        lib, library_id = None, "none"
        log(f"[factual] {err}: no stored-question check")
    path = questions_path(root, seed, per_set, n_probes, sets, library_id)
    if path.exists():
        return path, load_questions(path)
    qs = build_questions(cache_dir or default_cache_dir(), lib, per_set, n_probes, seed, sets, importance)
    save_questions(path, qs, {"format": FORMAT, "seed": int(seed), "per_set": int(per_set),
                              "probes": int(n_probes), "sets": list(sets), "library": library_id,
                              "library_dir": str(ldir) if lib is not None else None,
                              "stored_check": lib is not None,
                              "built": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"[factual] {len(qs)} questions written to {path}")
    return path, qs


# ------------------------------------------------------------------------------- stand-in
def standin_factual(questions: str, behaviour: str = "", key: str = "standin-factual",
                    fire: str | float = 0.5, memory: str | float = 0.3, mislead: str | float = 0.1):
    """A generator for tests and rehearsals (no model): it knows the answers of ``questions``
    (a question file). Per question it "remembers" the answer, the gate "fires" when the
    library is on, and a fired passage "misleads" it, as given in ``behaviour`` (a JSON file
    ``{question: {"memory", "fires", "misleads"}}``) or else by a hash of the question
    (probabilities ``memory``, ``fire``, ``mislead``). Right replies state the first answer."""
    answers = {q.q: q.answers for q in load_questions(questions)}
    table = read_json(behaviour) if behaviour else {}

    def u(q: str, what: str) -> float:
        return int(stable_hash([what, q], 8), 16) / 16 ** 8

    def generate(messages, settings, seed):
        q = messages[-1]["content"]
        b = table.get(q) or {"memory": u(q, "memory") < float(memory), "fires": u(q, "fires") < float(fire),
                             "misleads": u(q, "misleads") < float(mislead)}
        fired = bool(settings.get("use_library")) and bool(b["fires"])
        right = (not b["misleads"]) if fired else bool(b["memory"])
        ans = (answers.get(q) or ("",))[0]
        text = f"The answer is {ans}." if right and ans else "I am not sure about that."
        return {"text": text, "finish_reason": "stop", "completion_tokens": len(text.split()),
                "extra": {"library": f"passage for {q[:20]}" if fired else None}}

    generate.key = key            # type: ignore[attr-defined]
    return generate


__all__ = ["Question", "SETS", "QA_SETS", "ARMS", "build_questions", "qa_questions", "probe_questions",
           "key_facts", "first_sentence", "stored_question", "prepare", "load_questions", "save_questions",
           "as_prompts", "correct", "pair_rows", "block", "summarize", "to_markdown", "run_factual",
           "standin_factual"]
