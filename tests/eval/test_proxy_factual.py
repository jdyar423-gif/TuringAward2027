"""`python -m bes.eval.proxy factual`: factual questions with the library on and off.

Raw question files are written here in the layout of ``bes.library.sources.QA_SOURCES`` (so the
real loaders read them), probes come from a fake library, and the generator is the stand-in
whose behaviour per question is given explicitly, so every expected count is derived here from
that table, independently of the module's own counting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bes.eval.proxy import factual as F
from bes.eval.proxy.__main__ import main
from bes.eval.proxy.common import read_jsonl


# ------------------------------------------------------------------------------ raw data
def _parquet(path: Path, cols: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def raw_cache(tmp: Path, n: int = 30) -> Path:
    """Tiny copies of the four evaluation splits, with a question without answers and a
    TriviaQA question repeated (one row per evidence document)."""
    from bes.library.sources import QA_SOURCES as Q

    cache = tmp / "library"
    raw = cache / "raw"
    s = Q["nq_open_validation"]
    _parquet(raw / s.dir / s.filename, {"question": [f"who wrote book {i}" for i in range(n)] + ["no answer here"],
                                        "answer": [[f"Author {i}", f"A. {i}"] for i in range(n)] + [[]]})
    s = Q["trivia_qa_validation"]
    qs = [f"Which city hosts event {i}?" for i in range(n)]
    _parquet(raw / s.dir / s.filename, {"question": qs + qs[:5],
                                        "answer": [{"value": f"City {i}", "aliases": [f"The City {i}"],
                                                    "normalized_aliases": [f"city {i}"]} for i in range(n)]
                                        + [{"value": f"City {i}", "aliases": [], "normalized_aliases": []}
                                           for i in range(5)]})
    s = Q["popqa_test"]
    path = raw / s.dir / s.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["id\tquestion\tpossible_answers\ts_pop"] + [
        f"{i}\tWhat is the occupation of person {i}?\t{json.dumps([f'job {i}', f'trade {i}'])}\t{i}" for i in range(n)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    s = Q["web_questions_test"]
    _parquet(raw / s.dir / s.filename, {"question": [f"where is place {i}?" for i in range(n)],
                                        "answers": [[f"Country {i}"] for i in range(n)]})
    return cache


def test_questions_of_the_evaluation_splits_are_a_seeded_sample_with_normalised_answers(tmp_path):
    cache = raw_cache(tmp_path)
    nq = F.qa_questions(cache, "nq", 10, seed=0)
    assert len(nq) == 10 and len({q.id for q in nq}) == 10
    assert all(q.q != "no answer here" and q.set == "nq" for q in nq)
    i = int(nq[0].q.split()[-1])
    assert nq[0].answers == (f"author {i}", f"a {i}")                       # devsets.answers normalisation
    assert [q.id for q in F.qa_questions(cache, "nq", 10, seed=0)] == [q.id for q in nq]      # fixed
    assert [q.id for q in F.qa_questions(cache, "nq", 10, seed=1)] != [q.id for q in nq]
    tqa = F.qa_questions(cache, "tqa", 40, seed=0)
    assert len(tqa) == 30                                                     # repeated rows counted once
    assert "the city 3" in next(q for q in tqa if q.q.endswith("event 3?")).answers
    pop = F.qa_questions(cache, "popqa", 5, seed=0)
    assert all(len(q.answers) == 2 for q in pop)
    skipped = F.qa_questions(cache, "wq", 10, seed=0, skip=lambda q: q.endswith("place 7?"))
    assert len(skipped) == 10 and all(not q.q.endswith("place 7?") for q in skipped)
    built = F.build_questions(cache, None, 6, 5, seed=0)
    assert [q.set for q in built].count("nq") == 6 and not any(q.set == "probe" for q in built)


def test_first_sentence_and_key_facts():
    lead = ("Francis Scott Key Fitzgerald (September 24, 1896 - December 21, 1940) was an Irish American "
            "writer. He is remembered for The Great Gatsby.")
    assert F.first_sentence(lead).endswith("Irish American writer.")
    assert F.first_sentence("George Perry Floyd Jr. (born 1973) was an American man. He died.") == \
        "George Perry Floyd Jr. (born 1973) was an American man."
    assert F.first_sentence("It is in Washington, D.C. since 1994. More text.").endswith("1994.")
    facts = F.key_facts(lead, "F. Scott Fitzgerald", "Explain F. Scott Fitzgerald.")
    assert "1896" in facts and "1940" in facts and "irish american" in facts
    assert not any(f in ("september", "december", "fitzgerald", "great gatsby") for f in facts)
    assert F.key_facts("The Pacific Ocean is the largest ocean.", "Pacific Ocean", "What is Pacific Ocean?") == []
    assert F.key_facts("Lyon is a city in France, near Switzerland.", "Lyon", "Tell me about Lyon.") == \
        ["france", "switzerland"]
    assert F.key_facts("The 1984 Summer Olympics were held in Los Angeles, California, United States.",
                       "1984 Summer Olympics", "What is 1984 Summer Olympics?") == \
        ["los angeles", "california", "united states"]          # runs end at commas; title words left out


class FakeLib:
    """Titles, leads and a qa search that knows one stored question."""

    def __init__(self, stored=("What is Lyon?",)):
        self.stored = stored

    def retrieve_topk(self, q, k, kinds=None):
        from types import SimpleNamespace
        return [SimpleNamespace(title=s) for s in self.stored][:k]


def test_probes_of_the_evaluation_half_with_their_key_facts():
    from bes.library import devsets as D

    titles = [f"Town {i}" for i in range(300)] + ["Mercury (planet)"]
    importance = list(range(len(titles), 0, -1))
    leads = {i: f"Town {i} is a town in Countryland, founded in {1500 + i}." for i in range(300)}
    leads[300] = "Mercury is the smallest planet in the Solar System."
    qs = F.probe_questions(titles, importance, lambda tid: leads[tid], 20, seed=3)
    assert len(qs) == 20 and all(q.set == "probe" for q in qs)
    assert all(D.title_half(q.title) == "eval" and "(" not in q.title for q in qs)
    for q in qs:
        i = int(q.title.split()[-1])
        assert q.answers == ("countryland", str(1500 + i))
    # a probe equal to a stored question is left out
    lib = FakeLib(stored=(qs[0].q,))
    again = F.probe_questions(titles, importance, lambda tid: leads[tid], 20, seed=3,
                              skip=lambda q: F.stored_question(lib, q))
    assert qs[0].q not in {q.q for q in again} and len(again) == 20
    assert F.stored_question(FakeLib(), "what is LYON") and not F.stored_question(FakeLib(), "What is Paris?")
    assert not F.stored_question(None, "anything")


# ------------------------------------------------------------------------------- running
def _questions(tmp: Path, n: int = 16):
    qs = [F.Question(F.question_id("wq" if i % 2 else "nq", f"Question number {i}?"), "wq" if i % 2 else "nq",
                     f"Question number {i}?", (f"answer {i}",)) for i in range(n)]
    behaviour = {q.q: {"memory": i % 3 == 0, "fires": i % 2 == 0 or i == 5, "misleads": i % 4 == 0}
                 for i, q in enumerate(qs)}
    qfile, bfile = tmp / "questions.jsonl", tmp / "behaviour.json"
    F.save_questions(qfile, qs, {"test": True})
    bfile.write_text(json.dumps(behaviour))
    return qs, behaviour, qfile, bfile


def expected(qs, behaviour, ids=None):
    """Right with and without the library, from the stand-in's documented rule."""
    out = {}
    for q in qs:
        if ids is not None and q.id not in ids:
            continue
        b = behaviour[q.q]
        on = (not b["misleads"]) if b["fires"] else b["memory"]
        out[q.id] = (on, b["memory"], b["fires"])
    return out


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def wilson_ref(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _cli(root, qfile, bfile, *extra):
    return main(["--root", str(root), "factual", "--generator", "bes.eval.proxy.factual:standin_factual",
                 "--generator-arg", f"questions={qfile}", "--generator-arg", f"behaviour={bfile}",
                 "--questions", str(qfile), "--name", "t", *extra])


def test_library_on_and_off_paired_counts_sign_test_and_gate_split(tmp_path, capsys):
    qs, behaviour, qfile, bfile = _questions(tmp_path)
    assert _cli(tmp_path / "root", qfile, bfile, "--chunk", "5") == 0
    out = capsys.readouterr().out
    s = json.loads((tmp_path / "root" / "reports" / "factual" / "t.json").read_text())
    exp = expected(qs, behaviour)
    for name, keep in (("overall", lambda f: True), ("fired", lambda f: f), ("not_fired", lambda f: not f)):
        rows = [v for v in exp.values() if keep(v[2])]
        helped = sum(on and not off for on, off, _ in rows)
        hurt = sum(off and not on for on, off, _ in rows)
        b = s[name]
        assert (b["n"], b["helped"], b["hurt"], b["same"]) == (len(rows), helped, hurt, len(rows) - helped - hurt)
        assert b["lib_right"] == sum(on for on, _, _ in rows) and b["nolib_right"] == sum(off for _, off, _ in rows)
        assert b["sign_test_p"] == pytest.approx(sign_p(helped, hurt), abs=1e-6)
        if rows:
            assert b["lib_ci95"] == wilson_ref(b["lib_right"], len(rows))
    assert s["not_fired"]["helped"] == s["not_fired"]["hurt"] == 0
    assert s["not_fired_identical"] == s["not_fired"]["n"]           # same prompt, same seed, no passage
    assert s["n_paired"] == s["n_questions"] == len(qs) and s["gate_known"]
    assert set(s["by_set"]) == {"nq", "wq"} and s["by_set"]["nq"]["all"]["n"] == 8
    assert "| gate fired |" in out and "helped / hurt / same" in out
    # the two runs differ only in the library switch
    lib_meta = json.loads(next((tmp_path / "root" / "outputs").glob("*/factual-lib-*.meta.json")).read_text())
    nolib_meta = json.loads(next((tmp_path / "root" / "outputs").glob("*/factual-nolib-*.meta.json")).read_text())
    assert lib_meta["settings"]["extra"]["use_library"] is True and nolib_meta["settings"]["extra"]["use_library"] is False
    assert {k: v for k, v in lib_meta["settings"].items() if k != "extra"} == \
        {k: v for k, v in nolib_meta["settings"].items() if k != "extra"}


def test_resumable_and_stoppable(tmp_path, capsys):
    qs, behaviour, qfile, bfile = _questions(tmp_path)
    root = tmp_path / "root"
    cache = root / "cache" / "standin-factual.jsonl"
    assert _cli(root, qfile, bfile, "--minutes", "1e-9") == 0            # out of time at once
    s = json.loads((root / "reports" / "factual" / "t.json").read_text())
    assert s["n_paired"] == 0 and s["stopped_early"] and not cache.exists()
    assert _cli(root, qfile, bfile, "--limit", "6") == 0
    assert len(read_jsonl(cache)) == 12                                  # 6 questions x 2 arms
    s = json.loads((root / "reports" / "factual" / "t.json").read_text())
    first = [p.id for p in F.as_prompts(qs)][:6]
    assert s["n_paired"] == 6 and {"nq", "wq"} == set(s["by_set"])        # sets interleaved
    assert s["overall"]["helped"] == sum(on and not off for on, off, _ in expected(qs, behaviour, set(first)).values())
    assert _cli(root, qfile, bfile) == 0
    assert len(read_jsonl(cache)) == 2 * len(qs)                         # only the missing ones
    assert _cli(root, qfile, bfile) == 0
    assert len(read_jsonl(cache)) == 2 * len(qs)                         # nothing new
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _cli(root, qfile, bfile, "--set", "extra.use_library=false")


def test_probes_need_a_library_that_loads(tmp_path):
    with pytest.raises(ValueError, match="probes need the built library"):
        F.prepare(tmp_path, per_set=1, n_probes=5, seed=0, sets=("probe",), library_dir=tmp_path / "missing")
    cache = raw_cache(tmp_path)
    path, qs = F.prepare(tmp_path / "root", per_set=3, n_probes=0, seed=0, sets=("nq", "popqa"),
                         cache_dir=cache, library_dir=tmp_path / "missing", log=None)
    assert len(qs) == 6 and path.exists()
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["stored_check"] is False and meta["by_set"]["nq"] == 3
    again_path, again = F.prepare(tmp_path / "root", per_set=3, n_probes=0, seed=0, sets=("nq", "popqa"),
                                  cache_dir=tmp_path / "nowhere", library_dir=tmp_path / "missing", log=None)
    assert again_path == path and [q.id for q in again] == [q.id for q in qs]   # read back, not rebuilt
