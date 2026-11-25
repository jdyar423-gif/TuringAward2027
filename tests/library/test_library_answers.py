"""Answers library (bes/library/answers.py): matching against a brute-force reference, the
verbatim rule, exemplars, parts of a message, the training mode, saving and pickling."""
import math
import pickle

import numpy as np
import pytest

from bes.exact import constraints
from bes.library.answers import AnswerCfg, AnswerLibrary, AnswerRead, cut_answer, sub_queries
from bes.library.retrieve import Library
from bes.library.terms import bucket, terms

PAIRS = [
    ("Why does ice float on water?", "Ice is less dense than liquid water, so it floats.", "a"),
    ("Why does oil float on water?", "Oil is less dense than water and does not mix with it.", "a"),
    ("What is the capital of Australia?", "The capital of Australia is Canberra.", "b"),
    ("What is the capital of France?", "The capital of France is Paris.", "b"),
    ("Write a Python function that reverses a string.", "```python\ndef reverse(s):\n    return s[::-1]\n```", "c"),
    ("Explain how a rainbow forms, step by step.", "1. Sunlight enters a raindrop.\n2. It bends and splits into colours.", "a"),
    ("What are the three primary colors?", "Red, yellow and blue.", "b"),
    ("Name three countries in Europe.", "France, Germany and Spain.", "b"),
    ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet.", "c"),
    ("Write a play about Hamlet's ghost.", "ACT I. A castle at night. The ghost enters.", "c"),
    ("How many days are in a leap year?", "A leap year has 366 days.", "a"),
    ("What is 12 times 12?", "12 times 12 is 144.", "a"),
]


@pytest.fixture(scope="module")
def lib():
    return AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=12, verify_top=12))


def brute(lib, query):
    """Exact sym of every entry, from the term strings (the module's definition)."""
    n = lib.n_entries
    sets = [set(terms(q)) for q, _, _ in PAIRS]
    df = {}
    for s in sets:
        for b in {bucket(t, lib.acfg.n_bits) for t in s}:
            df[b] = df.get(b, 0) + 1
    idf = lambda t: math.log1p((n - df.get(bucket(t, lib.acfg.n_bits), 0) + 0.5) / (df.get(bucket(t, lib.acfg.n_bits), 0) + 0.5))
    q = set(terms(query))
    wq = sum(idf(t) for t in q)
    out = []
    for s in sets:
        m = sum(idf(t) for t in q & s)
        out.append(min(m / wq, m / sum(idf(t) for t in s)) if m else 0.0)
    return np.array(out)


def test_scores_match_the_brute_force_reference(lib):
    for query in ("Why does ice float?", "capital of Australia", "reverse a string in Python", "What are the primary colors?"):
        ref = brute(lib, query)
        got = {c.entry: c for c in lib.candidates(query)}
        assert got, query
        for e, c in got.items():
            raw = min(c.cov_q, c.cov_e)
            assert raw == pytest.approx(ref[e], abs=1e-9)
        best = max(got.values(), key=lambda c: min(c.cov_q, c.cov_e))
        assert min(best.cov_q, best.cov_e) == pytest.approx(ref.max(), abs=1e-9)


def test_verbatim_needs_the_same_question(lib):
    v = lib.verbatim("What is the capital of Australia?")
    assert v is not None and "Canberra" in v.answer
    assert lib.verbatim("What's the capital of France?").answer.endswith("Paris.")
    assert lib.verbatim("What is the capital of Germany?") is None           # a name is missing
    assert lib.verbatim("Who wrote the play Hamlet?").answer.startswith("William")
    assert lib.verbatim("Why do we have leap years?") is None                # another question type
    assert lib.verbatim("What is 13 times 12?") is None                      # other numbers


def test_verbatim_respects_explicit_constraints(lib):
    q = "What is the capital of Australia?"
    spec = constraints.compile(q + " Answer in exactly two sentences.")
    assert spec.constraints
    assert lib.verbatim(q, spec) is None                                      # one sentence stored
    assert lib.verbatim(q, constraints.compile(q)) is not None


def test_exemplars_and_parts(lib):
    ex = lib.exemplars("Why does ice float on water?")
    assert ex and ex[0].question.startswith("Why does ice")
    p = lib.retrieve("Why does ice float on water?")
    assert p is not None and p.kind == "qa" and p.text.rstrip().endswith("so it floats.")   # the best is last
    assert p.text.count("Q: ") in (1, 2)
    assert lib.retrieve("Tell me about zebra crossings in Norway.") is None
    assert sub_queries("Name three countries in Europe. Use exactly three bullet points.") == ["Name three countries in Europe."]
    assert sub_queries("In all lowercase letters, tell me what the three primary colors are.") == ["tell me what the three primary colors are."]
    assert sub_queries("What is the capital of Australia?") == []
    ex = lib.exemplars("Name three countries in Europe. Use exactly three bullet points.")
    assert ex and ex[0].part and ex[0].question == "Name three countries in Europe."
    assert lib.verbatim("Name three countries in Europe. Use exactly three bullet points.") is None


def test_a_floor_shows_loosely_related_exemplars_for_every_kind_of_message(lib):
    """``floor`` replaces both of the library's floors: an entry below them is shown, and the
    default call is unchanged. (Entries under ``min_partial`` of the query's mass are no
    candidates at all, whatever the floor.)"""
    cfg = lib.acfg
    loose = "Why do boats float on the sea water?"
    c = [x for x in lib.candidates(loose, 3) if x.contrast is None]
    assert c and cfg.min_partial < c[0].sym < cfg.exemplar_min
    assert lib.exemplars(loose) == [] and lib.retrieve(loose) is None
    ex = lib.exemplars(loose, floor=c[0].sym)
    assert ex and ex[0].entry == c[0].entry and len(ex) == 1          # a second one still needs second_min
    assert lib.exemplars(loose, floor=c[0].sym + 0.01) == []
    p = lib.retrieve(loose, floor=c[0].sym)
    assert p is not None and p.text.startswith("Q: ") and p.pid == c[0].entry
    # an arithmetic message: the library's own floor for it is the higher one
    maths = "What is 12 times 12 plus 1?"
    m = [x for x in lib.candidates(maths, 3) if x.contrast is None]
    assert m and m[0].sym < cfg.exemplar_min_math and lib.exemplars(maths) == []
    assert [x.entry for x in lib.exemplars(maths, floor=m[0].sym)] == [m[0].entry]
    # a close question is shown with or without a floor, and the same way
    close = "Why does ice float on water?"
    assert [x.entry for x in lib.exemplars(close)] == [x.entry for x in lib.exemplars(close, floor=0.2)]
    assert lib.retrieve(close).text == lib.retrieve(close, floor=0.2).text
    # nothing at all is no candidate under any floor
    assert lib.exemplars("Tell me about zebra crossings in Norway.", floor=0.01) == []


def test_follow_up_turns_retrieve_nothing(lib):
    assert lib.retrieve("And what about its population?", context=["What is the capital of Australia?"]) is None
    assert lib.retrieve("What is the capital of France?", context=["Hello there."]) is not None


def test_training_mode_skips_the_own_question(lib):
    t = lib.for_training()
    assert t.train and t._questions is not None
    ex = t.exemplars("Why does ice float on water?")
    assert all(c.question != "Why does ice float on water?" for c in ex)
    assert t.verbatim("Why does ice float on water?") is None
    assert lib.verbatim("Why does ice float on water?") is not None


def test_save_load_pickle_and_dispatch(lib, tmp_path):
    d = str(tmp_path / "answers")
    lib.save(d)
    again = Library.load(d)                              # the library loader dispatches
    assert isinstance(again, AnswerLibrary) and again.n_entries == len(PAIRS)
    assert again.accounting() == lib.accounting()
    assert again.entry(4) == (PAIRS[4][0], PAIRS[4][1])
    assert again.kind_range("qa") == range(0, len(PAIRS)) and len(again.kind_range("wiki")) == 0
    clone = pickle.loads(pickle.dumps(again.for_training()))
    assert clone.train and clone._questions is not None and clone.fingerprint == lib.fingerprint
    log = AnswerRead()
    again.exemplars("Why does ice float on water?", log=log)
    assert 0 < log.total <= again.read_bound()["total"]


def test_build_rejects_duplicates_and_multiline_questions():
    with pytest.raises(ValueError):
        AnswerLibrary.build([("What is A?", "x" * 5, "s"), ("what is a", "y" * 5, "s")], AnswerCfg(n_bits=10))
    with pytest.raises(ValueError):
        AnswerLibrary.build([("What is\nA?", "x", "s")], AnswerCfg(n_bits=10))


def test_cut_answer_keeps_code_fences_closed():
    text = "Intro line.\n\n```python\n" + "x = 1\n" * 100 + "```\nDone."
    cut = cut_answer(text, 200)
    assert len(cut) <= 210 and cut.count("```") % 2 == 0
    assert cut_answer("short", 100) == "short"


def test_read_cap_bounds_what_a_reply_reads(lib):
    """One log per reply: the verbatim rule and the retrieval together never read more than
    ``read_cap`` numbers, and a cap that leaves no room returns nothing instead of reading."""
    assert lib.read_bound()["total"] == lib.acfg.read_cap
    for query in ("Why does ice float on water?", "What are the primary colors? Answer in lowercase.",
                  "In exactly three bullet points, name three countries in Europe."):
        log = lib.new_read()
        assert log.cap == lib.acfg.read_cap - lib.SMALL_READS
        lib.verbatim(query, None, log=log)
        lib.retrieve(query, log=log)
        assert 0 < log.total <= lib.acfg.read_cap and log.refused == 0
        # the same reply again on the same log reads nothing new from lists and blocks
        before = (len(log.lists), len(log.blocks), log.text_blocks)
        lib.retrieve(query, log=log)
        assert (len(log.lists), len(log.blocks), log.text_blocks) == before
    tiny = AnswerRead(cap=8)
    assert lib.retrieve("Why does ice float on water?", log=tiny) is None
    assert lib.verbatim("What is the capital of Australia?", None, log=tiny) is None
    assert tiny.refused > 0 and tiny.text_blocks == 0 and not tiny.lists
    # room for the index but not for a text block: candidates cannot be verified
    no_text = AnswerRead(cap=lib.new_read().cap)
    lib.partial("Why does ice float on water?", log=no_text)
    no_text.cap = no_text.total + 1
    assert lib.candidates("Why does ice float on water?", log=no_text) == []
    assert no_text.refused > 0


def test_long_messages_are_not_looked_up(lib):
    long = "Why does ice float on water? " + "Please consider the following text carefully. " * 30
    assert len(long) > lib.acfg.max_query_chars
    assert lib.retrieve(long) is None and lib.verbatim(long) is None and lib.retrieve_topk(long, 3) == []
    assert lib.exemplars(long) == []


def test_accounting_names_its_header_numbers(lib):
    acc = lib.accounting()
    tables = sum(v for k, v in acc.items() if k.startswith("ans_array_"))
    assert tables + acc["index_header_numbers"] == acc["total_numbers"]
    assert acc["index_header_numbers"] == AnswerLibrary.HEADER_NUMBERS
