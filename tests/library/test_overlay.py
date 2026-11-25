"""Taught entries on top of an answers library (bes/library/overlay.py)."""
import json

import pytest

from bes.exact import constraints
from bes.library.answers import AnswerCfg, AnswerLibrary
from bes.library.overlay import LifelongAnswers, Overlay, OverlayFull, numbers_of_text, with_overlay

PAIRS = [
    ("Why does ice float on water?", "Ice is less dense than liquid water, so it floats.", "a"),
    ("Why does oil float on water?", "Oil is less dense than water and does not mix with it.", "a"),
    ("What is the capital of Australia?", "The capital of Australia is Canberra.", "b"),
    ("What is the capital of France?", "The capital of France is Paris.", "b"),
    ("Write a Python function that reverses a string.", "```python\\ndef reverse(s):\\n    return s[::-1]\\n```", "c"),
    ("Explain how a rainbow forms, step by step.", "1. Sunlight enters a raindrop.\\n2. It bends and splits into colours.", "a"),
    ("What are the three primary colors?", "Red, yellow and blue.", "b"),
    ("Name three countries in Europe.", "France, Germany and Spain.", "b"),
    ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet.", "c"),
    ("How many days are in a leap year?", "A leap year has 366 days.", "a"),
    ("What is 12 times 12?", "12 times 12 is 144.", "a"),
]
QUERIES = ["Why does ice float?", "capital of Australia", "reverse a string in Python", "What are the primary colors?",
           "Who wrote Hamlet?", "How many days does a leap year have?", "Tell me about rainbows and how they form."]


@pytest.fixture()
def base():
    return AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=12, verify_top=12, read_cap=50_000))


def snapshot(lib, query):
    """Everything a reply takes from the library for one message."""
    log = lib.new_read()
    spec = constraints.compile(query)
    v = lib.verbatim(query, spec, log=log)
    p = lib.retrieve(query, log=log)
    return (None if v is None else (v.entry, v.answer), None if p is None else (p.pid, p.text, p.score),
            [(c.entry, c.sym, c.cov_q, c.cov_e) for c in lib.candidates(query)], log.total, log.refused)


def test_an_overlay_keeps_entries_within_its_capacity_and_across_restarts(tmp_path):
    path = tmp_path / "memory" / "overlay.jsonl"
    ov = Overlay(path, capacity_numbers=60)
    r = ov.add("What is  my dog's name?", " Your dog's name is Pemberly. ")
    assert r == {"entry": 0, "replaced": False, "numbers": ov.numbers, "capacity_numbers": 60}
    assert ov.entries[0][:2] == ("What is my dog's name?", "Your dog's name is Pemberly.")
    assert ov.numbers == numbers_of_text("What is my dog's name?", "Your dog's name is Pemberly.")
    r = ov.add("what is my dog's name", "Your dog is called Biscuit now.")           # the same question: replaced
    assert r["replaced"] and len(ov) == 1 and ov.entries[0][1] == "Your dog is called Biscuit now."
    with pytest.raises(OverlayFull):
        ov.add("What is the name of the street where my grandmother lived for forty years?", "Larkspur Lane. " * 8)
    with pytest.raises(ValueError):
        ov.add("???", "an answer")
    with pytest.raises(ValueError):
        ov.add("What is my cat's name?", "   ")
    again = Overlay(path, capacity_numbers=60)
    assert [e[:2] for e in again.entries] == [e[:2] for e in ov.entries] and again.digest == ov.digest
    assert again.forget("What is my dog's name?") and len(Overlay(path)) == 0 and not again.forget("anything else here")
    path.write_text(json.dumps({"format": "something else"}) + "\\n")
    with pytest.raises(ValueError):
        Overlay(path)


def test_a_taught_entry_is_found_by_its_question_and_by_a_paraphrase(base):
    lib = with_overlay(base, None, 2000)
    assert isinstance(lib, LifelongAnswers) and with_overlay(lib, None) is lib and with_overlay(None, None) is None
    q = "Can you remind me what my dog is called?"
    assert lib.retrieve(q) is None and lib.verbatim(q, constraints.compile(q)) is None
    lib.teach("What is my dog's name?", "Your dog's name is Pemberly.")
    hit = lib.verbatim("What is my dog's name?", constraints.compile("What is my dog's name?"))
    assert hit is not None and hit.answer == "Your dog's name is Pemberly." and hit.entry == lib.n_stored
    p = lib.retrieve(q)                                    # a paraphrase: offered as an exemplar
    assert p is not None and "Pemberly" in p.text and p.pid == lib.n_stored
    assert lib.entry(lib.n_stored) == ("What is my dog's name?", "Your dog's name is Pemberly.")
    assert lib.question(lib.n_stored) == "What is my dog's name?"
    lib.teach("What is my dog's name?", "Your dog is called Biscuit.")
    assert "Biscuit" in lib.retrieve(q).text and "Pemberly" not in lib.retrieve(q).text
    assert lib.forget("What is my dog's name?") and lib.retrieve(q) is None


def test_a_write_changes_nothing_for_messages_that_share_no_term_with_it(base):
    """The text half of the exact locality: candidates, scores, the numbers read and the
    chosen entries of unrelated messages are identical before and after the writes."""
    lib = with_overlay(base, None, 2000)
    before = {q: snapshot(lib, q) for q in QUERIES}
    lib.teach("What is my dog's name?", "Your dog's name is Pemberly.")
    lib.teach("Which bakery does my sister Odalys own?", "Odalys owns the Marigold Bakery on Fennel Street.")
    assert {q: snapshot(lib, q) for q in QUERIES} == before
    assert lib.n_entries == base.n_entries and lib.n_total == base.n_entries + 2      # the index's counts do not move
    # and the stored library reads within its cap minus the overlay's capacity, written or not
    assert lib.new_read().cap == base.new_read().cap - 2000
    assert snapshot(lib, "Which bakery is owned by Odalys?")[1] is not None


def test_the_order_of_writes_does_not_matter(base):
    a, b = with_overlay(base, None, 2000), with_overlay(base, None, 2000)
    facts = [("What is my dog's name?", "Your dog's name is Pemberly."),
             ("Which bakery does my sister Odalys own?", "Odalys owns the Marigold Bakery."),
             ("What is the door code of the workshop?", "The workshop's door code is 4417.")]
    for f in facts:
        a.teach(*f)
    for f in reversed(facts):
        b.teach(*f)
    for q in ("What is my dog called?", "Which bakery does Odalys own?", "door code of the workshop", *QUERIES):
        pa, pb = a.retrieve(q), b.retrieve(q)
        assert (pa is None) == (pb is None) and (pa is None or (pa.title, pa.body, pa.score) == (pb.title, pb.body, pb.score))


def test_the_capacity_is_in_the_ledger_whether_used_or_not(base):
    lib = with_overlay(base, None, 2000)
    a0, a1 = base.accounting(), lib.accounting()
    assert a1["total_numbers"] == a0["total_numbers"] + 2000 and a1["overlay_capacity_numbers"] == 2000
    lib.teach("What is my dog's name?", "Your dog's name is Pemberly.")
    assert lib.accounting()["total_numbers"] == a1["total_numbers"] and lib.accounting()["overlay_entries"] == 1
    assert lib.read_bound()["total"] == base.read_bound()["total"]
    assert lib.fingerprint != with_overlay(base, None, 2000).fingerprint


def test_the_chat_commands_write_to_the_overlay(base):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bes_chat", Path(__file__).resolve().parents[2] / "scripts" / "chat.py")
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)
    lib = with_overlay(base, None, 2000)
    lines = iter(["/teach What is my dog's name? || Your dog's name is Pemberly.", "/teach only a question",
                  "What is my dog called?", "/forget What is my dog's name?", "/forget What is my dog's name?",
                  "/quit"])
    said, asked = [], []

    def reply(msgs, settings, seed):
        asked.append(msgs[-1]["content"])
        p = lib.retrieve(msgs[-1]["content"])
        return {"text": "from the overlay: " + p.body if p is not None else "no idea"}, None

    n = chat.chat_session(reply, {}, input_fn=lambda prompt: next(lines), print_fn=said.append,
                          teach_fn=lib.teach, forget_fn=lib.forget)
    assert n == 1 and asked == ["What is my dog called?"]
    text = "\n".join(said)
    assert "(remembered; the overlay holds" in text and "usage: /teach QUESTION || ANSWER" in text
    assert "bes> from the overlay: Your dog's name is Pemberly." in text
    assert "(forgotten)" in text and "(no such entry)" in text and len(lib.overlay) == 0
    said.clear()
    lines2 = iter(["/teach a || b", "/quit"])
    chat.chat_session(reply, {}, input_fn=lambda prompt: next(lines2), print_fn=said.append)
    assert "cannot be taught" in "\n".join(said)


def test_a_taught_entry_is_a_verbatim_reply_only_by_its_own_overlap(base):
    """Under the rank rule (``verbatim(min_rank=...)``) a taught entry counts with its own overlap:
    the lift that offers it as an exemplar does not make it the reply to a loosely related message."""
    lib = with_overlay(base, None, 400)
    lib.teach("What is my dog's name?", "Your dog's name is Pemberly.")
    loose = "What should I feed my dog in winter?"
    c = [x for x in lib.candidates(loose) if x.entry >= lib.n_stored]
    assert c and c[0].sym >= lib.acfg.exemplar_min                      # lifted: offered as an exemplar
    assert min(c[0].cov_q, c[0].cov_e) < 0.5
    assert lib.verbatim(loose, constraints.compile(loose), min_rank=0.5) is None
    assert lib.verbatim(loose, constraints.compile(loose)) is None
    same = "What is my dog's name?"
    hit = lib.verbatim(same, constraints.compile(same), min_rank=0.5)
    assert hit is not None and hit.answer == "Your dog's name is Pemberly." and hit.entry >= lib.n_stored
