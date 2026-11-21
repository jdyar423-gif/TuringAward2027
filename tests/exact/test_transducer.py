"""Tests of the byte-level transducer (bes.exact.transducer).

``independent_reference`` below is written without the transducer's state machine: it works in
separate passes over the whole text (fence lines first, then backtick runs found by looking
back, quotes, words by looking back, terminator runs found by looking ahead, list markers by
matching each line) and reports, for every byte, the events counted at it and the state after
it. The streaming transducer must agree with it byte for byte.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from bes.exact import transducer as tr

WS = b" \t\n\r\x0b\x0c"
CLOSERS = b"\"')]}"


def alnum(b: int) -> bool:
    return b >= 0x80 or chr(b).isalnum()


# ----------------------------------------------------------------- independent reference
def independent_reference(data: bytes) -> dict:
    """Per-byte events and state of one turn, computed in independent passes."""
    n = len(data)
    lines, start = [], 0
    for i, b in enumerate(data):
        if b == 0x0A:
            lines.append((start, i))
            start = i + 1
    lines.append((start, n))

    def content_start(a: int, e: int) -> int:
        while a < e and data[a] in b" \t":
            a += 1
        return a

    # Fences: the third backtick of a run of three or more at the start of a line.
    toggles = set()
    for a, e in lines:
        p = content_start(a, e)
        if p + 3 <= e and data[p:p + 3] == b"```":
            toggles.add(p + 2)
    fence_after, f = [], 0
    for i in range(n):
        f ^= int(i in toggles)
        fence_after.append(f)
    fence_before = [0] + fence_after[:-1]

    # Inline code: the first two backticks of every run flip it, outside fences.
    inline_after, x = [], 0
    for i in range(n):
        if data[i] == 0x60:
            k, j = 1, i - 1
            while j >= 0 and data[j] == 0x60:
                k, j = k + 1, j - 1
            if i in toggles:
                x = 0
            elif k <= 2 and not fence_before[i]:
                x ^= 1
        inline_after.append(x)
    inline_before = [0] + inline_after[:-1]

    quote_after, q = [], 0
    for i in range(n):
        if i in toggles:
            q = 0
        elif data[i] == 0x22 and not fence_before[i] and not inline_before[i]:
            q ^= 1
        quote_after.append(q)

    word = [int(alnum(data[i]) and not (i > 0 and (alnum(data[i - 1]) or (
        data[i - 1] == 0x27 and i > 1 and alnum(data[i - 2]))))) for i in range(n)]
    newline = [int(b == 0x0A) for b in data]

    # Line-start markers (only on lines that start outside a fence).
    item = [0] * n
    marker_dot, marker_paren = set(), set()
    for a, e in lines:
        if (fence_after[a - 1] if a > 0 else 0) or a >= e:
            continue
        p = content_start(a, e)
        m = None
        if p < e and data[p] in b"-*+":
            m = p + 1
        elif data[p:p + 3] == "•".encode():
            m = p + 3
        else:
            d = p
            while d < e and 0x30 <= data[d] <= 0x39:
                d += 1
            if d > p and d < e and data[d] in b".)":
                (marker_dot if data[d] == 0x2E else marker_paren).add(d)
                m = d + 1
        if m is not None and m < e and data[m] in b" \t":
            item[m] = 1

    # Sentences: maximal runs of terminators, decided by the byte after any emphasis bytes.
    term = [data[i] in b".!?" and not fence_before[i] and i not in marker_dot for i in range(n)]
    sentence, pending_at_end, i = [0] * n, False, 0
    while i < n:
        if not term[i]:
            i += 1
            continue
        j = i
        while j < n and term[j]:
            j += 1
        if i > 0 and data[i - 1] not in WS:
            r = j
            while r < n and data[r] in b"*_":
                r += 1
            if r == n:
                pending_at_end = True
            elif data[r] in WS or data[r] in CLOSERS:
                sentence[r] = 1
        i = j

    depth, d = [], 0
    for i in range(n):
        if not fence_before[i] and not inline_before[i]:
            if data[i] in b"([{":
                d += 1
            elif data[i] in b")]}" and i not in marker_paren:
                d = max(0, d - 1)
        depth.append(d)

    return {"fence": fence_after, "inline": inline_after, "quote": quote_after, "word": word,
            "sentence": sentence, "newline": newline, "item": item, "depth": depth,
            "pending_at_end": pending_at_end}


def streaming(data: bytes) -> dict:
    out = {k: [] for k in ("fence", "inline", "quote", "word", "sentence", "newline", "item", "depth")}
    state, d = tr.INIT, 0
    for b in data:
        state, ev = tr.step(state, b)
        d = max(0, d + ev.bracket)
        for k in ("fence", "inline", "quote"):
            out[k].append(getattr(state, k))
        for k in ("word", "sentence", "newline", "item"):
            out[k].append(getattr(ev, k))
        out["depth"].append(d)
    out["pending_at_end"] = bool(tr.flush(state)[1].sentence)
    return out


ATOMS = [b"a", b"b", b"Z", b"1", b"42", b" ", b" ", b"  ", b"\t", b"\n", b"\n", b"\r", b".", b"!",
         b"?", b"...", b"`", b"``", b"```", b"````", b'"', b"'", b"(", b")", b"[", b"]", b"{",
         b"}", b"*", b"**", b"_", b"-", b"+", b",", b":", b"#", b"\xe2\x80\xa2", b"\xe2\x80\x94",
         b"\xc3\xa9", b"\xe2", b"\x80", b"\xa2", b"1. ", b"- ", b"\n```\n", b"don't", b"3.14"]


def random_text(rng: random.Random, n_atoms: int) -> bytes:
    return b"".join(rng.choice(ATOMS) for _ in range(n_atoms))


HANDWRITTEN = [
    b"", b"Hello world. This is it.", b"Wait... what?! No.", b"e.g. this and 3.14 too",
    b"1. First\n2. Second\n10) Tenth\n- dash\n* star\n+ plus\n\xe2\x80\xa2 bullet\n",
    b"```python\nprint(\"hi\")  # (x\n```\nDone.", b"  ```\n  code.\n  ```\nAfter.",
    b"`a\n```\nx\n```", b"a ``` b `c` d", b'He said "Stop." Then left.', b"**Bold.** Next",
    b"(Yes.) No. [ok] {x", b"x  ... y . z", b"\n\n- a\n -b\n- \n", b"1.5\n1.\n1. \n",
    b"it's 'quoted' 'tis don''t", b"\xe2\x80\x94 dash \xe2\x80\x9cquote\xe2\x80\x9d",
]


@pytest.mark.parametrize("data", HANDWRITTEN)
def test_streaming_equals_independent_reference_on_handwritten_text(data):
    assert streaming(data) == independent_reference(data)


def test_streaming_equals_independent_reference_on_random_text():
    rng = random.Random(1)
    for _ in range(3000):
        data = random_text(rng, rng.randint(1, 40))
        assert streaming(data) == independent_reference(data), data


def test_streaming_equals_independent_reference_on_random_bytes():
    rng = random.Random(2)
    for _ in range(500):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(1, 60)))
        assert streaming(data) == independent_reference(data), data


# ------------------------------------------------------------------- hand-written counts
def counts(text: str | bytes, final: bool = True) -> dict:
    return tr.measure_counts(text, final=final)


@pytest.mark.parametrize("text, words", [
    ("", 0), ("hello", 1), ("don't stop", 2), ("'yes' and 'no'", 3), ("it's 3.14", 3),
    ("naïve café", 2), ("a-b c_d", 4), ("x''y", 2), ("e.g.", 2), ("  spaced   out  ", 2),
    ("```\ncode here\n```", 2), ("`inline code`", 2), ("1. one", 2),
])
def test_word_counts(text, words):
    assert counts(text)["words"] == words


@pytest.mark.parametrize("text, sentences", [
    ("Hi. Bye.", 2), ("Hi.Bye", 0), ("3.14 is pi.", 1), ("Wait... no", 1), ("Wait...no", 0),
    ("What?! No!", 2), (" ... x", 0), ("end . x", 0), ("(Yes.)", 1), ('"Go!" he said.', 2),
    ("'Go.' ok", 1), ("**Bold.** text", 1), ("Hello.*", 1), ("_Note._x", 0),
    ("```\nx. y. z.\n```", 0), ("```\nx. y.", 0), ("1. Item\n2. Item\n", 0), ("Done.\n", 1),
    ("a.\tb", 1), ("v1.2.3 now", 0), ("Ok.[1]", 0), ("Hi!!!", 1), ("x: y; z, w.", 1),
])
def test_sentence_counts(text, sentences):
    assert counts(text)["sentences"] == sentences


def test_sentence_needs_the_end_of_turn_to_flush():
    assert counts("Hello.", final=False) == {**counts("Hello."), "sentences": 0, "word_in_sent": 1,
                                             "pending_sentence": 1}
    assert counts("Hello", final=False)["pending_sentence"] == 0


@pytest.mark.parametrize("text, lines", [("", 0), ("a", 0), ("a\nb\n\nc", 3), ("\n", 1),
                                         ("```\n\n```\n", 3), ("a\r\nb", 1)])
def test_line_counts(text, lines):
    assert counts(text)["lines"] == lines


@pytest.mark.parametrize("text, items", [
    ("- a\n* b\n+ c\n• d\n1. e\n10) f\n  - g\n\t* h", 8), ("-a\n*b*\n1.5\n--- \n**x**", 0),
    ("```\n- x\n1. y\n```\n- z", 1), ("x - y", 0), ("- ", 1), ("-\n", 0), ("1.\tx", 1),
    ("a\n 2) b", 1), ("12 . x", 0), ("#. x", 0), ("•x", 0), ("— x", 0),
])
def test_item_counts(text, items):
    assert counts(text)["items"] == items


@pytest.mark.parametrize("text, depth", [
    ("((a)", 1), ("a))(", 1), ("`(`", 0), ("```\n(\n```", 0), ("1) x", 0), ("(1) x", 0),
    ("[{(", 3), ('"(" x', 1), ("`(` (", 1), ("(\n```\n)\n```", 1),
])
def test_bracket_depth(text, depth):
    assert counts(text)["depth"] == depth


@pytest.mark.parametrize("text, fence, inline, quote", [
    ("```py\ncode\n```", 0, 0, 0), ("```py\ncode", 1, 0, 0), ("  ```\n", 1, 0, 0),
    ("a```", 0, 0, 0), ("````\nx\n```", 0, 0, 0), ("`x`", 0, 0, 0), ("`x", 0, 1, 0),
    ("``x", 0, 0, 0), ("a ``` b", 0, 0, 0), ('say "hi', 0, 0, 1), ('say "hi"', 0, 0, 0),
    ('`"`', 0, 0, 0), ('`"', 0, 1, 0), ('"a\n```\nb', 1, 0, 0), ('```\n"\n```', 0, 0, 0),
    ("`a\n```\nx", 1, 0, 0), ('"`"', 0, 1, 1),
])
def test_final_formatting_state(text, fence, inline, quote):
    c = counts(text)
    assert (c["fence"], c["inline"], c["quote"]) == (fence, inline, quote)


def test_inline_state_is_eager_for_single_backticks():
    states = [s for s, _ in tr.trace(b"a `b` ``c`` ```d``` e")]
    inline = "".join(str(s.inline) for s in states)
    # bytes:           a _ ` b ` _ ` ` c ` ` _ ` ` ` d ` ` ` _ e
    assert inline == "0 0 1 1 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0".replace(" ", "")
    assert [s.ticks for s, _ in tr.trace(b"`````x")] == [1, 2, 3, 3, 3, 0]


def test_word_in_sentence_restarts_after_each_sentence():
    assert counts("One two. Three four five")["word_in_sent"] == 3
    assert counts("One two. Three four five.")["word_in_sent"] == 0
    assert counts("One two. Three four five.", final=False)["word_in_sent"] == 3


def test_measure_counts_batch_equals_measure_counts():
    rng = random.Random(4)
    texts = [random_text(rng, rng.randint(0, 80)) for _ in range(700)] + HANDWRITTEN
    texts += ["café. Done", "", "Hello.", "x " * 900]
    for final in (True, False):
        assert tr.measure_counts_batch(texts, final=final, chunk=64) == [
            tr.measure_counts(t, final=final) for t in texts]
    assert tr.measure_counts_batch([]) == []


def test_measure_counts_batch_on_real_text():
    from bes.tokenizer import samples

    (f,) = samples.files_for("chat", "heldout")
    if not f.local_path.exists():
        pytest.skip("samples not downloaded (python scripts/build_tokenizer.py)")
    texts = list(samples.iter_texts("chat", "heldout", 400_000))
    assert tr.measure_counts_batch(texts) == [tr.measure_counts(t) for t in texts]


def test_measure_counts_accepts_bytes_and_reports_all_fields():
    c = counts('Intro:\n- a (b\n```\ncode "x\n')
    assert c == {"words": 5, "sentences": 0, "lines": 4, "items": 1, "word_in_sent": 5,
                 "depth": 1, "fence": 1, "inline": 0, "quote": 0, "pending_sentence": 0}
    assert counts(b"Hi. There.") == counts("Hi. There.")


# ---------------------------------------------------------------------- compiled machine
def test_compiled_machine_tabulates_step_and_flush_exactly():
    m = tr.compile_machine()
    assert m.states[0] == tr.INIT and m.init == 0
    assert len(set(m.states)) == m.n_states <= 255
    for i, s in enumerate(m.states):
        assert tuple(m.components[i]) == tuple(s)
        for b in range(256):
            s2, ev = tr.step(s, b)
            assert m.next_state[i, b] == m.index(s2)
            assert tuple(m.events[i, b]) == tuple(ev)
        s2, ev = tr.flush(s)
        assert m.flush_state[i] == m.index(s2) and m.flush_sentence[i] == ev.sentence


def test_machine_is_closed_and_every_state_is_reachable():
    m = tr.compile_machine()
    seen, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in list(m.next_state[i]) + [m.flush_state[i]]:
            if int(j) not in seen:
                seen.add(int(j))
                frontier.append(int(j))
    assert seen == set(range(m.n_states))


def test_fingerprint_is_deterministic_and_tracks_the_tables():
    m = tr.compile_machine()
    tr.compile_machine.cache_clear()
    m2 = tr.compile_machine()
    assert m2.fingerprint == m.fingerprint
    assert np.array_equal(m2.next_state, m.next_state) and np.array_equal(m2.events, m.events)
    assert len(m.fingerprint) == 64


def test_trace_and_run_agree():
    rng = random.Random(3)
    for _ in range(200):
        data = random_text(rng, 30)
        tr_ = tr.trace(data)
        c = tr.run(data)
        assert c.state == (tr_[-1][0] if tr_ else tr.INIT)
        assert c.words == sum(ev.word for _, ev in tr_)
        assert c.sentences == sum(ev.sentence for _, ev in tr_)
        assert c.lines == data.count(b"\n")
        assert c.items == sum(ev.item for _, ev in tr_)
