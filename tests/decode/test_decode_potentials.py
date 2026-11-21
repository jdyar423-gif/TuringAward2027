"""Token-local potentials (bes.decode.potentials) against independent brute force.

References written here:

- ``accepted(data, spec)``: text-level acceptance of a complete response, from the byte-level
  transducer's reference ``step`` (fence, inline code, quote, bracket depth), plain Python
  checks (trim, control bytes, the ``***`` line restriction) and ``constraints.verify``;
- exhaustive search over token sequences of small synthetic vocabularies: a token is allowed
  iff some completion within the remaining budget is accepted (the potentials must agree
  exactly, budget-aware);
- on the real tokenizer, a backward breadth-first search over byte-level states built from
  ``transducer.compile_machine`` (byte tables, not the token tables) for unbounded
  reachability of word / sentence / item constraints.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache

import numpy as np
import pytest

from bes.decode import potentials as P
from bes.decode.lm import TinyTokenizer
from bes.exact import constraints as C
from bes.exact import registers as R
from bes.exact import transducer as tr
from bes.exact.constraints import ConstraintSpec, Interval
from bes.tokenizer import load

WS = b" \t\n\r\x0b\x0c"
ENFORCED = ("words", "sentences", "items", "lines", "paragraphs", "no_commas", "end_question",
            "code_block")


# ----------------------------------------------------------------------- references
def _r1_violation(text: str) -> bool:
    """A line outside a fence whose body (trailing CRs stripped) starts with spaces, tabs and
    at least three stars and then goes on with anything else."""
    fence = False
    for line in text.split("\n"):
        if not fence:
            body = line.rstrip("\r")
            n = len(body) - len(body.lstrip(" \t*"))
            if body[:n].count("*") >= 3 and n < len(body):
                return True
        if re.match(r"[ \t]*```", line):
            fence = not fence
    return False


def accepted(data: bytes, spec, max_depth: int = 8) -> bool:
    """Reference acceptance of a complete response (single-bracket tokens: the depth bound
    checked after every byte equals the bound checked after every token)."""
    if not data or data[0] in WS or data[-1] in WS:
        return False
    if any((b < 32 and b not in (9, 10)) or b == 127 for b in data):
        return False
    st, depth = tr.INIT, 0
    for b in data:
        st, ev = tr.step(st, b)
        depth = max(0, depth + ev.bracket)
        if depth > max_depth:
            return False
    if st.fence or st.inline or st.quote or depth:
        return False
    if spec is None:
        return True
    text = data.decode("utf-8", errors="replace")
    if spec.paragraphs is not None and _r1_violation(text):
        return False
    bad = [x for x in C.verify(spec, text) if x.split(":")[0] in ENFORCED]
    return not bad


# --------------------------------------------------------------- line machine vs measure
def test_line_machine_matches_measure():
    rng = random.Random(0)
    alpha = [" ", "\t", "\r", "*", "-", "_", "`", "a", "\n", "\x0b", "?", "\"", "»", "”", "x.",
             "   ", "***", "```", "- "]
    lm = P.LineMachine(lines=True, paragraphs=True, fenced=True)
    eq = P.EndQuestionMachine()
    n_dead = 0
    for _ in range(25000):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 12)))
        st, ev, ok = P.run_machine(lm, s.encode())
        m = C.measure(s)
        # the paragraph restriction forbids exactly the lines that start with >= 3 stars
        # (only spaces, tabs and stars before) and then go on with something else
        restricted = _r1_violation(s)
        assert ok == (not restricted), repr(s)
        if not ok:
            n_dead += 1
            continue
        lines, pars, chunks = ev
        has_div, fenced = lm.flag(st)
        assert lines == m["lines"], repr(s)
        assert (chunks if has_div else pars) == m["paragraphs"], repr(s)
        mc = tr.measure_counts(s)
        assert (fenced and mc["fence"] == 0) == (m["code_blocks"] >= 1 and not m["open_fence"]), repr(s)
        st2, _, _ = P.run_machine(eq, s.encode())
        assert eq.accept(st2) == m["ends_with_question"], repr(s)
    assert n_dead > 0


def test_aux_token_tables_compose_the_byte_machine():
    tok = load()
    tb = tok.all_token_bytes()
    machine = P.ProductMachine([P.LineMachine(lines=True, paragraphs=True, fenced=True),
                                P.EndQuestionMachine()])
    aux = P.aux_token_tables(P.compile_aux(machine, range(256)), tb)
    rng = np.random.default_rng(1)
    for _ in range(300):
        a = int(rng.integers(aux.n))
        v = int(rng.integers(tok.vocab_size - 16)) if rng.random() < 0.95 else tok.vocab_size
        data = tb[v] if v < tok.vocab_size else b" = "
        st, tot, ok = aux.states[a], [0, 0, 0], True
        for b in data:
            out = machine.step(st, b)
            if out is None:
                ok = False
                break
            st, e = out
            tot = [x + y for x, y in zip(tot, e)]
        if not ok:
            assert aux.next_tok[v, a] == aux.dead
            continue
        assert aux.states[aux.next_tok[v, a]] == st
        assert list(aux.ev_tok[v, a]) == tot


# ---------------------------------------------------------- tracker vs measure / registers
@pytest.mark.parametrize("unit", ["words", "sentences", "items", "lines", "paragraphs"])
def test_tracker_counts_match_measure_and_registers(unit):
    """Token by token on the real tokenizer: the transducer state and the bracket depth equal
    ``registers.compute``; after END every count equals ``constraints.measure``."""
    tok = load()
    prog = P.compile(ConstraintSpec(**{unit: Interval(0, 60 if unit != "paragraphs" else 12)}), tok)
    rng = random.Random(2)
    chunks = ["Hello world. ", "Yes!\n", "- item one\n", "1. first\n", "\n\n", "(a) ", "`x` ",
              "\"q\" ", "e.g. ", "Done.", "***\n", "a,b ", "```\ncode.\n```\n", "  ", "Why? ",
              "word", "* * *\n", "---\n", "\t- sub\n"]
    n_checked = 0
    for _ in range(40):
        text = "".join(rng.choice(chunks) for _ in range(rng.randint(1, 8))).lstrip()
        if unit == "paragraphs" and _r1_violation(text):
            continue
        ids = tok.encode(text)
        trk = prog.tracker(1, 10_000)
        regs = R.compute(np.array([[tok.assistant] + ids]), table=prog.table)
        for i, t in enumerate(ids):
            trk.advance(np.array([t]))
            assert int(trk.s[0]) == int(regs.raw["state"][0, i + 1])
            assert int(trk.d[0]) == int(regs.raw["depth"][0, i + 1])
        a = int(trk.a[0][0])
        trk.advance(np.array([prog.ids["end"]]))
        got = trk.counts(0)
        m = C.measure(text)
        if unit == "paragraphs":
            has_div = bool(prog.parts[0].auto.aux.flush_flag[a, 0])
            value = got["chunks" if has_div else "pars"]
        else:
            value = got[unit]
        assert value == m[unit], (text, got, m[unit])
        n_checked += 1
    assert n_checked >= 20


# ------------------------------------------------------------ exhaustive tiny vocabularies
def _tree_check(pieces, spec, horizon, max_check_len, max_depth=8):
    """Compare the budget-aware masks with exhaustive search over every prefix."""
    tok = TinyTokenizer(pieces)
    prog = P.compile(spec, tok, max_depth=max_depth)
    ids = list(range(len(pieces)))
    end = prog.ids["end"]
    tb = [p.encode() for p in pieces]

    @lru_cache(maxsize=None)
    def exists(data: bytes, budget: int) -> bool:
        if budget <= 0:
            return False
        if accepted(data, spec, max_depth):
            return True
        return budget > 1 and any(exists(data + tb[v], budget - 1) for v in ids)

    checked = 0
    stack = [((), b"", prog.tracker(1, horizon))]
    others = [t for t in range(tok.vocab_size) if t >= len(pieces) and
              t not in (end, prog.ids["calc"])]
    while stack:
        seq, data, trk = stack.pop()
        mask = trk.allowed()[0]
        rem = horizon - len(seq)
        assert mask[end] == (rem >= 1 and accepted(data, spec, max_depth)), (seq, data)
        assert not mask[others].any()
        for v in ids:
            want = exists(data + tb[v], rem - 1)
            assert mask[v] == want, (seq, data, pieces[v], rem)
            checked += 1
            if want and len(seq) + 1 <= max_check_len:
                child = trk.copy()
                child.advance(np.array([v]))
                stack.append((seq + (v,), data + tb[v], child))
    return checked


SPECS = [
    ("structure", ["a", "(", ")", "`", "\"", " "], None, 6),
    ("2 sentences", ["a", " b", ".", "?", ")", "\n"], ConstraintSpec(sentences=Interval(2, 2)), 7),
    ("<=3 words", ["a", " b", "a b", ".", "\n", "-"], ConstraintSpec(words=Interval(0, 3)), 6),
    ("1-2 sent, >=2 words", ["a", " b", ".", " ", "!"],
     ConstraintSpec(sentences=Interval(1, 2), words=Interval(2, None)), 6),
    ("2 items", ["- ", "a", "\n", " ", "1."], ConstraintSpec(items=Interval(2, 2)), 7),
    ("2 lines", ["a", "\n", " ", "-"], ConstraintSpec(lines=Interval(2, 2)), 7),
    ("2 paragraphs", ["a", "\n", " ", "***", "*", "---"], ConstraintSpec(paragraphs=Interval(2, 2)), 7),
    ("<=1 paragraph", ["a", "\n", "***", " *", "**"], ConstraintSpec(paragraphs=Interval(0, 1)), 6),
    ("end question", ["a", "?", "\"", ")", " "], ConstraintSpec(end_question=True), 6),
    ("code block", ["`", "```", "\n", "a"], ConstraintSpec(code_block=True), 8),
    ("no commas", ["a", ",", " b", "."], ConstraintSpec(no_commas=True, sentences=Interval(1, 1)), 6),
]


@pytest.mark.parametrize("name,pieces,spec,horizon", SPECS, ids=[s[0] for s in SPECS])
def test_reachability_is_exact_on_tiny_vocabularies(name, pieces, spec, horizon):
    checked = _tree_check(pieces, spec, horizon, max_check_len=4)
    assert checked > 50


def test_depth_bound_on_tiny_vocabulary():
    assert _tree_check(["a", "(", ")", " "], None, 6, 4, max_depth=1) > 20
    assert _tree_check(["a", "[", "]", "(", ")"], None, 6, 4, max_depth=2) > 20


# ------------------------------------------------- real tokenizer: byte-level backward search
def _byte_good(units: dict, max_depth: int, no_commas: bool = False):
    """Good set of byte-level states ``(s, depth, counts...)`` with unbounded budget: forward
    enumeration over all non-control bytes with the per-byte tables of
    ``transducer.compile_machine``, then backward reachability of an accepting flush."""
    m = tr.compile_machine()
    nxt, ev = m.next_state.astype(np.int64), m.events.astype(np.int64)
    comp = m.components.astype(np.int64)
    names = list(units)
    fld = {"words": 0, "sentences": 1, "items": 3}
    sizes = [(hi if hi is not None else lo) + 1 for lo, hi in (units[n] for n in names)]
    radix = [m.n_states, max_depth + 1] + sizes
    bytes_ok = np.array([b for b in range(256) if not ((b < 32 and b not in (9, 10)) or b == 127)
                         and not (no_commas and b == 44)])

    def enc(cols):
        code = np.zeros_like(cols[0])
        for c, r in zip(cols, radix):
            code = code * r + c
        return code

    def dec(code):
        out = []
        for r in reversed(radix):
            out.append(code % r)
            code = code // r
        return out[::-1]

    seen = {0}
    frontier = np.array([0], np.int64)
    src_l, dst_l = [], []
    while frontier.size:
        s, d, *c = dec(frontier)
        e = ev[s[:, None], bytes_ok[None, :]]                     # [f, B, 5]
        d2 = np.maximum(0, d[:, None] + e[..., 4])
        ok = d2 <= max_depth
        c2 = []
        for n, x, (lo, hi) in zip(names, c, (units[n] for n in names)):
            y = x[:, None] + e[..., fld[n]]
            if hi is not None:
                ok &= y <= hi
            else:
                y = np.minimum(y, lo)
            c2.append(y)
        s2 = nxt[s[:, None], bytes_ok[None, :]]
        code = enc([s2, np.minimum(d2, max_depth)] + [np.clip(y, 0, sz - 1) for y, sz in zip(c2, sizes)])
        src_l.append(np.broadcast_to(frontier[:, None], code.shape)[ok])
        dst_l.append(code[ok])
        new = np.unique(code[ok])
        new = np.array([x for x in new.tolist() if x not in seen], np.int64)
        seen.update(new.tolist())
        frontier = new
    allc = np.array(sorted(seen), np.int64)
    s, d, *c = dec(allc)
    acc = (comp[s, 0] == 0) & (comp[s, 1] == 0) & (comp[s, 2] == 0) & (d == 0)
    acc &= comp[s, 4] != tr.PREV_SPACE
    for n, x in zip(names, c):
        lo, hi = units[n]
        y = x + (m.flush_sentence[s].astype(np.int64) if n == "sentences" else 0)
        acc &= (y >= lo) & ((y <= hi) if hi is not None else True)
    pos = {int(code): i for i, code in enumerate(allc)}
    src = np.searchsorted(allc, np.concatenate(src_l))
    dst = np.searchsorted(allc, np.concatenate(dst_l))
    good = acc.copy()
    while True:
        new = good.copy()
        new[src[good[dst]]] = True
        if np.array_equal(new, good):
            break
        good = new
    return pos, good, nxt, ev, enc


REAL = [
    ("2 sentences", {"sentences": (2, 2)}, False),
    ("<=5 words", {"words": (0, 5)}, False),
    ("3 items", {"items": (3, 3)}, False),
    ("2 sent + <=6 words", {"sentences": (2, 2), "words": (0, 6)}, False),
    (">=4 words + 1 sent", {"words": (4, None), "sentences": (1, 1)}, False),
    ("no commas + 2 items", {"items": (2, 2)}, True),
]


@pytest.mark.parametrize("name,units,no_commas", REAL, ids=[r[0] for r in REAL])
def test_reachability_matches_byte_level_search_on_the_real_tokenizer(name, units, no_commas):
    tok = load()
    spec = ConstraintSpec(**{n: Interval(lo, hi) for n, (lo, hi) in units.items()},
                          no_commas=no_commas)
    prog = P.compile(spec, tok, budget_aware=False)
    index, good, nxt, ev, enc = _byte_good(units, 8, no_commas)
    names = list(units)
    fld = {"words": 0, "sentences": 1, "items": 3}
    tb = tok.all_token_bytes()
    n_ord = tok.vocab_size - 16
    rng = random.Random(3)
    chunks = ["Hi", " there", ".", " ", "\n", "- ", "1. ", "(", ")", "`", "\"", "!", "a", "?",
              "**", "e.g.", ", "]

    def run(state, data):
        s, d, *c = state
        for b in data:
            e = ev[s, b]
            d = max(0, d + int(e[4]))
            c = [x + int(e[fld[n]]) for n, x in zip(names, c)]
            s = int(nxt[s, b])
        if d > 8:
            return None
        c2 = []
        for n, x in zip(names, c):
            lo, hi = units[n]
            if hi is not None and x > hi:
                return None
            c2.append(min(x, lo) if hi is None else x)
        return (s, d) + tuple(c2)

    n_states = 0
    for _ in range(12):
        ids = tok.encode("".join(rng.choice(chunks) for _ in range(rng.randint(0, 5))).lstrip())
        trk = prog.tracker(1, 10_000)
        state = (0, 0) + (0,) * len(names)
        for t in ids:
            if not trk.allowed()[0, t]:
                break
            trk.advance(np.array([t]))
            state = run(state, tb[t])
        else:
            n_states += 1
            mask = trk.allowed()[0]
            ref = np.zeros(tok.vocab_size, bool)
            for v in range(n_ord):
                if any((b < 32 and b not in (9, 10)) or b == 127 for b in tb[v]):
                    continue
                if no_commas and b"," in tb[v]:
                    continue
                st2 = run(state, tb[v])
                if st2 is not None:
                    code = int(enc([np.array(x) for x in st2]))
                    ref[v] = bool(good[index[code]])
            if trk.n[0] == 0:
                ref[[v for v in range(n_ord) if tb[v][0] in WS]] = False
            diff = np.flatnonzero(mask[:n_ord] != ref[:n_ord])
            assert diff.size == 0, (name, ids, [tb[v] for v in diff[:5]])
    assert n_states >= 3


def _walks(prog, judge, n_walks, horizon, rng, nice=None):
    """Random walks choosing uniformly among the allowed tokens; returns how many walks ended
    with an accepted END and how many got stuck (stranded: no completion exists any more, as
    the exact programme ``judge`` says, or the budget ran out)."""
    ends = stuck = 0
    end = prog.ids["end"]
    for _ in range(n_walks):
        trk, ref = prog.tracker(1, horizon), judge.tracker(1, horizon)
        while True:
            mask = trk.allowed()[0]
            ids = np.flatnonzero(mask & nice) if nice is not None else np.flatnonzero(mask)
            if ids.size == 0:
                ids = np.flatnonzero(mask)          # the preferred subset is only a preference
            if ids.size == 0 or ref.distance()[0] > ref.remaining()[0]:
                stuck += 1
                break
            t = int(rng.choice(ids))
            trk.advance(np.array([t]))
            ref.advance(np.array([t]))
            if t == end:
                ends += 1
                break
    return ends, stuck


def test_naive_masking_strands_particles_and_exact_masks_never_do():
    """Masking only immediate violations strands walks; exact (budget-aware) pruning never
    does. Tiny vocabulary without word-free sentences; real tokenizer with a token budget."""
    rng = np.random.default_rng(4)
    tok = TinyTokenizer(["a", " b", ".", "\n"])
    spec = ConstraintSpec(sentences=Interval(2, 2), words=Interval(0, 2))
    exact = P.compile(spec, tok)
    naive = P.compile(spec, tok, reachability=False)
    e_ends, e_stuck = _walks(exact, exact, 200, 12, rng)
    n_ends, n_stuck = _walks(naive, exact, 200, 12, rng)
    assert e_stuck == 0 and e_ends == 200
    assert n_stuck >= 100, (n_ends, n_stuck)
    real = load()
    spec = ConstraintSpec(sentences=Interval(3, 3), words=Interval(0, 12))
    exact = P.compile(spec, real)
    naive = P.compile(spec, real, reachability=False)
    tb = real.all_token_bytes()
    nice = np.array([0 < len(b) <= 3 and all(32 <= c < 127 for c in b) for b in tb])
    nice[exact.ids["end"]] = True
    e_ends, e_stuck = _walks(exact, exact, 40, 24, rng, nice)
    n_ends, n_stuck = _walks(naive, exact, 40, 24, rng, nice)
    assert e_stuck == 0 and e_ends == 40
    assert n_stuck >= 20, (n_ends, n_stuck)


# ------------------------------------------------------------------------- calculator
def test_calculator_spans_masks_forcing_and_fallback():
    pieces = ["1", "2", "3", "+", " ", "a", ".", "*"]
    tok = TinyTokenizer(pieces)
    prog = P.compile(None, tok)
    sp = tok.special_tokens
    calc, eq, ce, end = sp["<|calc|>"], sp["<|eq|>"], sp["<|/calc|>"], sp["<|end|>"]
    trk = prog.tracker(1, 20)
    m = trk.allowed()[0]
    assert m[calc] and not m[eq] and not m[ce]
    trk.advance(np.array([calc]))
    m = trk.allowed()[0]
    assert not m[4] and not m[5] and m[0] and m[3]         # no leading space, no letters
    assert not m[end] and not m[calc] and not m[eq]        # empty expression: no EQ
    for t in (0, 3):                                       # "1+"
        trk.advance(np.array([t]))
    assert not trk.allowed()[0, eq]                        # does not evaluate
    trk.advance(np.array([1]))                             # "1+2"
    assert trk.allowed()[0, eq]
    probe = trk.copy()
    probe.advance(np.array([4]))                           # trailing space: EQ masked
    assert not probe.allowed()[0, eq]
    adv = trk.advance(np.array([eq]))
    assert adv.forced[0] == [2, ce]                        # "3" then CALC_END
    assert trk.calc[0] == 2 and not trk.allowed()[0].any()  # forced span: no mask asked
    for t in adv.forced[0]:
        trk.advance(np.array([t]))
    assert trk.calc[0] == 0 and trk.allowed()[0, end]      # "1+2 = 3" may end
    # without potentials the host still forces, with the fallback for a bad expression
    raw = P.Tracker(None, 1, 20, table=prog.table, tok=tok)
    for t in (calc, 0, 3):
        raw.advance(np.array([t]))
    assert raw.advance(np.array([eq])).forced[0] == [ce]
    # budget: EQ needs room for the forced tokens and an END
    short = prog.tracker(1, 6)
    for t in (calc, 0, 3, 1):
        short.advance(np.array([t]))
    assert not short.allowed()[0, eq]


def _syntactically_valid(x: str) -> bool:
    """The calculator's grammar checked with ``ast`` (numbers, + - * / // % **, unary signs,
    parentheses)."""
    import ast

    try:
        tree = ast.parse(x, mode="eval")
    except SyntaxError:
        return False
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
          ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)
    return all(isinstance(n, ok) for n in ast.walk(tree)) and all(
        isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
        for n in ast.walk(tree) if isinstance(n, ast.Constant))


def test_expression_recogniser_matches_the_grammar():
    """Dead prefixes have no valid completion (checked over many suffixes), viable prefixes
    have one, and the complete states are exactly the valid expressions."""
    rng = random.Random(7)
    chars = "0123456789.+-*/%() "
    closes = [")" * k for k in range(6)]
    suffixes = [""] + [a + c for a in ("1", ".5", "1.5", " 1", "*1", "+1", "1+1", "0", ".1")
                       for c in closes] + closes
    n_dead = n_viable = 0
    for _ in range(4000):
        e = "".join(rng.choice(chars) for _ in range(rng.randint(1, 7)))
        if e[0] == " ":
            continue                  # a leading space is masked by another rule
        mode, depth = P.X_ATOM, 0
        for b in e.encode():
            mode, depth = P._calc_step(mode, depth, b) if mode != P.X_DEAD else (P.X_DEAD, depth)
        found = any(_syntactically_valid(e + suf) for suf in suffixes)
        if mode == P.X_DEAD:
            n_dead += 1
            assert not found, e
        else:
            n_viable += 1
            assert found, e
            assert (mode in P.X_COMPLETE and depth == 0) == _syntactically_valid(e), e
    assert n_dead > 500 and n_viable > 500


# ------------------------------------------------------------- split programs, minimisation
def test_split_program_is_sound():
    tok = load()
    spec = ConstraintSpec(paragraphs=Interval(2, 2), words=Interval(0, 8))
    joint = P.compile(spec, tok)
    split = P.compile(spec, tok, joint_limit=1000)
    assert joint.joint and not split.joint and len(split.parts) == 2
    rng = random.Random(5)
    chunks = ["Hi", " there", ".", "\n", "\n\n", "***", " ", "a"]
    for _ in range(15):
        ids = tok.encode("".join(rng.choice(chunks) for _ in range(rng.randint(0, 6))).lstrip())
        tj, ts = joint.tracker(1, 64), split.tracker(1, 64)
        for t in ids:
            mj = tj.allowed()[0]
            ms = ts.allowed()[0]
            assert not (mj & ~ms).any()             # the split never masks a feasible token
            if not mj[t]:
                break
            tj.advance(np.array([t]))
            ts.advance(np.array([t]))


def test_minimisation_keeps_distances(monkeypatch):
    tok = load()
    spec = ConstraintSpec(sentences=Interval(1, 2), lines=Interval(2, 2))
    P._PROGRAMS.clear()
    small = P.compile(spec, tok)
    monkeypatch.setattr(P, "minimize", lambda a: a)
    P._PROGRAMS.clear()
    raw = P.compile(spec, tok)
    P._PROGRAMS.clear()
    a, b = raw.parts[0], small.parts[0]
    assert b.auto.n_q < a.auto.n_q
    mq = b.auto.qmap[a.auto.st_s, a.auto.st_d, a.auto.st_a]
    np.testing.assert_array_equal(a.dist[:a.auto.n_q], b.dist[mq])


def test_enforced_lists_what_the_masks_guarantee():
    spec = ConstraintSpec(words=Interval(0, 9), keywords=("ocean",), no_commas=True,
                          json_output=True)
    assert P.enforced(spec) == ["words: at most 9", "no_commas"]


# ---------------------------------------------------------------------------- twist
def test_twist_is_the_acceptance_probability_of_the_proxy():
    pieces = ["a", " b", ".", "\n"]
    tok = TinyTokenizer(pieces)
    spec = ConstraintSpec(sentences=Interval(1, 1))
    prog = P.compile(spec, tok)
    rng = np.random.default_rng(6)
    lp = np.log(rng.random(tok.vocab_size) + 0.05)
    lp -= np.log(np.exp(lp).sum())
    tw = P.Twist(prog, lp, horizon=5)
    end = prog.ids["end"]
    pr = np.exp(lp)

    def prob(data: bytes, budget: int) -> float:
        if budget <= 0:
            return 0.0
        out = pr[end] * accepted(data, spec)
        for v, p in enumerate(pieces):
            out += pr[v] * prob(data + p.encode(), budget - 1)
        return out

    for prefix in ([0], [0, 2], [0, 2, 1], [0, 3]):     # (the first-token rule is not in psi)
        trk = prog.tracker(1, 5 + len(prefix))
        for t in prefix:
            trk.advance(np.array([t]))
        data = b"".join(pieces[t].encode() for t in prefix)
        for b in range(1, 6):
            want = prob(data, b)
            got = tw.value(trk, np.array([0]), np.array([b]))[0]
            if want == 0:
                assert got == -np.inf
            else:
                assert got == pytest.approx(np.log(want), rel=1e-5, abs=1e-6)
