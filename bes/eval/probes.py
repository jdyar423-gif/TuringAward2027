"""Step-function probes (gate E4) and self-generation measurements (gate E5), ``docs/PLAN.md``
section 7.

E4: each exact mechanism must turn a task that the plain network cannot do reliably at micro
size into one it does (pass rule: accuracy > 90 % with the mechanism, < 50 % without). Tasks are
conversations in the real chat format (``bes.data.chat_format``); about 30 % of the micro
training rows are probe conversations (pools ``probe_calc`` and ``probe_plain``, registered with
``bes.eval.microbench``), and the probes are decoded greedily with the incremental exact
features (``bes.exact.features.IncrementalFeatures``).

=========  ==========================================================  ======================
probe      task                                                        mechanism on / off
=========  ==========================================================  ======================
``copy``   reproduce a random string of 20 single-token words given     candidate tape
           earlier in the conversation ("The code is:" + the words)
``count``  exactly N bullet items or N sentences, N in 1..8; the user     countdown registers
           turn compiles (``constraints.compile``) to the countdown
``fence``  one or two fenced code blocks (half without a language tag,    registers (fence state)
           so opening and closing fences look alike); every fence must
           be closed before ``END``
``arith``  ``a op b`` with 2-4 digit operands, ``op`` in + - *; the        calculator span with the
           answer is a calculator span ``CALC a op b EQ r CALC_END``      host forcing ``r``
           (``probe_calc``) or the same text with the model's own          (``probe_plain``: own
           digits (``probe_plain``)                                       digits)
=========  ==========================================================  ======================

Scoring (independent of how the reference answers were generated): ``copy`` the response tokens
equal the reference exactly; ``count`` ``constraints.verify(constraints.compile(user), text)``
is empty (the exact count of ``constraints.measure``); ``fence`` the response has at least two
fence lines, an even number, and ``measure(text)["open_fence"]`` is false; ``arith`` the last
``= number`` of the display text equals ``calc.evaluate(a op b)``. Every probe also requires
``END`` within the token budget.

Decoding: :func:`decode_batch` runs many prompts at once on ``bes.model.generate``'s compiled
``prefill`` / ``decode_step`` (left padding with ``seg = -1``), one feature provider per row,
greedy or temperature sampling, and optional forcing of calculator results through
``bes.decode.potentials.Tracker`` (the local :class:`CalcForcer` when that does not import).

E5: :func:`self_generation_metrics` gives, per sample, the repeated 4-gram rate, whether the
sample ends in an absorbing repetition loop (:func:`ends_in_loop`) and the share of tokens copied
verbatim from the prompt (:func:`copied_share`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from bes.config import BOS, CALC, CALC_END, END, EQ, PAD

PROBE_VERSION = 1
PROBES = ("copy", "count", "fence", "arith")
#: decoding budget (new tokens) per probe
MAX_NEW = {"copy": 40, "count": 112, "fence": 200, "arith": 40}
COPY_LEN = 20
#: E5 loop detector: a period of at most LOOP_MAX_PERIOD tokens repeated over the last
#: max(2 * period, LOOP_MIN_TAIL) tokens of a sample that did not stop
LOOP_MAX_PERIOD = 32
LOOP_MIN_TAIL = 32
#: E5: a generated token counts as copied when it lies in a run of at least this many tokens
#: that occurs verbatim in the prompt
COPY_MIN_RUN = 8

CATEGORIES = {
    "animals": ["cat", "dog", "horse", "rabbit", "tiger", "lion", "zebra", "eagle", "otter",
                "whale", "camel", "goat", "sheep", "mouse", "bear", "wolf", "fox", "deer"],
    "fruits": ["apple", "banana", "cherry", "grape", "lemon", "mango", "orange", "peach",
               "pear", "plum", "melon", "kiwi", "lime", "apricot", "fig", "papaya"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white",
               "brown", "pink", "gray", "silver", "gold", "violet", "teal", "navy"],
    "countries": ["France", "Spain", "Italy", "Japan", "China", "India", "Brazil", "Canada",
                  "Mexico", "Egypt", "Kenya", "Peru", "Chile", "Norway", "Sweden", "Greece"],
    "tools": ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "ladder", "shovel",
              "rake", "axe", "file", "clamp", "level", "brush", "knife", "needle"],
    "sports": ["tennis", "soccer", "rugby", "golf", "hockey", "boxing", "rowing", "cycling",
               "skiing", "surfing", "judo", "karate", "cricket", "baseball", "polo", "sailing"],
    "vegetables": ["carrot", "potato", "onion", "garlic", "pepper", "spinach", "lettuce",
                   "cabbage", "celery", "radish", "turnip", "pea", "bean", "leek", "corn", "kale"],
    "instruments": ["piano", "violin", "guitar", "flute", "drum", "trumpet", "cello", "harp",
                    "organ", "banjo", "oboe", "tuba", "clarinet", "viola", "lute", "bass"],
}
SUBJECTS = ["The cat", "My friend", "The old man", "A small bird", "The teacher", "Our team",
            "The river", "A tall tree", "The farmer", "The child", "My sister", "The baker"]
PREDICATES = ["sleeps in the sun", "walks to the market", "sings a quiet song",
              "likes warm bread", "reads a long book", "waits by the door", "runs every morning",
              "drinks cold water", "paints the fence", "plants new seeds", "watches the rain",
              "finds a red stone", "cooks a big meal", "opens the window", "fixes the old bike"]
TOPICS = ["the sea", "the city", "winter", "a village", "the morning", "friendship", "a garden",
          "the mountains", "a journey", "the market"]
CODE_NAMES = ["x", "y", "total", "count", "value", "result", "items", "name", "data", "index",
              "size", "score", "speed", "level"]
FENCE_INTROS = ["Here is the code:", "Sure, here it is:", "This snippet does it:",
                "You can use this:"]
FENCE_BETWEEN = ["And a second part:", "Then run this:", "After that:"]
FENCE_OUTROS = ["This prints the result.", "Run it to see the output.", "That is all.", ""]


# ============================================================================ items
@dataclass
class ProbeItem:
    """One probe instance: ``messages`` (user and reference assistant turn), the prompt ids
    (``BOS USER ... END ASSISTANT``), the reference answer ids (assistant content) and
    ``meta`` (task parameters: ``n`` for count, ``expected`` for arith, ...)."""

    kind: str
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    answer_ids: list[int]
    meta: dict = field(default_factory=dict)

    @property
    def user(self) -> str:
        return self.messages[0]["content"]


def copy_vocab(tok) -> list[int]:
    """Ids of single-token words: a space and 3-12 lowercase ASCII letters (sorted)."""
    out = []
    for i in range(256, tok.vocab_size):
        b = tok.token_bytes(i)
        if 4 <= len(b) <= 13 and b[:1] == b" " and b[1:].isascii() and b[1:].isalpha() and b[1:].islower():
            out.append(i)
    return out


_VOCAB_CACHE: dict[str, list[int]] = {}


def _vocab(tok) -> list[int]:
    key = str(getattr(tok, "sha256", id(tok)))
    if key not in _VOCAB_CACHE:
        _VOCAB_CACHE[key] = copy_vocab(tok)
    return _VOCAB_CACHE[key]


def _item(kind: str, user: str, answer: Any, tok, meta: dict) -> ProbeItem:
    from bes.data import chat_format as cf

    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    prompt, _ = cf.render_conversation(msgs[:1], tok, add_generation_prompt=True)
    full, _ = cf.render_conversation(msgs, tok)
    ans = [int(t) for t in full[len(prompt):-1]]            # assistant content (no END)
    return ProbeItem(kind, msgs, [int(t) for t in prompt], ans, meta)


def make_copy(rng: np.random.Generator, tok, n_words: int = COPY_LEN) -> ProbeItem:
    """Copy probe: ``n_words`` distinct single-token words, repeated after ``The code is:``."""
    vocab = _vocab(tok)
    while True:
        ids = [int(x) for x in rng.choice(vocab, n_words, replace=False)]
        s = tok.decode(ids)
        if tok.encode(s) == ids:
            break
    item = _item("copy", f"Remember this code:{s}\nRepeat the code.", f"The code is:{s}", tok,
                 {"code_ids": ids})
    if item.answer_ids[-n_words:] != ids:
        raise AssertionError("copy probe: the answer does not end with the code tokens")
    return item


def make_count(rng: np.random.Generator, tok, n: int | None = None, unit: str | None = None) -> ProbeItem:
    """Count probe: exactly ``n`` bullet items (``unit="items"``) or sentences."""
    n = int(rng.integers(1, 9)) if n is None else int(n)
    unit = ("items" if rng.random() < 0.5 else "sentences") if unit is None else unit
    if unit == "items":
        cat = sorted(CATEGORIES)[int(rng.integers(len(CATEGORIES)))]
        words = rng.choice(CATEGORIES[cat], n, replace=False)
        user = f"List {cat} in exactly {n} bullet {'point' if n == 1 else 'points'}."
        answer = "\n".join(f"- {w}" for w in words)
    else:
        topic = TOPICS[int(rng.integers(len(TOPICS)))]
        user = f"Write exactly {n} {'sentence' if n == 1 else 'sentences'} about {topic}."
        parts = []
        for _ in range(n):
            parts.append(f"{SUBJECTS[int(rng.integers(len(SUBJECTS)))]} "
                         f"{PREDICATES[int(rng.integers(len(PREDICATES)))]}.")
        answer = " ".join(parts)
    return _item("count", user, answer, tok, {"n": n, "unit": unit})


def _code_line(rng: np.random.Generator) -> str:
    a, b = (CODE_NAMES[int(i)] for i in rng.integers(len(CODE_NAMES), size=2))
    k = int(rng.integers(0, 100))
    form = int(rng.integers(4))
    if form == 0:
        return f"{a} = {b} + {k}"
    if form == 1:
        return f"print({a})"
    if form == 2:
        return f"{a} = [{k}, {int(rng.integers(0, 100))}]"
    return f"{a} = {b} * {k}"


def make_fence(rng: np.random.Generator, tok) -> ProbeItem:
    """Fence probe: one or two code blocks (each with or without a language tag)."""
    n_blocks = 1 + int(rng.random() < 0.4)
    lang = ["python", "Python", "a short Python"][int(rng.integers(3))]
    user = (f"Write {lang} code for this in a code block." if n_blocks == 1 else
            f"Write {lang} code for this in two code blocks.")
    parts = [FENCE_INTROS[int(rng.integers(len(FENCE_INTROS)))]]
    for k in range(n_blocks):
        tag = "python" if rng.random() < 0.5 else ""
        lines = [_code_line(rng) for _ in range(int(rng.integers(1, 9)))]
        if k:
            parts.append(FENCE_BETWEEN[int(rng.integers(len(FENCE_BETWEEN)))])
        parts.append(f"```{tag}\n" + "\n".join(lines) + "\n```")
    outro = FENCE_OUTROS[int(rng.integers(len(FENCE_OUTROS)))]
    if outro:
        parts.append(outro)
    return _item("fence", user, "\n\n".join(parts), tok, {"blocks": n_blocks})


def make_arith(rng: np.random.Generator, tok, calc: bool = True) -> ProbeItem:
    """Arithmetic probe ``a op b``: a calculator span (``calc``) or plain digits."""
    from bes.exact.calc import evaluate

    da, db = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    a = int(rng.integers(10 ** (da - 1), 10 ** da))
    b = int(rng.integers(10 ** (db - 1), 10 ** db))
    op = "+-*"[int(rng.integers(3))]
    expr = f"{a} {op} {b}"
    result = evaluate(expr)
    user = f"What is {expr}?"
    if calc:
        answer: Any = [{"type": "text", "text": "The answer is "},
                       {"type": "calc", "expr": expr, "result": result},
                       {"type": "text", "text": "."}]
    else:
        answer = f"The answer is {expr} = {result}."
    return _item("arith", user, answer, tok, {"expr": expr, "expected": result, "calc": calc})


def make_item(kind: str, rng: np.random.Generator, tok, calc: bool = True) -> ProbeItem:
    if kind == "copy":
        return make_copy(rng, tok)
    if kind == "count":
        return make_count(rng, tok)
    if kind == "fence":
        return make_fence(rng, tok)
    if kind == "arith":
        return make_arith(rng, tok, calc)
    raise KeyError(kind)


def eval_items(kind: str, n: int, tok, calc: bool = True, seed: int = 0) -> list[ProbeItem]:
    """The fixed evaluation items of a probe (``count``: N cycles through 1..8 and the unit
    alternates, so every N gets the same number of items)."""
    from bes.data.samples import rng_for

    out = []
    for i in range(n):
        rng = rng_for("probe-test", PROBE_VERSION, kind, seed, i)
        if kind == "count":
            out.append(make_count(rng, tok, n=1 + i % 8, unit="items" if (i // 8) % 2 == 0 else "sentences"))
        else:
            out.append(make_item(kind, rng, tok, calc))
    return out


def training_conversations(tok, calc: bool = True, seed: int = 0) -> Iterator[list[dict]]:
    """Endless probe conversations for training: the four probes in turn (keys disjoint from
    :func:`eval_items`)."""
    from bes.data.samples import rng_for

    i = 0
    while True:
        kind = PROBES[i % len(PROBES)]
        yield make_item(kind, rng_for("probe-train", PROBE_VERSION, seed, i), tok, calc).messages
        i += 1


def _register() -> None:
    from bes.eval import microbench as mb

    for variant, calc in (("probe_calc", True), ("probe_plain", False)):
        def build(split, T, n_rows, tok, samples_dir, calc=calc):
            del samples_dir
            convs = training_conversations(tok, calc, seed=0 if split == "train" else 1)
            return mb.sample_rows(mb.conversation_samples(convs, tok, T), T, n_rows)

        def describe(split, samples_dir, variant=variant):
            del samples_dir
            return {"generator": "bes.eval.probes", "version": PROBE_VERSION, "variant": variant,
                    "split": split}
        mb.register_pool(variant, build, describe)


_register()


# =========================================================================== scoring
def display(tokens: Sequence[int], tok) -> str:
    """User-facing text of generated tokens (``END`` and later dropped)."""
    from bes.data import chat_format as cf

    toks = list(tokens)
    if END in toks:
        toks = toks[: toks.index(END)]
    return cf.display_text(toks, tok)


def _content(tokens: Sequence[int]) -> tuple[list[int], bool]:
    toks = [int(t) for t in tokens]
    if END in toks:
        return toks[: toks.index(END)], True
    return toks, False


def score(item: ProbeItem, tokens: Sequence[int], tok) -> bool:
    """Exact-match verdict of a generated response (module docstring)."""
    content, stopped = _content(tokens)
    if not stopped:
        return False
    if item.kind == "copy":
        return content == item.answer_ids
    text = display(content, tok)
    if item.kind == "count":
        from bes.exact import constraints as C

        spec = C.compile(item.user)
        return not C.verify(spec, text)
    if item.kind == "fence":
        from bes.exact import constraints as C

        m = C.measure(text)
        return m["code_fences"] >= 2 and m["code_fences"] % 2 == 0 and not m["open_fence"]
    if item.kind == "arith":
        nums = re.findall(r"=\s*(-?\d[\d,]*)", text)
        return bool(nums) and nums[-1].replace(",", "") == item.meta["expected"]
    raise KeyError(item.kind)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion ``k / n``."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ===================================================================== calculator forcing
class CalcForcer:
    """Minimal host forcing of calculator results for ``B`` streams (used when
    ``bes.decode.potentials`` does not import): after ``CALC expr EQ`` the host forces
    ``encode(evaluate(expr)) + [CALC_END]`` (``[CALC_END]`` alone when ``expr`` does not
    evaluate). ``advance(tokens) -> list[list[int] | None]`` (forced tokens per stream)."""

    def __init__(self, n: int, tok):
        self.tok = tok
        self.state = [0] * n            # 0 outside, 1 in the expression, 2 in the forced result
        self.expr: list[list[int]] = [[] for _ in range(n)]

    def advance(self, tokens: Sequence[int], active: Sequence[bool] | None = None) -> list:
        from bes.exact.calc import evaluate

        out: list = [None] * len(self.state)
        for j, t in enumerate(tokens):
            if active is not None and not active[j]:
                continue
            t = int(t)
            st = self.state[j]
            if t == CALC:
                self.state[j], self.expr[j] = 1, []
            elif st == 1 and t == EQ:
                result = evaluate(self.tok.decode(self.expr[j]))
                out[j] = ([] if result is None else list(self.tok.encode(result))) + [CALC_END]
                self.state[j] = 2
            elif st == 1 and t < self.tok.vocab_size - 16:
                self.expr[j].append(t)
            elif t >= self.tok.vocab_size - 16:
                self.state[j] = 0
        return out


def make_forcer(n: int, max_new: int, tok):
    """``(forcer, name)``: ``bes.decode.potentials.Tracker`` without a program (it follows the
    calculator and returns the forced tokens) or the local :class:`CalcForcer`."""
    try:
        from bes.decode.potentials import Tracker, table_for

        tracker = Tracker(None, n, max_new, table=table_for(tok), tok=tok)

        class _Adapter:
            def advance(self, tokens, active=None):
                return tracker.advance(np.asarray(tokens), None if active is None else np.asarray(active)).forced
        return _Adapter(), "bes.decode.potentials.Tracker"
    except Exception:        # noqa: BLE001  (any import or table problem: local fallback)
        return CalcForcer(n, tok), "local CalcForcer"


# ========================================================================== decoding
@dataclass
class Decoded:
    """One decoded stream: new tokens (the stop token included), whether it stopped, and how
    many tokens the host forced."""

    tokens: list[int]
    stopped: bool
    forced: int = 0


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def decode_batch(params, consts, cfg, prompts: Sequence[Sequence[int]], *, max_new: int,
                 tok=None, features: bool = True, temperature: float = 0.0,
                 rngs: Sequence[np.random.Generator] | None = None, force_calc: bool = False,
                 stop_ids: Sequence[int] = (END,), chunk: int = 64, decoder=None,
                 provider_factory: Callable[[], Any] | None = None) -> list[Decoded]:
    """Decode ``len(prompts)`` streams at once.

    params / consts / cfg: the model (``cfg`` the ``ModelCfg``); prompts: token ids (normally
    ``BOS ... ASSISTANT``); features: feed the exact-layer rows (``IncrementalFeatures``, or
    ``provider_factory()``) else zero rows; temperature: 0 = greedy, else sampling from
    ``softmax(scores / temperature)`` with one ``numpy`` generator per stream (``rngs``);
    force_calc: the host forces calculator results after ``EQ``. Every ``BOS`` starts a new
    segment (as in training rows). Returns one :class:`Decoded` per prompt.
    """
    from bes.exact.features import IncrementalFeatures
    from bes.model import generate as G

    b = len(prompts)
    if b == 0:
        return []
    if temperature > 0 and (rngs is None or len(rngs) != b):
        raise ValueError("sampling needs one generator per stream")
    dec = decoder if decoder is not None else G.jitted(cfg)
    if provider_factory is None:
        provider_factory = (lambda: IncrementalFeatures(tok)) if features else G.ZeroFeatures
    lens = [len(p) for p in prompts]
    lp = _round_up(max(lens), chunk)
    tmax = _round_up(lp + max_new, G.TMAX_ROUND)
    tokens = np.full((b, lp), PAD, np.int32)
    seg = np.full((b, lp), -1, np.int32)
    pos = np.zeros((b, lp), np.int32)
    empty = G.empty_rows(lp)
    feat = {k: np.repeat(empty[k][None], b, 0) for k in G.FEATURE_KEYS}
    providers = []
    for i, p in enumerate(prompts):
        n = lens[i]
        o = lp - n
        tokens[i, o:] = p
        seg[i, o:] = G.prompt_segments(p, BOS)
        pos[i, o:] = np.arange(n)
        prov = provider_factory()
        rows = [prov.step(int(t)) for t in p]
        for k in G.FEATURE_KEYS:
            feat[k][i, o:] = np.stack([r[k] for r in rows])
        providers.append(prov)
    cache = G.init_cache(cfg, b, tmax)
    logits = None
    for lo in range(0, lp, chunk):
        sl = slice(lo, lo + chunk)
        f = {"tokens": tokens[:, sl], "seg": seg[:, sl], "pos": pos[:, sl],
             **{k: a[:, sl] for k, a in feat.items()}}
        logits, _, cache = dec.prefill(params, consts, cache, f)
    cur_seg = seg[:, -1].astype(np.int64)
    forcer = make_forcer(b, max_new, tok)[0] if force_calc else None
    queues: list[list[int]] = [[] for _ in range(b)]
    out = [Decoded([], False, 0) for _ in range(b)]
    done = np.zeros(b, bool)
    stops = {int(s) for s in stop_ids}
    for step in range(max_new):
        z = np.asarray(logits, np.float64)
        nxt = np.zeros(b, np.int64)
        for j in range(b):
            if done[j]:
                nxt[j] = PAD
                continue
            if queues[j]:
                nxt[j] = queues[j].pop(0)
                out[j].forced += 1
            elif temperature <= 0:
                nxt[j] = int(np.argmax(z[j]))
            else:
                s = z[j] / temperature
                s = np.exp(s - s.max())
                cdf = np.cumsum(s)
                nxt[j] = int(np.searchsorted(cdf, rngs[j].random() * cdf[-1], side="right"))
                nxt[j] = min(nxt[j], len(s) - 1)
        active = ~done
        if forcer is not None:
            forced = forcer.advance(np.where(active, nxt, PAD), active)
            for j in range(b):
                if active[j] and forced[j] is not None and not queues[j]:
                    queues[j] = [int(t) for t in forced[j]]
        for j in np.flatnonzero(active):
            out[j].tokens.append(int(nxt[j]))
            if int(nxt[j]) in stops:
                done[j] = True
                out[j].stopped = True
        if done.all() or step == max_new - 1:
            break
        rows_k = []
        for j in range(b):
            if done[j]:
                rows_k.append({k: v[0] for k, v in G.empty_rows(1).items()})
            else:
                cur_seg[j] += int(nxt[j] == BOS)
                rows_k.append(providers[j].step(int(nxt[j])))
        f = {k: np.stack([r[k] for r in rows_k]) for k in G.FEATURE_KEYS}
        f["seg"] = np.where(done, -1, cur_seg).astype(np.int32)
        feed = np.where(done, PAD, nxt).astype(np.int32)
        logits, _, cache = dec.decode_step(params, consts, cache, feed, f)
    return out


def run_probe(params, consts, cfg, items: Sequence[ProbeItem], tok, *, force_calc: bool = True,
              batch: int = 32, max_new: int | None = None, decoder=None) -> list[dict]:
    """Greedy responses and verdicts for ``items`` (one probe kind):
    ``[{"ok", "tokens", "text", "stopped", "forced", "meta"}]``."""
    res = []
    for lo in range(0, len(items), batch):
        part = items[lo:lo + batch]
        mn = max_new or MAX_NEW[part[0].kind]
        got = decode_batch(params, consts, cfg, [it.prompt_ids for it in part], max_new=mn,
                           tok=tok, force_calc=force_calc, decoder=decoder)
        for it, d in zip(part, got):
            res.append({"ok": bool(score(it, d.tokens, tok)), "tokens": d.tokens,
                        "text": display(d.tokens, tok), "stopped": d.stopped, "forced": d.forced,
                        "meta": {k: v for k, v in it.meta.items() if k != "code_ids"}})
    return res


# ================================================================ self-generation (E5)
def repeated_ngram_rate(tokens: Sequence[int], n: int = 4) -> float:
    """Share of the ``n``-gram occurrences of ``tokens`` that repeat an earlier ``n``-gram of
    the same sequence (``1 - distinct / total``; 0 with fewer than ``n`` tokens)."""
    t = [int(x) for x in tokens]
    total = len(t) - n + 1
    if total <= 0:
        return 0.0
    seen = set()
    rep = 0
    for i in range(total):
        g = tuple(t[i:i + n])
        if g in seen:
            rep += 1
        else:
            seen.add(g)
    return rep / total


def ends_in_loop(tokens: Sequence[int], stopped: bool, max_period: int = LOOP_MAX_PERIOD,
                 min_tail: int = LOOP_MIN_TAIL) -> bool:
    """Operational absorbing loop: the sample did **not** stop (no ``END`` within its budget)
    and its last ``R = max(2 p, min_tail)`` tokens are exactly periodic with some period
    ``1 <= p <= max_period`` (``x[i] == x[i - p]`` for every ``i`` of that window after its first
    ``p`` tokens): at least two full periods and ``min_tail`` tokens repeat exactly to the end."""
    if stopped:
        return False
    x = [int(v) for v in tokens]
    n = len(x)
    for p in range(1, max_period + 1):
        r = max(2 * p, min_tail)
        if r > n:
            break
        w = x[n - r:]
        if all(w[i] == w[i - p] for i in range(p, r)):
            return True
    return False


def copied_share(gen: Sequence[int], prompt: Sequence[int], min_run: int = COPY_MIN_RUN) -> float:
    """Share of the generated tokens that lie in a run of at least ``min_run`` consecutive
    generated tokens occurring verbatim (contiguously) in ``prompt``."""
    g = np.asarray([int(x) for x in gen], np.int64)
    p = np.asarray([int(x) for x in prompt], np.int64)
    n, m = g.size, p.size
    if n == 0 or m == 0:
        return 0.0
    eq = g[:, None] == p[None, :]
    run = np.zeros((n + 1, m + 1), np.int64)
    for i in range(n - 1, -1, -1):                 # run[i, j] = common run of g[i:] and p[j:]
        run[i, :m] = np.where(eq[i], run[i + 1, 1:] + 1, 0)
    best = run[:n, :m].max(axis=1)                 # longest prompt match starting at g[i]
    covered = np.zeros(n, bool)
    for i in np.flatnonzero(best >= min_run):
        covered[i:i + best[i]] = True
    return float(covered.mean())


def self_generation_metrics(gen: Sequence[int], prompt: Sequence[int], stopped: bool) -> dict:
    """E5 metrics of one sample: ``rep4``, ``loop`` (bool), ``copy`` and ``n`` (tokens)."""
    content = [int(t) for t in gen if int(t) != END]
    return {"rep4": repeated_ngram_rate(content, 4), "loop": ends_in_loop(content, stopped),
            "copy": copied_share(content, prompt), "n": len(content), "stopped": bool(stopped)}


def chat_prompts(n: int, tok, *, samples_dir=None, max_len: int = 384, skip: int = 2000) -> list[list[int]]:
    """Held-out chat prompts: the system and first user turn of smol-smoltalk test
    conversations (records from ``skip`` on, disjoint from the evaluation rows), rendered with
    a trailing ``ASSISTANT``."""
    from bes.data import chat_format as cf
    from bes.eval import microbench as mb

    out = []
    for i, msgs in enumerate(mb.source_texts("chat", "heldout", samples_dir or mb.SAMPLES_DIR)):
        if i < skip:
            continue
        msgs = list(msgs)
        head = [m for m in msgs[:2] if m.get("role") in ("system", "user")]
        if not head or head[-1]["role"] != "user":
            continue
        try:
            ids, _ = cf.render_conversation(head, tok, add_generation_prompt=True)
        except ValueError:
            continue
        if len(ids) <= max_len:
            out.append([int(t) for t in ids])
        if len(out) >= n:
            break
    return out


def web_prompts(n: int, tok, *, samples_dir=None, prefix: int = 96, skip: int = 2000) -> list[list[int]]:
    """Held-out web prompts: ``BOS`` + the first ``prefix`` tokens of ClimbMix ``shard_00997``
    documents (from record ``skip`` on; documents shorter than ``2 * prefix`` are skipped)."""
    from bes.eval import microbench as mb

    out = []
    for i, text in enumerate(mb.source_texts("web", "heldout", samples_dir or mb.SAMPLES_DIR)):
        if i < skip:
            continue
        ids = tok.encode(text)
        if len(ids) >= 2 * prefix:
            out.append([BOS] + [int(t) for t in ids[:prefix]])
        if len(out) >= n:
            break
    return out


# ============================================================================ reports
def probe_summary(results: Sequence[dict]) -> dict:
    """``{"n", "correct", "acc", "ci": (lo, hi), "by_n": {N: (correct, n)}}`` of one probe run."""
    n = len(results)
    k = sum(1 for r in results if r["ok"])
    by_n: dict = {}
    for r in results:
        key = r["meta"].get("n")
        if key is not None:
            c, t = by_n.get(key, (0, 0))
            by_n[key] = (c + int(r["ok"]), t + 1)
    return {"n": n, "correct": k, "acc": k / n if n else 0.0, "ci": wilson(k, n),
            "by_n": {str(a): list(b) for a, b in sorted(by_n.items())}}


def probe_verdict(on: dict | None, off: dict | None, hi: float = 0.9, lo: float = 0.5) -> str:
    """The E4 pass rule on point estimates: on > ``hi`` and off < ``lo``."""
    if on is None or off is None:
        return "incomplete"
    ok_on, ok_off = on["acc"] > hi, off["acc"] < lo
    if ok_on and ok_off:
        return "pass"
    why = []
    if not ok_on:
        why.append(f"on {on['acc']:.0%} <= {hi:.0%}")
    if not ok_off:
        why.append(f"off {off['acc']:.0%} >= {lo:.0%}")
    return "fail (" + ", ".join(why) + ")"


__all__ = ["PROBES", "ProbeItem", "make_copy", "make_count", "make_fence", "make_arith", "make_item",
           "eval_items", "training_conversations", "score", "wilson", "decode_batch", "run_probe",
           "CalcForcer", "make_forcer", "repeated_ngram_rate", "ends_in_loop", "copied_share",
           "self_generation_metrics", "chat_prompts", "web_prompts", "probe_summary",
           "probe_verdict", "display"]
