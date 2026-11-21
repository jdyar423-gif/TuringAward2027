"""The particle filter (bes.decode.smc) against exhaustive enumeration.

The reference target is computed here by recursion over every token sequence of a tiny
vocabulary up to the horizon, from the table model's probabilities, an independent
implementation of temperature and top-p, and text-level acceptance (reference byte-level
transducer ``step`` for code / quote / bracket state, ``constraints.verify`` for the counts): at
every prefix a token is allowed iff some completion within the remaining budget is accepted;
the increment is ``p~ * allowed`` (truncation before the mask) or ``alpha * q`` (after the mask:
``alpha`` the allowed mass of the tempered model, ``q`` the truncated renormalised allowed
distribution). ``Z`` is the sum over complete accepted sequences of the product of the
increments and ``pi`` their normalised products.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from bes.decode import potentials as P
from bes.decode import sampling as S
from bes.decode import smc
from bes.decode.lm import TableLM, TinyTokenizer
from bes.exact import constraints as C
from bes.exact import transducer as tr
from bes.exact.constraints import ConstraintSpec, Interval

WS = b" \t\n\r\x0b\x0c"
ENFORCED = ("words", "sentences", "items", "lines", "paragraphs", "no_commas", "end_question",
            "code_block")


def accepted(data: bytes, spec) -> bool:
    """Reference acceptance of a complete response (tiny vocabularies: single-byte brackets)."""
    if not data or data[0] in WS or data[-1] in WS:
        return False
    st, depth = tr.INIT, 0
    for b in data:
        st, ev = tr.step(st, b)
        depth = max(0, depth + ev.bracket)
        if depth > 8:
            return False
    if st.fence or st.inline or st.quote or depth:
        return False
    if spec is None:
        return True
    return not [x for x in C.verify(spec, data.decode()) if x.split(":")[0] in ENFORCED]


def ref_logq(logp: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logp, np.float64) / temperature
    m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


def ref_top_p(logq: np.ndarray, top_p: float, within: np.ndarray) -> np.ndarray:
    idx = [i for i in np.argsort(-logq, kind="stable") if within[i] and np.isfinite(logq[i])]
    if top_p >= 1.0:
        keep = idx
    else:
        tot = sum(math.exp(logq[i]) for i in idx)
        keep, cum = [], 0.0
        for i in idx:
            if cum >= top_p:
                break
            keep.append(i)
            cum += math.exp(logq[i]) / tot
    out = np.zeros(len(logq), bool)
    out[keep] = True
    return out


class Setting:
    """Tiny vocabulary, table model, spec, horizon and sampling settings, with the exact target."""

    def __init__(self, pieces, spec, horizon, *, seed=0, order=2, temperature=1.0, top_p=1.0,
                 after=True, boost_end=1.0, lm=None):
        self.tok = TinyTokenizer(pieces)
        self.pieces = [p.encode() for p in pieces]
        self.spec, self.horizon = spec, horizon
        sp = self.tok.special_tokens
        self.end = sp["<|end|>"]
        v = self.tok.vocab_size
        self.lm = lm or TableLM.random(v, order=order, seed=seed, boost={self.end: boost_end})
        self.prompt = [sp["<|bos|>"], sp["<|user|>"], 0, self.end, sp["<|assistant|>"]]
        self.settings = S.SamplingSettings(temperature=temperature, top_p=top_p,
                                           truncate_after_mask=after)
        self.program = P.compile(spec, self.tok)
        self.alphabet = list(range(len(pieces))) + [self.end]
        self.pi: dict[tuple, float] = {}
        self.proposal_only: dict[tuple, float] = {}
        self.z = self._z((), b"")
        for x in list(self.pi):
            self.pi[x] /= self.z

    @lru_cache(maxsize=None)
    def exists(self, data: bytes, budget: int) -> bool:
        if budget <= 0:
            return False
        if accepted(data, self.spec):
            return True
        return budget > 1 and any(self.exists(data + p, budget - 1) for p in self.pieces)

    def allowed(self, seq: tuple, data: bytes) -> np.ndarray:
        rem = self.horizon - len(seq)
        m = np.zeros(self.tok.vocab_size, bool)
        m[self.end] = rem >= 1 and accepted(data, self.spec)
        for v, p in enumerate(self.pieces):
            m[v] = self.exists(data + p, rem - 1)
        return m

    def increment(self, seq: tuple, data: bytes) -> np.ndarray:
        """Unnormalised target increment ``gamma[v]`` (float64) after the prefix ``seq``."""
        logp = self.lm.logp(self.prompt + list(seq)).astype(np.float32)
        lq = ref_logq(logp, self.settings.temperature)
        m = self.allowed(seq, data)
        if self.settings.truncate_after_mask:
            alpha = np.exp(lq)[m].sum()
            if alpha == 0:
                return np.zeros_like(lq)
            keep = ref_top_p(lq, self.settings.top_p, m)
            q = np.where(keep, np.exp(lq), 0.0)
            return alpha * q / q.sum()
        keep = ref_top_p(lq, self.settings.top_p, np.ones(len(lq), bool))
        pt = np.where(keep, np.exp(lq), 0.0)
        return np.where(m, pt / pt.sum(), 0.0)

    def _z(self, seq: tuple, data: bytes, path: float = 1.0, prop: float = 1.0) -> float:
        if len(seq) >= self.horizon:
            return 0.0
        g = self.increment(seq, data)
        tot = g.sum()
        z = 0.0
        for v in self.alphabet:
            if g[v] <= 0:
                continue
            if v == self.end:
                self.pi[seq + (v,)] = path * g[v]
                self.proposal_only[seq + (v,)] = prop * g[v] / tot
                z += g[v]
            else:
                z += g[v] * self._z(seq + (v,), data + self.pieces[v], path * g[v],
                                    prop * g[v] / tot)
        return z

    def run(self, k, seed, **kw):
        kw.setdefault("settings", self.settings)
        return smc.decode(self.lm, self.prompt, n_particles=k, max_new_tokens=self.horizon,
                          seed=seed, program=self.program, **kw)


def _mean_and_se(values):
    v = np.asarray(values)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


SPEC = ConstraintSpec(sentences=Interval(1, 1), words=Interval(0, 2))
PIECES = ["a", " b", ".", "\n"]


@pytest.fixture(scope="module")
def setting():
    return Setting(PIECES, SPEC, 5, seed=1)


@pytest.fixture(scope="module")
def setting_bigram_twist(setting):
    proxy = S.log_softmax(np.random.default_rng(9).standard_normal(setting.tok.vocab_size))
    return P.Twist(setting.program, proxy, horizon=setting.horizon)


# ----------------------------------------------------------------- normalising constant
@pytest.mark.parametrize("k", [1, 2, 8])
@pytest.mark.parametrize("resample", ["never", "always"])
@pytest.mark.parametrize("twist", [False, True])
def test_normalising_constant_is_unbiased(setting, setting_bigram_twist, k, resample, twist):
    runs = 300
    ess_frac = 0.0 if resample == "never" else 1.01
    tw = setting_bigram_twist if twist else None
    results = [setting.run(k, seed, ess_frac=ess_frac, twist=tw) for seed in range(runs)]
    zs = [math.exp(r.log_z) for r in results]
    n_resampled = sum(bool(r.resampled) for r in results)
    if resample == "never":
        assert n_resampled == 0
    else:
        assert n_resampled > runs // 2           # the configuration really resamples
    mean, se = _mean_and_se(zs)
    assert setting.z > 0
    assert abs(mean - setting.z) < 4 * se + 1e-12, (mean, setting.z, se)


@pytest.mark.parametrize("after,top_p,temperature", [(True, 0.8, 0.7), (False, 0.8, 0.7),
                                                     (False, 1.0, 1.0)])
def test_normalising_constant_is_unbiased_for_both_truncation_targets(after, top_p, temperature):
    st = Setting(PIECES, SPEC, 5, seed=2, temperature=temperature, top_p=top_p, after=after)
    zs = [math.exp(st.run(4, seed, ess_frac=0.7, resample_at="token").log_z) for seed in range(300)]
    mean, se = _mean_and_se(zs)
    assert abs(mean - st.z) < 4 * se + 1e-12, (mean, st.z, se)


def test_exact_twist_gives_a_zero_variance_estimate():
    """A unigram table model and the same unigram as the twist's proxy: psi is the optimal
    twist, every incremental weight is the same and log Z is exact for every seed."""
    tok = TinyTokenizer(PIECES)
    v = tok.vocab_size
    z = np.random.default_rng(3).standard_normal(v)
    z[tok.special_tokens["<|end|>"]] += 1.0
    lm = TableLM.from_table(z)
    st = Setting(PIECES, SPEC, 5, after=False, lm=lm)
    tw = P.Twist(st.program, S.log_softmax(z), horizon=5)
    plain, twisted = [], []
    for seed in range(40):
        plain.append(st.run(4, seed).log_z)
        twisted.append(st.run(4, seed, twist=tw).log_z)
    np.testing.assert_allclose(np.exp(twisted), st.z, rtol=1e-6)     # float32 model rows
    assert np.std(np.exp(plain)) > 1e-3 * st.z
    # a twist from a different proxy stays unbiased (above) and here lowers the variance
    proxy = S.log_softmax(z + np.random.default_rng(4).standard_normal(v) * 0.3)
    tw2 = P.Twist(st.program, proxy, horizon=5)
    approx = [math.exp(st.run(4, seed, twist=tw2).log_z) for seed in range(150)]
    plain = [math.exp(st.run(4, seed).log_z) for seed in range(150)]
    assert np.var(approx) < np.var(plain)


# --------------------------------------------------------------------------- convergence
def test_weighted_particles_converge_to_the_target(setting):
    tvs = []
    for k in (1, 4, 16, 64):
        per_run = []
        for seed in range(30):
            res = setting.run(k, 1000 + seed)
            est: dict[tuple, float] = {}
            for p, w in zip(res.particles, res.weights):
                if w > 0:
                    key = tuple(p.tokens)
                    est[key] = est.get(key, 0.0) + w
            keys = set(est) | set(setting.pi)
            per_run.append(0.5 * sum(abs(est.get(x, 0.0) - setting.pi.get(x, 0.0)) for x in keys))
        tvs.append(float(np.mean(per_run)))
    assert tvs[0] > tvs[1] > tvs[2] > tvs[3], tvs
    assert tvs[3] < 0.15, tvs
    for res_tokens in setting.pi:
        assert res_tokens[-1] == setting.end


def test_plain_masking_without_weights_does_not_sample_the_target():
    """First token "a" or "b" equally likely; after "a" the model wants "," (forbidden), after
    "b" it does not. Masking alone keeps P(first = a) = 1/2; the target gives it much less."""
    tok = TinyTokenizer(["a", "b", ",", "."])
    v = tok.vocab_size
    end = tok.special_tokens["<|end|>"]
    table = np.full((v, v), -8.0)
    table[:, end] = 0.0
    table[tok.special_tokens["<|assistant|>"], [0, 1]] = 3.0
    table[0, 2] = 4.0                     # after "a": mostly ","
    table[1, 3] = 1.0
    lm = TableLM.from_table(table)
    st = Setting(["a", "b", ",", "."], ConstraintSpec(no_commas=True), 4, lm=lm, after=False)
    keys = set(st.pi) | set(st.proposal_only)
    tv_mask = 0.5 * sum(abs(st.pi.get(x, 0) - st.proposal_only.get(x, 0)) for x in keys)
    pi_a = sum(p for x, p in st.pi.items() if x[0] == 0)
    mask_a = sum(p for x, p in st.proposal_only.items() if x[0] == 0)
    assert tv_mask > 0.3 and mask_a == pytest.approx(0.5, abs=1e-3) and pi_a < 0.2
    # unweighted single particles follow the masked proposal, weighted particles the target
    first = [st.run(1, seed).particles[0].tokens[0] for seed in range(400)]
    assert abs(np.mean(np.array(first) == 0) - 0.5) < 0.08
    w_a = []
    for seed in range(20):
        res = st.run(64, seed)
        w_a.append(sum(w for p, w in zip(res.particles, res.weights) if p.tokens[0] == 0))
    assert abs(np.mean(w_a) - pi_a) < 0.05


# ------------------------------------------------------------------- weights and bookkeeping
def test_incremental_weight_is_the_allowed_mass(setting):
    for seed in range(20):
        res = setting.run(6, seed, ess_frac=0.0, resample_at="token")
        for j, p in enumerate(res.particles):
            data, total = b"", 0.0
            for pos, t in enumerate(p.tokens):
                g = setting.increment(tuple(p.tokens[:pos]), data)
                a = res.log_alpha[pos][j]
                assert a == pytest.approx(math.log(g.sum()), abs=1e-9)
                total += a
                if t != setting.end:
                    data += setting.pieces[t]
            if p.finish == "stop":
                assert p.log_weight == pytest.approx(total, abs=1e-9)
                assert accepted(data, setting.spec)


def test_k1_without_potentials_is_ordinary_sampling():
    tok = TinyTokenizer(PIECES)
    end = tok.special_tokens["<|end|>"]
    lm = TableLM.random(tok.vocab_size, order=2, seed=5, boost={end: 1.5})
    prompt = [tok.special_tokens["<|bos|>"], tok.special_tokens["<|assistant|>"]]
    st = S.SamplingSettings(temperature=0.8, top_p=0.9)
    for seed in range(25):
        res = smc.decode(lm, prompt, n_particles=1, max_new_tokens=12, seed=seed, settings=st,
                         tok=tok)
        # plain sampling loop with the same random stream
        logp = lm.prefill(prompt, 1)
        out, lps = [], []
        for n in range(12):
            lg, _ = S.step_increment(logp, st)
            t = int(S.draw(lg, S.counter_uniform(seed, S.STREAM_TOKEN, np.array([0]),
                                                 np.array([n])))[0])
            out.append(t)
            lps.append(float(logp[0, t]))
            if t == end:
                break
            logp = lm.step(np.array([t]), np.array([True]))
        assert res.particles[0].tokens == out
        assert res.particles[0].token_logp == lps          # bit for bit
        assert abs(res.log_z) < 1e-9 or res.particles[0].finish != "stop"


def test_determinism_and_resampling_bookkeeping(setting):
    a = setting.run(8, 11, ess_frac=1.01)
    b = setting.run(8, 11, ess_frac=1.01)
    c = setting.run(8, 12, ess_frac=1.01)
    assert [p.tokens for p in a.particles] == [p.tokens for p in b.particles]
    assert a.log_z == b.log_z and np.array_equal(a.weights, b.weights)
    assert [p.tokens for p in a.particles] != [p.tokens for p in c.particles]
    for res in (a, c):
        assert res.resampled
        for p in res.particles:
            # each particle's history is consistent with the model state it was given
            np.testing.assert_allclose(p.token_logp,
                                       setting.lm.sequence_logprob(setting.prompt, p.tokens))
            if p.finish == "stop":
                data = b"".join(setting.pieces[t] for t in p.tokens[:-1])
                assert accepted(data, setting.spec)
    lm = TableLM.random(setting.tok.vocab_size, order=2, seed=1, boost={setting.end: 1.0})
    smc.decode(lm, setting.prompt, n_particles=8, max_new_tokens=5, seed=11,
               program=setting.program, settings=setting.settings, ess_frac=1.01)
    assert lm.reorders == sum(not np.array_equal(e["ancestors"], np.arange(8)) for e in a.resampled)


def test_dead_particles_are_replaced_and_all_dead_gives_zero():
    tok = TinyTokenizer(["a", "."])
    spec = ConstraintSpec(sentences=Interval(1, 1))
    prog = P.compile(spec, tok)
    end = tok.special_tokens["<|end|>"]
    table = np.full((tok.vocab_size,) * 2, -30.0)
    table[:, 0] = 0.0                 # the model only ever wants "a": "." has p ~ 1e-13
    table[:, end] = -1.0
    lm = TableLM.from_table(table)
    prompt = [tok.special_tokens["<|bos|>"], tok.special_tokens["<|assistant|>"]]
    res = smc.decode(lm, prompt, n_particles=4, max_new_tokens=4, seed=0, program=prog)
    for p in res.particles:
        if p.finish == "stop":
            assert tok.decode_bytes(p.tokens[:-1]).endswith(b".")
    assert np.isfinite(res.log_z)
    # nothing allowed at all: every particle dies, the estimate is zero
    never = ConstraintSpec(sentences=Interval(5, 5))
    res = smc.decode(lm, prompt, n_particles=3, max_new_tokens=3, seed=0,
                     program=P.compile(never, tok))
    assert all(p.finish == "dead" for p in res.particles)
    assert res.log_z == -math.inf and not res.weights.any()


# ------------------------------------------------------------------------- calculator
def test_forced_calculator_spans_have_weight_one():
    pieces = ["1", "2", "3", "+", "a", "."]
    tok = TinyTokenizer(pieces)
    sp = tok.special_tokens
    calc, eq, ce, end = sp["<|calc|>"], sp["<|eq|>"], sp["<|/calc|>"], sp["<|end|>"]
    v = tok.vocab_size
    table = np.full((v, v), -6.0)
    table[sp["<|assistant|>"], calc] = 5.0       # start with a calculator span
    table[calc, 0] = 5.0                         # "1"
    table[0, 3] = 3.0                            # "1+"
    table[3, 1] = 5.0                            # "1+2"
    table[1, eq] = 5.0                           # then "="
    table[ce, 5] = 3.0                           # after the span: "."
    table[5, end] = 5.0
    lm = TableLM.from_table(table)
    prog = P.compile(None, tok)
    prompt = [sp["<|bos|>"], sp["<|assistant|>"]]
    res = smc.decode(lm, prompt, n_particles=3, max_new_tokens=12, seed=0, program=prog)
    seen = 0
    for j, p in enumerate(res.particles):
        if eq not in p.tokens:
            continue
        seen += 1
        i = p.tokens.index(eq)
        expr = tok.decode_bytes(p.tokens[p.tokens.index(calc) + 1:i]).decode()
        result = str(sum(int(x) for x in expr.split("+")))     # the test's own arithmetic
        assert p.tokens[i + 1:i + 2 + len(tok.encode(result))] == tok.encode(result) + [ce]
        assert p.forced[i + 1:i + 2 + len(tok.encode(result))] == [True] * (len(tok.encode(result)) + 1)
        assert not any(p.forced[:i + 1])
        assert (i + 1, i + 2 + len(tok.encode(result))) in res.forced_spans[j]
        for pos in range(i + 1, i + 2 + len(tok.encode(result))):
            assert math.isnan(res.log_alpha[pos][j])  # no weight for forced tokens
    assert seen >= 2
    # without potentials the host still forces the result
    raw = smc.decode(lm, prompt, n_particles=2, max_new_tokens=12, seed=0, tok=tok)
    assert any(eq in p.tokens and p.tokens[p.tokens.index(eq) + 1] == 2 for p in raw.particles)


# ------------------------------------------------------------- our model, end to end
def test_our_model_end_to_end_with_real_tokenizer_features_and_constraints():
    import jax
    import dataclasses

    from bes.config import load_config
    from bes.data.chat_format import display_text, render_conversation
    from bes.decode.lm import OurLM
    from bes.exact.features import exact_features
    from bes.model import model as M
    from bes.tokenizer import load

    from pathlib import Path

    tok = load()
    cfg = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "smoke.py")).model
    cfg = dataclasses.replace(cfg, n_loops=2, max_seq=256)
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    params["head"]["out"] = params["head"]["out"] + rng.standard_normal(
        params["head"]["out"].shape).astype(np.float32) * 0.5
    consts = M.make_consts(cfg)
    prompt, _ = render_conversation([{"role": "user", "content":
                                      "Answer in exactly 2 sentences and at most 12 words."}],
                                    tok, add_generation_prompt=True)
    spec = ConstraintSpec(sentences=Interval(2, 2), words=Interval(0, 12))
    prog = P.compile(spec, tok)
    lm = OurLM(params, consts, cfg, tok, max_steps=96)
    res = smc.decode(lm, prompt, n_particles=4, max_new_tokens=24, seed=0, program=prog,
                     tok=tok, settings=S.SamplingSettings(temperature=1.0), no_repeat_ngram=6,
                     max_copy_run=12)
    finished = [p for p in res.particles if p.finish == "stop"]
    assert finished, [p.finish for p in res.particles]
    for p in finished:
        text = display_text(p.tokens, tok)
        assert not [x for x in C.verify(spec, text) if x.split(":")[0] in ENFORCED], text
    # reported log-probabilities equal a full forward pass over prompt + response
    for p in res.particles:
        if not p.tokens:
            continue
        seq = np.asarray(list(prompt) + p.tokens, np.int32)[None]
        feats = exact_features(seq, None, tok)
        feats = {k: np.asarray(v) for k, v in feats.items()}
        feats["tokens"] = seq
        logits = np.asarray(jax.jit(M.logits_fn, static_argnames="cfg")(params, consts, feats,
                                                                        cfg=cfg))[0]
        lp = S.log_softmax(logits)
        n = len(prompt)
        want = [lp[n - 1 + i, t] for i, t in enumerate(p.tokens)]
        np.testing.assert_allclose(p.token_logp, want, atol=1e-4)


# ------------------------------------------------------- guards, batching invariance
def _cycle_lm(tok, first, cycle, end_logit=-1.0, extra=()):
    """Order-2 table: the assistant token leads to ``first``; ``cycle[i]`` leads to
    ``cycle[i + 1]`` (strongly), ``extra`` holds more ``(a, b)`` transitions, END has
    ``end_logit`` everywhere."""
    v = tok.vocab_size
    sp = tok.special_tokens
    table = np.full((v, v), -6.0)
    table[:, sp["<|end|>"]] = end_logit
    table[sp["<|assistant|>"], first] = 6.0
    for a, b in list(zip(cycle, cycle[1:] + cycle[:1])) + list(extra):
        table[a, b] = 6.0
    for a in range(len(tok.pieces)):
        table[a, (a + 1) % len(tok.pieces)] += 2.0       # a clear second choice
    return TableLM.from_table(table)


def test_ngram_blocking_outside_quotes_only():
    tok = TinyTokenizer(["a", " b", "\"", ".", " c"])
    sp = tok.special_tokens
    prompt = [sp["<|bos|>"], sp["<|assistant|>"]]
    lm = _cycle_lm(tok, 0, [0, 1])                      # a b a b ...
    res = smc.decode(lm, prompt, n_particles=1, max_new_tokens=10, seed=0, tok=tok,
                     no_repeat_ngram=2, greedy=True)
    toks = [t for t in res.particles[0].tokens if t < len(tok.pieces)]
    grams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
    assert len(grams) == len(set(grams)), toks            # no repeated bigram
    plain = smc.decode(lm, prompt, n_particles=1, max_new_tokens=10, seed=0, tok=tok, greedy=True)
    assert plain.particles[0].tokens[:6] == [0, 1, 0, 1, 0, 1]
    quoted = _cycle_lm(tok, 2, [0, 1], extra=[(2, 0)])  # '"' then a b a b ... inside the quote
    res = smc.decode(quoted, prompt, n_particles=1, max_new_tokens=8, seed=0, tok=tok,
                     no_repeat_ngram=2, greedy=True)
    t = res.particles[0].tokens
    assert t[0] == 2 and t[1:6] == [0, 1, 0, 1, 0], t      # inside the quote nothing is blocked


def test_copy_cap_limits_verbatim_runs_from_the_prompt():
    pieces = ["a", "b", "c", "d", "e", "f", "Z", "."]
    tok = TinyTokenizer(pieces)
    sp = tok.special_tokens
    source = [0, 1, 2, 3, 4, 5]                          # "abcdef" in the user turn
    prompt = [sp["<|bos|>"], sp["<|user|>"], *source, sp["<|end|>"], sp["<|assistant|>"]]
    lm = _cycle_lm(tok, 0, [0, 1, 2, 3, 4, 5])
    capped = smc.decode(lm, prompt, n_particles=1, max_new_tokens=9, seed=0, tok=tok,
                        max_copy_run=3, greedy=True).particles[0].tokens
    free = smc.decode(lm, prompt, n_particles=1, max_new_tokens=9, seed=0, tok=tok,
                      greedy=True).particles[0].tokens
    assert free[:6] == source
    runs = {tuple(source[i:i + 4]) for i in range(len(source) - 3)}
    body = [t for t in capped if t < len(pieces)]
    assert not any(tuple(body[i:i + 4]) in runs for i in range(len(body) - 3)), capped
    assert body[:3] == source[:3]                        # copying up to the cap is fine
    # names and numbers are exempt: a capitalised token may continue a copy
    lm2 = _cycle_lm(tok, 0, [0, 1, 2, 6])
    src2 = [0, 1, 2, 6, 7]
    prompt2 = [sp["<|bos|>"], sp["<|user|>"], *src2, sp["<|end|>"], sp["<|assistant|>"]]
    out = smc.decode(lm2, prompt2, n_particles=1, max_new_tokens=6, seed=0, tok=tok,
                     max_copy_run=3, greedy=True).particles[0].tokens
    assert out[:4] == [0, 1, 2, 6]


def test_batching_never_changes_a_particles_draws(setting):
    for seed in range(5):
        small = setting.run(2, seed, ess_frac=0.0)
        big = setting.run(5, seed, ess_frac=0.0)
        for j in range(2):
            assert small.particles[j].tokens == big.particles[j].tokens
            assert small.particles[j].log_weight == big.particles[j].log_weight


def test_twist_with_a_short_horizon_stays_unbiased(setting):
    """A horizon shorter than the budget: far states get the positive floor value."""
    proxy = S.log_softmax(np.random.default_rng(10).standard_normal(setting.tok.vocab_size))
    tw = P.Twist(setting.program, proxy, horizon=2)
    zs = [math.exp(setting.run(4, seed, twist=tw).log_z) for seed in range(300)]
    mean, se = _mean_and_se(zs)
    assert abs(mean - setting.z) < 4 * se + 1e-12, (mean, setting.z, se)
    naive = P.compile(setting.spec, setting.tok, budget_aware=False)
    with pytest.raises(ValueError):
        smc.decode(setting.lm, setting.prompt, n_particles=2, max_new_tokens=5, program=naive,
                   twist=P.Twist(naive, proxy, horizon=5))


def test_run_reads_the_decode_config_and_library_turns_are_found():
    from bes.config import DecodeCfg

    tok = TinyTokenizer(PIECES)
    sp = tok.special_tokens
    lm = TableLM.random(tok.vocab_size, order=2, seed=3, boost={sp["<|end|>"]: 1.0})
    prompt = [sp["<|bos|>"], sp["<|library|>"], 0, 1, sp["<|end|>"], sp["<|user|>"], 0,
              sp["<|end|>"], sp["<|assistant|>"]]
    cfg = DecodeCfg(particles=3, max_new_tokens=6, temperature=1.0, top_p=1.0,
                    no_repeat_ngram=2, max_copy_run=3, resample_at="token")
    res = smc.run(lm, prompt, spec=SPEC, tok=tok, cfg=cfg, seed=4)
    assert len(res.particles) == 3 and res.steps <= 6
    for p in res.particles:
        if p.finish == "stop":
            assert accepted(b"".join(tok.token_bytes(t) for t in p.tokens[:-1]), SPEC)
    table = P.table_for(tok)
    assert smc.library_passages(prompt, table) == [[0, 1]]
