"""The particle filter: Feynman-Kac decoding of one prompt (``docs/MATH.md`` F1-F5, PLAN 4G).

Target: ``pi(x) ~ prod_t gamma_t(x_t | x_<t)`` over complete responses (ending with ``END``
within ``max_new_tokens``), where ``gamma_t`` is the target increment of
``bes.decode.sampling.step_increment``: the (tempered, possibly truncated) model times the
0/1 potentials of ``bes.decode.potentials``. Tokens forced by the host (calculator results)
have ``gamma = 1`` on the forced token: they are written with weight 1 and are never masked.

One synchronous step, for every running particle ``j`` that is not inside a forced span:

1. model log-probabilities (from the previous LM call) -> ``step_increment`` with the
   potential mask (``Tracker.allowed``) and the n-gram / copy-cap keep mask
   -> ``log_gamma[j]`` and ``log_alpha[j] = logsumexp(log_gamma[j])``;
2. optional twist: ``log_gamma += log psi(next) - log psi(current)`` (cancels at ``END``);
3. draw ``x_j`` from ``log_gamma[j]`` with the counter-based uniform
   ``U(seed, STREAM_TOKEN, j, n_j)`` (``n_j``: tokens the particle has emitted), and add
   ``log_alpha[j]`` to its log-weight; a particle with nothing allowed is **dead** (weight 0);
4. feed every new token to the language model (``LanguageModel.step``) in one batch.

Resampling (``resample_at``):

- ``"sentence"`` (default, PLAN 4G): a particle that completes a sentence (the registers: a
  terminator run opens or is counted outside a calculator span), emits ``END``, dies, or has
  written ``unit_max_tokens`` tokens since the last resampling point **waits**; when no particle
  is running, every particle is at such a point and the filter resamples (systematic, with
  ``U(seed, STREAM_RESAMPLE, 0, event)``) if ``ESS < ess_frac * K``. Finished particles wait
  for good and may be selected (their copies are finished); dead ones are replaced. Waiting
  costs synchronous steps (padding rows for the LM), so a particle waits only while the steps
  left (``max_steps``) can absorb a whole unit (at most ``min(unit_max_tokens, largest remaining
  budget)`` steps) plus the largest remaining token budget; after that the filter steps without
  waiting (and without resampling) so that no particle is ever cut by ``max_steps``. Every rule here depends only on the current particle system, so the
  filter stays exact.
- ``"token"``: the same test after every synchronous step (no waiting).

Resampling copies everything a particle owns: LM cache rows and feature providers
(``LanguageModel.reorder``), the exact tracker, n-gram and copy-cap histories, pending forced
tokens, the stored next-token log-probabilities and the twist value.

The estimate of the normalising constant ``Z = sum_x prod_t gamma_t`` is
``log_z = sum over resampling events of log mean(w) + log mean(w_final * finished)``; it is
unbiased (F1) and the weighted finished particles converge to ``pi`` as ``K`` grows. ``K = 1``
without potentials, blocking or copy cap is ordinary sampling, draw for draw.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from bes.decode import sampling as S
from bes.decode.potentials import Program, Tracker, Twist, compile, table_for
from bes.exact import instr_table as it

RUN, WAIT, STOP, LENGTH, DEAD = range(5)
_FINISH = {STOP: "stop", LENGTH: "length", DEAD: "dead", RUN: "length", WAIT: "length"}


@dataclass
class ParticleResult:
    """One particle at the end: its response tokens (``END`` included when it finished), its
    final log-weight (since the last resampling; ``-inf`` when dead or unfinished), the model's
    log-probability of every token (forced ones included) and their sum, which tokens were
    forced, and the finish reason (``"stop"``, ``"length"`` or ``"dead"``)."""

    tokens: list[int]
    log_weight: float
    logp_model: float
    token_logp: list[float]
    forced: list[bool]
    finish: str

    @property
    def finished(self) -> bool:
        return self.finish == "stop"


@dataclass
class SMCResult:
    """All particles, the normalising-constant estimate and per-step diagnostics.

    ``weights``: normalised weights of the particles (0 for unfinished and dead ones; all 0
    when none finished); ``ess``: effective sample size after every synchronous step;
    ``log_alpha``: ``float64[K]`` log incremental weight of every step (NaN where a particle
    did not sample: waiting, finished, forced); ``resampled``: ``{"step", "ess", "ancestors"}``
    per resampling event; ``forced_spans``: per particle ``(start, stop)`` token positions of the
    forced spans; ``steps``: synchronous steps (LM calls after the prefill)."""

    particles: list[ParticleResult]
    log_z: float
    weights: np.ndarray
    ess: list[float]
    log_alpha: list[np.ndarray]
    resampled: list[dict]
    forced_spans: list[list[tuple[int, int]]]
    steps: int
    extra: dict = field(default_factory=dict)

    def best(self) -> ParticleResult | None:
        """The finished particle with the largest weight (``None`` when none finished)."""
        if not np.any(self.weights > 0):
            return None
        return self.particles[int(np.argmax(self.weights))]


def _logmeanexp(lw: np.ndarray) -> float:
    lw = np.asarray(lw, np.float64)
    m = np.max(lw)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.mean(np.exp(lw - m))))


def library_passages(prompt_ids: Sequence[int], table: it.InstrTable) -> list[list[int]]:
    """Content tokens of every ``LIBRARY`` turn of a prompt (up to its ``END``)."""
    out, cur = [], None
    lib = [i for i in np.flatnonzero(table.role_code == 4)]
    lib_id = int(lib[0]) if lib else -1
    kind = table.kind
    for t in prompt_ids:
        t = int(t)
        if t == lib_id:
            cur = []
            continue
        if cur is not None:
            if kind[t] != it.KIND_ORDINARY:
                out.append(cur)
                cur = None
            else:
                cur.append(t)
    if cur:
        out.append(cur)
    return out


def decode(lm, prompt_ids: Sequence[int], *, n_particles: int, max_new_tokens: int, seed: int = 0,
           settings: S.SamplingSettings = S.SamplingSettings(), program: Program | None = None,
           tok=None, table: it.InstrTable | None = None, twist: Twist | None = None,
           ess_frac: float = 0.5, resample_at: str = "sentence", unit_max_tokens: int = 32,
           no_repeat_ngram: int = 0, max_copy_run: int = 0,
           library: Sequence[Sequence[int]] | None = None,
           sources: Sequence[Sequence[int]] | None = None, max_steps: int | None = None,
           greedy: bool = False) -> SMCResult:
    """Run the particle filter on one prompt (module docstring).

    lm: a ``bes.decode.lm.LanguageModel``; program: compiled potentials (``None``: none; give
    ``tok`` and/or ``table`` so that calculator spans, sentence boundaries and the code / quote
    exemptions still work); twist: an optional ``potentials.Twist`` of ``program``;
    no_repeat_ngram / max_copy_run: sizes of the degeneracy guards (0 = off); library: LIBRARY
    passages exempt from n-gram blocking (default: the prompt's LIBRARY turns); sources: copy
    sources of the copy cap (default: the whole prompt); max_steps: cap on synchronous steps
    (waiting included; default ``4 * max_new_tokens``, at most the LM's ``capacity`` when it has
    one, e.g. ``OurLM``); greedy: arg-max instead of drawing
    (deterministic, not a sampler: diagnostics only).
    """
    k = int(n_particles)
    t_max = int(max_new_tokens)
    if k < 1 or t_max < 1:
        raise ValueError("n_particles and max_new_tokens must be >= 1")
    if resample_at not in ("sentence", "token"):
        raise ValueError("resample_at must be 'sentence' or 'token'")
    if program is not None:
        tok, table = program.tok, program.table
    elif table is None and tok is not None:
        table = table_for(tok)
    if twist is not None and (program is None or twist.p is not program):
        raise ValueError("the twist must belong to the program")
    if twist is not None and not (program.budget_aware and program.reachability):
        raise ValueError("a twist needs a budget-aware program with reachability pruning "
                         "(psi must be positive wherever a token is allowed)")
    v = lm.vocab_size
    tracker = program.tracker(k, t_max) if program is not None else \
        Tracker(None, k, t_max, table=table, tok=tok)
    special = (table.kind != it.KIND_ORDINARY) if table is not None else None
    end_id = int(table.special(it.KIND_END)) if table is not None else None
    if end_id is None and tok is not None:
        end_id = int(tok.special_tokens["<|end|>"])
    if end_id is None:
        raise ValueError("give a program, a tokenizer or a table (the END id is needed)")
    blocker = capper = None
    if (no_repeat_ngram > 0 or max_copy_run > 0) and table is not None and tok is not None:
        tbytes = program.token_bytes if program is not None else \
            [tok.token_bytes(i) for i in range(table.vocab_size)]
        word, exempt_tok = S.token_classes(tbytes, special)
        if no_repeat_ngram > 0:
            lib = library_passages(prompt_ids, table) if library is None else library
            blocker = S.NgramBlocker(no_repeat_ngram, k, word=word, special=special, library=lib)
        if max_copy_run > 0:
            src = [list(prompt_ids)] if sources is None else sources
            capper = S.CopyCap(max_copy_run, k, src, exempt=exempt_tok, special=special)
    max_steps = 4 * t_max if max_steps is None else int(max_steps)

    logp = np.asarray(lm.prefill(list(prompt_ids), k), np.float32)
    if getattr(lm, "capacity", None) is not None:
        max_steps = min(max_steps, int(lm.capacity))
    status = np.full(k, RUN, np.int64)
    logw = np.zeros(k)
    psi_cur = np.zeros(k)
    if twist is not None:
        psi_cur = twist.value(tracker, np.arange(k), np.full(k, t_max))
        logw += psi_cur
    log_z = 0.0
    tokens: list[list[int]] = [[] for _ in range(k)]
    token_lp: list[list[float]] = [[] for _ in range(k)]
    forced_flags: list[list[bool]] = [[] for _ in range(k)]
    spans: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    queue: list[list[int]] = [[] for _ in range(k)]
    unit_len = np.zeros(k, np.int64)
    ess_trace: list[float] = []
    alpha_trace: list[np.ndarray] = []
    events: list[dict] = []
    n_events = 0
    steps = 0

    def resample_if_needed():
        nonlocal logw, logp, status, log_z, n_events, tokens, token_lp, forced_flags, spans, \
            queue, unit_len, psi_cur
        e = S.ess(logw)
        if not (e < ess_frac * k) or not np.isfinite(np.max(logw)):
            return
        log_z += _logmeanexp(logw)
        u = float(S.counter_uniform(seed, S.STREAM_RESAMPLE, 0, n_events)[()])
        anc = S.systematic(logw, u)
        n_events += 1
        events.append({"step": steps, "ess": e, "ancestors": anc.copy()})
        lm.reorder(anc)
        tracker.select(anc)
        if blocker is not None:
            blocker.select(anc)
        if capper is not None:
            capper.select(anc)
        logp = logp[anc]
        status = status[anc]
        unit_len = unit_len[anc]
        psi_cur = psi_cur[anc]
        tokens = [list(tokens[a]) for a in anc]
        token_lp = [list(token_lp[a]) for a in anc]
        forced_flags = [list(forced_flags[a]) for a in anc]
        spans = [list(spans[a]) for a in anc]
        queue = [list(queue[a]) for a in anc]
        logw = np.zeros(k)

    while True:
        running = status == RUN
        if not running.any():
            waiting = status == WAIT
            if not waiting.any():
                break
            resample_if_needed()
            status = np.where(status == WAIT, RUN, status)
            unit_len[:] = 0
            continue
        if steps >= max_steps:
            break
        x = np.full(k, -1, np.int64)
        alpha = np.full(k, np.nan)
        has_q = np.array([bool(queue[j]) for j in range(k)])
        free = running & ~has_q
        frows = np.flatnonzero(free)
        if frows.size:
            allowed = tracker.allowed(free)
            keep = None
            if blocker is not None or capper is not None:
                keep = np.ones((len(frows), v), bool)
                ex = tracker.exempt()
                for r, j in enumerate(frows):
                    if ex[j]:
                        continue
                    for guard in (blocker, capper):
                        if guard is not None:
                            ids = guard.blocked(int(j))
                            if ids:
                                keep[r, ids] = False
            lg, lw = S.step_increment(logp[frows], settings,
                                      None if allowed is None else allowed[frows], keep)
            if twist is not None:
                rem = tracker.remaining()[frows]
                tv = twist.token_values(tracker, frows, rem - 1)
                ids = program.ids
                for r, j in enumerate(frows):
                    tv[r, ids["end"]] = 0.0
                    tv[r, ids["calc"]] = twist.value(tracker, np.array([j]),
                                                     np.array([rem[r] - 1]))[0]
                    if np.isfinite(lg[r, ids["eq"]]):
                        fz = tracker.forced_after_eq(int(j))
                        st = tracker.simulate(int(j), [table.eq_row] + fz)
                        tv[r, ids["eq"]] = twist.state_value(st, int(rem[r] - 1 - len(fz)))
                lg = lg + tv - psi_cur[frows][:, None]
                lg = np.where(np.isnan(lg), -np.inf, lg)
                lw = S.logsumexp(lg)
            if greedy:
                drawn = S.greedy(lg)
            else:
                u = S.counter_uniform(seed, S.STREAM_TOKEN, frows, tracker.n[frows])
                drawn = S.draw(lg, u)
            dead = drawn < 0
            alpha[frows] = lw
            x[frows] = drawn
            logw[frows] = logw[frows] + np.where(dead, -np.inf, lw)
            if twist is not None:
                ok = np.flatnonzero(~dead)
                psi_cur[frows[ok]] = tv[ok, drawn[ok]]      # the numerator of this step
            status[frows[dead]] = DEAD
            logw[frows[dead]] = -np.inf
        qrows = np.flatnonzero(running & has_q)
        for j in qrows:
            x[j] = queue[j].pop(0)
        stepping = running & (x >= 0)
        srows = np.flatnonzero(stepping)
        for j in srows:
            tokens[j].append(int(x[j]))
            token_lp[j].append(float(logp[j, x[j]]))
            forced_flags[j].append(bool(has_q[j]))
            if blocker is not None:
                blocker.push(int(j), int(x[j]))
            if capper is not None:
                capper.push(int(j), int(x[j]))
        adv = tracker.advance(x, stepping)
        for j in srows:
            fz = adv.forced[j]
            if fz:
                queue[j] = list(fz)
                start = len(tokens[j])
                spans[j].append((start, start + len(fz)))
        unit_len[srows] += 1
        status[adv.finished] = STOP
        over = stepping & (status == RUN) & (tracker.n >= t_max)
        status[over] = LENGTH
        logw[over] = -np.inf
        if resample_at == "sentence":
            # a unit lasts at most min(unit cap, largest remaining budget) more steps; after it
            # every particle must still fit its whole remaining budget into max_steps
            live = (status == RUN) | (status == WAIT)
            need_steps = int(tracker.remaining()[live].max()) if live.any() else 0
            unit_cost = min(unit_max_tokens, need_steps)
            if steps + 1 + unit_cost + need_steps <= max_steps:
                wait = stepping & (status == RUN) & (adv.boundary | (unit_len >= unit_max_tokens))
                status[wait] = WAIT
        need = stepping & ((status == RUN) | (status == WAIT))
        if need.any():
            new = np.asarray(lm.step(np.where(need, x, 0), need), np.float32)
            logp[need] = new[need]
        steps += 1
        alpha_trace.append(alpha)
        ess_trace.append(S.ess(logw))
        if resample_at == "token":
            resample_if_needed()
    unfinished = (status == RUN) | (status == WAIT)
    status[unfinished] = LENGTH
    logw[unfinished] = -np.inf
    final = np.where(status == STOP, logw, -np.inf)
    log_z += _logmeanexp(final)
    if np.isfinite(np.max(final)):
        w = np.exp(final - np.max(final))
        weights = w / w.sum()
    else:
        weights = np.zeros(k)
    particles = [ParticleResult(tokens[j], float(logw[j]), float(np.sum(token_lp[j])), token_lp[j],
                                forced_flags[j], _FINISH[int(status[j])]) for j in range(k)]
    return SMCResult(particles, float(log_z), weights, ess_trace, alpha_trace, events, spans,
                     steps)


def run(lm, prompt_ids: Sequence[int], *, spec=None, tok=None, cfg=None, seed: int = 0,
        n_particles: int | None = None, potentials: bool = True, twist: Twist | None = None,
        **overrides) -> SMCResult:
    """:func:`decode` configured from a ``bes.config.DecodeCfg`` (``cfg``; default values when
    ``None``): temperature / top-p / top-k / min-p / truncate_after_mask, particles,
    max_new_tokens, ess_frac, no_repeat_ngram, max_copy_run, and the potential options
    ``max_depth``, ``budget_aware``, ``resample_at``, ``unit_max_tokens`` when the config has
    them. ``spec``: a ``ConstraintSpec`` (or ``None``); ``potentials=False`` turns every potential
    off. Keyword ``overrides`` go to :func:`decode`."""
    from bes.config import DecodeCfg

    cfg = DecodeCfg() if cfg is None else cfg
    if tok is None:
        from bes.tokenizer.tok import load as load_tokenizer

        tok = load_tokenizer()
    program = None
    if potentials:
        program = compile(spec, tok, max_depth=getattr(cfg, "max_depth", 8),
                          budget_aware=getattr(cfg, "budget_aware", True))
    kw = dict(n_particles=n_particles or cfg.particles, max_new_tokens=cfg.max_new_tokens,
              seed=seed, settings=S.SamplingSettings.from_cfg(cfg), program=program, tok=tok,
              twist=twist, ess_frac=cfg.ess_frac,
              resample_at=getattr(cfg, "resample_at", "sentence"),
              unit_max_tokens=getattr(cfg, "unit_max_tokens", 32),
              no_repeat_ngram=cfg.no_repeat_ngram, max_copy_run=cfg.max_copy_run)
    kw.update(overrides)
    return decode(lm, prompt_ids, **kw)


__all__ = ["decode", "run", "SMCResult", "ParticleResult", "library_passages"]
