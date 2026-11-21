"""The chat entry point: one reply to a conversation (people and the evaluation harness).

``Responder(lm_factory, tok, cfg, library=None).reply(messages, settings, seed)`` returns
``{"text", "finish_reason", "prompt_tokens", "completion_tokens", "logprob_sum",
"logprob_count", "extra"}``:

1. **Messages.** ``[{"role": "system" | "user" | "assistant", "content": str}, ...]`` ending
   with a user message (earlier assistant messages are the visible texts of earlier replies).
   Library messages given by the caller are dropped; consecutive messages of one role are
   joined with a blank line.
2. **Constraints** of the last user message: ``bes.exact.constraints.compile`` (an empty spec
   when nothing explicit is asked).
3. **Library gate** (``library.retrieve(user_text) -> Passage | None``, when a library is given
   and ``use_library`` is on): at most one ``LIBRARY`` turn, placed right before the last user
   turn; earlier turns never carry a passage.
4. **Prompt**: the chat format renders the conversation with a generation prompt
   (:class:`OurChat`: ``bes.data.chat_format``; ``bes.decode.gguf_lm.GGUFChat`` for stand-ins).
   It must fit ``context - max_new_tokens`` tokens: the oldest exchanges (user + assistant) are
   dropped first, then the passage, then the system message, then the middle of the last user
   message (its head and tail, where instructions are, are kept).
5. **Feasibility**: when no accepted response fits ``max_new_tokens`` (the program's
   ``start_distance()``, ``END`` included, exceeds it, e.g. "at least 900 words" with a 768-token
   budget), the count constraints that cannot be met are dropped (each one that cannot be met
   alone, then, while the rest still cannot be met together, the one needing the most tokens);
   the rest is kept and ``extra["infeasible"]`` lists what was dropped. Selection then uses the
   reduced spec and no length band.
6. **Particle filter** ``bes.decode.smc.decode`` with the compiled potentials of the spec
   (``potentials`` on; off: plain sampling), the decoding settings (below) and the seed. The
   language model is told the request's ``self_evidence_weight`` (multiplier of tape evidence
   from the turn being generated; ``extra["self_evidence_weight"]``, ``None`` for a model
   without a tape) and, by the decoder, the temperature (``OurLM`` tempers its network's
   distribution only; ``extra["tempered_by"]``). The copy cap (``max_copy_run``) acts on
   LIBRARY passages only and is switched off for specs that need copied text (repeat the
   request, answer with one of the options, start or end with a given phrase).
7. **Selection** ``bes.decode.select.select`` among the particles; the reply is the chosen
   particle's display text (calculator spans shown as ``expr = result``), stripped. When every
   particle died, or the reply would be empty because particles died, the decoder runs again
   with the structural potentials only (``extra["rerun"]``), so the reply is never empty for
   that reason.

Deterministic given ``(messages, settings, seed)``: the decoder's random numbers are
counter based on ``seed``, retrieval and selection are deterministic.

Settings (a plain dict, the contract of ``bes/eval/proxy/run_ours.py``): ``temperature``,
``top_p``, ``top_k``, ``min_p`` override the decoding config; ``max_tokens`` caps
``max_new_tokens`` (the smaller one is used unless ``max_new_tokens`` itself is given);
``temperature <= 0`` decodes greedily (diagnostics). Every field of ``bes.config.DecodeCfg``
may be given by name (``particles``, ``use_library``, ``no_repeat_ngram``, ``resample_at``,
...). Decoder switches of this module: ``potentials`` (default on), ``band`` (length band in
the selection, default on), ``utility`` (consensus utility, default ``"f12"``), ``trim`` (the
no-leading/trailing-whitespace potential, default on). ``repeat_penalty`` is not implemented
(n-gram blocking is the decoder's repetition control) and ``use_critic`` is ignored (there is
no critic in the decoder); both are recorded in ``extra["ignored"]`` when set, as is an
explicit ``self_evidence_weight`` for a model without a candidate tape.

``logprob_sum`` / ``logprob_count`` sum the decoder's per-token log-probabilities
(``smc.ParticleResult.token_logp``): under the distribution the decoder samples from, the
model at the request's temperature (the model's own distribution at temperature 1).

:func:`our_generator` is the harness factory (``--generator bes.decode.respond:our_generator
--generator-arg checkpoint=... --generator-arg config=... [--generator-arg library=...]``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import DecodeCfg

REPO = Path(__file__).resolve().parents[2]
#: Bump when a change alters replies for the same checkpoint and settings.
#: 2: temperature on the network only, self-evidence weight applied, feasibility check and
#: structural rerun, copy cap on LIBRARY passages only, token-local potentials, smooth mass cap.
RESPOND_VERSION = "2"
#: count constraints the feasibility check may drop (``bes.exact.constraints.COUNT_UNITS``)
COUNT_FIELDS = ("words", "sentences", "items", "paragraphs", "lines")
_DECODE_FIELDS = {f.name: f for f in dataclasses.fields(DecodeCfg)}
_SWITCHES = {"potentials": True, "band": True, "utility": "f12", "trim": True}
_HARNESS_ONLY = ("max_tokens", "system_prompt")
_IGNORED_DEFAULTS = {"repeat_penalty": 1.0, "use_critic": False}
_ROLES = ("system", "user", "assistant")


# ============================================================================ chat formats
class OurChat:
    """The project's chat format (``bes.data.chat_format``) for the committed tokenizer."""

    supports_library = True

    def __init__(self, tok):
        from bes.config import END

        self.tok = tok
        self.end_id = END

    def render(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        from bes.data.chat_format import render_conversation

        ids, _ = render_conversation(list(messages), self.tok, add_generation_prompt=True)
        return [int(t) for t in ids]

    def display(self, tokens: Sequence[int]) -> str:
        from bes.data.chat_format import display_text

        return display_text(list(tokens), self.tok)


# ================================================================================ settings
def _coerce(default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value)) if isinstance(value, str) else int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


@dataclasses.dataclass(frozen=True)
class Resolved:
    """Decoding configuration of one reply: the ``DecodeCfg`` after the settings, the switches
    of this module, greedy decoding, and the settings that were ignored."""

    decode: DecodeCfg
    potentials: bool = True
    band: bool = True
    utility: str = "f12"
    trim: bool = True
    greedy: bool = False
    ignored: tuple[tuple[str, Any], ...] = ()


def resolve(cfg: DecodeCfg, settings: Mapping[str, Any] | None) -> Resolved:
    """Merge the harness settings into ``cfg`` (module docstring)."""
    s = dict(settings or {})
    values = dataclasses.asdict(cfg)
    switches = dict(_SWITCHES)
    ignored = []
    for name, value in s.items():
        if value is None:
            continue
        if name in _DECODE_FIELDS and name not in _IGNORED_DEFAULTS:
            values[name] = _coerce(getattr(cfg, name), value)
        elif name in switches:
            switches[name] = _coerce(_SWITCHES[name], value)
        elif name in _IGNORED_DEFAULTS:
            if _coerce(_IGNORED_DEFAULTS[name], value) != _IGNORED_DEFAULTS[name]:
                ignored.append((name, value))
        elif name not in _HARNESS_ONLY:
            ignored.append((name, value))
    if s.get("max_tokens") is not None and s.get("max_new_tokens") is None:
        values["max_new_tokens"] = min(int(values["max_new_tokens"]), int(s["max_tokens"]))
    greedy = float(values["temperature"]) <= 0.0
    if greedy:
        values["temperature"] = 1.0
    if values["particles"] < 1 or values["max_new_tokens"] < 1:
        raise ValueError("particles and max_new_tokens must be >= 1")
    return Resolved(DecodeCfg(**values), bool(switches["potentials"]), bool(switches["band"]),
                    str(switches["utility"]), bool(switches["trim"]), greedy,
                    tuple(sorted((str(k), v) for k, v in ignored)))


# ============================================================================ conversation
def clean_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Validated copy: roles in system / user / assistant (library messages dropped), string
    contents, consecutive messages of one role joined with a blank line, a system message only
    first, and a last message from the user."""
    out: list[dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, Mapping):
            raise ValueError(f"message {i} is not a mapping")
        role, content = m.get("role"), m.get("content")
        if role == "library":
            continue
        if role not in _ROLES:
            raise ValueError(f"message {i}: unknown role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message {i}: content must be a string")
        if role == "system" and out:
            raise ValueError("a system message may only come first")
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n\n" + content}
        else:
            out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "user":
        raise ValueError("the conversation must end with a user message")
    return out


def _cut_middle(text: str, keep: int) -> str:
    if keep >= len(text):
        return text
    head = max(1, keep // 2)
    tail = max(1, keep - head)
    return text[:head].rstrip() + "\n...\n" + text[len(text) - tail:].lstrip()


def fit_prompt(chat, messages: Sequence[Mapping[str, str]], passage: str | None,
               budget: int) -> tuple[list[int], dict[str, Any]]:
    """Render ``messages`` (+ the passage before the last user turn) within ``budget`` tokens
    (module docstring, step 4). Returns ``(ids, info)`` with ``info = {"dropped_turns",
    "dropped_passage", "dropped_system", "cut_user"}``."""
    if budget < 1:
        raise ValueError("no room for the prompt")
    msgs = [dict(m) for m in messages]
    system = msgs[0] if msgs and msgs[0]["role"] == "system" else None
    body = msgs[1:] if system is not None else msgs
    last = body[-1]
    history = body[:-1]
    info = {"dropped_turns": 0, "dropped_passage": False, "dropped_system": False,
            "cut_user": False}

    def render(hist, psg, sys_msg, user):
        conv = ([sys_msg] if sys_msg is not None else []) + list(hist)
        if psg is not None:
            conv.append({"role": "library", "content": psg})
        conv.append(user)
        return chat.render(conv)

    ids = render(history, passage, system, last)
    while len(ids) > budget and history:
        cut = 2 if len(history) >= 2 and history[0]["role"] == "user" else 1
        history = history[cut:]
        while history and history[0]["role"] == "assistant":
            history = history[1:]
            cut += 1
        info["dropped_turns"] += cut
        ids = render(history, passage, system, last)
    if len(ids) > budget and passage is not None:
        passage, info["dropped_passage"] = None, True
        ids = render(history, passage, system, last)
    if len(ids) > budget and system is not None:
        system, info["dropped_system"] = None, True
        ids = render(history, passage, system, last)
    if len(ids) > budget:
        content = last["content"]
        lo, hi = 0, len(content)
        best = None
        while lo <= hi:                              # longest kept prefix + suffix that fits
            mid = (lo + hi) // 2
            cand = render(history, passage, system, {"role": "user",
                                                     "content": _cut_middle(content, mid)})
            if len(cand) <= budget:
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError(f"the prompt cannot fit {budget} tokens")
        ids, info["cut_user"] = best, True
    return ids, info


# ======================================================================== decoding helpers
def needs_copied_text(spec) -> bool:
    """The spec asks for text taken verbatim from the prompt (repeat the request, answer with
    one of given options, start or end with a given phrase): the copy cap must stay off."""
    if spec is None:
        return False
    return bool(getattr(spec, "repeat", None) or getattr(spec, "one_of", ())
                or getattr(spec, "starts_with", ()) or getattr(spec, "end_phrase", None))


def set_self_evidence(lm, weight: float) -> bool:
    """Give the language model the self-evidence weight (``set_self_evidence_weight``); False
    when it has no candidate tape (the weight does nothing there)."""
    setter = getattr(lm, "set_self_evidence_weight", None)
    return bool(setter(float(weight))) if callable(setter) else False


def drop_counts(spec, units: Sequence[str]):
    """``spec`` without the count constraints ``units``."""
    return dataclasses.replace(spec, **{u: None for u in units}) if units else spec


# ============================================================================== responder
class Responder:
    """One reply per call (module docstring).

    lm_factory: ``lm_factory(max_new_tokens) -> LanguageModel`` (``bes.decode.lm``), called
    once per reply (it may return the same object every time); tok: the tokenizer the
    potentials and the copy guards are compiled for (``bes.tokenizer.load()`` for our model, a
    ``GGUFVocab`` for a stand-in); cfg: a ``bes.config.Cfg`` (``cfg.decode`` and
    ``cfg.model.max_seq``) or a ``DecodeCfg``; library: an object with ``retrieve(text) ->
    Passage | None`` (``bes.library.retrieve.Library``), or ``None``; chat: the chat format
    (default :class:`OurChat`); context: tokens of prompt + reply (default
    ``cfg.model.max_seq``, else 2048).
    """

    def __init__(self, lm_factory: Callable[[int], Any], tok, cfg=None, library=None, *,
                 chat=None, context: int | None = None):
        self.lm_factory = lm_factory
        self.tok = tok
        decode = getattr(cfg, "decode", cfg)
        self.cfg = decode if isinstance(decode, DecodeCfg) else DecodeCfg()
        model = getattr(cfg, "model", None)
        self.context = int(context) if context is not None else \
            int(getattr(model, "max_seq", 2048) if model is not None else 2048)
        self.library = library
        self.chat = chat if chat is not None else OurChat(tok)
        if library is not None and not getattr(self.chat, "supports_library", False):
            raise ValueError("this chat format cannot hold a library passage")

    # -- pieces ---------------------------------------------------------------------------
    def compile_spec(self, user_text: str):
        from bes.exact import constraints

        try:
            return constraints.compile(user_text), None
        except Exception as err:                    # a compiler bug must not kill the reply
            return constraints.ConstraintSpec(), f"{type(err).__name__}: {err}"

    def retrieve(self, user_text: str):
        if self.library is None:
            return None
        return self.library.retrieve(user_text)

    def program(self, spec, r: Resolved):
        from bes.decode import potentials

        return potentials.compile(spec if spec is not None and not spec.empty else None, self.tok,
                                  max_depth=r.decode.max_depth, budget_aware=r.decode.budget_aware,
                                  trim=r.trim)

    def feasible_program(self, spec, r: Resolved, budget: int):
        """``(program, spec_used, dropped)``: the program of ``spec`` when an accepted response
        fits ``budget`` tokens (``program.start_distance() <= budget``). Otherwise the count
        constraints that cannot be met are dropped (module docstring, step 5): first every count
        that cannot be met on its own (with the spec's other constraints), then, while the rest
        still cannot be met together, the remaining count needing the most tokens; the last
        resort is the structural program. ``dropped``: the descriptions of what was dropped
        (``ConstraintSpec.constraints`` wording)."""
        prog = self.program(spec, r)
        if spec is None or prog.start_distance() <= budget:
            return prog, spec, ()
        units = [u for u in COUNT_FIELDS if getattr(spec, u, None) is not None]
        need = {}
        for u in units:
            need[u] = self.program(drop_counts(spec, [x for x in units if x != u]), r).start_distance()
        drop = [u for u in units if need[u] > budget]
        keep = [u for u in units if u not in drop]
        while True:
            reduced = drop_counts(spec, drop)
            prog = self.program(reduced, r)
            if prog.start_distance() <= budget or not keep:
                break
            worst = max(keep, key=lambda u: (need[u], -COUNT_FIELDS.index(u)))
            keep.remove(worst)
            drop.append(worst)
        dropped = [f"{u}: {getattr(spec, u)}" for u in COUNT_FIELDS if u in drop]
        if prog.start_distance() > budget:          # not even without counts: structure only
            dropped = list(spec.constraints)
            reduced, prog = None, self.program(None, r)
        return prog, reduced, tuple(dropped)

    # -- one reply ------------------------------------------------------------------------
    def reply(self, messages: Sequence[Mapping[str, Any]], settings: Mapping[str, Any] | None = None,
              seed: int = 0) -> dict[str, Any]:
        from bes.decode import sampling as S
        from bes.decode import select as sel_mod
        from bes.decode import smc
        from bes.decode.potentials import enforced

        r = resolve(self.cfg, settings)
        d = r.decode
        msgs = clean_messages(messages)
        user_text = msgs[-1]["content"]
        history = [m["content"] for m in msgs[:-1] if m["role"] == "user"]
        spec, spec_error = self.compile_spec(user_text)
        passage = self.retrieve(user_text) if d.use_library else None
        budget = self.context - d.max_new_tokens
        prompt, fit = fit_prompt(self.chat, msgs, None if passage is None else passage.text, budget)
        program, prog_error = None, None
        spec_used, infeasible = spec, ()
        if r.potentials:
            try:
                program, spec_used, infeasible = self.feasible_program(spec, r, d.max_new_tokens)
            except Exception as err:               # fall back to the structural potentials
                prog_error = f"{type(err).__name__}: {err}"
                program, spec_used = self.program(None, r), spec
        if spec_used is None:
            from bes.exact import constraints

            spec_used = constraints.ConstraintSpec()
        lm = self.lm_factory(d.max_new_tokens)
        self_evidence = set_self_evidence(lm, d.self_evidence_weight)
        copy_cap = 0 if needs_copied_text(spec) else d.max_copy_run
        sampling = S.SamplingSettings.from_cfg(d)

        def run(prog):
            return smc.decode(lm, prompt, n_particles=d.particles, max_new_tokens=d.max_new_tokens,
                              seed=int(seed), settings=sampling, program=prog, tok=self.tok,
                              ess_frac=d.ess_frac, resample_at=d.resample_at,
                              unit_max_tokens=d.unit_max_tokens, no_repeat_ngram=d.no_repeat_ngram,
                              max_copy_run=copy_cap, greedy=r.greedy)

        def choose(res, sel_spec, band):
            cands = [sel_mod.Candidate(text=self.chat.display(q.tokens).strip(),
                                       weight=float(res.weights[j]), finish=q.finish,
                                       n_tokens=len(q.tokens), logp=q.logp_model)
                     for j, q in enumerate(res.particles)]
            return sel_mod.select(cands, sel_spec, user_text, history=history, band=band,
                                  utility_kind=r.utility, budget=d.max_new_tokens)

        band = "auto" if (r.band and not infeasible) else None
        res = run(program)
        choice = choose(res, spec_used, band)
        rerun = None
        dead = [q.finish == "dead" for q in res.particles]
        if program is not None and any(dead) and (all(dead) or not choice.text):
            # never an empty reply because the potentials killed the particles: decode again
            # with the structural potentials only (the spec is still used by the selection)
            rerun = {"reason": "all particles died" if all(dead) else "empty reply",
                     "finishes": dict(Counter(q.finish for q in res.particles))}
            res = run(self.program(None, r))
            choice = choose(res, spec_used, band)
        p = res.particles[choice.index]
        free = [lp for lp, f in zip(p.token_logp, p.forced) if not f]
        extra = {
            "particles": d.particles, "potentials": r.potentials, "rule": choice.rule,
            "n_survivors": choice.n_survivors,
            "n_finished": sum(1 for q in res.particles if q.finish == "stop"),
            "finishes": dict(Counter(q.finish for q in res.particles)),
            "finish": p.finish, "prompt_type": choice.prompt_type,
            "band": None if choice.band is None else str(choice.band), "answer": choice.answer,
            "constraints": list(spec.constraints),
            "enforced": enforced(spec_used) if program is not None and rerun is None else [],
            "violations": [x for x in choice.reasons.get(choice.index, []) if x.startswith("violation")],
            "library": None if passage is None else passage.title,
            "calc_spans": len(res.forced_spans[choice.index]),
            "tokens_all": int(sum(len(q.tokens) for q in res.particles)),
            "tokens_sampled": int(sum(int(np.count_nonzero(~np.isnan(a))) for a in res.log_alpha)),
            "steps": int(res.steps), "resampled": len(res.resampled),
            "ess_min": round(float(min(res.ess)), 4) if res.ess else None,
            "log_z": None if not np.isfinite(res.log_z) else round(float(res.log_z), 4),
            "logprob_sum_free": float(np.sum(free)), "logprob_count_free": len(free),
            "temperature": res.extra.get("temperature"), "tempered_by": res.extra.get("tempered_by"),
            "self_evidence_weight": d.self_evidence_weight if self_evidence else None,
            "copy_cap": copy_cap, "fit": fit, "seed": int(seed), "greedy": r.greedy,
        }
        if infeasible:
            extra["infeasible"] = list(infeasible)
        if rerun is not None:
            extra["rerun"] = rerun
        ignored = {k: v for k, v in r.ignored}
        if not self_evidence and (settings or {}).get("self_evidence_weight") is not None:
            ignored["self_evidence_weight"] = (settings or {})["self_evidence_weight"]
        if ignored:
            extra["ignored"] = ignored
        for name, err in (("spec_error", spec_error), ("program_error", prog_error)):
            if err:
                extra[name] = err
        return {"text": choice.text, "finish_reason": "stop" if p.finish == "stop" else "length",
                "prompt_tokens": len(prompt), "completion_tokens": len(p.tokens),
                "logprob_sum": float(np.sum(p.token_logp)), "logprob_count": len(p.token_logp),
                "extra": extra}

    __call__ = reply


# ========================================================================== our generator
def _sha_files(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(REPO)).encode() + b"\x00" + path.read_bytes() + b"\x00")
    return h.hexdigest()


def code_hash() -> str:
    """SHA-256 of the sources that decide a reply (decoder, exact layer, chat format, model,
    retrieval, tokenizer)."""
    files = [p for d in ("bes/decode", "bes/exact", "bes/model") for p in (REPO / d).glob("*.py")]
    files += [REPO / "bes/data/chat_format.py", REPO / "bes/library/retrieve.py",
              REPO / "bes/tokenizer/tok.py", REPO / "bes/config.py"]
    return _sha_files([p for p in files if p.exists()])


def find_checkpoint(path: str | Path) -> Path:
    """A checkpoint directory (with ``manifest.json``), or the newest complete one under a
    checkpoint root (``<run>/ckpt``) or a run directory."""
    from bes.train import checkpoint as ck

    p = Path(path)
    if (p / ck.MANIFEST).exists():
        return p
    for root in (p, p / "ckpt"):
        if root.is_dir():
            info = ck.latest(root)
            if info is not None:
                return info.path
    raise FileNotFoundError(f"no checkpoint at {p}")


def checkpoint_fingerprint(path: str | Path) -> str:
    """Hash of the manifest and of every array member's CRC-32 and size (cheap, exact)."""
    from bes.train import checkpoint as ck

    p = Path(path)
    h = hashlib.sha256((p / ck.MANIFEST).read_bytes())
    with zipfile.ZipFile(p / ck.ARRAYS) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            h.update(f"{info.filename}:{info.CRC}:{info.file_size}\n".encode())
    return h.hexdigest()


def load_checkpoint(path: str | Path, model_cfg) -> tuple[dict, dict, dict[str, Any]]:
    """``(params, consts, info)`` of a trainer checkpoint (``bes.train.checkpoint``: the tree
    ``{"state": TrainState(step, params, opt_state, consts), "rng"}``). The parameters must
    match ``bes.model.model.init_params(model_cfg)`` in paths, shapes and dtypes; the stored
    consts are used when they have the structure of ``make_consts(model_cfg)``, else the
    defaults."""
    import jax
    import jax.numpy as jnp

    from bes.model import model as M
    from bes.train import checkpoint as ck

    ckpt = find_checkpoint(path)
    tree, man = ck.load(ckpt, None, device=False)
    state = tree.get("state", tree)
    if "params" not in state:
        raise ValueError(f"{ckpt}: no params in the checkpoint")
    params = state["params"]
    want = jax.eval_shape(lambda: M.init_params(model_cfg, jax.random.PRNGKey(0)))
    got_l, got_t = jax.tree_util.tree_flatten_with_path(params)
    want_l, want_t = jax.tree_util.tree_flatten_with_path(want)
    if got_t != want_t:
        raise ValueError(f"{ckpt}: parameter tree differs from the model configuration")
    for (pth, a), (_, b) in zip(got_l, want_l):
        if tuple(a.shape) != tuple(b.shape) or np.dtype(a.dtype) != np.dtype(b.dtype):
            raise ValueError(f"{ckpt}: {jax.tree_util.keystr(pth)} is {a.dtype}{tuple(a.shape)}, "
                             f"the configuration wants {b.dtype}{tuple(b.shape)}")
    params = jax.tree.map(jnp.asarray, params)
    default = M.make_consts(model_cfg)
    stored = state.get("consts")
    consts = default
    if stored is not None and jax.tree.structure(stored) == jax.tree.structure(default) and all(
            np.shape(a) == np.shape(b) for a, b in zip(jax.tree.leaves(stored),
                                                     jax.tree.leaves(default))):
        consts = jax.tree.map(jnp.asarray, stored)
    info = {"path": str(ckpt), "step": man.get("step"), "seq": man.get("seq"),
            "config_hash": man.get("config_hash"),
            "consts": "stored" if consts is not default else "default"}
    return params, consts, info


def _library_fingerprint(path: str | Path) -> str:
    meta = Path(path) / "library.json"
    return hashlib.sha256(meta.read_bytes()).hexdigest() if meta.exists() else "none"


def our_generator(checkpoint: str, config: str, library: str = "", overrides: str = "",
                  key: str = "") -> Callable[..., dict[str, Any]]:
    """Harness factory (``bes/eval/proxy/run_ours.py`` contract; string keyword arguments):
    checkpoint: a checkpoint directory, checkpoint root or run directory; config: a
    ``configs/<name>.py`` file; library: a built library directory (empty: none);
    overrides: comma-separated ``section.field=value`` config overrides (``decode.particles=4``);
    key: cache key (default: derived from the checkpoint, the configuration, the code and the
    library). Returns ``generate(messages, settings, seed) -> dict`` with a ``key`` attribute."""
    from bes.config import config_hash, load_config, to_dict
    from bes.decode.lm import OurLM, cache_slots
    from bes.tokenizer import load as load_tokenizer

    items = [x.strip() for x in str(overrides).split(",") if x.strip()]
    cfg = load_config(str(config), items)
    params, consts, info = load_checkpoint(checkpoint, cfg.model)
    tok = load_tokenizer()
    lib = None
    if library:
        from bes.library.retrieve import Library

        lib = Library.load(str(library))

    n_cap = int(cfg.decode.max_new_tokens)

    def lm_factory(max_new_tokens: int):
        # about 2 x max_new_tokens synchronous steps (waiting included) is enough; the cache
        # holds any prompt that fits the context plus those steps for every request of up to
        # n_cap new tokens, so one compiled decoder per particle count serves every reply
        n = int(max_new_tokens)
        return OurLM(params, consts, cfg.model, tok, max_steps=2 * n,
                     tmax=cache_slots(responder.context, max(n, n_cap)),
                     self_evidence_weight=cfg.decode.self_evidence_weight)

    responder = Responder(lm_factory, tok, cfg, library=lib)

    def generate(messages, settings, seed):
        return responder.reply(messages, settings, seed)

    if not key:
        blob = {"v": RESPOND_VERSION, "ckpt": checkpoint_fingerprint(info["path"]),
                "cfg": config_hash(cfg), "decode": to_dict(cfg)["decode"], "code": code_hash(),
                "library": _library_fingerprint(library) if library else "none"}
        digest = hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()
        key = f"ours-step{info['step']}-{digest[:16]}"
    generate.key = key                     # type: ignore[attr-defined]
    generate.responder = responder         # type: ignore[attr-defined]
    generate.checkpoint = info             # type: ignore[attr-defined]
    return generate


__all__ = ["Responder", "OurChat", "Resolved", "resolve", "clean_messages", "fit_prompt",
           "our_generator", "load_checkpoint", "find_checkpoint", "checkpoint_fingerprint",
           "code_hash", "RESPOND_VERSION", "needs_copied_text", "set_self_evidence", "drop_counts"]
