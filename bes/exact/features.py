"""All exact-layer features of a batch: registers, clocks and the candidate tape.

``exact_features(tokens, seg, tok)`` is what the training loader and data relabelling call;
:class:`IncrementalFeatures` gives the same rows one token at a time for generation.

=============  ==============================  ===============================================
key            dtype, shape                    source
=============  ==============================  ===============================================
``reg``        int16 ``[B, T, N_REG]``         ``registers.compute`` (``bes.config.REG_FIELDS``)
``clock``      int32 ``[B, T, N_CLOCKS]``      ``registers.compute``
``cand_tok``   int32 ``[B, T, 16]``            ``tape.compute`` (-1 = empty slot)
``cand_feat``  int8 ``[B, T, 16, N_CAND_FEAT]``  ``tape.compute``
=============  ==============================  ===============================================

The countdown registers need per-position targets ``int32[B, T, 3]`` (sentences, items, words;
-1 = none), which ``constraints.targets_for_tokens(tokens, seg, tok)`` extracts from the user
turn that precedes each assistant turn. The registers and constraints modules are imported
lazily and guarded: when ``bes.exact.constraints`` is not importable the countdowns are left
without targets (``HAS_CONSTRAINTS`` is then False and a warning is issued once); when
``bes.exact.registers`` is missing, the functions raise ``RuntimeError``.
"""
from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np

from bes.config import BOS, CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG, ROLE_TOKENS
from bes.exact import tape

FEATURE_KEYS = ("reg", "clock", "cand_tok", "cand_feat")
_TURN_STARTS = frozenset((BOS, *ROLE_TOKENS))
_warned: set[str] = set()


def _optional(name: str):
    try:
        return importlib.import_module(f"bes.exact.{name}")
    except ImportError:
        return None


def _registers():
    mod = _optional("registers")
    if mod is None:
        raise RuntimeError("bes.exact.registers is not importable; registers cannot be computed")
    return mod


def _constraints():
    mod = _optional("constraints")
    if mod is None or not hasattr(mod, "targets_for_tokens"):
        if "constraints" not in _warned:
            _warned.add("constraints")
            warnings.warn("bes.exact.constraints.targets_for_tokens is not available: the "
                          "countdown registers (rem_sent, rem_item, rem_word) get no targets",
                          RuntimeWarning, stacklevel=3)
        return None
    return mod


def has_constraints() -> bool:
    """True when ``bes.exact.constraints.targets_for_tokens`` can be imported."""
    mod = _optional("constraints")
    return mod is not None and hasattr(mod, "targets_for_tokens")


def _instr_table(tok):
    """Compiled transducer tables of ``tok`` (``None``: the committed tokenizer)."""
    from bes.exact import instr_table

    return instr_table.load(tok)


def targets_for(tokens: np.ndarray, seg: np.ndarray | None, tok) -> np.ndarray | None:
    """``int32[B, T, 3]`` countdown targets, or ``None`` without the constraints module."""
    mod = _constraints()
    if mod is None:
        return None
    if tok is None:
        from bes.tokenizer.tok import load as load_tokenizer

        tok = load_tokenizer()
    return np.asarray(mod.targets_for_tokens(tokens, seg, tok), dtype=np.int32)


def exact_features(tokens: np.ndarray, seg: np.ndarray | None = None, tok=None,
                   targets: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Exact-layer features of ``tokens int[B, T]`` with segment ids ``seg int[B, T]``.

    tok: the tokenizer (``None``: the committed one); targets: ``int[B, T, 3]`` to use instead
    of ``constraints.targets_for_tokens``. Returns the dict of the module docstring.
    """
    tokens = np.asarray(tokens)
    registers = _registers()
    if targets is None:
        targets = targets_for(tokens, seg, tok)
    regs = registers.compute(tokens, seg, targets, table=_instr_table(tok))
    cand = tape.compute(tokens, seg)
    return {"reg": regs.reg, "clock": regs.clock, "cand_tok": cand.cand_tok,
            "cand_feat": cand.cand_feat}


class IncrementalFeatures:
    """The rows of :func:`exact_features` for one stream, one token at a time.

    ``step(token, new_segment=False)`` returns ``{"reg": int16[N_REG], "clock":
    int32[N_CLOCKS], "cand_tok": int32[16], "cand_feat": int8[16, N_CAND_FEAT]}`` for the
    position of ``token``. Segments start at the first token, at every ``BOS`` and where
    ``new_segment`` is True (pass it where ``seg`` changes). :meth:`fork` copies the state
    (particle decoding).

    Targets: the countdown targets are recomputed with ``constraints.targets_for_tokens`` on the
    tokens since the last segment start given by ``new_segment`` (what the batch call sees of
    the row) whenever a turn starts (``BOS`` or a role token), and kept for the rest of the
    turn. This equals the batch values because a turn's targets come from the user turn before
    it (``docs/SPEC.md`` 2.1), so they cannot change inside the turn.
    """

    def __init__(self, tok=None, targets: Any = "auto"):
        """targets: ``"auto"`` (from the constraints module when importable) or ``None``
        (no countdown targets)."""
        if not (targets is None or (isinstance(targets, str) and targets == "auto")):
            raise ValueError('targets must be "auto" or None')
        registers = _registers()
        self._tok = tok
        self._use_targets = targets is not None and _constraints() is not None
        self._regs = registers.Incremental(1, _instr_table(tok))
        self._tape = tape.Incremental()
        self._started = False
        self._history: list[int] = []             # tokens since the last new_segment
        self._target_row: np.ndarray | None = None

    def fork(self) -> "IncrementalFeatures":
        other = IncrementalFeatures.__new__(IncrementalFeatures)
        other._tok, other._use_targets, other._started = self._tok, self._use_targets, self._started
        other._regs = self._regs.copy()
        other._tape = self._tape.fork()
        other._history = list(self._history)
        other._target_row = None if self._target_row is None else self._target_row.copy()
        return other

    def _targets_at(self, token: int) -> np.ndarray | None:
        if not self._use_targets:
            return None
        if self._target_row is None or token in _TURN_STARTS:
            hist = np.asarray(self._history, dtype=np.int32)[None, :]
            self._target_row = targets_for(hist, None, self._tok)[0, -1].astype(np.int64)
        return self._target_row

    def step(self, token: int, new_segment: bool = False) -> dict[str, np.ndarray]:
        token = int(token)
        fresh = bool(new_segment) or not self._started    # what a change of ``seg`` means
        self._started = True
        if fresh:                  # the history is the current seg run, as the batch call sees it
            self._history = []
            self._target_row = None
        self._history.append(token)
        tgt = self._targets_at(token)
        reg, clock = self._regs.step(token, targets=tgt, new_segment=np.array([fresh]))
        cand_tok, cand_feat = self._tape.step(token, new_segment=fresh)
        return {"reg": np.asarray(reg, dtype=np.int16).reshape(N_REG),
                "clock": np.asarray(clock, dtype=np.int32).reshape(N_CLOCKS),
                "cand_tok": cand_tok, "cand_feat": cand_feat}

    def extend(self, tokens, seg=None) -> dict[str, np.ndarray]:
        """Step through ``tokens int[n]`` (``seg int[n]``: a change starts a segment); returns
        the rows stacked: ``reg [n, N_REG]``, ``clock [n, N_CLOCKS]``, ``cand_tok [n, 16]``,
        ``cand_feat [n, 16, F]``."""
        tokens = np.asarray(tokens).reshape(-1).tolist()
        segs = None if seg is None else np.asarray(seg).reshape(-1).tolist()
        out = {"reg": np.zeros((len(tokens), N_REG), np.int16),
               "clock": np.zeros((len(tokens), N_CLOCKS), np.int32),
               "cand_tok": np.full((len(tokens), CAND_SLOTS), -1, np.int32),
               "cand_feat": np.zeros((len(tokens), CAND_SLOTS, N_CAND_FEAT), np.int8)}
        for i, token in enumerate(tokens):
            new = segs is not None and (i == 0 or segs[i] != segs[i - 1])
            row = self.step(token, new_segment=new)
            for key in FEATURE_KEYS:
                out[key][i] = row[key]
        return out


__all__ = ["exact_features", "IncrementalFeatures", "targets_for", "has_constraints",
           "FEATURE_KEYS"]
