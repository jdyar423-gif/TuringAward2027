"""Roles of parameter leaves: lookup or dense, and optimiser group.

``role_of(path) -> Role(kind, group)``:

* ``kind``: ``"dense"`` (every number is read for every token) or ``"lookup"`` (rows are read
  by index; ``bes.budget`` counts one row per read);
* ``group``: optimiser group, one of :data:`GROUPS`:
  ``matrix``  spectral optimiser (MuonH / Muon); every such matrix is non-zero at init,
              stored ``[..., d_in, d_out]`` (also the token-neuron projections ``page/pin``,
              ``page/pout``);
  ``embed``   Adam, vocabulary-indexed input tables (token and value embeddings, the
              token-neuron keys ``page/key``, which are used RMS-normalised like ``emb/tok``);
  ``unembed`` Adam, output embedding and the trust feature tables of the output law;
  ``feature`` Adam at a much lower rate, the zero-initialised feature tables: registers,
              candidate input (token and feature tables) and the trust biases;
  ``norm``    Adam, RMSNorm gains;
  ``scalar``  Adam, lanes, attention gains, sink scalars, the value-embedding gate and the
              zero-initialised attention output gate ``core/attn/gate_w``, ``core/attn/gate_b``
              (a zero matrix cannot be in the spectral group);
  ``frozen``  not updated by the main optimiser (the critic, fitted in a separate stage);
  ``gain``    Adam with its own rate and decay, the zero-initialised per-token gain tables
              ``gain/*`` (one row of multiplicative deltas per vocabulary id);
  ``page``    Adam with its own rate and decay, the zero-initialised token-neuron values
              ``page/val`` (one row per vocabulary id).

Paths are ``/``-joined dict keys (``"core/attn/wq"``); tuples of keys and JAX key paths are
accepted too. Every leaf of ``bes.model.model.init_params`` must match a rule (tested);
later tracks may add rules with :func:`register`.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable

import jax

KINDS = ("dense", "lookup")
GROUPS = ("matrix", "embed", "unembed", "feature", "norm", "scalar", "frozen", "gain", "page")


@dataclass(frozen=True)
class Role:
    kind: str
    group: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.group not in GROUPS:
            raise ValueError(f"unknown group {self.group!r}")


_LOOKUP_EMBED = Role("lookup", "embed")
_LOOKUP_FEATURE = Role("lookup", "feature")
_LOOKUP_UNEMBED = Role("lookup", "unembed")
_MATRIX = Role("dense", "matrix")
_SCALAR = Role("dense", "scalar")
_NORM = Role("dense", "norm")
_FROZEN = Role("dense", "frozen")

#: (glob pattern, role); the first matching rule wins.
_RULES: list[tuple[str, Role]] = [
    ("emb/tok", _LOOKUP_EMBED),
    ("emb/ve", _LOOKUP_EMBED),
    ("emb/reg/*", _LOOKUP_FEATURE),
    ("cand/in_tok", _LOOKUP_FEATURE),
    ("cand/in_feat/*", _LOOKUP_FEATURE),
    ("cand/in_proj", _MATRIX),
    ("core/attn/wq", _MATRIX),
    ("core/attn/wk", _MATRIX),
    ("core/attn/wv", _MATRIX),
    ("core/attn/wo", _MATRIX),
    ("core/attn/q_gain", _SCALAR),
    ("core/attn/sink_logit", _SCALAR),
    ("core/attn/sink_value", _SCALAR),
    ("core/attn/ve_gate", _SCALAR),
    ("core/mlp/w_up", _MATRIX),
    ("core/mlp/w_down", _MATRIX),
    ("core/norm/g1", _NORM),
    ("core/norm/g2", _NORM),
    ("loop/g", _NORM),
    ("lanes/alpha", _SCALAR),
    ("lanes/mix", _SCALAR),
    ("lanes/beta", _SCALAR),
    ("head/g", _NORM),
    ("head/proj", _MATRIX),
    ("head/out", Role("dense", "unembed")),
    ("trust/proj", _MATRIX),
    ("trust/feat/*", _LOOKUP_UNEMBED),
    ("trust/bias/*", _LOOKUP_FEATURE),
    ("critic/w1", _FROZEN),
    ("critic/w2", _FROZEN),
    ("gain/*", Role("lookup", "gain")),
    ("core/attn/gate_w", _SCALAR),
    ("core/attn/gate_b", _SCALAR),
    ("page/pin", _MATRIX),
    ("page/pout", _MATRIX),
    ("page/key", _LOOKUP_EMBED),
    ("page/val", Role("lookup", "page")),
]


def _key_name(entry: Any) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


def path_str(path: Any) -> str:
    """Normalise a path (``"a/b"``, ``("a", "b")`` or a JAX key path) to ``"a/b"``."""
    if isinstance(path, str):
        return path.strip("/")
    return "/".join(_key_name(p) for p in path)


def register(pattern: str, role: Role) -> None:
    """Add a rule (checked before the built-in rules) for leaves added by later tracks."""
    _RULES.insert(0, (pattern, role))


def role_of(path: Any) -> Role:
    """Role of a parameter leaf; raises ``KeyError`` for an unknown path."""
    p = path_str(path)
    for pattern, role in _RULES:
        if fnmatch.fnmatchcase(p, pattern):
            return role
    raise KeyError(f"no role for parameter path {p!r}")


def leaves_with_paths(tree: Any) -> list[tuple[str, Any]]:
    """``[(path, leaf), ...]`` for a nested dict of arrays (or shape structs)."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(path_str(path), leaf) for path, leaf in flat]


def roles_tree(tree: Any) -> Any:
    """A tree of the same structure with a :class:`Role` at every leaf."""
    return jax.tree_util.tree_map_with_path(lambda path, _: role_of(path), tree)


def group_paths(tree: Any, groups: Iterable[str]) -> list[str]:
    """Paths of the leaves whose optimiser group is in ``groups``."""
    wanted = set(groups)
    return [p for p, _ in leaves_with_paths(tree) if role_of(p).group in wanted]
