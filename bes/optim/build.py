"""Optimiser assembly: parameter groups, group learning rates and one pure update for the tree.

Group set-up follows ``kschwethelm/looped-lm-scaling`` ``nanochat/gpt.py``
(``GPT.setup_optimizer``, lines 591-669, MIT licence): matrices of the blocks go to the spectral
optimiser with ``lr_matrix``; embeddings, unembedding, norm gains and the hyper-connection scalars
go to Adam with their own learning rates multiplied by ``(d_model / d_ref) ** -0.5`` (lines
634-649). As in that file the width factor is *not* applied to the matrix group unless
``OptimCfg.width_scale_matrix`` is set. The training script multiplies every group's learning rate
by one schedule factor (``scripts/base_train.py`` lines 612-616); here that factor is ``lr_scale``.

Groups come from ``bes.roles.role_of(path)`` with ``path`` the ``/``-joined key path of the leaf
(``"core/attn/wq"``); ``role_fn`` may be passed instead and may return either an object with a
``group`` attribute or the group name itself.

========  ==========================  ========================================================
group     optimiser                   learning rate (before ``lr_scale``)
========  ==========================  ========================================================
matrix    MuonH (``"muonh"``) or      ``lr_matrix`` (x width factor if ``width_scale_matrix``;
          Muon (``"muon"``)           ``"muon"`` also applies ``sqrt(max(1, d_out/d_in))``)
embed     Adam                        ``lr_embed * w``
unembed   Adam                        ``lr_unembed * w``
norm      Adam                        ``lr_norm * w``
scalar    Adam                        ``lr_scalar * w``
frozen    none (never moves)          --
========  ==========================  ========================================================

with ``w = (d_model / lr_width_ref) ** -0.5``. Adam: betas ``(adam_b1, adam_b2)``, eps
``adam_eps``, no weight decay. If ``grad_clip > 0`` the gradients of all non-frozen leaves are
clipped jointly to that global norm inside :meth:`Optimizer.update` (the training step must not
clip again).

All per-leaf decisions (group, learning rate, shapes) are made once in :func:`build`; ``init`` and
``update`` are pure functions of arrays and can be jit-compiled, differentiated and donated.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import Cfg, OptimCfg
from bes.optim import adam, muon_normuon, muonh

GROUPS = ("matrix", "embed", "unembed", "norm", "scalar", "frozen")
ADAM_GROUPS = ("embed", "unembed", "norm", "scalar")
KINDS = ("muonh", "muon")
NS_DTYPES = ("float32", "bfloat16")


class OptState(NamedTuple):
    """Optimiser state (a pytree of arrays).

    count: int32 ``[]``, number of updates applied so far.
    slots: per-leaf state keyed by the leaf path. matrix/MuonH: ``{"mom", "norm"}``;
      matrix/Muon: ``{"mom", "v"}``; Adam groups: ``{"m", "v"}``; frozen: ``{}``.
      All float32. For MuonH, ``norm`` (the cached initial norms) is part of the state and must be
      checkpointed with it.
    """

    count: jax.Array
    slots: dict[str, dict[str, jax.Array]]


@dataclasses.dataclass(frozen=True)
class LeafPlan:
    """Static decisions for one leaf, made once at build time."""

    path: str
    group: str
    shape: tuple[int, ...]
    base_lr: float  # learning rate before ``lr_scale`` (and before the Muon shape factor)


def path_str(keypath: tuple[Any, ...]) -> str:
    """``/``-joined name of a ``jax.tree_util`` key path (dict keys, indices, attributes)."""
    parts = []
    for k in keypath:
        if isinstance(k, jax.tree_util.DictKey):
            parts.append(str(k.key))
        elif isinstance(k, jax.tree_util.SequenceKey):
            parts.append(str(k.idx))
        elif isinstance(k, jax.tree_util.GetAttrKey):
            parts.append(str(k.name))
        elif isinstance(k, jax.tree_util.FlattenedIndexKey):
            parts.append(str(k.key))
        else:
            parts.append(str(k))
    return "/".join(parts)


def default_role_fn(path: str) -> Any:
    """``bes.roles.role_of`` imported lazily (the roles module belongs to the model track)."""
    from bes.roles import role_of

    return role_of(path)


def width_factor(cfg: Cfg) -> float:
    """``(d_model / lr_width_ref) ** -0.5``."""
    return (cfg.model.d_model / cfg.optim.lr_width_ref) ** -0.5


def group_lrs(cfg: Cfg) -> dict[str, float]:
    """Learning rate of every group before ``lr_scale`` (see the module table)."""
    o, w = cfg.optim, width_factor(cfg)
    return {
        "matrix": o.lr_matrix * (w if o.width_scale_matrix else 1.0),
        "embed": o.lr_embed * w,
        "unembed": o.lr_unembed * w,
        "norm": o.lr_norm * w,
        "scalar": o.lr_scalar * w,
        "frozen": 0.0,
    }


def global_norm(tree: Any) -> jax.Array:
    """float32 ``[]``: square root of the sum of squares of every leaf (accumulated in float32)."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((), jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves))


def _group_name(role: Any, path: str) -> str:
    group = getattr(role, "group", role)
    if group not in GROUPS:
        raise ValueError(f"leaf {path!r}: unknown optimiser group {group!r} (expected {GROUPS})")
    return group


def _is_concrete(x: Any) -> bool:
    tracer = getattr(jax.core, "Tracer", ())
    if isinstance(x, tracer):
        return False
    return isinstance(x, (np.ndarray, np.generic, jax.Array))


def _resolve_ns_dtype(cfg: Cfg) -> str:
    name = cfg.model.act_dtype if cfg.optim.ns_dtype == "auto" else cfg.optim.ns_dtype
    if name not in NS_DTYPES:
        raise ValueError(f"Polar Express dtype must be one of {NS_DTYPES}, got {name!r}")
    return name


@dataclasses.dataclass(frozen=True)
class Optimizer:
    """The whole-tree optimiser. Build it with :func:`build`; all methods are pure."""

    kind: str
    cfg: OptimCfg
    plan: tuple[LeafPlan, ...]
    treedef: Any
    ns_dtype: str
    layout: str

    @property
    def groups(self) -> dict[str, str]:
        """``{path: group}`` for every leaf."""
        return {lp.path: lp.group for lp in self.plan}

    def _flatten(self, tree: Any, what: str) -> list[Any]:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        if treedef != self.treedef:
            raise ValueError(f"{what} do not have the tree structure the optimiser was built for")
        return leaves

    def init(self, params: Any) -> OptState:
        """Fresh state for ``params`` (for MuonH this caches the Frobenius norm of every matrix)."""
        slots = {}
        for lp, p in zip(self.plan, self._flatten(params, "params")):
            if lp.group == "frozen":
                slots[lp.path] = {}
            elif lp.group == "matrix":
                slots[lp.path] = (muonh.init(p) if self.kind == "muonh"
                                  else muon_normuon.init(p, self.layout))
            else:
                slots[lp.path] = adam.init(p)
        return OptState(count=jnp.zeros((), jnp.int32), slots=slots)

    def update(self, grads: Any, state: OptState, params: Any, lr_scale: jax.Array | float = 1.0,
               *, momentum: jax.Array | float | None = None,
               wd_scale: jax.Array | float = 1.0) -> tuple[Any, OptState]:
        """One step for the whole tree.

        grads, params: trees of the structure given to :func:`build`. lr_scale: schedule factor
        multiplying every group's learning rate (scalar, may be traced). momentum: optional
        override of ``OptimCfg.momentum`` for the matrix group (e.g. a momentum warm-up); wd_scale:
        multiplies ``OptimCfg.weight_decay`` (``"muon"`` only, e.g. a decay schedule).
        Returns ``(new_params, new_state)``; parameter dtypes are preserved.
        """
        o = self.cfg
        p_leaves = self._flatten(params, "params")
        g_leaves = self._flatten(grads, "grads")
        lr_scale = jnp.asarray(lr_scale, jnp.float32)
        mu = jnp.asarray(o.momentum if momentum is None else momentum, jnp.float32)
        count = state.count + 1
        if o.grad_clip > 0:
            norm = global_norm([g for lp, g in zip(self.plan, g_leaves) if lp.group != "frozen"])
            factor = jnp.minimum(1.0, o.grad_clip / (norm + 1e-6))
            g_leaves = [g.astype(jnp.float32) * factor for g in g_leaves]
        new_params, slots = [], {}
        for lp, p, g in zip(self.plan, p_leaves, g_leaves):
            slot = state.slots[lp.path]
            lr = lp.base_lr * lr_scale
            if lp.group == "frozen":
                new_p, new_slot = p, slot
            elif lp.group == "matrix" and self.kind == "muonh":
                new_p, new_slot = muonh.update(p, g, slot, lr, mu, ns_steps=o.ns_steps,
                                               ns_dtype=self.ns_dtype)
            elif lp.group == "matrix":
                new_p, new_slot = muon_normuon.update(
                    p, g, slot, lr, mu, o.muon_beta2, o.weight_decay * jnp.asarray(wd_scale),
                    ns_steps=o.ns_steps, ns_dtype=self.ns_dtype, layout=self.layout)
            else:
                new_p, new_slot = adam.update(p, g, slot, lr, count, b1=o.adam_b1, b2=o.adam_b2,
                                              eps=o.adam_eps)
            new_params.append(new_p)
            slots[lp.path] = new_slot
        return (jax.tree_util.tree_unflatten(self.treedef, new_params),
                OptState(count=count, slots=slots))


def build(cfg: Cfg, params: Any, role_fn: Callable[[str], Any] | None = None, *,
          check_init: bool = True) -> Optimizer:
    """Assign every leaf of ``params`` to a group and fix its learning rate.

    cfg: the full configuration (``cfg.optim`` plus ``cfg.model.d_model`` and, for
    ``ns_dtype="auto"``, ``cfg.model.act_dtype``). params: the parameter tree (arrays, or
    ``jax.ShapeDtypeStruct`` for compile-only use). role_fn: ``path -> role``; defaults to
    ``bes.roles.role_of``. check_init: with MuonH, refuse matrices whose initial Frobenius norm is
    not positive and finite (such a matrix would stay at zero forever); skipped for abstract leaves.
    """
    o = cfg.optim
    if o.optimizer not in KINDS:
        raise ValueError(f"optim.optimizer must be one of {KINDS}, got {o.optimizer!r}")
    if o.matrix_layout not in muon_normuon.LAYOUTS:
        raise ValueError(f"optim.matrix_layout must be one of {muon_normuon.LAYOUTS}")
    role_fn = default_role_fn if role_fn is None else role_fn
    lrs = group_lrs(cfg)
    with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    plan, bad = [], []
    for keypath, leaf in with_path:
        path = path_str(keypath)
        group = _group_name(role_fn(path), path)
        shape = tuple(int(d) for d in leaf.shape)
        if group == "matrix":
            if len(shape) < 2:
                raise ValueError(f"leaf {path!r} of shape {shape} is in the matrix group but is not "
                                 "a (stack of) matrices [..., m, n]")
            if check_init and o.optimizer == "muonh" and _is_concrete(leaf):
                mats = np.asarray(leaf, np.float64).reshape((-1,) + shape[-2:])
                norms = np.sqrt((mats * mats).sum(axis=(1, 2)))
                if not np.all(np.isfinite(norms) & (norms > 0)):
                    bad.append(path)
        plan.append(LeafPlan(path=path, group=group, shape=shape, base_lr=lrs[group]))
    if bad:
        raise ValueError("MuonH keeps every matrix on the sphere of its initial Frobenius norm, so a "
                         f"matrix initialised at zero never moves; non-zero init required for {bad}")
    paths = [lp.path for lp in plan]
    if len(set(paths)) != len(paths):
        raise ValueError("leaf paths are not unique; cannot key the optimiser state by path")
    return Optimizer(kind=o.optimizer, cfg=o, plan=tuple(plan), treedef=treedef,
                     ns_dtype=_resolve_ns_dtype(cfg), layout=o.matrix_layout)
