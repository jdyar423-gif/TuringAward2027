"""Helpers of the tests of the v9 write-side paths (token gains, head-wise attention output gate,
token neurons).

:func:`reference_logits` is an independent numpy (float64) forward of the plain looped
transformer with those paths, written from the formulas of the specification (SPEC_V9 section 3,
part B section B1, part C section C3), not from the model code; every older feature path is off
(no registers, candidates, sink, confidence channel or clocks: plain RoPE).
:func:`check_directional_grads` compares reverse-mode gradients with central differences.
"""
from __future__ import annotations

import jax
import numpy as np

from bes.model import model as M


def rms(x: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return y if g is None else y * g


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def token_neurons(x: np.ndarray, pin: np.ndarray, pout: np.ndarray, key: np.ndarray,
                  val: np.ndarray) -> np.ndarray:
    """``P_out^T sum_j relu(<rmsnorm(key_j), rmsnorm(P_in^T x)> / sqrt(dk))^2 val_j``.

    x: ``[B, T, d]``; pin ``[d, dk]``; pout ``[dk, d]``; key, val: ``[B, T, rho, dk]`` (raw table
    rows: the keys are normalised here). Returns ``[B, T, d]``."""
    dk = pin.shape[-1]
    z = rms(x @ pin)
    a = np.einsum("btc,btjc->btj", z, rms(key)) / np.sqrt(dk)
    return np.einsum("btj,btjc->btc", np.maximum(a, 0.0) ** 2, val) @ pout


def reference_logits(P: dict, tokens: np.ndarray, seg: np.ndarray, cfg) -> np.ndarray:
    """Logits ``[B, T, V]`` for the parameters ``P`` (float64 numpy tree), int tokens and segment
    ids ``[B, T]``. The gain, gate and token-neuron paths are used when their leaves are in ``P``
    (with every switch at 1)."""
    B, T = tokens.shape
    H, hd, nb, L, N = cfg.n_heads, cfg.head_dim, cfg.n_blocks, cfg.n_loops, cfg.n_lanes
    dc = hd - cfg.bias_dims
    n_planes = dc // 2
    ang = np.arange(T)[:, None] * cfg.rope_base ** (-np.arange(n_planes) / n_planes)[None]
    cos, sin = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]

    def rope(x):
        x1, x2 = x[..., :n_planes], x[..., n_planes:]
        return np.concatenate([x1 * cos + x2 * sin, x2 * cos - x1 * sin], axis=-1)

    pos = np.arange(T)
    mask = (pos[None, :, None] >= pos[None, None, :]) & (seg[:, :, None] == seg[:, None, :])
    gains = {name: P["gain"][name][tokens] for name in P.get("gain", {})}   # [B, T, P, nb, w]
    page = P.get("page")
    if page is not None:
        keys, vals = page["key"][tokens], page["val"][tokens]                 # [B, T, nb, rho, dk]
    a, m, bt = P["lanes"]["alpha"], P["lanes"]["mix"], P["lanes"]["beta"]
    at, ml, nm = P["core"]["attn"], P["core"]["mlp"], P["core"]["norm"]
    ve_all = P["emb"]["ve"][tokens]
    n_ve = cfg.ve_dim // hd
    lanes = [rms(P["emb"]["tok"][tokens]) for _ in range(N)]
    for r in range(L):
        ph = r * cfg.gain_phases // L
        u = sum(a[r, n] * lanes[n] for n in range(N))
        for i in range(nb):
            x = rms(u, nm["g1"][i])
            q = rope((x @ at["wq"][i]).reshape(B, T, H, dc))
            k = rope((x @ at["wk"][i]).reshape(B, T, H, dc))
            v = x @ at["wv"][i]
            if "v" in gains:
                v = v * (1.0 + gains["v"][:, :, ph, i])
            v = v.reshape(B, T, H, hd)
            if i == 0 and n_ve:
                g = 3.0 * sigmoid(x[..., :12] @ at["ve_gate"])
                v[:, :, :n_ve] += g[..., None] * ve_all[:, :, r].reshape(B, T, n_ve, hd)
            q = q / np.linalg.norm(q, axis=-1, keepdims=True) * at["q_gain"][i][None, None, :, None]
            k = k / np.linalg.norm(k, axis=-1, keepdims=True)
            logits = np.where(mask[:, None], np.einsum("bqhd,bkhd->bhqk", q, k), -np.inf)
            p = np.exp(logits - logits.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            y = np.einsum("bhqk,bkhd->bqhd", p, v)                                # [B, T, H, hd]
            if "gate_w" in at:
                y = y * (1.0 + (2.0 * sigmoid(x @ at["gate_w"][i] / x.shape[-1] + at["gate_b"][r, i]) - 1.0))[..., None]
            y = y.reshape(B, T, H * hd)
            if "o" in gains:
                y = y * (1.0 + gains["o"][:, :, ph, i])
            u = u + y @ at["wo"][i]
            x = rms(u, nm["g2"][i])
            hh = np.maximum(x @ ml["w_up"][i], 0.0) ** 2
            if "ffn" in gains:
                n = gains["ffn"].shape[-1]
                hh[..., :n] = hh[..., :n] * (1.0 + gains["ffn"][:, :, ph, i])
            f = hh @ ml["w_down"][i]
            if page is not None:
                f = f + token_neurons(x, page["pin"][i], page["pout"][i], keys[:, :, i], vals[:, :, i])
            u = u + f
        s = rms(u, P["loop"]["g"][r])
        lanes = [sum(m[r, j, n] * lanes[n] for n in range(N)) + bt[r, j] * s for j in range(N)]
    h = rms(sum(lanes), P["head"]["g"])
    z = (h @ P["head"]["proj"]) @ P["head"]["out"].T
    return cfg.softcap * np.tanh(z / cfg.softcap)


def switches_on(cfg, **sw) -> dict:
    """Consts with the older feature paths off and the v9 write-side paths on."""
    return M.make_consts(cfg, **{**M.switches_off(), "gain": 1.0, "gate": 1.0, "page": 1.0, **sw})


def check_directional_grads(f, grad_f, tree: dict, seed: int, eps: float = 1e-6,
                            rtol: float = 1e-5) -> dict[str, float]:
    """Reverse-mode gradient of a scalar ``f(tree)`` against central differences, leaf by leaf.

    f, grad_f: jitted ``tree -> float64 []`` and its gradient; tree: ``{name: float64 array}``
    (needs ``jax.enable_x64``). For every leaf a random unit direction ``e`` (only that leaf
    moves): ``(f(t + eps e) - f(t - eps e)) / (2 eps)`` must equal ``<grad, e>``. Returns
    ``{name: directional derivative}``; raises ``AssertionError`` on a mismatch.
    """
    rng = np.random.default_rng(seed)
    grads = grad_f(tree)
    out = {}
    for name, leaf in tree.items():
        e = rng.standard_normal(np.shape(leaf))
        e /= np.linalg.norm(e)
        plus = {**tree, name: leaf + eps * e}
        minus = {**tree, name: leaf - eps * e}
        num = (float(f(plus)) - float(f(minus))) / (2 * eps)
        ana = float(np.sum(np.asarray(grads[name]) * e))
        np.testing.assert_allclose(ana, num, rtol=rtol, atol=1e-9, err_msg=name)
        out[name] = ana
    return out


def tree_f64(tree):
    """A tree of float64 jax arrays (inside ``jax.enable_x64``)."""
    return jax.tree.map(lambda a: jax.numpy.asarray(a, jax.numpy.float64), tree)
