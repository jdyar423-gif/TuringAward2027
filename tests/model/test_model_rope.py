"""Rotary planes driven by the token position and by the structural clocks."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import config_model, tiny_cfg
from bes.config import N_CLOCKS
from bes.model import rope_clocks


def test_plain_rope_frequencies_when_no_clock_planes():
    cfg = tiny_cfg(clock_planes=0)
    dc = cfg.head_dim - cfg.bias_dims
    table = rope_clocks.freq_table(cfg)
    want = 1.0 / (cfg.rope_base ** (np.arange(0, dc, 2) / dc))    # reference RoPE
    np.testing.assert_allclose(table[0], want, rtol=1e-6)
    assert np.all(table[1:] == 0)


def test_final_config_uses_plain_rope():
    """The frozen configuration switches the structural clocks off (docs/CHANGES.md section 14):
    all 63 planes rotate with the token position."""
    cfg = config_model("final")
    assert cfg.clock_planes == 0
    assert rope_clocks.plane_layout(cfg) == (63, 63, 0)
    table = rope_clocks.freq_table(cfg)
    assert table.shape == (1 + N_CLOCKS, 63)
    assert np.all(table[0] > 0) and np.all(table[1:] == 0)


def test_clock_plane_layout():
    """The layout with 15 clock planes (the configuration before the gates): 48 position planes
    and 5 planes per structural clock."""
    import dataclasses
    cfg = dataclasses.replace(config_model("final"), clock_planes=15)
    n_planes, n_pos, per = rope_clocks.plane_layout(cfg)
    assert (n_planes, n_pos, per) == (63, 48, 5)
    table = rope_clocks.freq_table(cfg)
    assert table.shape == (1 + N_CLOCKS, 63)
    assert np.all((table != 0).sum(0) == 1)            # every plane has exactly one source
    assert np.all(table[0, :48] > 0) and np.all(table[0, 48:] == 0)
    for c in range(N_CLOCKS):
        planes = table[1 + c, 48 + 5 * c: 48 + 5 * (c + 1)]
        np.testing.assert_allclose(planes, cfg.rope_base ** (-np.arange(5) / 5), rtol=1e-6)
        assert np.all(np.delete(table[1 + c], np.arange(48 + 5 * c, 48 + 5 * (c + 1))) == 0)


def test_angles_match_formula():
    cfg = tiny_cfg(head_dim=32, clock_planes=6)
    table = rope_clocks.freq_table(cfg)
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 2048, size=(2, 9))
    clock = rng.integers(0, 3000, size=(2, 9, N_CLOCKS))
    n_planes, n_pos, per = rope_clocks.plane_layout(cfg)
    want = np.zeros((2, 9, n_planes))
    want[..., :n_pos] = pos[..., None] * table[0, :n_pos].astype(np.float64)
    for c in range(N_CLOCKS):
        sl = slice(n_pos + c * per, n_pos + (c + 1) * per)
        want[..., sl] = clock[..., c:c + 1] * table[1 + c, sl].astype(np.float64)
    got = rope_clocks.angles(jnp.asarray(pos), jnp.asarray(clock), jnp.asarray(table), 1.0)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)
    off = np.asarray(rope_clocks.angles(jnp.asarray(pos), jnp.asarray(clock), jnp.asarray(table), 0.0))
    np.testing.assert_allclose(off[..., :n_pos], want[..., :n_pos], rtol=1e-6)
    assert np.all(off[..., n_pos:] == 0)


def test_rotation_is_relative():
    """<R(a) x, R(b) y> = <R(a - b) x, y> and rotations preserve norms."""
    rng = np.random.default_rng(1)
    P = 7
    x, y = rng.normal(size=(2, 2 * P))
    a, b = rng.normal(size=(2, P)) * 5

    def rot(v, ang):
        return np.asarray(rope_clocks.rotate(jnp.asarray(v), jnp.cos(jnp.asarray(ang)), jnp.sin(jnp.asarray(ang))))

    np.testing.assert_allclose(np.dot(rot(x, a), rot(y, b)), np.dot(rot(x, a - b), y), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(rot(x, a)), np.linalg.norm(x), rtol=1e-5)


def test_invalid_layouts_raise():
    with pytest.raises(ValueError):
        rope_clocks.plane_layout(tiny_cfg(head_dim=32, bias_dims=1))     # odd content dims
    with pytest.raises(ValueError):
        rope_clocks.plane_layout(tiny_cfg(clock_planes=4))               # not a multiple of N_CLOCKS
    with pytest.raises(ValueError):
        rope_clocks.plane_layout(tiny_cfg(head_dim=8, clock_planes=6))   # more clock planes than planes
