"""Tests of bes.data.mixture: water-filling and the two-level plan against an iterative
brute-force reference written here, the epoch caps, availability, and the transform policy."""
from __future__ import annotations

import math
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from bes.config import BOS, Cfg, DataCfg, MixCfg
from bes.data import mixture as M
from bes.data import sources
from bes.data.samples import rng_for
from bes.train.schedule import stage_at

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_KINDS = ("base", "chat", "rag", "math")
LIB_TASKS = ("library_read", "library_qa")


def stage(chat_frac: float, library_frac_base: float = 0.15, library_frac_chat: float = 0.4):
    return SimpleNamespace(chat_frac=chat_frac, library_frac_base=library_frac_base,
                           library_frac_chat=library_frac_chat)


# ------------------------------------------------------------------------ references
def ref_fill(weights: dict, caps: dict, total: float) -> tuple[dict, float]:
    """Iterative water-filling: share in proportion, clip every item above its cap, hand the
    excess to the others, repeat."""
    x = {k: 0.0 for k in weights}
    free = {k for k, v in weights.items() if v > 0}
    remaining = float(total)
    while free and remaining > 1e-15:
        ws = sum(weights[k] for k in free)
        prop = {k: remaining * weights[k] / ws for k in free}
        over = {k for k in free if prop[k] > caps[k]}
        if not over:
            x.update(prop)
            remaining = 0.0
            break
        for k in over:
            x[k] = caps[k]
            remaining -= caps[k]
            free.remove(k)
    return x, max(0.0, remaining)


def ref_plan(st, avail: dict, horizon: float, data: DataCfg, mix: MixCfg, has_lib: bool) -> dict:
    """Source shares of the module docstring, computed with ref_fill on both levels."""
    weights = {}
    for kind in SOURCE_KINDS:
        w = {s: v for s, v in sources.mixture_weights(kind, data.permissive_only).items() if avail.get(s, 0) > 0}
        weights[kind] = {s: v / sum(w.values()) for s, v in w.items()}
    lib = {"library_read": mix.read_weight, "library_qa": mix.qa_weight}
    lib = {s: v for s, v in lib.items() if has_lib and avail.get(s, 0) > 0 and v > 0}
    weights["library"] = {s: v / sum(lib.values()) for s, v in lib.items()}
    mu = {k: v for k, v in (("chat", mix.chat_weight), ("rag", mix.rag_weight), ("math", mix.math_weight),
                            ("library", mix.library_weight)) if v > 0 and weights[k]}
    c = st.chat_frac if mu else 0.0
    mu = {k: v / sum(mu.values()) for k, v in mu.items()}
    L = st.library_frac_base if weights["library"] else 0.0
    targets = {"base": (1 - c) * (1 - L), "library": (1 - c) * L + c * mu.get("library", 0.0),
               **{k: c * mu.get(k, 0.0) for k in ("chat", "rag", "math")}}
    caps = {s: mix.max_epochs * avail[s] / horizon for k in weights for s in weights[k]}
    kcap = {k: sum(caps[s] for s in weights[k]) for k in targets}
    x, unplaced = ref_fill(targets, kcap, 1.0)
    if unplaced > 1e-12:
        scale = 1.0 / sum(kcap[k] for k in targets if targets[k] > 0)
        caps = {s: v * scale for s, v in caps.items()}
        kcap = {k: v * scale for k, v in kcap.items()}
        x, _ = ref_fill(targets, kcap, 1.0)
    out = {}
    for k, w in weights.items():
        if w:
            y, _ = ref_fill(w, {s: caps[s] for s in w}, x[k])
            out.update(y)
    return out


def random_availability(rng: np.random.Generator) -> dict:
    names = [s.name for k in SOURCE_KINDS for s in sources.by_kind(k)] + list(LIB_TASKS)
    out = {}
    for name in names:
        if rng.random() < 0.2:
            out[name] = 0.0
        else:
            out[name] = float(10 ** rng.uniform(4, 10))
    return out


# ---------------------------------------------------------------------- water-filling
def test_water_fill_matches_iterative_reference():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 9))
        w = rng.uniform(0, 1, n) * (rng.random(n) < 0.85)
        c = rng.uniform(0, 1, n) * (rng.random(n) < 0.9)
        total = float(rng.uniform(0, 2))
        x, unplaced = M.water_fill(w, c, total)
        rx, runplaced = ref_fill(dict(enumerate(w)), dict(enumerate(c)), total)
        np.testing.assert_allclose(x, [rx[i] for i in range(n)], atol=1e-12)
        assert unplaced == pytest.approx(runplaced, abs=1e-12)
        assert (x <= c + 1e-12).all() and (x[w == 0] == 0).all()
        assert x.sum() + unplaced == pytest.approx(total, abs=1e-12)
        if unplaced > 0 and w.sum() > 0:
            assert np.allclose(x[w > 0], c[w > 0])           # nothing left unplaced while room remains
        free = (x < c - 1e-12) & (w > 0)
        if free.any():                       # one common level for every item below its cap
            lam = x[free] / w[free]
            assert np.allclose(lam, lam[0]) and (x[w > 0] / w[w > 0] <= lam[0] + 1e-9).all()
    with pytest.raises(ValueError):
        M.water_fill([1, -1], [1, 1], 1.0)
    with pytest.raises(ValueError):
        M.water_fill([1], [1, 2], 1.0)


# ------------------------------------------------------------------------------- plan
def check_plan(mx: M.Mixture, avail: dict, horizon: float, mix: MixCfg) -> None:
    assert sum(mx.share.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(mx.kinds.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(mx.rows.values()) == pytest.approx(1.0, abs=1e-9) and min(mx.rows.values()) >= -1e-12
    if mx.rows["chat"] > 1e-12:
        assert sum(mx.chat_rows.values()) == pytest.approx(1.0, abs=1e-9)
    for kind, within in mx.sources.items():
        assert sum(within.values()) == pytest.approx(1.0, abs=1e-9)
    for s, v in mx.share.items():
        assert v * horizon <= mx.epoch_limit * avail[s] * (1 + 1e-9) + 1e-6
        assert mx.epochs[s] <= mx.epoch_limit * (1 + 1e-9)
    assert mx.epoch_limit >= mix.max_epochs
    for s in mx.missing:
        assert avail.get(s, 0.0) == 0.0 and s not in mx.share


def test_plan_matches_brute_force_reference():
    rng = np.random.default_rng(1)
    mix, checked = MixCfg(), 0
    seen = {"capped": 0, "moved": 0, "raised": 0}
    for trial in range(300):
        avail = random_availability(rng)
        horizon = float(10 ** rng.uniform(7, 11))
        st = stage(float(rng.choice([0.0, 0.25, 0.5, 0.7, 1.0, rng.random()])),
                   float(rng.choice([0.0, 0.15, 0.5])))
        data = DataCfg(permissive_only=bool(rng.random() < 0.3))
        has_lib = any(avail[t] > 0 for t in LIB_TASKS)
        try:
            mx = M.plan(st, avail, horizon, data=data, mix=mix)
        except ValueError:
            continue
        checked += 1
        check_plan(mx, avail, horizon, mix)
        want = ref_plan(st, avail, horizon, data, mix, has_lib)
        assert set(mx.share) == set(want)
        for s in want:
            assert mx.share[s] == pytest.approx(want[s], abs=1e-9), (trial, s)
        seen["capped"] += bool(mx.capped)
        seen["moved"] += any(abs(v) > 1e-9 for v in mx.moved.values())
        seen["raised"] += mx.epoch_limit > mix.max_epochs
    assert checked >= 250
    assert seen["capped"] > 100 and seen["moved"] > 50 and seen["raised"] > 5     # all branches exercised


def test_plan_without_binding_caps_equals_the_targets():
    avail = {s.name: 1e15 for s in sources.SOURCES}
    avail.update({"library_read": 1e15, "library_qa": 1e15})
    mix, data = MixCfg(), DataCfg()
    for c, L in [(0.25, 0.15), (0.7, 0.15), (0.0, 0.15), (1.0, 0.15), (0.4, 0.0)]:
        mx = M.plan(stage(c, L), avail, 2e10, data=data, mix=mix)
        mu_total = mix.chat_weight + mix.rag_weight + mix.math_weight + mix.library_weight
        assert mx.rows["chat"] == pytest.approx(c) and mx.rows["base"] == pytest.approx((1 - c) * (1 - L))
        assert mx.rows["library"] == pytest.approx((1 - c) * L)
        if c > 0:
            assert mx.chat_rows["math"] == pytest.approx(mix.math_weight / mu_total)
            assert mx.chat_rows["library"] == pytest.approx(mix.library_weight / mu_total)
        for kind in SOURCE_KINDS:
            want = sources.mixture_weights(kind)
            if mx.kinds[kind] > 0:
                assert mx.sources[kind] == pytest.approx(want)
        assert mx.capped == () and mx.missing == ()
        assert all(abs(v) < 1e-12 for v in mx.moved.values()) and mx.epoch_limit == mix.max_epochs
        assert mx.share == pytest.approx(mx.target)


def test_caps_redistribute_within_the_kind_first():
    avail = {s.name: 1e15 for s in sources.SOURCES}
    avail.update({"library_read": 1e15, "library_qa": 0.0, "smoltalk_everyday": 1e6, "gsm8k": 1e6})
    horizon = 1e11
    mx = M.plan(stage(0.5), avail, horizon)
    assert "smoltalk_everyday" in mx.capped and "gsm8k" in mx.capped
    assert mx.epochs["smoltalk_everyday"] == pytest.approx(8.0) and mx.epochs["gsm8k"] == pytest.approx(8.0)
    assert mx.kinds["chat"] == pytest.approx(mx.kind_target["chat"])       # absorbed inside the kind
    w = sources.mixture_weights("chat")
    free = [s for s in w if s != "smoltalk_everyday"]
    ratios = [mx.share[s] / w[s] for s in free]
    assert np.allclose(ratios, ratios[0])                                   # proportional to the weights
    assert mx.share["smoltalk_everyday"] < mx.target["smoltalk_everyday"]
    assert "library_qa" in mx.missing and mx.sources["library"] == {"library_read": 1.0}


def test_a_fully_capped_kind_moves_its_share_to_the_others():
    avail = {s.name: 1e15 for s in sources.SOURCES}
    avail.update({"library_read": 1e7, "library_qa": 1e6})
    horizon = 2e10
    mx = M.plan(stage(0.25, 0.15), avail, horizon)
    cap = MixCfg().max_epochs * (1e7 + 1e6) / horizon
    assert mx.kinds["library"] == pytest.approx(cap) and mx.moved["library"] < 0
    others = [k for k in M.KIND_ORDER if k != "library"]
    gains = {k: mx.moved[k] for k in others}
    assert sum(gains.values()) == pytest.approx(-mx.moved["library"])
    for k in others:                                   # in proportion to the other kinds' targets
        assert mx.kinds[k] / mx.kind_target[k] == pytest.approx(mx.kinds["base"] / mx.kind_target["base"])
    assert any("library" in n for n in mx.notes) and set(mx.capped) >= {"library_read", "library_qa"}
    lib_rows_part = 0.75 * 0.15 / mx.kind_target["library"]
    assert mx.rows["library"] == pytest.approx(mx.kinds["library"] * lib_rows_part)


def test_missing_sources_are_skipped_and_the_chat_fraction_kept():
    avail = {"climbmix": 1e15, "smol_smoltalk": 1e15}
    mx = M.plan(stage(0.3), avail, 1e10)
    assert mx.rows["chat"] == pytest.approx(0.3) and mx.rows["base"] == pytest.approx(0.7)
    assert mx.sources["chat"] == {"smol_smoltalk": 1.0} and mx.sources["base"] == {"climbmix": 1.0}
    assert mx.chat_rows == {"chat": 1.0} and mx.rows["library"] == 0.0
    assert "fineweb_edu" in mx.missing and "gsm8k" in mx.missing and "library_read" not in mx.missing
    only_base = M.plan(stage(0.3), {"climbmix": 1e15}, 1e10)
    assert only_base.rows == {"base": 1.0, "library": 0.0, "chat": 0.0} and only_base.notes
    with pytest.raises(ValueError):
        M.plan(stage(0.3), {}, 1e10)
    with pytest.raises(ValueError):
        M.plan(stage(0.3), {"climbmix": 1e15}, 0.0)


def test_permissive_only_and_library_switches():
    avail = {s.name: 1e15 for s in sources.SOURCES}
    avail.update({"library_read": 1e15, "library_qa": 1e15})
    strict = M.plan(stage(0.25), avail, 1e10, data=DataCfg(permissive_only=True))
    assert "climbmix" not in strict.share and strict.sources["base"] == pytest.approx(
        sources.mixture_weights("base", permissive_only=True))
    no_lib = M.plan(stage(0.25), avail, 1e10, data=DataCfg(use_library=False))
    assert no_lib.rows["library"] == 0.0 and "library" not in no_lib.sources
    assert no_lib.rows["chat"] == pytest.approx(0.25) and no_lib.rows["base"] == pytest.approx(0.75)
    told = M.plan(stage(0.25), avail, 1e10, library=False)
    assert told.share == pytest.approx(no_lib.share)


def test_epoch_limit_is_raised_when_all_data_is_too_small():
    avail = {"climbmix": 1e6, "smol_smoltalk": 1e6, "library_read": 1e5}
    mx = M.plan(stage(0.5), avail, 1e9)
    assert mx.epoch_limit > MixCfg().max_epochs and mx.notes
    assert sum(mx.share.values()) == pytest.approx(1.0)
    assert all(e == pytest.approx(mx.epoch_limit) for e in mx.epochs.values())
    check_plan(mx, avail, 1e9, MixCfg())


def test_plan_over_the_real_schedule():
    cfg = Cfg()
    avail = {s.name: 1e15 for s in sources.SOURCES}
    avail.update(M.library_tokens(540_000, 0))
    for p in np.linspace(0, 1, 21):
        st = stage_at(float(p), cfg)
        mx = M.plan(st, avail, 2e10)
        check_plan(mx, avail, 2e10, MixCfg())
        assert mx.rows["chat"] >= st.chat_frac - 1e-9          # only the library cap moves shares
        lib_cap = MixCfg().max_epochs * avail["library_read"] / 2e10
        assert mx.kinds["library"] == pytest.approx(min(lib_cap, mx.kind_target["library"]))


def test_segment_probs_reproduce_token_shares():
    shares = {"chat": 0.6, "rag": 0.2, "math": 0.2}
    lengths = {"chat": 600.0, "rag": 400.0, "math": 200.0}
    p = M.segment_probs(shares, lengths)
    assert sum(p.values()) == pytest.approx(1.0)
    tokens = {k: p[k] * lengths[k] for k in p}
    total = sum(tokens.values())
    assert {k: v / total for k, v in tokens.items()} == pytest.approx(shares)
    assert M.segment_probs({"a": 1.0, "b": 0.0}, {"a": 5.0, "b": 3.0}) == {"a": 1.0}
    assert M.segment_probs({"a": 1.0}, {"a": 0.0}) == {}


def test_library_and_disk_availability(tmp_path):
    from bes.data import shards

    fake = SimpleNamespace(n_passages=10, kind_range=lambda k: range(0, 7) if k == "wiki" else range(7, 10))
    mix = MixCfg()
    assert M.library_available(fake, mix) == {"library_read": 7 * mix.read_task_tokens,
                                              "library_qa": 3 * mix.qa_task_tokens}
    assert M.library_available(None) == {}
    d = tmp_path / "tok" / "gsm8k"
    w = shards.ShardWriter(d, 0, with_roles=True)
    w.add([BOS, 5, 6, 7], [0, 2, 2, 2])
    w.add([BOS, 9], [0, 3])
    shards.append_manifest(d, w.close())
    got = M.tokens_on_disk(tmp_path, ["gsm8k", "orca_math"])
    assert got == {"gsm8k": 6.0, "orca_math": 0.0}
    assert set(M.tokens_on_disk(tmp_path)) == {s.name for s in sources.SOURCES}


@pytest.mark.skipif(not (ROOT / "data_cache" / "tok").exists(), reason="no token shards in data_cache")
def test_plan_with_what_is_on_disk():
    avail = M.tokens_on_disk(ROOT / "data_cache")
    lib_dir = ROOT / "data_cache" / "library" / "lib_60M"
    if (lib_dir / "library.json").exists():
        from bes.library.retrieve import Library

        avail.update(M.library_available(Library.load(str(lib_dir))))
    if not any(avail.get(s.name, 0) > 0 for s in sources.by_kind("base")):
        avail["climbmix"] = 1e12                        # only conversations tokenised so far
    mx = M.plan(stage_at(0.3, Cfg()), avail, 2e10)
    check_plan(mx, avail, 2e10, MixCfg())
    assert mx.missing                                   # most sources are not downloaded here


# ---------------------------------------------------------------- transform policy
def test_transform_probabilities():
    cfg = Cfg()
    st = stage_at(0.3, cfg)
    d = cfg.data
    assert M.transform_probs("chat", st, d) == {"hindsight": d.hindsight_frac, "retrieval": st.library_frac_chat}
    assert M.transform_probs("rag", st, d) == {"hindsight": 0.0, "retrieval": st.library_frac_chat}
    for kind in ("math", "library", "base"):
        assert M.transform_probs(kind, st, d) == {"hindsight": 0.0, "retrieval": 0.0}
    assert M.transform_probs("chat", st, d, library=False)["retrieval"] == 0.0
    assert M.transform_probs("chat", st, DataCfg(use_library=False))["retrieval"] == 0.0


def test_draw_transforms_frequencies():
    st = stage(0.25, 0.15, 0.4)
    n = 20000
    plans = [M.draw_transforms("chat", st, rng_for("policy", i)) for i in range(n)]
    h = np.array([p.hindsight for p in plans])
    r = np.array([p.retrieval for p in plans])
    for rate, p in ((h.mean(), 0.12), (r.mean(), 0.4), ((h & r).mean(), 0.12 * 0.4)):
        assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / n)
    assert plans[:50] == [M.draw_transforms("chat", st, rng_for("policy", i)) for i in range(50)]
    assert not any(M.draw_transforms("math", st, rng_for(i)).retrieval for i in range(200))
