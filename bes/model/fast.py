"""The model's speed switches in one place: read a setting, apply it, say what is on.

The switches are module attributes read when a program is traced (plain Python ``if``), not
configuration: each keeps the values of the reference path to float rounding (their tests say how
close), so the batches, the journal, the checkpoints and the configuration hash do not change.

    kernel=onetile:save relu2=1 ce=1 io=0

* ``kernel``: :data:`bes.model.attention_splash.FAST_KERNEL` (``""``: the configured kernels);
* ``relu2``: :data:`bes.model.layers.FAST_RELU2`;
* ``ce``: :data:`bes.model.head.FAST_CE`;
* ``io``: :data:`bes.model.attention_splash.FAST_SPLASH_IO` (the configured splash kernels only).

A run takes its setting from ``<run>/fast`` (``scripts/run_24h.py``); ``scripts/tpu_kernels.py``
times combinations on the chip. A program JAX has already traced keeps what it was traced with:
apply a setting before the first trace of a process (or clear JAX's caches).
"""
from __future__ import annotations

import importlib

#: name in a setting -> (module, attribute) of a boolean switch
FLAGS = {"relu2": ("bes.model.layers", "FAST_RELU2"),
         "ce": ("bes.model.head", "FAST_CE"),
         "io": ("bes.model.attention_splash", "FAST_SPLASH_IO")}
_TRUE, _FALSE = ("1", "true", "on", "yes"), ("0", "false", "off", "no")


def parse(text: str) -> dict:
    """``"kernel=onetile:save relu2=1"`` -> ``{"kernel": "onetile:save", "relu2": True}``: only
    the names the text gives (the others are left as they are by :func:`apply`). Raises
    ``ValueError`` for an unknown name, an unknown kernel or a value that is not a boolean."""
    from bes.model import attention_splash

    out: dict = {}
    for item in str(text).split():
        name, eq, value = item.partition("=")
        if not eq:
            raise ValueError(f"speed switches: {item!r} is not name=value")
        if name == "kernel":
            attention_splash.parse_switch(value)            # unknown kernels fail here
            out["kernel"] = value
        elif name in FLAGS:
            v = value.lower()
            if v not in _TRUE + _FALSE:
                raise ValueError(f"speed switches: {name}={value!r} is neither 1 nor 0")
            out[name] = v in _TRUE
        else:
            raise ValueError(f"speed switches: unknown name {name!r} (kernel, {', '.join(FLAGS)})")
    return out


def current() -> dict:
    """What is on now, every name."""
    from bes.model import attention_splash

    out: dict = {"kernel": attention_splash.FAST_KERNEL}
    for name, (mod, attr) in FLAGS.items():
        out[name] = bool(getattr(importlib.import_module(mod), attr))
    return out


def apply(setting: dict) -> dict:
    """Set the switches ``setting`` names; returns their values before (for :func:`apply` again)."""
    from bes.model import attention_splash

    before = {k: v for k, v in current().items() if k in setting}
    for name, value in setting.items():
        if name == "kernel":
            attention_splash.parse_switch(value)
            attention_splash.FAST_KERNEL = str(value)
        elif name in FLAGS:
            mod, attr = FLAGS[name]
            setattr(importlib.import_module(mod), attr, bool(value))
        else:
            raise ValueError(f"speed switches: unknown name {name!r}")
    return before


def describe(setting: dict | None = None) -> str:
    """The canonical text of a setting (default: what is on now)."""
    s = current() if setting is None else setting
    parts = [f"kernel={s['kernel']}"] if "kernel" in s else []
    return " ".join(parts + [f"{k}={int(bool(s[k]))}" for k in FLAGS if k in s])


__all__ = ["FLAGS", "parse", "current", "apply", "describe"]
