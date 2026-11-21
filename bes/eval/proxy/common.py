"""Shared types and helpers of the proxy blind-test harness (``docs/EVAL_PROTOCOL.md``).

Every model, ours or an opponent, is driven through one small interface:

    backend.generate(messages, settings, seed) -> Reply

``messages`` is a list of ``{"role": "system" | "user" | "assistant", "content": str}`` that
ends with a user message; ``settings`` is a :class:`GenSettings`; the backend must be
deterministic given ``(messages, settings, seed)``. A :class:`Reply` carries the visible text
plus token counts, latency, the finish reason and (when the backend can provide it) the sum of
the log-probabilities of the generated tokens under the model's own unmodified distribution.

Backends are built inside worker processes from a picklable *spec* (a plain dict), see
:func:`make_backend`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bes.config import RunCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
#: Bump when a change alters generations, verdicts or cache keys.
HARNESS_VERSION = "1"

Message = Mapping[str, str]


def data_root() -> Path:
    """``<repo>/data_cache`` (``bes.config.RunCfg.data_dir``)."""
    return REPO_ROOT / RunCfg().data_dir


def default_root() -> Path:
    """Default output directory of the harness: ``data_cache/proxy``."""
    return data_root() / "proxy"


def models_dir() -> Path:
    """Where opponent and judge GGUF files live: ``data_cache/opponents``."""
    return data_root() / "opponents"


# -------------------------------------------------------------------------------- hashing
def stable_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no spaces, UTF-8 kept)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_hash(obj: Any, n: int = 16) -> str:
    """First ``n`` hex digits of the SHA-256 of :func:`stable_json` of ``obj``."""
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:n]


def derive_seed(base_seed: int, *parts: Any) -> int:
    """A 31-bit seed that depends only on ``base_seed`` and ``parts`` (order matters)."""
    digest = hashlib.sha256(stable_json([int(base_seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------- settings
SELECT_RULES = ("first", "logprob_mean", "filtered_logprob_mean")


@dataclass(frozen=True)
class GenSettings:
    """Generation settings of one arm.

    Sampling fields follow llama.cpp semantics (``top_k <= 0`` and ``top_p = 1`` disable the
    filter, ``temperature = 0`` is greedy). ``max_tokens`` counts the model's own tokens,
    including any hidden reasoning. ``system_prompt=None`` sends no system message (a chat
    template may still add its own default). ``strip_think`` removes ``<think>...</think>``
    blocks from the visible reply. ``samples`` candidates are drawn per turn and one is kept by
    ``select`` (see :func:`select_candidate`). ``extra`` holds backend-specific options (for our
    generator: particles, library on or off, ...) as sorted ``(key, value)`` pairs.
    """

    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 1024
    system_prompt: str | None = None
    strip_think: bool = False
    samples: int = 1
    select: str = "first"
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.select not in SELECT_RULES:
            raise ValueError(f"unknown select rule {self.select!r}; expected one of {SELECT_RULES}")
        if self.samples < 1 or self.max_tokens < 1:
            raise ValueError("samples and max_tokens must be >= 1")
        if isinstance(self.extra, Mapping):
            object.__setattr__(self, "extra", tuple(sorted(self.extra.items())))
        else:
            object.__setattr__(self, "extra", tuple(sorted((str(k), v) for k, v in self.extra)))

    @property
    def extra_dict(self) -> dict[str, Any]:
        return dict(self.extra)

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["extra"] = self.extra_dict
        return out

    def core_dict(self) -> dict[str, Any]:
        """Fields that determine one single generation (``samples`` and ``select`` excluded),
        so that runs with different sample counts share cached generations."""
        out = self.as_dict()
        out.pop("samples")
        out.pop("select")
        return out

    def hash(self) -> str:
        return stable_hash(self.as_dict(), 10)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GenSettings:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown settings fields {sorted(unknown)}")
        return cls(**dict(data))

    def replace(self, **changes: Any) -> GenSettings:
        return dataclasses.replace(self, **changes)

    def with_overrides(self, items: Sequence[str]) -> GenSettings:
        """Apply ``["temperature=0.7", "extra.particles=4", "system_prompt=none"]``."""
        current = self.as_dict()
        extra = dict(current.pop("extra"))
        for item in items:
            key, sep, text = item.partition("=")
            key, text = key.strip(), text.strip()
            if not sep:
                raise ValueError(f"override {item!r} is not key=value")
            if key.startswith("extra."):
                extra[key[6:]] = _parse_scalar(text)
                continue
            if key not in current:
                raise KeyError(f"unknown settings field {key!r}")
            old = current[key]
            if key == "system_prompt":
                current[key] = None if text.lower() in ("none", "null") else text
            elif isinstance(old, bool):
                current[key] = text.lower() in ("1", "true", "yes", "on")
            elif isinstance(old, int):
                current[key] = int(text)
            elif isinstance(old, float):
                current[key] = float(text)
            else:
                current[key] = text
        current["extra"] = extra
        return GenSettings.from_dict(current)


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


# --------------------------------------------------------------------------------- replies
@dataclass
class Reply:
    """One generated reply.

    ``text`` is what the user sees (stripped, reasoning removed when the settings ask for it);
    ``raw_text`` is everything the model generated. ``finish_reason`` is ``"stop"`` (end of turn
    token), ``"length"`` (``max_tokens`` reached) or ``"error"``. ``logprob_sum`` is the sum of
    ``log p(token)`` under the model's unmodified distribution over all generated tokens
    including the end-of-turn token, ``logprob_count`` the number of such tokens (``None`` and
    0 when the backend cannot provide them).
    """

    text: str
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = "stop"
    logprob_sum: float | None = None
    logprob_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Reply:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def logprob_mean(self) -> float | None:
        if self.logprob_sum is None or self.logprob_count <= 0:
            return None
        return self.logprob_sum / self.logprob_count


class Backend(Protocol):
    """A model the harness can query. ``key`` identifies the model (weights, chat template and
    code version) for the cache: two backends with the same key must give the same replies."""

    key: str

    def generate(self, messages: Sequence[Message], settings: GenSettings, seed: int) -> Reply:
        ...


# ----------------------------------------------------------------------------- reasoning
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> tuple[str, dict[str, Any]]:
    """Remove ``<think>...</think>`` blocks. An unclosed ``<think>`` hides everything after it
    (the reasoning was cut off, so there is no visible answer). A stray ``</think>`` without an
    opening tag (templates that put ``<think>`` in the prompt) hides everything before it.

    Returns ``(visible_text, info)`` with ``info = {"think_blocks", "think_unclosed",
    "think_chars"}``.
    """
    n_blocks = len(_THINK_BLOCK.findall(text))
    visible = _THINK_BLOCK.sub("", text)
    unclosed = False
    if "<think>" in visible:
        unclosed = True
        n_blocks += 1
        visible = visible[: visible.index("<think>")]
    elif "</think>" in visible:
        n_blocks += 1
        visible = visible.rpartition("</think>")[2]
    info = {"think_blocks": n_blocks, "think_unclosed": unclosed,
            "think_chars": len(text) - len(visible)}
    return visible.strip(), info


# ---------------------------------------------------------------------- candidate selection
def select_candidate(cands: Sequence[Reply], rule: str, is_degenerate=None) -> int:
    """Index of the candidate to keep.

    - ``first``: candidate 0 (single-sample arms).
    - ``logprob_mean``: highest mean log-probability per generated token; candidates cut by
      ``max_tokens`` rank after finished ones; candidates without log-probabilities rank last.
    - ``filtered_logprob_mean``: as ``logprob_mean`` after dropping candidates that
      ``is_degenerate(reply) -> bool`` flags (empty, looping, truncated, unclosed fence), unless
      all are flagged. This is the control arm that mirrors our own failure filter.
    Ties break towards the lower index, so the rule is deterministic.
    """
    if not cands:
        raise ValueError("no candidates")
    if rule == "first" or len(cands) == 1:
        return 0
    if rule not in SELECT_RULES:
        raise ValueError(f"unknown select rule {rule!r}")
    pool = list(range(len(cands)))
    if rule == "filtered_logprob_mean" and is_degenerate is not None:
        kept = [i for i in pool if not is_degenerate(cands[i])]
        pool = kept or pool

    def rank(i: int) -> tuple:
        r = cands[i]
        mean = r.logprob_mean
        return (r.finish_reason == "stop", mean is not None, mean if mean is not None else 0.0, -i)

    return max(pool, key=rank)


# ------------------------------------------------------------------------------- backends
def resolve_callable(target: str) -> Any:
    """``"package.module:attr"`` -> the attribute."""
    module, sep, attr = target.partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"expected 'module:attribute', got {target!r}")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_backend(spec: Mapping[str, Any]) -> Backend:
    """Build a backend from a picklable spec.

    - ``{"kind": "gguf", "path": ..., "n_ctx": ..., "n_threads": ..., "chat_template": ...,
      "key": ...}`` -> :class:`bes.eval.proxy.run_gguf.GGUFBackend`;
    - ``{"kind": "callable", "factory": "module:attr", "kwargs": {...}, "key": ...}`` -> the
      factory's generator wrapped by :class:`bes.eval.proxy.run_ours.CallableBackend`;
    - ``{"kind": "backend", "factory": "module:attr", "kwargs": {...}}`` -> whatever the factory
      returns (it must already implement :class:`Backend`).
    """
    kind = spec.get("kind")
    if kind == "gguf":
        from bes.eval.proxy.run_gguf import GGUFBackend
        args = {k: v for k, v in spec.items() if k != "kind"}
        return GGUFBackend(**args)
    if kind == "callable":
        from bes.eval.proxy.run_ours import CallableBackend
        factory = resolve_callable(spec["factory"])
        generator = factory(**dict(spec.get("kwargs") or {}))
        return CallableBackend(generator, key=spec.get("key"))
    if kind == "backend":
        factory = resolve_callable(spec["factory"])
        return factory(**dict(spec.get("kwargs") or {}))
    raise ValueError(f"unknown backend kind {kind!r}")


# ---------------------------------------------------------------------------------- files
def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """All complete JSON lines of ``path`` (missing file -> ``[]``). A truncated or corrupt
    line (an interrupted append) is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
    return out


def append_jsonl(path: str | os.PathLike, records: Iterable[Mapping[str, Any]], sync: bool = True) -> None:
    """Append records as JSON lines; with ``sync`` the data reaches the disk before returning.
    If the file does not end with a newline (an earlier append was interrupted), a newline is
    written first so the new records stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
    if not lines:
        return
    with open(path, "a+b") as f:
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(lines.encode("utf-8"))
        f.flush()
        if sync:
            os.fsync(f.fileno())


def write_text_atomic(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: str | os.PathLike, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def write_jsonl_atomic(path: str | os.PathLike, records: Iterable[Mapping[str, Any]]) -> None:
    write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records))


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_name(text: str) -> str:
    """A file-name-safe version of ``text`` (letters, digits, ``.-_`` kept)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"
