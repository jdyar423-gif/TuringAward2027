"""Supervisor of the single 24-hour run (``docs/PLAN.md`` sections 6 and 9).

What lives here (``scripts/run_24h.py`` wires it to the real stages):

* **Run directory and manifest** (:class:`RunPaths`, :func:`write_manifest`): ``manifest.json``
  is written atomically at the start (configuration, overrides, budget, argv, code commit,
  host and device report, core plan) and extended by the benchmark (execution plan, horizon,
  fallback-ladder rung) and by the decode selection. A second run in the same directory is
  refused unless ``--resume``; a live :class:`RunLock` refuses two supervisors at once.
* **Preflight** (:func:`host_report`, :func:`device_report`, :func:`check_devices`).
* **Host core plan** (:func:`core_plan`): a small static division of the host's cores, read at
  run time (``os.sched_getaffinity``):

  ==================  =============================  =====  =====  ======
  share               rule                           n=4    n=8    n=44
  ==================  =============================  =====  =====  ======
  trainer process     2 (device driver, loader       2      2      2
                      coordinator, Python)
  loader workers      clamp(n // 4, 2, 6)            2      2      6
  loader maximum      max(workers, min(12, n - 4))   2      4      12
  tokenisation        clamp(n // 8, 1, 4)            1      1      4
  downloads           1 core, 4 threads              1      1      1
  library build       2                              2      2      2
  side-car            1 (nice 19)                    1      1      1
  opponents (GGUF)    the rest, 2 threads/worker     1x1    1x1    14x2
  ==================  =============================  =====  =====  ======

  Background processes are pinned (CPU affinity) to their own cores counted from the top of the
  core list and run at a lower priority (``nice``); the trainer process and its loader workers
  are not pinned (the device compiler uses many threads) and keep the default priority. When
  there are fewer cores than shares the assignment wraps around (shared cores). With the core
  list of this run, the benchmark may raise the loader to ``loader_max`` workers.
* **Background processes** (:class:`ProcessSupervisor`): children started by one monitor
  thread, restarted on failure with exponential back-off, logs in ``logs/<task>.log``, state in
  ``background.json``; a resumed run starts every task again (each is idempotent and cheap
  once its work is on disk, and a restarted machine may have lost that work); children die
  with the supervisor (Linux ``PR_SET_PDEATHSIG``) and get SIGTERM when it exits.
* **The stage loop** (:class:`Supervisor`): stages in :data:`bes.train.stages.ORDER`, each with
  a slice from :func:`bes.train.stages.allocate`, recorded in the stage ledger; ``--resume``
  continues at the first unfinished stage; SIGTERM / SIGINT request a stop (the trainer stops
  after its step and saves); a stage that cannot run raises :class:`StageSkipped`; a failed
  optional stage is recorded and the run goes on; the export always runs last.
* **The library gate** (:class:`LibraryGate`): the trainer's data callable passes every stage to
  the loader with the library fractions held at 0 until the library has been built and passed
  :func:`library_self_check`; then it calls ``loader.set_library`` once, in the main thread.
* **Background tasks**: ``python -m bes.train.supervisor task <name> --run <out_dir>`` for
  ``downloads``, ``tokenize``, ``library``, ``gguf``, ``opponents`` and ``sidecar``.

This module imports no JAX at import time (background tasks never touch the accelerator).
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import fcntl
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from bes.config import Cfg, LibraryCfg, load_config
from bes.train import stages as ST
from bes.train.ledger import atomic_write_json

REPO = Path(__file__).resolve().parents[2]
FORMAT = 1
GB = 1e9

#: Fallback ladder of the canary, switched off one group at a time, cumulatively. First the v9
#: paths (``docs/V9.md``): the head-wise attention output gate, selective attention, the token
#: gain pages and the token neurons; then plan section 6: sink, confidence channel, n-gram count
#: heads, library passages, candidate tilt, registers, calculator spans. Each rung: (group,
#: make_consts switches, config overrides). The model has no switch for calculator spans (they
#: are data and a decoding rule), so that rung changes nothing; it is kept so the ladder reads as
#: the plan does. A rung whose paths a configuration does not have changes nothing for it either:
#: the canary does not run such a rung again (``bes.train.benchmark.canary``), and the rung
#: numbers are the same for every configuration.
LADDER: tuple[tuple[str, dict[str, float], tuple[str, ...]], ...] = (
    ("gate", {"gate": 0.0}, ()),
    ("sel", {"sel": 0.0}, ()),
    ("gain", {"gain": 0.0}, ()),
    ("page", {"page": 0.0}, ()),
    ("sink", {"sink": 0.0}, ()),
    ("conf", {"conf": 0.0}, ()),
    ("ngram", {"cand_in": 0.0}, ()),
    ("library", {}, ("data.library_frac_base=0", "data.library_frac_chat=0")),
    ("tilt", {"cand_out": 0.0}, ()),
    ("registers", {"reg": 0.0, "clock": 0.0}, ()),
    ("calc", {}, ()),
)


class PreflightError(RuntimeError):
    """An assumption of the run does not hold on this machine (raised before anything starts)."""


class StageSkipped(Exception):
    """Raised by a stage that cannot run (a missing module, nothing to do); the reason is
    recorded and the run goes on."""


class Interrupted(Exception):
    """Raised by a stage that stopped on a stop request before it was complete."""


class FatalStageError(RuntimeError):
    """An essential stage failed; the run stops (``--resume`` retries that stage)."""


class RunLocked(RuntimeError):
    """Another supervisor holds the run directory."""


# ============================================================================ run directory
@dataclass(frozen=True)
class RunPaths:
    """Files and directories of one run."""

    out: Path

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.json"

    @property
    def stages(self) -> Path:
        return self.out / "stages.json"

    @property
    def lock(self) -> Path:
        return self.out / ".supervisor.lock"

    @property
    def logs(self) -> Path:
        return self.out / "logs"

    @property
    def state(self) -> Path:
        return self.out / "state"

    @property
    def background(self) -> Path:
        return self.out / "background.json"

    @property
    def train(self) -> Path:
        return self.out / "train"

    @property
    def bench(self) -> Path:
        return self.out / "benchmark"

    @property
    def post(self) -> Path:
        return self.out / "post"

    @property
    def final(self) -> Path:
        return self.out / "final"

    @property
    def health(self) -> Path:
        return self.out / "health"

    @property
    def jax_cache(self) -> Path:
        return self.out / "jax_cache"

    def marker(self, task: str) -> Path:
        """``state/<task>.json``: written by a background task when it has finished its job."""
        return self.state / f"{task}.json"


def tok_root(cfg: Cfg, out: str | Path) -> Path:
    """Directory holding ``tok/<source>/`` (``RunCfg.tok_root``: "" = ``data_dir``, ``"run"`` =
    ``<out>/data``, else that path)."""
    if cfg.run.tok_root == "run":
        return Path(out) / "data"
    return Path(cfg.run.tok_root) if cfg.run.tok_root else Path(cfg.run.data_dir)


def proxy_root(cfg: Cfg, out: str | Path) -> Path:
    """Output root of the proxy evaluation (``RunCfg.proxy_root``)."""
    out = Path(out)
    if cfg.run.proxy_root == "run":
        return out / "proxy"
    return Path(cfg.run.proxy_root) if cfg.run.proxy_root else Path(cfg.run.data_dir) / "proxy"


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def read_manifest(out: str | Path) -> dict | None:
    return read_json(RunPaths(Path(out)).manifest)


def write_manifest(out: str | Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(RunPaths(Path(out)).manifest, _jsonable(dict(manifest)))


def update_manifest(out: str | Path, **fields: Any) -> dict:
    """Read, update the top-level ``fields`` and atomically rewrite the manifest."""
    man = read_manifest(out) or {}
    man.update(fields)
    write_manifest(out, man)
    return man


def write_marker(paths: RunPaths, task: str, **fields: Any) -> None:
    atomic_write_json(paths.marker(task), _jsonable({"task": task, "time": time.time(), **fields}))


def read_marker(paths: RunPaths, task: str) -> dict | None:
    return read_json(paths.marker(task))


def move_deadline(out: str | Path, deadline: float, *, now: float | None = None,
                  reason: str = "") -> dict:
    """Move the deadline of the run in ``out`` (no run may be running there) and return
    ``{"from", "to", "moved_s", "train_end", "left_s", "files"}``.

    A run is one budget of wall time, and a restart counts its downtime against that budget.
    This is the exception for a run that continues on another machine whose own lifetime decides
    how long it may still run: the time between the machines is not the run's (the deadline
    moves later), or the new machine does not live as long as the run had left (it moves
    earlier). The schedule needs no other change: progress is ``T / (T + remaining)``, so it
    stretches or compresses into the time that is left (``bes.train.schedule``). ``budget_s``,
    the plan the stage table shows, is kept; every move is recorded in the manifest
    (``budget.moved``). The stage ledger and the trainer's ``run.json`` hold the same deadline
    and are rewritten with it. The new deadline must lie in the future."""
    paths = RunPaths(Path(out))
    man = read_manifest(out)
    if man is None or "budget" not in man:
        raise ValueError(f"{out}: no run (no manifest.json with a budget)")
    now = time.time() if now is None else float(now)
    new, old = float(deadline), float(man["budget"]["deadline"])
    if not math.isfinite(new) or new <= now:
        raise ValueError(f"the new deadline ({new:.0f}) is not in the future (now {now:.0f})")
    files = []
    stages = read_json(paths.stages)
    if isinstance(stages, dict) and "deadline" in stages:
        stages["deadline"] = new
        atomic_write_json(paths.stages, stages)
        files.append(paths.stages.name)
    run_json = paths.train / "run.json"
    info = read_json(run_json)
    if isinstance(info, dict) and "deadline" in info:
        info["deadline"] = new
        atomic_write_json(run_json, info)
        files.append("train/run.json")
    budget = dict(man["budget"])
    budget["deadline"] = new
    budget["moved"] = list(budget.get("moved", [])) + [
        {"time": now, "from": old, "to": new, "reason": str(reason)}]
    man["budget"] = budget
    write_manifest(out, man)
    files.append(paths.manifest.name)
    post_s = base_config(man).run.post_reserve_hours * 3600.0
    return {"from": old, "to": new, "moved_s": new - old, "train_end": new - post_s,
            "left_s": new - now, "files": files}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return _jsonable(x.item())
        if isinstance(x, np.ndarray):
            return _jsonable(x.tolist())
    except ImportError:  # pragma: no cover
        pass
    return repr(x)


class RunLock:
    """Exclusive, non-blocking lock of a run directory (``fcntl.flock``); released when the
    process ends, however it ends."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fd: int | None = None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.pread(fd, 64, 0).decode().strip()
            except OSError:
                pass
            os.close(fd)
            raise RunLocked(f"another supervisor (pid {holder or '?'}) is running in "
                            f"{self.path.parent}") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode(), 0)
        self.fd = fd
        return self

    def release(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    __enter__ = acquire

    def __exit__(self, *exc: Any) -> None:
        self.release()


def code_commit(repo: str | Path = REPO) -> dict[str, Any]:
    """``{"commit", "dirty"}`` from read-only git commands (``"unknown"`` without git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                capture_output=True, text=True, timeout=10).stdout
        return {"commit": commit or "unknown", "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


# ================================================================================= preflight
def host_cores() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def _meminfo() -> dict[str, float]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[key] = int(rest.split()[0]) * 1024 / GB
    except OSError:
        pass
    return out


def host_report(out: str | Path) -> dict[str, Any]:
    """Cores, memory, free disk under ``out``, whether ``HF_TOKEN`` is set, Python, platform."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(out)
    mem = _meminfo()
    return {"cores": len(host_cores()), "core_ids": host_cores(), "cpu_count": os.cpu_count(),
            "ram_gb": round(mem.get("MemTotal", 0.0), 2),
            "ram_available_gb": round(mem.get("MemAvailable", 0.0), 2),
            "disk_free_gb": round(du.free / GB, 2), "disk_total_gb": round(du.total / GB, 2),
            "hf_token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "python": platform.python_version(), "platform": platform.platform(),
            "hostname": platform.node()}


def device_report() -> dict[str, Any]:
    """JAX version, backend, devices (kind, platform) and the first device's memory limit."""
    import jax
    import jaxlib

    devs = jax.devices()
    rep: dict[str, Any] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                           "backend": jax.default_backend(),
                           "devices": [{"id": d.id, "platform": d.platform, "kind": d.device_kind}
                                       for d in devs]}
    try:
        stats = devs[0].memory_stats() or {}
        rep["memory_limit_gb"] = round(stats.get("bytes_limit", 0) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001 - not every backend reports memory
        rep["memory_limit_gb"] = None
    try:
        import libtpu  # noqa: F401
        rep["libtpu"] = getattr(sys.modules["libtpu"], "__version__", "present")
    except ImportError:
        rep["libtpu"] = None
    return rep


PINNED_JAX = "0.11.1"


def check_devices(rep: Mapping[str, Any], allow_cpu: bool) -> list[str]:
    """Raise :class:`PreflightError` when the accelerator assumption fails; return warnings.

    The run expects one TPU (a v6e chip, ``device_kind`` "TPU v6 lite"); a CPU-only machine is
    accepted only with ``allow_cpu`` (rehearsals)."""
    warnings = []
    backend = rep.get("backend")
    kinds = [d["kind"] for d in rep.get("devices", [])]
    if backend != "tpu":
        if not allow_cpu:
            raise PreflightError(
                f"no TPU: JAX backend is {backend!r} with devices {kinds}. The 24-hour run needs "
                "one TPU v6e; pass --allow-cpu only for a rehearsal on a CPU host.")
        warnings.append(f"running on {backend} ({kinds}) because --allow-cpu was given: a rehearsal")
    else:
        if not any("v6" in k for k in kinds):
            warnings.append(f"device kind {kinds} is not a TPU v6e: memory limits and throughput "
                            "assumptions of the plan may not hold")
        if len(kinds) != 1:
            warnings.append(f"{len(kinds)} devices found; the run uses device 0 only")
    if rep.get("jax") != PINNED_JAX:
        warnings.append(f"jax {rep.get('jax')} is not the pinned {PINNED_JAX}")
    return warnings


SHARD_BYTES_PER_RAW_BYTE = 1.4   # 2 bytes per token; a ClimbMix parquet of 92 MB holds 63.8 M tokens
DISK_OTHER_GB = 6.0              # library, checkpoints (three of the parameters and both moments),
                                 # compilation cache, post-training outputs
#: bytes per stored number of a token page in one checkpoint: the float32 parameter and its two
#: Adam moments
PAGE_CKPT_BYTES = 12


def page_numbers(model_cfg: Any) -> int:
    """Stored numbers of the v9 token pages of a ``ModelCfg`` (``gain/*``: ``V x P x n_blocks x
    width`` per table; ``page/key`` and ``page/val``: ``V x n_blocks x rho x dk`` each, plus the
    projections ``page/pin``, ``page/pout``); 0 without them. Plain arithmetic, no JAX."""
    m = model_cfg
    hw = m.n_heads * m.head_dim
    width = m.gain_ffn + (hw if m.gain_o else 0) + (hw if m.gain_v else 0)
    gain = m.vocab_size * m.gain_phases * m.n_blocks * width
    page = 0
    if m.tok_neurons:
        page = (2 * m.vocab_size * m.n_blocks * m.tok_neurons * m.tok_dim
                + 2 * m.n_blocks * m.d_model * m.tok_dim)
    return gain + page


def disk_need(cfg: Cfg, tokenize_workers: int = 4) -> dict[str, Any]:
    """The run's disk estimate, stated in the manifest by the preflight (GB = 1e9 bytes).

    Downloads ``D`` at the registry caps; token shards ``1.4 D``; raw files left on disk: all of
    them, or with ``run.delete_raw_after_tokenize`` (a raw file this run downloaded is deleted
    once its whole shard is committed, :func:`prune_raw`) only the newest file of each source
    the sampling stage reads. With deletion the disk holds ``downloaded + 0.4 x tokenised`` raw
    and shard bytes at any time, so the peak is ``1.4 D`` plus the kept files, however far the
    tokenisation lags behind the downloads. Plus the shard temporaries of the tokenisation
    workers (one largest file each) and :data:`DISK_OTHER_GB` for everything else; the v9 token
    pages (:func:`page_numbers`, 0.76 GB per checkpoint for ``configs/v9.py``) add their bytes in
    ``run.keep_ckpts`` checkpoints and the one being written to ``other_gb``."""
    from bes.data import sources

    sizes: dict[str, float] = {}
    largest = 0.0
    kept = 0.0
    for src, max_files, _ in download_plan(cfg):
        s = sources.get(src)
        cap = sources.download_cap(s, cfg.data.permissive_only)
        size = float(s.approx_bytes if cap is None else min(cap, s.approx_bytes or cap))
        if max_files is not None and s.n_files:
            size = min(size, s.approx_bytes / s.n_files * max_files)
        per_file = size if not s.n_files else min(size, s.approx_bytes / s.n_files)
        largest = max(largest, per_file)
        if src not in sizes and s.kind in ("chat", "math") and s.fmt == "messages":
            kept += per_file                  # the newest file stays for the sampling prompts
        sizes[src] = max(sizes.get(src, 0.0), size)      # a source can appear in both phases
    total = sum(sizes.values())
    delete = bool(cfg.run.delete_raw_after_tokenize)
    raw_left = min(total, kept) if delete else total
    temp = SHARD_BYTES_PER_RAW_BYTE * largest * max(1, int(tokenize_workers))
    need = SHARD_BYTES_PER_RAW_BYTE * total + raw_left + temp
    pages = (cfg.run.keep_ckpts + 1) * PAGE_CKPT_BYTES * page_numbers(cfg.model) / GB
    other = round(DISK_OTHER_GB + pages, 3)
    rule = ("1.4 x downloads (shards) + raw files left (all of them without deletion, the "
            "newest file of each sampled chat/math source with it) + one largest file per "
            "tokenisation worker + 6 GB (library, checkpoints, compilation cache)")
    if pages:
        rule += f" + {pages:.2f} GB (the token pages in {cfg.run.keep_ckpts + 1} checkpoints)"
    return {"downloads_gb": round(total / GB, 3),
            "shards_gb": round(SHARD_BYTES_PER_RAW_BYTE * total / GB, 3),
            "raw_left_gb": round(raw_left / GB, 3), "tokenize_temp_gb": round(temp / GB, 3),
            "other_gb": other, "total_gb": round(need / GB + other, 3),
            "delete_raw_after_tokenize": delete, "rule": rule}


def disk_need_gb(cfg: Cfg, tokenize_workers: int = 4) -> float:
    """``disk_need(cfg)["total_gb"]``."""
    return float(disk_need(cfg, tokenize_workers)["total_gb"])


#: Mount points of network drives the data must never be written to (Colab's Google Drive).
DRIVE_PREFIXES: tuple[str, ...] = ("/content/drive",)


def data_locations(cfg: Cfg, out: str | Path) -> dict[str, Path]:
    """Where the run writes (absolute): the run directory, the data root (raw downloads, the
    library's sources), the token shards, the library, the proxy outputs and the opponents'
    models. All but the run directory are relative to the working directory or the code."""
    locs = {"run": Path(out), "data": Path(cfg.run.data_dir), "tok": tok_root(cfg, out),
            "library": Path(LibraryCfg().library_dir), "proxy": proxy_root(cfg, out)}
    try:
        from bes.eval.proxy.common import models_dir

        locs["models"] = Path(models_dir())
    except Exception:  # noqa: BLE001 - the harness is optional here
        pass
    return {k: Path(os.path.abspath(os.path.realpath(v))) for k, v in locs.items()}


def on_drive(path: str | Path) -> bool:
    """``path`` lies on a network drive of :data:`DRIVE_PREFIXES`."""
    p = os.path.abspath(os.path.realpath(path))
    return any(p == d or p.startswith(d.rstrip("/") + "/") for d in DRIVE_PREFIXES)


def free_gb(path: str | Path) -> float:
    """Free GB of the file system that holds ``path`` (its nearest existing parent)."""
    p = Path(os.path.abspath(path))
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free / GB


def _committed_whole(entry: Mapping[str, Any] | None, path: str, size: int) -> bool:
    """The committed shard ``entry`` was made from the whole raw file ``path`` of ``size`` bytes
    (not cut at ``run.tokenize_max_records``)."""
    if not entry or entry.get("max_records") is not None:
        return False
    inp = entry.get("input") or {}
    return inp.get("path") == path and int(inp.get("size", -1)) == int(size)


def _fetched_since(raw_dir: Path, t0: float) -> set[str]:
    """Paths the download manifest of ``raw_dir`` recorded at or after ``t0`` (fetched by this
    run); an entry whose time cannot be read counts as older."""
    from bes.data import download

    out: set[str] = set()
    p = raw_dir / download.MANIFEST
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
            when = time.mktime(time.strptime(e["time"], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
        if isinstance(e.get("path"), str) and when >= t0 - 1.0:
            out.add(e["path"])
    return out


def prune_raw(paths: RunPaths, cfg: Cfg, name: str, log: Callable[[str], None]) -> list[str]:
    """With ``run.delete_raw_after_tokenize``: delete the raw downloads of source ``name`` whose
    whole token shard is committed. Kept: a file this run did not download (its download record
    is older than the run: a shared cache), a file whose shard was cut at
    ``run.tokenize_max_records``, and the newest downloaded file of a chat or math source with
    messages (the sampling stage reads its rows). Returns the deleted paths."""
    if not cfg.run.delete_raw_after_tokenize:
        return []
    from bes.data import download, shards, sources, tokenize_worker

    t0 = ((read_manifest(paths.out) or {}).get("budget") or {}).get("t0")
    if t0 is None:
        return []
    src = sources.get(name)
    try:
        files = download.completed_files(name, cfg.run.data_dir)
    except Exception:  # noqa: BLE001 - no listing yet
        return []
    done = shards.read_manifest(tokenize_worker.tok_dir(name, tok_root(cfg, paths.out)))
    keep = {max(f.index for f in files)} if files and src.kind in ("chat", "math") \
        and src.fmt == "messages" else set()
    owned = _fetched_since(download.repo_dir(src, cfg.run.data_dir), float(t0))
    deleted = []
    for f in files:
        if f.index in keep or f.path not in owned or not _committed_whole(done.get(f.index), f.path,
                                                                           f.size):
            continue
        try:
            os.unlink(f.local)
            deleted.append(f.path)
        except FileNotFoundError:
            pass
        except OSError as e:
            log(f"tokenize: {name}: could not delete {f.local}: {e}")
    if deleted:
        log(f"tokenize: {name}: deleted {len(deleted)} raw files whose shards are committed")
    return deleted


def _download_source(paths: RunPaths, cfg: Cfg, name: str, max_files: int | None,
                     log: Callable[[str], None]) -> int:
    """``download.download_source`` (4 threads, the registry cap) that, with
    ``run.delete_raw_after_tokenize``, does not fetch again a file whose whole shard is committed
    and whose raw file is gone (:func:`prune_raw`). Returns the number of selected files."""
    from concurrent.futures import ThreadPoolExecutor

    from bes.data import download, shards, sources, tokenize_worker

    cap = sources.download_cap(name, cfg.data.permissive_only)
    if not cfg.run.delete_raw_after_tokenize:
        return len(download.download_source(name, cfg.run.data_dir, max_files=max_files,
                                            max_bytes=cap, threads=4, log=log))
    src = sources.get(name)
    rdir = download.repo_dir(src, cfg.run.data_dir)
    commit, files = download.source_listing(src, cfg.run.data_dir, log=log)
    chosen = download.select_files(files, max_files, cap)
    index = {f.path: i for i, f in enumerate(files)}
    tdir = tokenize_worker.tok_dir(name, tok_root(cfg, paths.out))

    def one(rf) -> None:
        if not (rdir / rf.path).exists() and _committed_whole(
                shards.read_manifest(tdir).get(index[rf.path]), rf.path, rf.size):
            return                            # tokenised whole, raw file deleted on purpose
        download.download_file(src.repo, commit, rf, rdir, log=log)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, chosen))
    return len(chosen)


# ================================================================================ core plan
@dataclass(frozen=True)
class CorePlan:
    """Static division of the host cores (module docstring). Tuples are CPU ids."""

    cores: tuple[int, ...]
    trainer: tuple[int, ...]
    loader_workers: int
    loader_max: int
    tokenize: tuple[int, ...]
    downloads: tuple[int, ...]
    library: tuple[int, ...]
    sidecar: tuple[int, ...]
    opponents: tuple[int, ...]
    opponent_workers: int
    opponent_threads: int
    judge_threads: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorePlan":
        kw = {}
        for f in dataclasses.fields(cls):
            v = d[f.name]
            kw[f.name] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)


def core_plan(cores: Sequence[int] | int | None = None) -> CorePlan:
    """The core plan for this host (or for ``cores``: a CPU id list or a count)."""
    if cores is None:
        ids = host_cores()
    elif isinstance(cores, int):
        ids = list(range(cores))
    else:
        ids = sorted(int(c) for c in cores)
    n = len(ids)
    if n < 1:
        raise ValueError("no CPU cores")
    loader = min(6, max(2, n // 4))
    loader_max = max(loader, min(12, n - 4))
    tok = min(4, max(1, n // 8))
    trainer = tuple(ids[:min(2, n)])
    # background shares are taken from the top of the core list, wrapping around when short
    top = list(reversed(ids))
    pos = 0

    def take(k: int) -> tuple[int, ...]:
        nonlocal pos
        out = tuple(sorted({top[(pos + i) % n] for i in range(k)}))
        pos += k
        return out

    downloads = take(1)
    tokenize = take(tok)
    library = take(2)
    sidecar = take(1)
    rest = n - 2 - loader - tok - 1 - 2 - 1
    opponents = take(max(1, rest)) if rest > 0 else tuple(top[:1])
    opp_threads = 2 if len(opponents) >= 2 else 1
    return CorePlan(cores=tuple(ids), trainer=trainer, loader_workers=loader, loader_max=loader_max,
                    tokenize=tokenize, downloads=downloads, library=library, sidecar=sidecar,
                    opponents=opponents, opponent_workers=max(1, len(opponents) // opp_threads),
                    opponent_threads=opp_threads, judge_threads=max(1, n - 2))


# ======================================================================= effective config
def parse_switches(text: str) -> dict[str, float]:
    """``"sink:0 conf:0 src_5:0.25"`` -> ``{"sink": 0.0, ...}`` (``make_consts`` keywords)."""
    out: dict[str, float] = {}
    for item in str(text).replace(",", " ").split():
        name, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"switch {item!r}: expected name:value")
        out[name.strip()] = float(value)
    return out


def format_switches(sw: Mapping[str, float]) -> str:
    return " ".join(f"{k}:{float(v):g}" for k, v in sw.items())


def ladder_rung(k: int) -> tuple[dict[str, float], list[str], list[str]]:
    """Rung ``k`` of :data:`LADDER` (``k`` groups switched off, ``0`` = full): ``(switches,
    config overrides, groups)``."""
    if not 0 <= k <= len(LADDER):
        raise ValueError(f"ladder rung {k} outside 0..{len(LADDER)}")
    sw: dict[str, float] = {}
    ov: list[str] = []
    groups: list[str] = []
    for name, s, o in LADDER[:k]:
        sw.update(s)
        ov.extend(o)
        groups.append(name)
    return sw, ov, groups


def safe_switches() -> dict[str, float]:
    """Every feature path off (``bes.model.model.switches_off``)."""
    from bes.model.model import switches_off

    return dict(switches_off())


def make_consts_fn(switches: Mapping[str, float] | str) -> Callable:
    """``consts_fn(model_cfg)`` for the trainer: ``make_consts`` with the given switches."""
    sw = parse_switches(switches) if isinstance(switches, str) else dict(switches)

    def consts_fn(model_cfg):
        from bes.model import model as M

        return M.make_consts(model_cfg, **sw)

    return consts_fn


def run_overrides(man: Mapping[str, Any]) -> list[str]:
    """The overrides of the effective configuration, in order: the command line (budget
    included), the benchmark's execution plan and the fallback-ladder rung (the horizon is
    applied by :func:`effective_config` with ``schedule.with_horizon``)."""
    out = list(man.get("overrides", []))
    bench = man.get("benchmark") or {}
    out += list(bench.get("plan_overrides", []))
    ladder = bench.get("ladder") or {}
    out += list(ladder.get("overrides", []))
    if ladder.get("switches") is not None:
        out.append(f"run.switches={ladder['switches']}")
    return out


def horizon_of(man: Mapping[str, Any]) -> float | None:
    """The benchmark's horizon estimate (expected training tokens), ``None`` before it."""
    h = (man.get("benchmark") or {}).get("horizon_tokens")
    return None if h is None else float(h)


def base_config(man: Mapping[str, Any]) -> Cfg:
    """The configuration of the run before the benchmark (file + command-line overrides)."""
    return load_config(str(man["config"]), list(man.get("overrides", [])))


def effective_config(man: Mapping[str, Any]) -> Cfg:
    """The configuration the trainer and every later stage use: :func:`run_overrides`, then
    ``schedule.with_horizon`` with the benchmark's horizon (before any optimiser is built; the
    same value on every resume)."""
    from bes.train.schedule import with_horizon

    cfg = load_config(str(man["config"]), run_overrides(man))
    h = horizon_of(man)
    return cfg if h is None else with_horizon(cfg, h)


def generator_overrides(man: Mapping[str, Any], extra: Sequence[str] = ()) -> str:
    """Comma-separated ``model.*`` / ``decode.*`` overrides for ``bes.decode.respond``'s
    generator (other sections do not change a reply)."""
    keep = [o for o in list(run_overrides(man)) + list(extra)
            if o.split("=", 1)[0].strip().split(".")[0] in ("model", "decode")]
    for o in keep:
        if "," in o:
            raise ValueError(f"override {o!r} contains a comma; the generator cannot take it")
    return ",".join(keep)


# ======================================================================= library readiness
#: The library task's record of the settings a library was built with, written into the library
#: directory after the build (``bes.export`` reads it into the model card and never ships it).
LIBRARY_SETTINGS = "build_settings.json"


def library_settings(cfg: Cfg) -> dict[str, Any]:
    """The settings the run's library task builds a library with: ``permissive_only``
    (``data.permissive_only``), the question-answer sources such a build stores (the store
    sources of ``bes.library.sources``, those without a clear licence left out by a permissive
    build), the SHA-256 of the tokenizer the passage lengths are measured with, and
    ``LibraryCfg`` (as JSON)."""
    from bes.library.sources import DEFAULT_QA_STORE, QA_SOURCES
    from bes.tokenizer import load as load_tokenizer

    perm = bool(cfg.data.permissive_only)
    qa = [n for n in DEFAULT_QA_STORE
          if QA_SOURCES[n].role == "store" and (QA_SOURCES[n].permissive or not perm)]
    return {"permissive_only": perm, "qa_sources": sorted(qa),
            "tokenizer_sha256": load_tokenizer().sha256,
            "library_cfg": json.loads(json.dumps(dataclasses.asdict(LibraryCfg())))}


def recorded_library_settings(path: str | Path) -> dict[str, Any]:
    """What a built library records about its build: ``permissive_only`` (``library.json``
    ``build``, else ``qa.json``; ``None`` when neither says), ``qa_stored`` (the sources of its
    question-answer layer, ``qa.json``; ``[]`` without one) and, from the library task's record
    :data:`LIBRARY_SETTINGS` (absent for a library built by hand with
    ``scripts/build_library.py``: ``recorded`` False), the sources asked for, the tokenizer's
    SHA-256 and ``LibraryCfg``."""
    root = Path(path)
    meta = read_json(root / "library.json") or {}
    qa = read_json(root / "qa.json") if (root / "qa.json").exists() else None
    stamp = read_json(root / LIBRARY_SETTINGS) if (root / LIBRARY_SETTINGS).exists() else None
    perm = (meta.get("build") or {}).get("permissive_only")
    if perm is None and qa is not None:
        perm = qa.get("permissive_only")
    return {"permissive_only": perm, "qa_stored": sorted((qa or {}).get("sources") or []),
            "qa_sources": (stamp or {}).get("qa_sources"),
            "tokenizer_sha256": (stamp or {}).get("tokenizer_sha256"),
            "library_cfg": (stamp or {}).get("library_cfg"), "recorded": stamp is not None}


def library_settings_mismatch(path: str | Path, want: Mapping[str, Any], *,
                              allow_unrecorded: bool = False) -> list[str]:
    """Why the library at ``path`` was not built with the run's settings ``want``
    (:func:`library_settings`); ``[]`` when it was. Compared: the permissive flag, the stored
    question-answer sources (a source the run does not allow always disqualifies), and from the
    library task's record the sources asked for, the tokenizer and ``LibraryCfg``. A library
    without that record is refused (its tokenizer and ``LibraryCfg`` are unknown) unless
    ``allow_unrecorded`` (``run.library_reuse_unrecorded``); it must then store exactly the
    run's sources."""
    got = recorded_library_settings(path)
    reasons = []
    if got["permissive_only"] is None:
        reasons.append("the library does not record whether it was built permissive-only")
    elif bool(got["permissive_only"]) != bool(want["permissive_only"]):
        reasons.append(f"built with permissive_only={bool(got['permissive_only'])}, the run has "
                       f"permissive_only={bool(want['permissive_only'])}")
    extra = sorted(set(got["qa_stored"]) - set(want["qa_sources"]))
    if extra:
        reasons.append(f"stores question-answer sources the run does not allow: {extra}")
    if not got["recorded"]:
        if not allow_unrecorded:
            reasons.append(f"no {LIBRARY_SETTINGS}: built outside the run's library task, so its "
                           "tokenizer and LibraryCfg are unknown (run.library_reuse_unrecorded=true "
                           "reuses such a library)")
        elif got["qa_stored"] != list(want["qa_sources"]):
            reasons.append(f"stores the question-answer sources {got['qa_stored']}, the run builds "
                           f"{list(want['qa_sources'])}")
        return reasons
    if got["qa_sources"] != list(want["qa_sources"]):
        reasons.append(f"built from the question-answer sources {got['qa_sources']}, the run builds "
                       f"{list(want['qa_sources'])}")
    if got["tokenizer_sha256"] != want["tokenizer_sha256"]:
        reasons.append(f"built with tokenizer {str(got['tokenizer_sha256'])[:12]}, the run's is "
                       f"{str(want['tokenizer_sha256'])[:12]}")
    if got["library_cfg"] != want["library_cfg"]:
        a, b = got["library_cfg"] or {}, want["library_cfg"]
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        reasons.append(f"built with another LibraryCfg (fields {diff})")
    return reasons


def library_self_check(path: str | Path, *, cap: int | None = None, max_read: int | None = None,
                       probes: Sequence[str] = ("What is DNA?", "Who was Napoleon?",
                                                "Tell me about Paris."),
                       min_probe_hits: int = 1, sample_passages: int = 8) -> dict[str, Any]:
    """Load a built library and check it: stored numbers within the cap, read bound within the
    limit, a spread of compressed blocks decompress with matching CRC, the probes retrieve at
    least ``min_probe_hits`` entries. Returns ``{"ok", "reasons", ...}``; never raises."""
    L = LibraryCfg()
    cap = L.total_numbers_cap if cap is None else cap
    max_read = L.max_read_numbers if max_read is None else max_read
    rep: dict[str, Any] = {"path": str(path), "ok": False, "reasons": []}
    try:
        from bes.data.loader import library_fingerprint
        from bes.library.retrieve import Library

        lib = Library.load(str(path))
        acc = lib.accounting()
        rep.update(n_wiki=lib.n_wiki, n_qa=lib.n_qa, total_numbers=int(acc["total_numbers"]))
        if acc["total_numbers"] > cap:
            rep["reasons"].append(f"stores {acc['total_numbers']} numbers, above the cap {cap}")
        bound = lib.read_bound()
        rep["read_bound"] = int(bound.get("total", 0))
        if rep["read_bound"] > max_read:
            rep["reasons"].append(f"read bound {rep['read_bound']} above {max_read}")
        n = lib.n_passages
        step = max(1, n // max(1, sample_passages))
        for pid in range(0, n, step):
            p = lib.passage(pid)
            if not p.text:
                rep["reasons"].append(f"passage {pid} is empty")
                break
        hits = 0
        for q in probes:
            if lib.retrieve(q) is not None:
                hits += 1
        rep["probe_hits"] = hits
        if hits < min_probe_hits:
            rep["reasons"].append(f"{hits} of {len(probes)} probe questions retrieved an entry")
        rep["fingerprint"] = library_fingerprint(lib)
    except Exception as e:  # noqa: BLE001 - any failure means "not ready"
        rep["reasons"].append(f"{type(e).__name__}: {e}")
    rep["ok"] = not rep["reasons"]
    return rep


class LibraryGate:
    """The trainer's data callable: ``gate(index, stage) -> loader(index, stage')``.

    Until ``ready()`` returns a library (checked at most every ``check_every_s``), ``stage'`` is
    ``stage`` with ``library_frac_base = library_frac_chat = 0``; once it does,
    ``loader.set_library(lib)`` is called (here, in the trainer's thread, between two batches)
    and every later stage passes unchanged. ``active=True``: the loader already has the library
    (a resumed run). Every stage reaches the loader, so the loader sees every stage change."""

    def __init__(self, loader: Any, ready: Callable[[], Any], *, active: bool = False,
                 check_every_s: float = 30.0, clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] | None = None) -> None:
        self.loader = loader
        self.ready = ready
        self.active = bool(active)
        self.check_every_s = float(check_every_s)
        self.clock = clock
        self.log = log or (lambda msg: None)
        self._last = -math.inf
        self.activated_at_index: int | None = None

    def _check(self, index: int) -> None:
        now = self.clock()
        if self.active or now - self._last < self.check_every_s:
            return
        self._last = now
        lib = self.ready()
        if lib is not None:
            self.loader.set_library(lib)
            self.active = True
            self.activated_at_index = int(index)
            self.log(f"library ready: loader.set_library at data index {index}")

    def __call__(self, index: int, stage: Any) -> Any:
        self._check(index)
        if not self.active and (stage.library_frac_base or stage.library_frac_chat):
            stage = dataclasses.replace(stage, library_frac_base=0.0, library_frac_chat=0.0)
        return self.loader(index, stage)

    get = __call__

    def stats(self) -> dict[str, Any]:
        return self.loader.stats()


# ==================================================================== labelling of samples
def label_sample(user_text: str, text: str, finish: str, reference: str | None = None,
                 tol: float | None = None) -> dict[str, Any]:
    """Exact label of one sampled response (higher ``score`` is better, in ``[0, 1]``).

    ``score`` is 0 for a degenerate response (``bes.eval.exact_score.degenerate_stats``: empty,
    cut at the token limit, looping, repeated lines, unclosed fence, leaked role markers);
    otherwise the share of the user message's explicit constraints that hold
    (``bes.exact.constraints.compile`` / ``verify``; 1 without constraints), times 0 for a wrong
    final number when a ``reference`` answer is known (``exact_score.math_correct``)."""
    from bes.eval import exact_score
    from bes.exact import constraints as C

    deg = exact_score.degenerate_stats(text, "stop" if finish == "stop" else "length")
    out: dict[str, Any] = {"degenerate": bool(deg["degenerate"]), "violations": [],
                           "n_constraints": 0, "math_correct": None}
    try:
        spec = C.compile(user_text)
    except Exception:  # noqa: BLE001 - a compiler failure means "no constraint"
        spec = C.ConstraintSpec()
    if not spec.empty:
        out["n_constraints"] = max(1, len(spec.constraints))
        out["violations"] = list(C.verify(spec, text))
    if reference is not None:
        out["math_correct"] = bool(exact_score.math_correct(text, reference, tol,
                                                            context=[user_text])["correct"])
    score = 0.0 if out["degenerate"] else 1.0
    if out["n_constraints"]:
        score *= max(0.0, 1.0 - len(out["violations"]) / out["n_constraints"])
    if out["math_correct"] is False:
        score = 0.0
    out["score"] = score
    return out


# ================================================================== background processes
@dataclass(frozen=True)
class TaskSpec:
    """A supervised background process.

    argv: the command; after: tasks that must have finished successfully first; restarts:
    restarts after failures before giving up; backoff_s / backoff_max_s: first and largest
    delay before a restart (doubling); nice: priority increment; cores: CPU affinity (empty:
    inherited); threads: ``OMP_NUM_THREADS`` and friends; env: extra environment; timeout_s:
    a run longer than this is killed and counts as a failure (0 = none)."""

    name: str
    argv: tuple[str, ...]
    after: tuple[str, ...] = ()
    restarts: int = 5
    backoff_s: float = 5.0
    backoff_max_s: float = 300.0
    nice: int = 10
    cores: tuple[int, ...] = ()
    threads: int = 1
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float = 0.0


def _set_pdeathsig(sig: int = signal.SIGTERM) -> None:
    """Ask Linux to send ``sig`` to this process when its parent thread exits."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(sig), 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except (OSError, AttributeError):
        pass


def child_setup(nice: int = 0, cores: Sequence[int] = (), parent_pid: int | None = None) -> None:
    """First thing a supervised child does: die with the supervisor, lower its priority, pin
    itself to its cores."""
    _set_pdeathsig()
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(3)                         # the supervisor died before prctl took effect
    if nice:
        try:
            os.nice(int(nice))
        except OSError:
            pass
    if cores:
        try:
            os.sched_setaffinity(0, set(int(c) for c in cores))
        except (AttributeError, OSError, ValueError):
            pass


def _thread_env(threads: int) -> dict[str, str]:
    t = str(max(1, int(threads)))
    return {"OMP_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t, "MKL_NUM_THREADS": t,
            "NUMEXPR_NUM_THREADS": t, "RAYON_NUM_THREADS": t, "TOKENIZERS_PARALLELISM": "false"}


class ProcessSupervisor:
    """Starts, watches and restarts background processes (module docstring).

    ``poll()`` does one round (start what may start, reap what ended, schedule restarts);
    ``start()`` runs it in a monitor thread every ``period_s``. State goes to ``state_path``
    after every change. A task found ``finished`` there is started again unless
    ``keep_finished``: the tasks of the run are idempotent and cheap when their work is done,
    and a resumed run may be on a machine whose disk lost that work."""

    def __init__(self, log_dir: str | Path, state_path: str | Path | None = None, *,
                 clock: Callable[[], float] = time.monotonic, period_s: float = 2.0,
                 log: Callable[[str], None] | None = None, keep_finished: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = None if state_path is None else Path(state_path)
        self.clock = clock
        self.period_s = float(period_s)
        self.say = log or (lambda msg: None)
        self.keep_finished = bool(keep_finished)
        self.specs: dict[str, TaskSpec] = {}
        self.state: dict[str, dict[str, Any]] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self._files: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        old = read_json(self.state_path) if self.state_path is not None else None
        self._previous = (old or {}).get("tasks", {})

    # -------------------------------------------------------------------- set-up
    def add(self, spec: TaskSpec) -> None:
        with self._lock:
            self.specs[spec.name] = spec
            prev = self._previous.get(spec.name, {})
            done = self.keep_finished and prev.get("state") == "finished"
            self.state[spec.name] = {"state": "finished" if done else "waiting", "pid": None,
                                     "runs": int(prev.get("runs", 0)), "failures": 0,
                                     "exit_codes": list(prev.get("exit_codes", []))[-5:],
                                     "next_start": 0.0, "started": None, "ended": None,
                                     "error": "", "log": str(self.log_dir / f"{spec.name}.log"),
                                     "from_previous_run": done}
            self._save()

    # -------------------------------------------------------------------- queries
    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self.state.items()}

    def finished(self, name: str) -> bool:
        with self._lock:
            return self.state.get(name, {}).get("state") == "finished"

    def failed(self, name: str) -> bool:
        with self._lock:
            return self.state.get(name, {}).get("state") == "failed"

    def ended(self, name: str) -> bool:
        return self.finished(name) or self.failed(name)

    # -------------------------------------------------------------------- one round
    def _save(self) -> None:
        if self.state_path is not None:
            atomic_write_json(self.state_path, {"tasks": _jsonable(self.state),
                                                "time": time.time()})

    def _spawn(self, name: str) -> None:
        spec = self.specs[name]
        st = self.state[name]
        f = open(self.log_dir / f"{name}.log", "ab", buffering=0)
        f.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} run {st['runs'] + 1}\n".encode())
        env = dict(os.environ)
        env.update(_thread_env(spec.threads))
        env.update({k: str(v) for k, v in spec.env.items()})
        env["BES_SUPERVISOR_PID"] = str(os.getpid())
        try:
            proc = subprocess.Popen(list(spec.argv), stdout=f, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(REPO), start_new_session=True)
        except OSError as e:
            f.close()
            st.update(state="backoff", error=f"cannot start: {e}", failures=st["failures"] + 1,
                      next_start=self.clock() + spec.backoff_s)
            return
        self.procs[name] = proc
        self._files[name] = f
        st.update(state="running", pid=proc.pid, runs=st["runs"] + 1, started=time.time(),
                  ended=None, start_mono=self.clock())
        self.say(f"background: started {name} (pid {proc.pid})")

    def _deps_ok(self, spec: TaskSpec) -> tuple[bool, str]:
        for dep in spec.after:
            s = self.state.get(dep, {}).get("state")
            if s == "failed":
                return False, f"dependency {dep} failed"
            if s != "finished":
                return False, ""
        return True, ""

    def poll(self) -> None:
        with self._lock:
            now = self.clock()
            changed = False
            for name, spec in self.specs.items():
                st = self.state[name]
                proc = self.procs.get(name)
                if st["state"] == "running" and proc is not None:
                    code = proc.poll()
                    if code is None and spec.timeout_s and now - st.get("start_mono", now) > spec.timeout_s:
                        proc.kill()
                        code = proc.wait()
                        st["error"] = f"killed after {spec.timeout_s:.0f} s"
                    if code is None:
                        continue
                    self.procs.pop(name, None)
                    fh = self._files.pop(name, None)
                    if fh is not None:
                        fh.close()
                    st["exit_codes"] = (st["exit_codes"] + [code])[-5:]
                    st["ended"] = time.time()
                    st["pid"] = None
                    changed = True
                    if code == 0:
                        st.update(state="finished", error="")
                        self.say(f"background: {name} finished")
                        continue
                    st["failures"] += 1
                    st["error"] = st["error"] or f"exit code {code} (see {st['log']})"
                    if st["failures"] > spec.restarts:
                        st["state"] = "failed"
                        self.say(f"background: {name} FAILED after {st['failures']} failures: "
                                 f"{st['error']}")
                        continue
                    delay = min(spec.backoff_max_s, spec.backoff_s * 2 ** (st["failures"] - 1))
                    st.update(state="backoff", next_start=now + delay)
                    self.say(f"background: {name} exited with {code}; restart in {delay:.1f} s")
                if st["state"] in ("waiting", "backoff"):
                    if st["state"] == "backoff" and now < st["next_start"]:
                        continue
                    ok, why = self._deps_ok(spec)
                    if why:
                        st.update(state="failed", error=why)
                        changed = True
                        continue
                    if ok:
                        if st["state"] == "backoff":
                            st["error"] = ""
                        self._spawn(name)
                        changed = True
            if changed:
                self._save()

    # -------------------------------------------------------------------- thread
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as e:  # noqa: BLE001 - supervision must not die
                self.say(f"background: monitor error {e!r}")
            self._stop.wait(self.period_s)

    def start(self) -> "ProcessSupervisor":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="bes-background", daemon=True)
            self._thread.start()
        return self

    def wait(self, names: Iterable[str] | None = None, timeout: float | None = None) -> bool:
        """Block until the tasks ``names`` (default: all) have ended; ``False`` on timeout."""
        names = list(self.specs) if names is None else list(names)
        t_end = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._thread is None:
                self.poll()
            if all(self.ended(n) for n in names):
                return True
            if t_end is not None and time.monotonic() >= t_end:
                return False
            time.sleep(min(0.2, self.period_s))

    def stop(self, timeout: float = 20.0) -> None:
        """Stop the monitor thread and terminate every child (SIGTERM, then SIGKILL)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            for proc in self.procs.values():
                try:
                    proc.terminate()
                except OSError:
                    pass
            t_end = time.monotonic() + timeout
            for name, proc in list(self.procs.items()):
                try:
                    proc.wait(timeout=max(0.1, t_end - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                st = self.state[name]
                if st["state"] == "running":
                    st.update(state="stopped", pid=None, ended=time.time())
                fh = self._files.pop(name, None)
                if fh is not None:
                    fh.close()
            self.procs.clear()
            self._save()


# ============================================================================ stage loop
@dataclass
class StageContext:
    """What a stage function gets: its slice and the supervisor's services."""

    name: str
    sup: "Supervisor"
    started: float
    slice_s: float
    hard_s: float

    @property
    def deadline(self) -> float:
        """End of this stage's slice (wall time)."""
        return self.started + self.slice_s

    @property
    def hard_deadline(self) -> float:
        return self.started + max(self.slice_s, self.hard_s)

    @property
    def run_deadline(self) -> float:
        return self.sup.deadline

    @property
    def train_end(self) -> float:
        return self.sup.train_end

    def now(self) -> float:
        return self.sup.ledger.now()

    def left(self) -> float:
        return self.deadline - self.now()

    def stop_requested(self) -> bool:
        return self.sup.stop_requested

    def should_stop(self) -> bool:
        """A stop was requested or the slice is used up."""
        return self.sup.stop_requested or self.now() >= self.deadline

    def check(self) -> None:
        """Raise :class:`Interrupted` when a stop was requested."""
        if self.sup.stop_requested:
            raise Interrupted(f"stop requested during {self.name}")

    def log(self, msg: str, **fields: Any) -> None:
        self.sup.log(f"[{self.name}] {msg}")
        if fields:
            self.sup.ledger.event(self.name, **fields)


class Supervisor:
    """Runs the stages of one run directory with resume, slices, signals and the final export.

    out_dir: the run directory; specs: :func:`bes.train.stages.stage_specs`; stages:
    ``{name: fn(ctx) -> result dict | None}`` for every name of ``order``; t0 / deadline: the
    budget of a new run (an existing ``stages.json`` keeps its own); post_reserve_s: training ends
    ``post_reserve_s`` before the deadline; essential: stages whose failure stops the run (the
    others are recorded as failed and the run goes on); clock: wall clock; on_boundary:
    ``fn(name)`` called before every stage starts (tests use it to simulate a kill)."""

    def __init__(self, out_dir: str | Path, specs: Mapping[str, ST.StageSpec],
                 stages: Mapping[str, Callable[[StageContext], Mapping[str, Any] | None]], *,
                 t0: float | None = None, deadline: float | None = None,
                 post_reserve_s: float | None = None,
                 essential: Iterable[str] = ("preflight", "background", "benchmark", "train"),
                 order: Sequence[str] = ST.ORDER, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] | None = None,
                 on_boundary: Callable[[str], None] | None = None) -> None:
        self.paths = RunPaths(Path(out_dir))
        self.paths.out.mkdir(parents=True, exist_ok=True)
        self.specs = dict(specs)
        self.fns = dict(stages)
        missing = [n for n in order if n not in self.fns]
        if missing:
            raise ValueError(f"no stage function for {missing}")
        self.order = tuple(order)
        self.ledger = ST.StageLedger(self.paths.stages, t0=t0, deadline=deadline, order=order,
                                     clock=clock)
        self.deadline = self.ledger.deadline
        if post_reserve_s is None:
            post_reserve_s = sum(self.specs[n].planned_s for n in ST.POST + ("buffer",)
                                 if n in self.specs)
        self.post_reserve_s = float(post_reserve_s)
        self.train_end = self.deadline - self.post_reserve_s
        self.essential = frozenset(essential)
        self._log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self.on_boundary = on_boundary
        self.stop_requested = False
        self._stop_callbacks: list[Callable[[], None]] = []
        self.current: str | None = None

    # ------------------------------------------------------------------ services
    def log(self, msg: str) -> None:
        t = self.ledger.now() - self.ledger.t0
        self._log(f"[{t / 60:7.2f} min] {msg}")

    def on_stop(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Register ``fn`` to be called when a stop is requested; returns a remover."""
        self._stop_callbacks.append(fn)

        def remove() -> None:
            if fn in self._stop_callbacks:
                self._stop_callbacks.remove(fn)
        return remove

    def request_stop(self, reason: str = "stop requested") -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.log(f"stop requested ({reason}); the current stage stops at its next check")
            self.ledger.event("stop_requested", reason=reason, stage=self.current)
        for fn in list(self._stop_callbacks):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.log(f"stop callback failed: {e!r}")

    def install_signal_handlers(self) -> None:
        """SIGTERM / SIGINT: the first one requests a stop, a second one exits at once."""
        def handler(signum, frame):  # noqa: ARG001
            if self.stop_requested:
                raise KeyboardInterrupt
            self.request_stop(signal.Signals(signum).name)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handler)

    # --------------------------------------------------------------------- loop
    def pending(self) -> list[str]:
        led = self.ledger
        return [n for n in self.order if led.stages[n]["status"] not in ST.DONE
                and not (led.stages[n]["status"] == "failed" and n not in self.essential)]

    def _heartbeat_loop(self, stop: threading.Event, period_s: float) -> None:
        while not stop.wait(period_s):
            try:
                self.ledger.heartbeat()
            except Exception:  # noqa: BLE001 - monitoring only
                pass

    def run(self, heartbeat_s: float = 10.0) -> dict[str, Any]:
        """Run every pending stage; returns ``{"status", "stages", "table"}`` where status is
        ``"complete"`` or ``"interrupted"``. A thread persists the ledger's ``last_seen`` every
        ``heartbeat_s`` seconds (0 = no heartbeat), so a kill costs at most that much
        attribution accuracy."""
        stop = threading.Event()
        beat = None
        if heartbeat_s > 0:
            beat = threading.Thread(target=self._heartbeat_loop, args=(stop, heartbeat_s),
                                    name="bes-heartbeat", daemon=True)
            beat.start()
        try:
            return self._run()
        finally:
            stop.set()
            if beat is not None:
                beat.join(timeout=5.0)

    def _run(self) -> dict[str, Any]:
        self.ledger.event("supervisor_start", pending=self.pending())
        while True:
            pend = self.pending()
            if not pend:
                break
            name = pend[0]
            if self.on_boundary is not None:
                self.on_boundary(name)
            if self.stop_requested:
                return self._interrupted(name, "stop requested between stages", started=False)
            now = self.ledger.now()
            alloc = ST.allocate(self.specs, pend, now, self.deadline, self.train_end)
            if name in alloc.skipped and name not in ST.NEVER_SHED:
                self.ledger.skip(name, now, alloc.skipped[name])
                self.log(f"skip {name}: {alloc.skipped[name]}")
                continue
            slice_s = alloc.slices.get(name, self.specs[name].planned_s)
            hard_s = alloc.hard.get(name, slice_s)
            self.ledger.start(name, now, planned_s=self.specs[name].planned_s, slice_s=slice_s,
                              hard_s=hard_s)
            self.current = name
            self.log(f"start {name}: slice {slice_s:.0f} s (planned {self.specs[name].planned_s:.0f} s), "
                     f"{self.deadline - now:.0f} s left in the run")
            ctx = StageContext(name, self, now, slice_s, hard_s)
            try:
                result = self.fns[name](ctx) or {}
            except StageSkipped as e:
                self.ledger.skip(name, self.ledger.now(), str(e))
                self.log(f"{name} skipped itself: {e}")
                continue
            except Interrupted as e:
                return self._interrupted(name, str(e), started=True)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                self.ledger.fail(name, self.ledger.now(), f"{type(e).__name__}: {e}")
                self.log(f"{name} FAILED: {type(e).__name__}: {e}\n{tb}")
                if name in self.essential:
                    self._emergency_export(f"{name} failed: {type(e).__name__}: {e}")
                    raise FatalStageError(f"stage {name} failed: {type(e).__name__}: {e}") from e
                continue
            finally:
                self.current = None
            self.ledger.finish(name, self.ledger.now(), result)
            self.log(f"finished {name} in {self.ledger.stages[name]['used_s']:.0f} s")
        self.ledger.event("supervisor_end")
        return {"status": "complete", "stages": self.ledger.stages,
                "table": self.ledger.format_table(self.specs)}

    def _interrupted(self, name: str, reason: str, started: bool) -> dict[str, Any]:
        now = self.ledger.now()
        if started:
            self.ledger.interrupt(name, now, reason)
        self.log(f"interrupted in {name}: {reason}")
        export = self.specs.get("export")
        if export is not None and name != "export" and \
                self.deadline - now < 2.0 * export.planned_s:
            self._emergency_export(f"interrupted near the deadline in {name}")
        return {"status": "interrupted", "stage": name, "reason": reason,
                "stages": self.ledger.stages, "table": self.ledger.format_table(self.specs)}

    def _emergency_export(self, why: str) -> None:
        """Run the export stage out of turn (its ledger record is not changed)."""
        fn = self.fns.get("export")
        if fn is None:
            return
        self.log(f"emergency export: {why}")
        ctx = StageContext("export", self, self.ledger.now(), self.specs["export"].planned_s,
                           self.specs["export"].planned_s)
        try:
            res = fn(ctx)
            self.ledger.event("emergency_export", why=why, result=res or {})
        except Exception as e:  # noqa: BLE001
            self.ledger.event("emergency_export_failed", why=why, error=f"{type(e).__name__}: {e}")
            self.log(f"emergency export failed: {e!r}")


# ======================================================================= background tasks
def download_plan(cfg: Cfg) -> list[tuple[str, int | None, int]]:
    """``(source, max_files, phase)`` in download order.

    ``RunCfg.download_sources`` (``"name[:max_files] ..."``) gives an explicit list (phase 1).
    Otherwise phase 1 is the first file of the main web source (ClimbMix; FineWeb-Edu with
    ``permissive_only``) and of every chat, rag and math source, so training can start early;
    phase 2 is everything up to the registry caps (``sources.download_cap``): the main web
    source, the chat-like sources, the other base sources, then the preference data."""
    from bes.data import sources

    if cfg.run.download_sources.strip():
        out = []
        for item in cfg.run.download_sources.replace(",", " ").split():
            name, _, n = item.partition(":")
            sources.get(name)
            out.append((name, int(n) if n else None, 1))
        return out
    permissive = cfg.data.permissive_only
    main = "fineweb_edu" if permissive else "climbmix"
    chat_like = []
    for kind in ("chat", "rag", "math"):
        for name in sorted(sources.mixture_weights(kind, permissive),
                           key=lambda n: -sources.mixture_weights(kind, permissive)[n]):
            if name not in chat_like:
                chat_like.append(name)
    bases = [n for n in sorted(sources.mixture_weights("base", permissive)) if n != main]
    prefs = sorted(sources.mixture_weights("prefs", permissive))
    plan: list[tuple[str, int | None, int]] = [(main, 1, 1)] + [(n, 1, 1) for n in chat_like]
    plan += [(main, None, 2)] + [(n, None, 2) for n in chat_like]
    plan += [(n, None, 2) for n in bases + prefs]
    return plan


def task_downloads(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None]) -> int:
    failures = []
    done: dict[str, int] = {}
    for name, max_files, phase in download_plan(cfg):
        try:
            done[name] = _download_source(paths, cfg, name, max_files, log)
            log(f"downloads: {name} phase {phase}: {done[name]} files")
        except Exception as e:  # noqa: BLE001 - the other sources go on
            failures.append(f"{name}: {type(e).__name__}: {e}")
            log(f"downloads: {name} FAILED: {e!r}")
        if phase == 1:
            write_marker(paths, "downloads_phase1", sources=done)
    if failures:
        log(f"downloads: {len(failures)} failures: {failures[:5]}")
        return 1
    write_marker(paths, "downloads", sources=done)
    return 0


def _tok_sources(cfg: Cfg) -> list[str]:
    seen = []
    for name, _, _ in download_plan(cfg):
        if name not in seen:
            seen.append(name)
    from bes.data import sources
    return [n for n in seen if sources.get(n).kind != "library"]


def task_tokenize(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None],
                  poll_s: float = 10.0) -> int:
    """Tokenise every downloaded file (round after round) until the downloads have ended and a
    round finds nothing new."""
    from bes.data import download, tokenize_worker

    root = tok_root(cfg, paths.out)
    max_records = cfg.run.tokenize_max_records or None
    workers = max(1, len(cp.tokenize))
    failed_rounds = 0
    while True:
        downloads_over = read_marker(paths, "downloads") is not None or _task_ended(paths, "downloads")
        new = 0
        errors = []
        for name in _tok_sources(cfg):
            if not (download.repo_dir(name, cfg.run.data_dir) / download.MANIFEST).exists():
                continue                  # nothing downloaded yet (and no listing calls from here)
            try:
                files = download.completed_files(name, cfg.run.data_dir)
            except Exception as e:  # noqa: BLE001 - no listing yet
                if downloads_over:
                    errors.append(f"{name}: {e!r}")
                continue
            if not files:
                continue
            try:
                stats = tokenize_worker.run(name, cfg.run.data_dir, files=files,
                                            out_dir=tokenize_worker.tok_dir(name, root),
                                            workers=workers if len(files) > 1 else 0,
                                            max_records=max_records, strict=False, log=log)
                new += stats.committed
                if stats.failed:
                    errors.append(f"{name}: {len(stats.failed)} files failed")
                prune_raw(paths, cfg, name, log)    # raw files whose whole shard is committed
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
                log(f"tokenize: {name} FAILED: {e!r}")
        write_marker(paths, "tokenize_progress", new=new, errors=errors[:10])
        if downloads_over and new == 0:
            if errors:
                failed_rounds += 1
                if failed_rounds >= 3:
                    log(f"tokenize: giving up on {errors[:5]}")
                    return 1
            else:
                write_marker(paths, "tokenize")
                return 0
        time.sleep(poll_s if new == 0 else 0.5)


def _task_ended(paths: RunPaths, name: str) -> bool:
    st = read_json(paths.background) or {}
    return st.get("tasks", {}).get(name, {}).get("state") in ("finished", "failed")


def _library_usable(lib_dir: Path, want: Mapping[str, Any], allow_unrecorded: bool) -> dict[str, Any]:
    """:func:`library_self_check` plus :func:`library_settings_mismatch`: ``ok`` only when the
    library works and was built with the run's settings."""
    rep = library_self_check(lib_dir)
    if rep["ok"]:
        why = library_settings_mismatch(lib_dir, want, allow_unrecorded=allow_unrecorded)
        if why:
            rep = {**rep, "ok": False, "reasons": why}
    return rep


def task_library(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None]) -> int:
    """Use the configured library when it passes its self-check and was built with the run's
    settings (:func:`library_settings`: permissive flag, question-answer sources, tokenizer,
    ``LibraryCfg``); otherwise move it aside and build it with ``scripts/build_library.py``,
    recording the settings in :data:`LIBRARY_SETTINGS` (when the configuration uses a library).
    Writes the ready marker."""
    import runpy

    lib_dir = Path(LibraryCfg().library_dir)
    if not cfg.data.use_library:
        write_marker(paths, "library", path=None, reason="the configuration has no library")
        return 0
    want = library_settings(cfg)
    rep = _library_usable(lib_dir, want, bool(cfg.run.library_reuse_unrecorded))
    if not rep["ok"]:
        log(f"library: {lib_dir} not usable ({rep['reasons']}); building it")
        if lib_dir.exists():
            aside = lib_dir.with_name(f"{lib_dir.name}.aside-{time.strftime('%Y%m%d-%H%M%S')}-"
                                      f"{os.urandom(3).hex()}")
            os.replace(lib_dir, aside)
            log(f"library: moved the old directory to {aside}")
        tmp = lib_dir.with_name(f"{lib_dir.name}.building")
        if tmp.exists():
            shutil.rmtree(tmp)
        argv = ["build_library.py", "--cache-dir", str(Path(cfg.run.data_dir) / "library"),
                "--out", str(tmp)]
        if cfg.data.permissive_only:
            argv.append("--permissive-only")
        old = sys.argv
        sys.argv = argv
        try:
            runpy.run_path(str(REPO / "scripts" / "build_library.py"), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        finally:
            sys.argv = old
        atomic_write_json(tmp / LIBRARY_SETTINGS, _jsonable({**want, "argv": argv, "built": time.time(),
                                                             "code": code_commit()}))
        os.replace(tmp, lib_dir)
        rep = _library_usable(lib_dir, want, False)
    if not rep["ok"]:
        log(f"library: not usable after the build: {rep['reasons']}")
        return 1
    write_marker(paths, "library", path=str(lib_dir), self_check=rep,
                 settings=recorded_library_settings(lib_dir))
    log(f"library: ready ({rep.get('n_wiki')} wiki passages, {rep.get('n_qa')} qa entries, "
        f"{rep.get('total_numbers')} numbers)")
    return 0


def _arms(cfg: Cfg) -> list[tuple[str, str]]:
    out = []
    for item in cfg.run.eval_opponents.replace(",", " ").split():
        name, _, preset = item.partition("=")
        out.append((name, preset))
    return out


def task_gguf(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None]) -> int:
    """Download and verify the opponents' GGUF files and the judge model."""
    from bes.eval.proxy import opponents

    names = [n for n, _ in _arms(cfg) if n in opponents.MODELS]
    judge = cfg.run.eval_judge
    if judge in opponents.MODELS:
        names.append(judge)
    got = {}
    for name in dict.fromkeys(names):
        got[name] = str(opponents.fetch(name, None, download=True, log=log))
        log(f"gguf: {name} -> {got[name]}")
    write_marker(paths, "gguf", files=got)
    return 0


def task_opponents(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None]) -> int:
    """Precompute the opponents' replies to the proxy prompts (test split first, then dev);
    resumable through the harness cache, bounded by the time until the end of training."""
    from bes.eval.proxy import __main__ as cli
    from bes.eval.proxy import prompts as prompt_set
    from bes.eval.proxy import runner

    man = read_manifest(paths.out) or {}
    budget = man.get("budget", {})
    deadline = float(budget.get("deadline", time.time() + 3600.0))
    specs = ST.stage_specs(cfg.run, float(budget.get("budget_s", 3600.0)))
    # done before the judged comparison starts (in the plan: after sample .. select)
    end = deadline - sum(specs[n].planned_s for n in ("judge", "export", "buffer"))
    root = proxy_root(cfg, paths.out)
    ps = prompt_set.load()
    runs = {}
    for split in ("test", "dev"):
        todo = prompt_set.select(ps, split, None, cfg.run.eval_limit or None)
        for name, preset in _arms(cfg):
            minutes = (end - time.time()) / 60.0
            if minutes <= 0.1:
                log("opponents: out of time")
                write_marker(paths, "opponents_progress", runs=runs)
                return 0
            spec = cli.build_spec(name, preset or None, [], None, 0, None, None, cp.opponent_threads,
                                  None, download=False)
            res = runner.run(todo, spec, root, workers=cp.opponent_workers if len(todo) > 1 else 0,
                             n_threads=cp.opponent_threads, minutes=minutes, all_prompts=ps, log=log)
            runs[f"{name}={preset}/{split}"] = {"run_id": res.run_id, "done": res.done,
                                               "of": len(todo), "failed": len(res.failed)}
            write_marker(paths, "opponents_progress", runs=runs)
    write_marker(paths, "opponents", runs=runs)
    return 0


SIDECAR_ROWS = 4          # probe segments per source
SIDECAR_LEN = 256         # tokens per probe row (at most the run's base length)
SIDECAR_NEW = 64          # new tokens of each sample generation


def task_sidecar(paths: RunPaths, cfg: Cfg, cp: CorePlan, log: Callable[[str], None],
                 ckpt: str) -> int:
    """Health report of one checkpoint on the CPU (``health/health-<step>.json`` and
    ``health/latest.json``): probe loss by source on ``SIDECAR_ROWS`` segments of each source:
    for chat, rag and math sources conversations of the loader's held-out split
    (``bes.data.loader.heldout_samples``, never trained on), for the other sources the first
    segments of the newest shard (in the training universe: read those numbers as a trend;
    ``probe_heldout`` says which is which), and two fixed sample generations on dev proxy
    prompts (one math, one with explicit constraints) with exact scores. It runs at the lowest
    priority on its own core; the trainer skips it when late and kills it when it outlives the
    interval."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np

    jax.config.update("jax_platforms", "cpu")
    try:                                 # a cache is a speed-up only: never fail the report on it
        paths.jax_cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(paths.jax_cache))
    except Exception as e:  # noqa: BLE001
        log(f"sidecar: no persistent compilation cache ({type(e).__name__}: {e})")
    from bes.config import END, PAD, apply_overrides
    from bes.data import shards, sources
    from bes.data.loader import HELDOUT_KINDS, heldout_samples
    from bes.decode import respond, smc
    from bes.decode import sampling as SMP
    from bes.decode.lm import OurLM
    from bes.eval import exact_score
    from bes.eval.proxy import prompts as prompt_set
    from bes.exact.features import exact_features
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    t0 = time.time()
    man = read_manifest(paths.out) or {}
    ecfg = apply_overrides(effective_config(man), ["model.attn_impl=xla", "model.remat=false",
                                                   "model.act_dtype=float32"])
    params, consts, info = respond.load_checkpoint(ckpt, ecfg.model)
    tok = load_tokenizer()
    T = min(ecfg.data.seq_len, SIDECAR_LEN)
    loss = jax.jit(lambda p, c, b: M.loss_fn(p, c, b, ecfg.model)[1])
    report: dict[str, Any] = {"checkpoint": info, "probe_loss": {}, "probe_heldout": {},
                              "samples": []}
    troot = tok_root(cfg, paths.out) / "tok"
    for d in sorted(troot.iterdir()) if troot.exists() else []:
        man_s = shards.read_manifest(d)
        if not man_s:
            continue
        entry = man_s[max(man_s)]
        if entry.get("pairs"):
            continue
        segs = []
        try:
            if sources.get(d.name).kind in HELDOUT_KINDS:
                segs = [np.asarray(s.ids, np.int32)
                        for s in heldout_samples(troot.parent, d.name, SIDECAR_ROWS, seed=0,
                                                 data=cfg.data)]
        except (KeyError, ValueError):     # not a registry source: its first segments
            segs = []
        report["probe_heldout"][d.name] = bool(segs)
        if not segs:
            sh = shards.open_shard(d, entry)
            segs = [np.asarray(sh.segment(i), np.int32) for i in range(min(SIDECAR_ROWS, len(sh)))]
        rows = []
        for seg in segs:
            seg = seg[:T + 1]
            row = np.full(T + 1, PAD, np.int32)
            row[:len(seg)] = seg
            rows.append((row, len(seg)))
        if not rows:
            continue
        tokens = np.stack([r for r, _ in rows])
        w = np.zeros((len(rows), T), np.float32)
        seg_ids = np.ones((len(rows), T), np.int32)
        for j, (_, n) in enumerate(rows):
            w[j, :max(0, n - 1)] = 1.0
            seg_ids[j, :n] = 0
        feats = exact_features(tokens[:, :T], seg_ids, tok)
        batch = {"tokens": tokens, "w": w, "seg": seg_ids, **feats}
        m = jax.device_get(loss(params, consts, batch))
        report["probe_loss"][d.name] = round(float(m["ce_sum"]) / max(1e-9, float(m["weight"])), 4)
    ps = [p for p in prompt_set.select(prompt_set.load(), "dev") if not p.multi_turn]
    picks = [next((p for p in ps if p.category == c), None) for c in ("math", "instruct")]
    chat = respond.OurChat(tok)
    # the run's decode settings, explicitly: temperature and truncation through the settings
    # (smc.decode hands the temperature to the model), the self-evidence weight to the model
    settings = SMP.SamplingSettings.from_cfg(ecfg.decode)
    for p in [q for q in picks if q is not None]:
        prompt = chat.render([{"role": "user", "content": p.turns[0]}])
        lm = OurLM(params, consts, ecfg.model, tok, max_steps=4 * SIDECAR_NEW,
                   self_evidence_weight=ecfg.decode.self_evidence_weight)
        res = smc.decode(lm, prompt, n_particles=1, max_new_tokens=SIDECAR_NEW, seed=0, tok=tok,
                         settings=settings)
        part = res.particles[0]
        text = chat.display([t for t in part.tokens if t != END]).strip()
        sc = exact_score.score_conversation(p, [text], [part.finish])
        report["samples"].append({"prompt_id": p.id, "category": p.category, "text": text[:600],
                                  "finish": part.finish, "math_correct": sc.get("math_correct"),
                                  "constraints_ok": sc.get("constraints_ok"),
                                  "degenerate": sc["turns"][0]["degenerate"]["degenerate"]})
    report["seconds"] = round(time.time() - t0, 1)
    step = info.get("step")
    paths.health.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.health / f"health-{int(step or 0):09d}.json", _jsonable(report))
    atomic_write_json(paths.health / "latest.json", _jsonable(report))
    log(f"sidecar: step {step}: probe loss {report['probe_loss']} in {report['seconds']} s")
    return 0


TASKS = {"downloads": task_downloads, "tokenize": task_tokenize, "library": task_library,
         "gguf": task_gguf, "opponents": task_opponents}


def task_argv(name: str, out: str | Path, nice: int = 0, cores: Sequence[int] = (),
              extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Command line of a background task (``python -m bes.train.supervisor task ...``)."""
    argv = [sys.executable, "-m", "bes.train.supervisor", "task", name, "--run", str(out),
            "--nice", str(int(nice))]
    if cores:
        argv += ["--cores", ",".join(str(int(c)) for c in cores)]
    return tuple(argv + list(extra))


def _main_task(args: argparse.Namespace) -> int:
    parent = os.environ.get("BES_SUPERVISOR_PID")
    child_setup(args.nice, [int(c) for c in args.cores.split(",") if c.strip()] if args.cores else (),
                int(parent) if parent else None)
    paths = RunPaths(Path(args.run))
    man = read_manifest(paths.out)
    if man is None:
        print(f"task {args.name}: no manifest in {paths.out}", file=sys.stderr)
        return 2
    cfg = base_config(man)
    cp = CorePlan.from_dict(man["core_plan"]) if man.get("core_plan") else core_plan()
    given = [int(c) for c in args.cores.split(",") if c.strip()] if args.cores else []
    if args.name == "tokenize" and len(given) > len(cp.tokenize):
        # a run catching up on a recycled machine gives the task more cores than the plan's share
        cp = dataclasses.replace(cp, tokenize=tuple(given))

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.name == "sidecar":
        return task_sidecar(paths, cfg, cp, log, args.ckpt)
    if args.name not in TASKS:
        print(f"unknown task {args.name!r}; known: {sorted(TASKS) + ['sidecar']}", file=sys.stderr)
        return 2
    return TASKS[args.name](paths, cfg, cp, log)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Background tasks of the 24-hour run.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("task")
    p.add_argument("name")
    p.add_argument("--run", required=True)
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--cores", default="")
    p.add_argument("--ckpt", default="")
    sub.add_parser("plan", help="print the core plan of this host")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print(json.dumps(core_plan().to_dict(), indent=1))
        return 0
    return _main_task(args)


__all__ = ["RunPaths", "RunLock", "RunLocked", "PreflightError", "StageSkipped", "Interrupted",
           "FatalStageError", "LADDER", "ladder_rung", "parse_switches", "format_switches",
           "safe_switches", "make_consts_fn", "run_overrides", "horizon_of", "base_config",
           "effective_config",
           "generator_overrides", "CorePlan", "core_plan", "host_report", "device_report",
           "check_devices", "disk_need", "disk_need_gb", "prune_raw", "code_commit",
           "read_manifest", "write_manifest", "update_manifest", "write_marker", "read_marker",
           "library_self_check", "LibraryGate", "label_sample", "TaskSpec", "ProcessSupervisor", "child_setup", "StageContext",
           "Supervisor", "download_plan", "task_argv", "tok_root", "proxy_root", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
