#!/usr/bin/env python3
"""Our model's replies to the proxy prompts of one split (default ``test``: the 200 held-out
prompts of the blind comparison), generated in parallel CPU processes from the exported bundle
and stored in the layout of the proxy harness (``bes.eval.proxy``).

    python scripts/gen_test_replies.py runs/main16/final --out OUT --workers 42 --minutes 24
    python scripts/gen_test_replies.py runs/main16 --out OUT                 # its final/ export
    python scripts/gen_test_replies.py SNAPSHOT --out OUT --make-export      # parts.json + run.tar.*
    python scripts/gen_test_replies.py EXPORT --out OUT --ids chat-025,math-017 --workers 2
    python scripts/gen_test_replies.py OUT --merge-into data_cache/proxy     # next to the opponents

**Source.** An export directory (``export.json``, ``bes.export``) is loaded with the bundle
loader, :func:`bes.export.load_export` (``SHA256SUMS`` checked once, in this process), and every
worker builds :func:`bes.export.export_generator` on it: the exported 16-bit weights, the exported
configuration, tokenizer and library, the export's own responder. The decoding settings are the
run's choice recorded in the export (``config.json`` ``decode_choice``: harness overrides of the
``ours`` preset ``default``, applied as ``bes.eval.arena.ours_contestant`` applies them, with
``extra.use_library=false`` when the export holds no library), plus ``--set`` overrides. A run
directory means its ``final/`` export when that exports the run's newest weights; otherwise (or
with ``--make-export``) those weights are exported first with ``bes.export.export_run`` into
``OUT/export``. A snapshot (``parts.json`` + ``run.tar.*`` of ``scripts/colab_cli.py snapshot``)
is checked against its SHA-256 sums, unpacked into ``OUT/source`` and treated as a run directory.

**Replies.** Every conversation goes through ``bes.eval.proxy.runner.converse``, exactly as
``python -m bes.eval.proxy run --model ours --generator bes.export:export_generator
--generator-arg export=DIR`` runs it: for each user turn the model sees the earlier user turns
and its own earlier visible replies, the seed of turn ``t`` is ``derive_seed(seed, prompt id,
t, 0)``, the cache keys are the harness's. A reply here therefore equals the single-process
harness's reply for the same prompt and seed on the same machine.

**Elastic cache** (``--cache elastic``, the default). The export's decoder allocates a key-value
cache of ``max_seq + 4 * max_new_tokens`` slots (5,248 for the final configuration) and every
decoding step attends over all of them: about 0.2 ms per slot and step at 6 particles on one
core, i.e. most of a step's cost. Here the same decoder (``OurLM`` with the export's arguments,
:func:`elastic_class`) prefills the prompt exactly as the export does (on the full length: a
prefill's results depend on the cache length), then decodes on a shorter cache that grows with
the reply, with the export's step budget. Masked slots add exact zeros, but XLA:CPU processes the
slot axis in blocks (1,024 slots on the dev container): a step on ``L`` slots gives the full
cache's rows only while the filled slots stay within ``L``'s whole blocks, so the candidate
lengths are 256, 512, 768, 1,024, 2,048 and 3,072, and equal rows are checked, not assumed
(:func:`calibrate_elastic`): worker 0 checks every candidate on the workers' machine and thread
layout, on real model states of a ~3,000-token prompt, at filled spans every 256 slots up to the
length, bitwise against the export's own cache; replies then decode only on the lengths
that passed and the spans they were checked for, else on the full length. Meanwhile the other
workers decode speculatively (every candidate); each reply reports the (length, span) pairs it
stepped on and is kept only when the calibration covers all of them, else dropped and handed out
again (``--no-speculate``: they wait). If no length passes, every reply decodes with the export's
own cache (exact; 5-10 times slower; as many workers at once as the memory holds); ``--cache
elastic-always`` keeps the elastic cache all the same under a key of its own, ``--cache full``
skips the calibration. The result is kept in ``OUT/calibration.json``; a restarted call on the
same kind of machine (CPU, JAX, threads, export, settings, this script) reuses it
(``--recalibrate``: check again).

**Workers.** ``W`` processes, each pinned to its own cores (``os.sched_setaffinity`` before its
interpreter starts; the SMT siblings of a core stay in one worker when a worker has whole cores,
otherwise one thread per core first), with ``JAX_PLATFORMS=cpu`` (the machine's accelerator is
never touched) and the thread counts of OpenMP / BLAS / Rayon set in the environment before
anything is imported; XLA sizes its thread pool from the affinity. Each worker loads the bundle
once and compiles once (the shapes of the first replies), then takes one prompt at a
time from the parent (two-turn prompts and long-reply categories first). Every finished reply is
appended at once (fsync) to the harness cache ``cache/<key>.jsonl``, so a restarted call
generates only what is missing. A worker that dies is restarted (``--restarts``) and its prompt
handed out again (once); one above ``--max-rss-gb`` is killed. After ``--minutes`` no prompt is
handed out any more and ``--grace-minutes`` later the workers are stopped; SIGTERM / SIGINT stop
them at once. In every case the outputs are written from what is complete.

**Output** (``OUT``): ``proxy/`` (a harness root: ``cache/<key>.jsonl``, ``outputs/<key>/<label>-
<settings hash>-s<seed>.jsonl`` + ``.meta.json``; ``--root`` names another root), ``replies.jsonl``
(one line per complete prompt of the split: id, category, turns, replies, finish information,
seconds), ``testgen.json`` (summary and cost model: start-up per worker, seconds per reply,
memory), ``testgen.log`` (the progress lines), ``logs/worker-NN.log`` and ``testgen.tar.gz`` (all
of these, for one download). Progress lines say done/total, replies per minute and the ETA; the
last line printed is ``TESTGEN {json}`` (``status``: complete, partial or failed; ``failed``:
prompts whose generation raised, recorded with their error in ``testgen.json`` and skipped).
Exit status: 0 when at least one prompt is complete, 1 when none is, 2 when nothing could start.

**Merge** (``--merge-into ROOT``, no model needed): appends the cache lines of ``OUT`` that
``ROOT`` lacks and writes the run's outputs file there (``bes.eval.proxy.runner.assemble``), so
``python -m bes.eval.proxy --root ROOT pairs --x RUN --y OPPONENT_RUN ...`` pairs it with the
opponents' runs of that root.
"""
from __future__ import annotations

import os


def _cpu_only_environment(env) -> None:
    """JAX on the CPU only, for this process and every process it starts: ``JAX_PLATFORMS=cpu``
    (no accelerator backend is ever initialised), no TPU flag in ``XLA_FLAGS`` (a CPU-only XLA
    stops at start-up on an unknown flag), no ``LIBTPU_INIT_ARGS``."""
    env["JAX_PLATFORMS"] = "cpu"
    flags = [f for f in env.get("XLA_FLAGS", "").split() if "xla_tpu" not in f]
    if flags:
        env["XLA_FLAGS"] = " ".join(flags)
    else:
        env.pop("XLA_FLAGS", None)
    env.pop("LIBTPU_INIT_ARGS", None)


_cpu_only_environment(os.environ)        # before anything can import JAX: never the accelerator

import argparse
import collections
import ctypes
import dataclasses
import gc
import hashlib
import json
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Thread-count variables set for every worker before its interpreter starts.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
#: Hand-out order inside a number of turns: categories with long replies first (longest work
#: first shortens the tail of a parallel run; the order never changes a reply).
CATEGORY_ORDER = ("writing", "code", "chat", "instruct", "math")
EXPORT_FACTORY = "bes.export:export_generator"
SUMMARY_TAG = "TESTGEN"
SUMMARY_JSON = "testgen.json"
REPLIES_JSONL = "replies.jsonl"


class _Missing(Exception):
    """A generation that is not in the cache (replaying conversations from the cache only)."""


def _missing(_messages, _seed):
    raise _Missing()


def _jline(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


# ======================================================================== machine facts
def proc_status(pid: int | str = "self") -> dict[str, int]:
    """``{"VmRSS": kB, "VmHWM": kB}`` of a process (empty when it is gone)."""
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, value = line.split(":", 1)
                    out[name] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    return out


def mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except (OSError, ValueError):
        pass
    return None


def parse_cpu_list(text: str) -> set[int]:
    """``"0-3,8,10-11"`` -> ``{0, 1, 2, 3, 8, 10, 11}``."""
    out: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def sibling_groups(cpus: Sequence[int], sysfs: str = "/sys/devices/system/cpu") -> list[list[int]]:
    """The allowed CPUs grouped by physical core (``thread_siblings_list``), in CPU order."""
    allowed = set(int(c) for c in cpus)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for c in sorted(allowed):
        if c in seen:
            continue
        try:
            sib = parse_cpu_list(Path(f"{sysfs}/cpu{c}/topology/thread_siblings_list").read_text())
        except (OSError, ValueError):
            sib = {c}
        group = sorted((sib & allowed) | {c})
        seen.update(group)
        groups.append(group)
    return groups


def plan_cores(cpus: Sequence[int], workers: int, per_worker: int,
               groups: Sequence[Sequence[int]] | None = None) -> list[list[int]]:
    """Cores of every worker. When a worker gets whole cores (``per_worker`` a multiple of the
    threads per core) the siblings of a core stay together; otherwise the first thread of every
    core is handed out before any second one (a half-used machine uses every core)."""
    groups = [list(g) for g in (groups if groups is not None else sibling_groups(cpus))]
    smt = max((len(g) for g in groups), default=1)
    if per_worker % smt == 0:
        order = [c for g in groups for c in g]
    else:
        order = [g[i] for i in range(smt) for g in groups if i < len(g)]
    need = int(workers) * int(per_worker)
    if workers < 1 or per_worker < 1 or need > len(order):
        raise ValueError(f"{workers} workers x {per_worker} cores need {need} cores; "
                         f"{len(order)} are allowed ({sorted(cpus)})")
    return [order[i * per_worker:(i + 1) * per_worker] for i in range(workers)]


def _set_pdeathsig(sig: int = signal.SIGTERM) -> None:
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(sig), 0, 0, 0)   # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _child_setup(cores: Sequence[int], nice: int, parent: int = 0) -> None:
    """First thing a worker does, before it imports anything that starts threads (NumPy's BLAS,
    JAX): die with the parent, pin itself to its cores, lower its priority. (Not a
    ``preexec_fn``: the parent holds JAX's threads, and Python code between fork and exec of a
    threaded process can deadlock.)"""
    _set_pdeathsig()
    if parent and os.getppid() != parent:
        os._exit(3)                                 # the parent died before prctl took effect
    if cores:
        os.sched_setaffinity(0, set(int(c) for c in cores))
    if nice:
        try:
            os.nice(int(nice))
        except OSError:
            pass


def worker_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({v: str(int(threads)) for v in THREAD_VARS})
    _cpu_only_environment(env)
    env.update({"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": os.pathsep.join([str(REPO)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])})
    return env


# ============================================================================== sources
class _Parts:
    """A read-only file object over the concatenation of files (a split tar stream)."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)
        self.f = None
        self.i = -1

    def read(self, n: int = -1) -> bytes:
        out = []
        want = n if n is not None and n >= 0 else None
        while want is None or want > 0:
            if self.f is None:
                self.i += 1
                if self.i >= len(self.paths):
                    break
                self.f = open(self.paths[self.i], "rb")
            chunk = self.f.read(-1 if want is None else want)
            if not chunk:
                self.f.close()
                self.f = None
                continue
            out.append(chunk)
            if want is not None:
                want -= len(chunk)
        return b"".join(out)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()


def unpack_snapshot(snap: Path, dest: Path, log: Callable[[str], None]) -> Path:
    """The run directory of a snapshot (``parts.json`` + parts): every part checked against its
    SHA-256 and size, the stream unpacked into ``dest`` (once: an unpacked copy of the same
    snapshot is reused)."""
    meta = json.loads((snap / "parts.json").read_text())
    parts = meta["parts"]
    ident = meta.get("sha256") or hashlib.sha256(_jline(parts).encode()).hexdigest()
    stamp = dest / ".unpacked.json"
    if stamp.exists():
        try:
            if json.loads(stamp.read_text()).get("sha256") == ident:
                return _run_dir_in(dest)
        except (ValueError, FileNotFoundError):
            pass
    t0 = time.time()
    whole = hashlib.sha256()
    for p in parts:
        path = snap / p["name"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 22):
                h.update(block)
                whole.update(block)
        if h.hexdigest() != p["sha256"] or path.stat().st_size != int(p["bytes"]):
            raise SystemExit(f"snapshot part {path} does not match parts.json (SHA-256 or size)")
    if meta.get("sha256") and whole.hexdigest() != meta["sha256"]:
        raise SystemExit(f"snapshot {snap}: the parts together do not match parts.json")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    stream = _Parts([snap / p["name"] for p in parts])
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:                       # a Python without extraction filters
                tar.extractall(dest)
    finally:
        stream.close()
    stamp.write_text(_jline({"sha256": ident, "snapshot": str(snap), "checkpoints": meta.get("checkpoints"),
                             "steps": meta.get("steps"), "tokens": meta.get("tokens")}))
    log(f"snapshot {snap} (steps {meta.get('steps')}, {meta.get('checkpoints')}) checked and unpacked "
        f"into {dest} in {time.time() - t0:.0f} s")
    return _run_dir_in(dest)


def _run_dir_in(dest: Path) -> Path:
    for cand in [dest / "run"] + sorted(p for p in dest.iterdir() if p.is_dir()):
        if (cand / "manifest.json").is_file():
            return cand
    raise SystemExit(f"{dest} holds no run directory (no manifest.json)")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def export_of_run(run: Path, out: Path, args: argparse.Namespace,
                  log: Callable[[str], None]) -> tuple[Path, dict[str, Any]]:
    """The export to use for a run directory (module docstring) and how it was found."""
    from bes import export as EX
    from bes.decode.respond import checkpoint_fingerprint

    man = _read_json(run / "manifest.json") or {}
    newest, source = EX._latest_checkpoint(run, man)     # what scripts/run_24h.py exports
    fp = checkpoint_fingerprint(newest)
    final = run / "final"
    rec = _read_json(final / EX.EXPORT_JSON)
    if rec and (rec.get("source") or {}).get("fingerprint") == fp and not args.make_export:
        return final, {"how": "the run's final export", "checkpoint": str(newest)}
    if rec and not args.make_export:
        log(f"{final} exports step {(rec.get('source') or {}).get('step')}, not the run's newest weights "
            f"({newest}): exporting those (name {final} itself to use it as it is)")
    dst = out / "export"
    old = _read_json(dst / EX.EXPORT_JSON)
    if old and (old.get("source") or {}).get("fingerprint") == fp and not EX.verify_export(dst):
        return dst, {"how": "exported earlier by this script", "checkpoint": str(newest)}
    t0 = time.time()
    library = str(Path(args.library).resolve()) if args.library else None
    data_dir = str(Path(args.data_dir).resolve()) if args.data_dir else None
    cwd = os.getcwd()
    os.chdir(REPO)                       # a run manifest names its configuration relative to the repository
    try:
        res = EX.export_run(run, final_dir=dst, library=library, download_heldout=not args.no_download,
                            data_dir=data_dir, log=log)
    finally:
        os.chdir(cwd)
    took = round(time.time() - t0, 1)
    log(f"exported {newest} ({source} weights) into {dst} in {took} s (held-out check {res.get('check_ok')})")
    return dst, {"how": "exported by bes.export.export_run", "checkpoint": str(newest),
                 "export_seconds": took, "check_ok": res.get("check_ok")}


def resolve_source(src: Path, out: Path, args: argparse.Namespace,
                   log: Callable[[str], None]) -> tuple[Path, dict[str, Any]]:
    """``(export directory, provenance)`` of the source argument (module docstring)."""
    from bes import export as EX

    if (src / EX.EXPORT_JSON).is_file():
        return src, {"how": "export directory"}
    info: dict[str, Any] = {}
    if (src / "parts.json").is_file():
        t0 = time.time()
        run = unpack_snapshot(src, out / "source", log)
        info["snapshot"] = {"path": str(src), "unpacked": str(run), "seconds": round(time.time() - t0, 1)}
    elif (src / "manifest.json").is_file():
        run = src
    else:
        raise SystemExit(f"{src} is neither an export (export.json), a run directory (manifest.json) "
                         "nor a snapshot (parts.json)")
    path, how = export_of_run(run, out, args, log)
    return path, {**info, **how, "run": str(run)}


# ============================================================================ generators
_ELASTIC: dict[str, Any] = {}


#: Longest cache length an elastic cache takes; a reply that needs more slots decodes on the
#: plain cache itself (first prompts of the test split have at most 154 tokens, second ones up to
#: about 1,000, and a reply at 6 particles takes up to about 1,100 decoding steps: spans beyond
#: 1,987 slots, the 2,048-slot limit, happen in long second turns only). A longer cap costs the
#: calibration a longer probe prompt (every candidate is checked up to its length).
MAX_ELASTIC = 3072
#: The slot axis of a decoding step's attention is processed by XLA:CPU in blocks of this many
#: slots (measured on the dev container, 1 and 6 particles): a step on a cache of ``L`` slots gives
#: the plain cache's rows bitwise while the filled slots stay within the whole blocks of ``L``
#: (``L`` below one block: within ``L``), and other low bits once they reach a partial block (1,280
#: slots with 1,089 filled differs; 2,048 or 3,072 slots with 1,984 or 3,008 filled does not).
#: Hence the candidate lengths: whole ``--cache-step`` below one block, whole blocks above.
SLOT_BLOCK = 1024
#: The calibration checks every candidate length at filled spans every CHECK_EVERY slots (up to
#: 64 below the length), CHECK_STEPS decoding steps each, on real model states.
CHECK_EVERY = 256
CHECK_STEPS = 4


def candidate_lengths(full: int, step: int) -> list[int]:
    """Cache lengths an elastic cache may take below the plain length ``full``."""
    small = list(range(int(step), SLOT_BLOCK, int(step)))
    big = list(range(SLOT_BLOCK, min(MAX_ELASTIC, int(full) - 1) + 1, SLOT_BLOCK))
    return [t for t in small + big if t < int(full)]


def span_limit(length: int) -> int:
    """The longest filled span a length is used for: the last span its check covers."""
    return int(length) - 64 + CHECK_STEPS - 1


def check_spans(length: int) -> list[int]:
    """Filled slots at which the calibration checks ``length``: ``length - 64`` and every
    CHECK_EVERY slots below it (whole prefill chunks)."""
    return [c for c in range(int(length) - 64, 63, -CHECK_EVERY)][::-1]


def default_lengths(full: int, step: int) -> dict[int, int]:
    """``{length: longest span}`` of every candidate (the speculative workers' guess)."""
    return {t: span_limit(t) for t in candidate_lengths(full, step)}


def first_slots(cache: Mapping[str, Any], n: int) -> dict[str, Any]:
    """The first ``n`` slots of a key-value cache in the layout of ``bes.model.generate``
    (``k``/``v`` ``[loops, blocks, B, H, T, head_dim]``, ``seg``/``pos``/``bos`` ``[B, T]``,
    ``sel`` ``[loops, blocks, B, T]``; ``t``, ``next_pos`` and ``cur_seg`` are kept): a copy, no
    arithmetic."""
    out = dict(cache)
    for name in ("k", "v"):
        out[name] = cache[name][:, :, :, :, :n]
    for name in ("seg", "pos", "bos"):
        if name in cache:
            out[name] = cache[name][:, :n]
    if "sel" in cache:
        out["sel"] = cache["sel"][:, :, :, :n]
    return out


def _int_keys(d: Mapping[Any, Any] | None) -> dict[int, int] | None:
    return None if d is None else {int(k): int(v) for k, v in d.items()}


def elastic_class():
    """``ElasticLM``: :class:`bes.decode.lm.OurLM` whose decoding steps attend over a key-value
    cache that grows with the reply instead of the plain class's ``max_seq + 4 * max_new_tokens``
    slots (5,248 for the final configuration; a step costs about 0.2 ms per slot at 6 particles on
    one core of the dev container).

    - **Prefill on the plain length.** The prompt is prefilled exactly as the plain class does it,
      on a one-row cache of the plain length: a prefill chunk's results depend on the cache length
      on XLA:CPU (a 1,088-token prompt prefilled on 1,280 or 1,536 slots gives other low bits than
      on 5,248, at 1 and at 6 particles, on the dev container). Only the copy of that row into the
      ``K`` particle rows keeps the first ``start`` slots (every later slot is still empty), so the
      state after the prefill is the plain state's used part, bit for bit.
    - **Lengths.** ``lengths`` maps every length the cache may take to the longest filled span
      (slots up to and including the one a step writes) a decoding step on it gives the plain
      rows for (:func:`calibrate_elastic`; ``None``: :func:`default_lengths`, the guess). Before
      each step the cache takes the shortest such length that holds the step's span, never a
      shorter one than it has; past all of them, the plain length itself. New slots are filled
      as an unused slot of the plain cache is (keys and values 0, segment -1, position 0).
    - **Budget.** The step budget (``capacity``) is the plain class's, so the decoder stops where
      it would.
    - ``usage``: ``{length: longest span stepped on it}`` of this instance (the parent keeps a
      speculative reply only when the calibration covers it)."""
    if "cls" in _ELASTIC:
        return _ELASTIC["cls"]
    import jax.numpy as jnp

    from bes.decode.lm import OurLM

    class ElasticLM(OurLM):
        cache_step = 256
        lengths: dict[int, int] | None = None

        def length_table(self) -> dict[int, int]:
            if self.lengths is None:
                return default_lengths(self.full_slots, self.cache_step)
            return self.lengths

        def length_for(self, span: int, at_least: int = 0) -> int:
            """The shortest usable length of at least ``at_least`` slots for ``span`` (class docstring)."""
            for t, lim in sorted(self.length_table().items()):
                if t >= at_least and t < self.full_slots and int(span) <= lim:
                    return int(t)
            return self.full_slots

        def _tmax(self, n_prompt):
            n_pad, full = OurLM._tmax(self, n_prompt)     # the plain cache: bound and step budget
            self.full_slots, self.base_slot = int(full), int(n_pad + n_prompt)
            self.start_slots = self.length_for(self.base_slot + 1)
            return n_pad, full                            # the prompt is prefilled on the plain length

        def prefill(self, prompt_ids, n_particles):
            plain = self.dec

            def reorder(cache, idx):                      # the 1 -> K copy right after the prefill
                return plain.reorder(first_slots(cache, self.start_slots), idx)

            self.dec = plain._replace(reorder=reorder)
            try:
                out = OurLM.prefill(self, prompt_ids, n_particles)
            finally:
                self.dec = plain
            self.slots = int(self.cache["seg"].shape[1])
            self.grown = 0
            if not hasattr(self, "usage"):
                self.usage = {}
            return out

        def step(self, tokens, active=None):
            span = self.base_slot + self.steps + 1        # filled slots once this step wrote its own
            if self.slots < self.full_slots and self.steps < self.capacity and \
                    span > self.length_table().get(self.slots, 0):
                self.grow(span)
            self.usage[self.slots] = max(self.usage.get(self.slots, 0), span)
            return OurLM.step(self, tokens, active)

        def grow(self, span: int) -> None:
            self.grow_to(self.length_for(span, at_least=self.slots + 1))

        def grow_to(self, new: int) -> None:
            pad = int(new) - self.slots
            if pad <= 0:
                return
            c = self.cache
            out = dict(c)
            for name in ("k", "v"):                       # [loops, blocks, B, H, T, head_dim]
                out[name] = jnp.pad(c[name], [(0, 0)] * 4 + [(0, pad), (0, 0)])
            out["seg"] = jnp.pad(c["seg"], ((0, 0), (0, pad)), constant_values=-1)
            out["pos"] = jnp.pad(c["pos"], ((0, 0), (0, pad)))
            if "sel" in c:                                # [loops, blocks, B, T]
                out["sel"] = jnp.pad(c["sel"], [(0, 0)] * 3 + [(0, pad)])
            if "bos" in c:
                out["bos"] = jnp.pad(c["bos"], ((0, 0), (0, pad)))
            self.cache = out
            self.slots = int(new)
            self.grown += 1

    _ELASTIC["cls"] = ElasticLM
    return ElasticLM


def make_elastic(lm, cache_step: int, lengths: Mapping[Any, Any] | None = None):
    """The (not yet prefilled) ``OurLM`` ``lm`` turned into an ``ElasticLM`` (every constructor
    argument kept) on the lengths ``lengths`` (``None``: :func:`default_lengths`)."""
    if int(cache_step) < 1 or int(cache_step) % 64:
        raise ValueError("--cache-step must be a positive multiple of 64")
    lm.__class__ = elastic_class()
    lm.cache_step = int(cache_step)
    lm.lengths = _int_keys(lengths)
    lm.usage = {}
    return lm


CALIBRATION_JSON = "calibration.json"


def machine_fingerprint(threads: int, spec, cache_step: int) -> str:
    """What a calibration's result depends on: the CPU (vendor, model, stepping, cache, flags),
    JAX and jaxlib, ``XLA_FLAGS``, threads per worker, the generator key (weights, configuration,
    decoding code, library) and settings, this script. A restarted call on the same kind of
    machine reuses the calibration (``OUT/calibration.json``) instead of checking again."""
    from importlib import metadata

    cpu: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            name, _, value = line.partition(":")
            name = name.strip()
            if name in ("vendor_id", "cpu family", "model", "model name", "stepping", "cache size", "flags") \
                    and name not in cpu:
                cpu[name] = value.strip()
    except OSError:
        pass
    versions = {}
    for name in ("jax", "jaxlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    blob = {"cpu": cpu, "versions": versions, "xla_flags": os.environ.get("XLA_FLAGS"), "threads": int(threads),
            "key": spec.key, "settings": spec.settings.hash(), "cache_step": int(cache_step),
            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(_jline(blob).encode()).hexdigest()[:24]


def usage_ok(usage: Mapping[Any, Any] | None, lengths: Mapping[Any, Any] | None, full: int) -> bool:
    """Every (length, span) a reply stepped on is the plain length or within ``lengths``."""
    table = _int_keys(lengths) or {}
    return all(int(t) >= int(full) or (int(t) in table and int(s) <= table[int(t)])
               for t, s in (usage or {}).items())


_SENTENCES = ("The harbour master counted {n} boats before the storm reached the northern pier.",
              "In {n} the council rebuilt the bridge with stone taken from the old mill.",
              "A recipe from {n} calls for flour, two eggs, butter and a pinch of salt.",
              "She measured the river at {n} metres wide where it bends below the hill.",
              "Volunteers planted {n} oak trees along the road to the railway station.",
              "The library kept {n} letters written by sailors to their families at home.")


def long_prompt(chat, tokens: int) -> list[int]:
    """A rendered one-turn prompt of at most ``tokens`` tokens (as many sentences of notes as fit)."""
    def render(n: int) -> list[int]:
        notes = " ".join(_SENTENCES[i % len(_SENTENCES)].format(n=1000 + 37 * i) for i in range(n))
        return chat.render([{"role": "user", "content": "Summarise these notes in three sentences.\n\n" + notes}])

    lo, hi = 0, max(1, int(tokens) // 4)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(mid)) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return render(lo)


def _settings_of(gen, settings) -> tuple[int, int, float]:
    from bes.decode.respond import resolve
    from bes.eval.proxy.run_ours import settings_for_generator

    res = resolve(gen.responder.cfg, settings_for_generator(settings))
    return int(res.decode.max_new_tokens), int(res.decode.particles), float(res.decode.temperature)


def _same(a: Sequence, b: Sequence) -> bool:
    import numpy as np

    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def calibrate_elastic(gen, settings, cache_step: int) -> dict[str, Any]:
    """Worker 0, before any reply is kept: which candidate lengths (:func:`candidate_lengths`)
    give, on this machine and thread layout, with the reply's particle count, the decoding rows
    of the export's own cache (the plain length) bitwise, on real model states.

    One long prompt (notes of made-up numbers, up to the longest check) is prefilled chunk by
    chunk on the plain length, as the export prefills. At every filled span of
    :func:`check_spans` of a candidate, the state is copied into ``K`` particle rows twice, on
    the plain length and on the candidate's first slots (:func:`first_slots`), and both take
    :data:`CHECK_STEPS` decoding steps (the same tokens, one particle paused once); every row
    must be equal. A candidate is kept only when all its checks pass, for spans up to
    :func:`span_limit`. Then the real class (:func:`elastic_class`) is checked end to end on a
    short prompt against the plain model. ``lengths`` (``{length: longest span}``) is what every
    reply then decodes on; an empty one means the plain cache for every reply."""
    import numpy as np

    from bes.model import generate as G

    t0 = time.time()
    n_new, k, temperature = _settings_of(gen, settings)
    r = gen.responder
    lm = r.lm_factory(n_new)
    lm.set_temperature(temperature)
    full = int(lm._tmax(64)[1])
    cands = [t for t in candidate_lengths(full, cache_step) if check_spans(t)]
    spans = {t: check_spans(t) for t in cands}
    points = sorted({c for cs in spans.values() for c in cs})
    out: dict[str, Any] = {"role": "calibrate", "elastic_ok": False, "lengths": {}, "candidates": cands,
                           "reference_slots": full, "particles": k, "max_new_tokens": n_new,
                           "cache_step": int(cache_step), "failed": [], "checks": 0}
    m = lm.cfg
    out["slot_bytes"] = 2 * int(m.n_loops) * int(m.n_blocks) * int(m.n_heads) * int(m.head_dim) * 4
    if not points:
        out["seconds"] = round(time.time() - t0, 1)
        return out
    ids = long_prompt(r.chat, max(points) + 63)
    n = len(ids)
    chunk = lm.chunk
    n_pad = (-n) % chunk
    provider = lm._provider()
    rows = [provider.step(int(t)) for t in ids]
    feat = {key: np.concatenate([G.empty_rows(n_pad)[key], np.stack([np.asarray(x[key]) for x in rows])])
            for key in G.FEATURE_KEYS}
    tokens = np.concatenate([np.full(n_pad, lm.pad_id, np.int32), np.asarray(ids, np.int32)])
    seg_all = np.concatenate([np.full(n_pad, -1, np.int32), G.prompt_segments(ids, lm.bos_id)])
    pos = np.concatenate([np.zeros(n_pad, np.int32), np.arange(n, dtype=np.int32)])
    plain_ids = [int(t) for t in ids if int(t) != lm.bos_id] or [int(ids[-1])]

    def steps(cache, c: int) -> list:
        seg = int(max(seg_all[:c].max(), 0))
        got = []
        for j in range(CHECK_STEPS):
            tok = np.array([plain_ids[(c + 7 * i + 3 * j) % len(plain_ids)] for i in range(k)], np.int32)
            act = np.ones(k, bool)
            if j == 1 and k > 1:
                act[0] = False
            f = {key: np.asarray(v) for key, v in G.empty_rows(k).items()}
            f["seg"] = np.where(act, seg, -1).astype(np.int32)
            logits, _, cache = lm.dec.decode_step(lm.params, lm.consts, cache, np.where(act, tok, lm.pad_id).astype(np.int32),
                                                  f, lm.temperature)
            got.append(np.asarray(logits)[act])
        return got

    ok = {t: True for t in cands}
    cache = G.init_cache(lm.cfg, 1, full)
    for lo in range(0, n_pad + n, chunk):
        sl = slice(lo, lo + chunk)
        fc = {"tokens": tokens[None, sl], "seg": seg_all[None, sl], "pos": pos[None, sl],
              **{key: a[None, sl] for key, a in feat.items()}}
        _, _, cache = lm.dec.prefill(lm.params, lm.consts, cache, fc, lm.temperature)
        c = lo + chunk
        todo = [t for t in cands if ok[t] and c in spans[t]]
        if not todo:
            continue
        zeros = np.zeros(k, np.int32)
        want = steps(lm.dec.reorder(cache, zeros), c)       # the export's own (plain) length
        for t in todo:
            out["checks"] += 1
            if not _same(steps(lm.dec.reorder(first_slots(cache, t), zeros), c), want):
                ok[t] = False
                out["failed"].append([int(t), int(c)])
        if c >= points[-1]:
            break
    del cache
    lengths = {int(t): span_limit(t) for t in cands if ok[t]}
    out["probe_tokens"] = n
    e2e = False
    if lengths:
        # the class itself end to end (prefill copy, growth through every kept length) on a short prompt
        p = _Probe(gen, settings, 0)
        el = make_elastic(r.lm_factory(n_new), cache_step, lengths)
        el._tmax(len(p.ids))
        grow = [t for t in sorted(lengths) if t >= el.start_slots]
        p.steps = len(grow) + 2
        want = p.run(r.lm_factory(n_new))
        e2e = _same(p.run(make_elastic(r.lm_factory(n_new), cache_step, lengths), grow), want)
        out["end_to_end"] = {"ok": e2e, "lengths": grow}
    out.update(elastic_ok=bool(lengths) and e2e, lengths=lengths if e2e else {},
               seconds=round(time.time() - t0, 1))
    return out


class _Probe:
    """A prefill and steps (one particle paused once, a cache growth before each step of an
    elastic model, a resampling gather in the middle when ``gather``) of a short check prompt;
    the first run picks the tokens every later run feeds."""

    def __init__(self, gen, settings, steps: int, gather: bool = False):
        self.r = gen.responder
        self.n, self.k, self.temperature = _settings_of(gen, settings)
        self.ids = self.r.chat.render([{"role": "user", "content": "Write two sentences about the sea."}])
        self.steps = int(steps)
        self.gather = bool(gather)
        self.feeds: list = []

    def run(self, lm, grow: Sequence[int] = ()) -> list:
        import numpy as np

        k = self.k
        lm.set_temperature(self.temperature)
        out, acts = [lm.prefill(self.ids, k)], [np.ones(k, bool)]
        for s in range(self.steps):
            if s < len(grow):
                lm.grow_to(grow[s])
            if len(self.feeds) <= s:
                top = np.argsort(-out[-1], axis=1)[:, :3]
                self.feeds.append(np.array([top[j, (j + s) % 3] for j in range(k)], np.int64))
            act = np.ones(k, bool)
            if s == 1:
                act[0] = False
            out.append(lm.step(self.feeds[s], act))
            acts.append(act)
            if self.gather and s == self.steps // 2 and k > 1:
                lm.reorder(np.roll(np.arange(k), 1))
        lm.close()
        return [x[a] for x, a in zip(out, acts)]       # rows of paused particles are unspecified


def warm_elastic(gen, settings, cache_step: int) -> dict[str, Any]:
    """The other workers, meanwhile: compile the shapes of the first replies (the prefill on the
    plain length; steps and gathers at the first two lengths)."""
    p = _Probe(gen, settings, 4, gather=True)
    el = make_elastic(p.r.lm_factory(p.n), cache_step)
    el._tmax(len(p.ids))
    sizes = sorted({el.start_slots, el.length_for(1, at_least=el.start_slots + 1)})
    t0 = time.time()
    p.run(el, [sizes[0], sizes[0], sizes[-1]])
    return {"role": "warm", "compiled_slots": [int(s) for s in sizes], "seconds": round(time.time() - t0, 1)}


def build_generator(export_dir: str, verify: bool):
    from bes.export import export_generator

    return export_generator(export=export_dir, verify=str(bool(verify)).lower())


def warm_up(gen, settings) -> dict[str, Any]:
    """Compile what every reply uses (prefill chunk, the 1 -> K and K -> K cache gathers, the
    K-row decoding step) on a throw-away model instance of the reply's own shapes."""
    import numpy as np

    from bes.decode.respond import resolve
    from bes.eval.proxy.run_ours import settings_for_generator

    r = gen.responder
    res = resolve(r.cfg, settings_for_generator(settings))
    n, k = int(res.decode.max_new_tokens), int(res.decode.particles)
    lm = r.lm_factory(n)
    ids = r.chat.render([{"role": "user", "content": "Hello there."}])
    lm.prefill(ids, k)
    lm.step(np.full(k, int(ids[-1]), np.int64), np.ones(k, bool))
    if k > 1:
        lm.reorder(np.roll(np.arange(k), 1))
    slots = int(lm.cache["seg"].shape[1]) if getattr(lm, "cache", None) is not None else None
    lm.close()
    return {"max_new_tokens": n, "particles": k, "cache_slots": slots}


# ================================================================================ worker
def worker_main() -> int:
    """A worker process: one ``init`` line on stdin, then ``task`` lines; JSON lines on the
    original stdout (``ready``, ``line`` per finished reply, ``done`` per prompt, ``fatal``)."""
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)                                 # anything else printed goes to the worker log
    sys.stdout = sys.stderr

    def send(obj: Mapping[str, Any]) -> None:
        proto.write(_jline(obj) + "\n")
        proto.flush()

    t0 = time.time()
    line = sys.stdin.readline()
    if not line:
        return 0
    init = json.loads(line)
    try:
        import jax
        import numpy  # noqa: F401

        from bes.eval.proxy import runner
        from bes.eval.proxy.common import GenSettings
        from bes.eval.proxy.prompts import Prompt
        from bes.eval.proxy.run_ours import CallableBackend
        jax.config.update("jax_platforms", "cpu")
        backend_name = jax.default_backend()
        platforms = sorted({d.platform for d in jax.devices()})
        if backend_name != "cpu" or platforms != ["cpu"]:
            raise RuntimeError(f"JAX runs on {backend_name} {platforms} here, not on the CPU only")
        t_import = time.time() - t0
        t1 = time.time()
        gen = build_generator(init["export"], verify=False)
        t_load = time.time() - t1
        want = str(init["key"])
        if init.get("cache") == "elastic-always" and want.endswith("-elastic"):
            want = want[:-len("-elastic")]
        if gen.key != want:
            raise RuntimeError(f"this worker's generator has key {gen.key}, the parent's is {init['key']}")
        settings = GenSettings.from_dict(init["settings"])
        t2 = time.time()
        cache = init.get("cache", "elastic")
        step = int(init.get("cache_step", 256))
        warm: dict[str, Any] = {}
        if cache == "full":
            if init.get("warmup", True):
                warm = warm_up(gen, settings)
        elif init.get("role") == "calibrate":
            warm = calibrate_elastic(gen, settings, step)
        elif init.get("warmup", True):
            warm = warm_elastic(gen, settings, step)
        t_compile = time.time() - t2
        plain_factory = gen.responder.lm_factory
        backend = CallableBackend(gen)
        st = proc_status()
        send({"type": "ready", "pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0)),
              "backend": backend_name, "platforms": platforms, "jax_platforms": os.environ.get("JAX_PLATFORMS"),
              "import_s": round(t_import, 2), "load_s": round(t_load, 2),
              "compile_s": round(t_compile, 2), "ready_s": round(time.time() - t0, 2), "warmup": warm,
              "cache": cache, "rss_mb": round(st.get("VmRSS", 0) / 1024, 1),
              "threads": {v: os.environ.get(v) for v in THREAD_VARS[:3]}})
    except Exception as e:  # noqa: BLE001 - reported to the parent, which decides
        traceback.print_exc()
        send({"type": "fatal", "error": f"{type(e).__name__}: {e}"})
        return 3
    seed, key = int(init["seed"]), str(init["key"])
    fail_ids = set(init.get("fail_ids") or ())            # tests only (--fail-ids)
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg.get("type") == "stop":
            break
        if msg.get("type") != "task":
            continue
        d = dict(msg["prompt"])
        p = Prompt(**{**d, "turns": tuple(d["turns"]), "constraints": tuple(d.get("constraints") or ()),
                      "answer": tuple(d["answer"]) if isinstance(d.get("answer"), list) else d.get("answer")})
        mode = msg.get("cache") or {"mode": "full"}       # decided by the parent from worker 0's calibration
        made: list = []                                   # the models of the reply being generated
        if mode.get("mode") == "elastic":
            lengths = mode.get("lengths")                 # None: the guess (speculative)

            def factory(n, lengths=lengths):
                made.append(make_elastic(plain_factory(n), step, lengths))
                return made[-1]

            gen.responder.lm_factory = factory
        else:
            gen.responder.lm_factory = plain_factory

        def on_new(ln, pid=p.id):
            usage: dict[int, int] = {}
            for lm in made:                               # every (length, span) the reply stepped on
                for t, s in getattr(lm, "usage", {}).items():
                    usage[int(t)] = max(usage.get(int(t), 0), int(s))
            made.clear()
            send({"type": "line", "prompt_id": pid, "line": ln, "usage": usage})

        w0, c0 = time.time(), time.process_time()
        record, error = None, None
        try:
            if p.id in fail_ids:
                raise RuntimeError(f"failure injected for {p.id} (--fail-ids)")
            record = runner.converse(p, settings, seed, key, lambda m, s: backend.generate(m, settings, s),
                                     msg.get("cached") or {}, on_new=on_new)
        except Exception as e:  # noqa: BLE001 - the prompt stays incomplete; the worker goes on
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"
        st = proc_status()
        send({"type": "done", "prompt_id": p.id, "record": record, "error": error,
              "seconds": round(time.time() - w0, 2), "cpu_s": round(time.process_time() - c0, 2),
              "rss_mb": round(st.get("VmRSS", 0) / 1024, 1), "hwm_mb": round(st.get("VmHWM", 0) / 1024, 1)})
    return 0


# ========================================================================= the parent
@dataclasses.dataclass
class Worker:
    index: int
    cores: list[int]
    proc: subprocess.Popen | None = None
    buf: bytes = b""
    ready: bool = False
    task: Any = None
    task_t0: float = 0.0
    started: float = 0.0
    restarts: int = 0
    dead: bool = False
    info: dict = dataclasses.field(default_factory=dict)
    max_rss_kb: int = 0
    prompts: int = 0
    busy_s: float = 0.0
    killed_for: str = ""
    ended: bool = False
    spec: bool = False                 # the task in flight was handed out before the calibration decided
    dropped: bool = False              # a reply of the task in flight was not covered: hand it out again

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self.dead

    def send(self, obj: Mapping[str, Any]) -> bool:
        try:
            self.proc.stdin.write((_jline(obj) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            return False

    def read(self) -> list[dict] | None:
        """Complete messages read so far; ``None`` at end of file (the worker is gone)."""
        try:
            chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
        except BlockingIOError:
            return []
        except OSError:
            return None
        if not chunk:
            return None
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for ln in lines:
            if ln.strip():
                out.append(json.loads(ln.decode("utf-8")))
        return out


class Generation:
    """The parent: workers, hand-out, cache, progress and the end (module docstring)."""

    def __init__(self, args: argparse.Namespace, spec, export_dir: Path, prompts, all_prompts, root: Path,
                 out: Path, log: Callable[[str], None], t_start: float):
        from bes.eval.proxy import runner

        self.args, self.spec, self.export_dir = args, spec, export_dir
        self.prompts, self.all_prompts, self.root, self.out, self.log = prompts, all_prompts, root, out, log
        self.t_start = t_start
        self.runner = runner
        self.cpath = runner.cache_path(root, spec.key)
        self.cached = runner.load_cache(root, spec.key)
        self.by_prompt: dict[str, dict[str, Any]] = {}
        for k, line in self.cached.items():
            self.by_prompt.setdefault(line.get("prompt_id", ""), {})[k] = line
        self.records: dict[str, dict] = {}
        self.failed: dict[str, str] = {}
        self.attempts: collections.Counter = collections.Counter()
        self.reused = 0
        self.new_lines = 0
        self.reply_rows: list[dict] = []
        self.prompt_rows: list[dict] = []
        self.workers: list[Worker] = []
        self.stopping = ""
        self.dispatch_closed = ""
        self.t_first_ready: float | None = None
        self.deadline = t_start + 60.0 * args.minutes if args.minutes else None
        self.hard_deadline = (self.deadline + 60.0 * args.grace_minutes) if self.deadline else None
        self.events: list[str] = []
        self.last_progress = 0.0
        self.setup_error = ""
        self.unstarted: list[str] = []
        # how every reply decodes: {"mode": "elastic", "lengths": {length: longest span}} or
        # {"mode": "full"}; with an elastic cache no reply is kept before worker 0's calibration
        # decided it (every worker has the same machine and thread layout)
        self.mode: dict[str, Any] | None = {"mode": "full"} if args.cache == "full" else None
        self.calibration: dict[str, Any] = {}
        self.max_busy = 10 ** 9
        # meanwhile the other workers decode speculatively on every candidate length; their
        # replies are held back and kept only when the calibration covers every (length, span)
        # they stepped on, else handed out again
        self.guess = ({"mode": "elastic", "lengths": None}
                      if args.cache in ("elastic", "elastic-always") and not getattr(args, "no_speculate", False)
                      else None)
        self.held: list[tuple[str, Worker, dict, Any]] = []
        self.speculation: dict[str, Any] = {"tasks": 0}
        self.requeue: collections.deque = collections.deque()      # the hand-out queue of run()

    # ------------------------------------------------------------------ bookkeeping
    def complete(self, p) -> dict | None:
        try:
            return self.runner.converse(p, self.spec.settings, self.spec.seed, self.spec.key, _missing,
                                        self.by_prompt.get(p.id, {}))
        except _Missing:
            return None

    def remaining_replies(self) -> int:
        return sum(max(0, len(p.turns) - len(self.by_prompt.get(p.id, {})))
                   for p in self.prompts if p.id not in self.records)

    def add_line(self, line: Mapping[str, Any]) -> None:
        from bes.eval.proxy.common import append_jsonl

        append_jsonl(self.cpath, [line])
        self.cached[line["key"]] = dict(line)
        self.by_prompt.setdefault(line["prompt_id"], {})[line["key"]] = dict(line)
        self.new_lines += 1
        rep = line.get("reply") or {}
        ex = rep.get("extra") or {}
        self.reply_rows.append({"prompt_id": line["prompt_id"], "turn": line.get("turn"),
                                "seconds": rep.get("latency_s"), "tokens": rep.get("completion_tokens"),
                                "prompt_tokens": rep.get("prompt_tokens"), "steps": ex.get("steps"),
                                "particles": ex.get("particles"), "finish": rep.get("finish_reason")})

    def progress(self, force: bool = False) -> None:
        t = time.time()
        if not force and t - self.last_progress < self.args.progress_s:
            return
        self.last_progress = t
        n, total = len(self.records), len(self.prompts)
        busy = sum(1 for w in self.workers if w.task is not None)
        ready = sum(1 for w in self.workers if w.ready and w.alive)
        t0 = self.t_first_ready or t
        minutes = max(1e-9, (t - t0) / 60.0)
        rate = self.new_lines / minutes if self.t_first_ready else 0.0
        left = self.remaining_replies()
        eta = f"{left / rate:.1f} min" if rate > 0 else "?"
        self.log(f"{n}/{total} prompts done ({self.reused} from cache, {len(self.failed)} failed), "
                 f"{self.new_lines} new replies, {rate:.2f} replies/min, {left} replies left, ETA {eta}; "
                 f"workers {ready} ready, {busy} busy; {(t - self.t_start) / 60:.1f} min since start")

    # ------------------------------------------------------------------ workers
    def start_worker(self, w: Worker, init: Mapping[str, Any], env: Mapping[str, str]) -> None:
        logs = self.out / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-cores",
                ",".join(str(c) for c in w.cores), "--nice", str(int(self.args.nice)), "--worker-parent",
                str(os.getpid())]
        with open(logs / f"worker-{w.index:02d}.log", "ab") as errf:
            w.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                                      env=dict(env), cwd=str(REPO), close_fds=True)
        os.set_blocking(w.proc.stdout.fileno(), False)
        w.buf, w.ready, w.task, w.started, w.dead = b"", False, None, time.time(), False
        w.killed_for, w.ended, w.spec, w.dropped = "", False, False, False
        w.send({**init, "index": w.index, "role": "calibrate" if w.index == 0 and self.mode is None else "warm"})

    def stop_worker(self, w: Worker, hard: bool = False) -> None:
        if w.proc is None:
            return
        if w.proc.poll() is None:
            if hard:
                w.proc.kill()
            else:
                w.send({"type": "stop"})
                try:
                    w.proc.stdin.close()
                except OSError:
                    pass
        w.dead = True

    def run(self) -> None:
        args = self.args
        pending = [p for p in self.prompts if p.id not in self.records]
        if not pending:
            self.log("every prompt is complete in the cache: nothing to generate")
            return
        position = {p.id: i for i, p in enumerate(self.prompts)}
        cat = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        queue = collections.deque(sorted(pending, key=lambda p: (-len(p.turns), cat.get(p.category, len(cat)),
                                                                 position[p.id])))
        self.requeue = queue
        cpus = sorted(os.sched_getaffinity(0))
        per = int(args.cores_per_worker)
        n_workers = int(args.workers) if args.workers else max(1, len(cpus) // per)
        n_workers = max(1, min(n_workers, len(pending)))
        plan = plan_cores(cpus, n_workers, per)
        threads = int(args.threads) if args.threads else per
        env = worker_env(threads)
        init = {"type": "init", "export": str(self.export_dir), "cache": args.cache, "cache_step": args.cache_step,
                "key": self.spec.key, "settings": self.spec.settings.as_dict(), "seed": int(self.spec.seed),
                "warmup": not args.no_warmup,
                "fail_ids": [x for x in (getattr(args, "fail_ids", None) or "").split(",") if x]}
        self.log(f"{len(pending)} prompts to generate ({sum(len(p.turns) for p in pending)} turns, "
                 f"{self.remaining_replies()} replies not cached) with {n_workers} workers x {per} cores "
                 f"({threads} threads each) on CPUs {cpus}; cores {plan}")
        self.fingerprint = machine_fingerprint(threads, self.spec, args.cache_step)
        saved = _read_json(self.out / CALIBRATION_JSON) or {}
        if self.mode is None and not getattr(args, "recalibrate", False) and saved.get("fingerprint") == self.fingerprint:
            self.calibration = dict(saved.get("result") or {})
            self._decide({}, self.calibration)     # same machine type, code and settings: no new check
            self.calibration["reused_from"] = saved.get("written")
            self.log(f"calibration of {saved.get('written')} reused ({CALIBRATION_JSON}: same CPU, JAX, threads, "
                     "export and settings)")
        self.workers = [Worker(i, cores) for i, cores in enumerate(plan)]
        sel = selectors.DefaultSelector()
        for w in self.workers:
            self.start_worker(w, init, env)
            sel.register(w.proc.stdout, selectors.EVENT_READ, w)
        last_rss = 0.0

        def handle_end(w: Worker, why: str) -> None:
            if w.ended:
                return
            w.ended = True
            try:
                sel.unregister(w.proc.stdout)
            except (KeyError, ValueError):
                pass
            rc = w.proc.wait() if w.proc.poll() is None and w.dead else w.proc.poll()
            if rc is None:
                try:
                    rc = w.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    w.proc.kill()
                    rc = w.proc.wait()
            w.dead = True
            if w.task is not None:
                p = w.task
                w.task = None
                self.attempts[p.id] += 1
                if self.attempts[p.id] < 2 and not self.stopping:
                    queue.appendleft(p)
                    self.log(f"worker {w.index} ended ({why}, exit {rc}) during {p.id}: it is handed out again")
                else:
                    self.failed[p.id] = f"worker ended ({why}, exit {rc})"
                    self.log(f"worker {w.index} ended ({why}, exit {rc}) during {p.id} again: given up")
            elif not self.stopping and why != "stopped":
                self.log(f"worker {w.index} ended ({why}, exit {rc})")
            self.events.append(f"worker {w.index}: {why} (exit {rc})")
            calibrating = w.index == 0 and self.mode is None       # the others wait for it
            if (not self.stopping and not self.dispatch_closed and (queue or calibrating)
                    and w.restarts < args.restarts and why != "stopped"):
                w.restarts += 1
                self.log(f"worker {w.index}: restart {w.restarts} of {args.restarts}")
                self.start_worker(w, init, env)
                sel.register(w.proc.stdout, selectors.EVENT_READ, w)
            elif w.index == 0 and self.mode is None and not self.stopping:
                self.stopping = self.setup_error = (f"worker 0 ended ({why}) before its calibration of the elastic "
                                                    f"cache: see {self.out}/logs/worker-00.log")
                self.log(f"ABORT: {self.stopping}")

        while True:
            now = time.time()
            if self.stopping:
                break
            if self.deadline and now > self.deadline and not self.dispatch_closed:
                self.dispatch_closed = f"--minutes {args.minutes} reached"
                self.log(f"{self.dispatch_closed}: no more prompts are handed out; {len(queue)} not started")
            if self.hard_deadline and now > self.hard_deadline:
                self.stopping = f"--grace-minutes {args.grace_minutes} after --minutes reached"
                self.log(f"{self.stopping}: stopping the workers")
                break
            for w in self.workers:
                mode = self.mode or self.guess
                if w.ready and w.alive and w.task is None and mode is not None:
                    busy = sum(1 for x in self.workers if x.task is not None)
                    if queue and not self.dispatch_closed and busy < self.max_busy:
                        p = queue.popleft()
                        task = {"type": "task", "prompt": _prompt_dict(p), "cached": self.by_prompt.get(p.id, {}),
                                "cache": mode}
                        w.task, w.task_t0, w.spec, w.dropped = p, time.time(), self.mode is None, False
                        self.speculation["tasks"] += int(w.spec)
                        if not w.send(task):
                            handle_end(w, "its input closed")
                    elif not w.dead and (not queue or self.dispatch_closed) and self.mode is not None:
                        self.stop_worker(w)                  # nothing left for it
            undecided = self.mode is None and bool(self.held)    # speculative replies wait for the calibration
            if not any(w.task is not None for w in self.workers) and (not queue or self.dispatch_closed) \
                    and not undecided:
                break
            if all(w.dead for w in self.workers):
                self.log(f"every worker ended; {len(queue)} prompts not started")
                break
            for skey, _ in sel.select(timeout=1.0):
                w = skey.data
                try:
                    msgs = w.read()
                except ValueError as e:                 # a corrupt message: the worker is not trusted
                    w.proc.kill()
                    handle_end(w, f"unreadable message ({e})")
                    continue
                if msgs is None:
                    handle_end(w, w.killed_for or ("stopped" if w.dead else "its output closed"))
                    continue
                for m in msgs:
                    self.handle(w, m)
            if time.time() - last_rss >= 2.0:
                last_rss = time.time()
                for w in self.workers:
                    if not w.alive:
                        continue
                    if not w.ready and time.time() - w.started > args.start_timeout:
                        w.killed_for = f"not ready after {args.start_timeout:.0f} s"
                        self.log(f"worker {w.index}: {w.killed_for}: killed")
                        w.proc.kill()
                        continue
                    kb = proc_status(w.proc.pid).get("VmRSS", 0)
                    w.max_rss_kb = max(w.max_rss_kb, kb)
                    if args.max_rss_gb and kb > args.max_rss_gb * 2 ** 20:
                        w.killed_for = f"memory {kb / 2 ** 20:.1f} GB > --max-rss-gb {args.max_rss_gb}"
                        self.log(f"worker {w.index}: {w.killed_for}: killed")
                        w.proc.kill()
            for w in self.workers:
                if w.proc is not None and w.proc.poll() is not None and not w.ended:
                    try:
                        rest = w.read() or []
                    except ValueError:
                        rest = []
                    for m in rest:
                        self.handle(w, m)
                    handle_end(w, w.killed_for or ("stopped" if w.dead else f"exited {w.proc.returncode}"))
            self.progress()
        # the end: idle workers were told to stop; busy ones (deadline, signal) are stopped now
        for w in self.workers:
            if w.alive:
                self.stop_worker(w, hard=w.task is not None)
        t_end = time.time() + 15.0
        for w in self.workers:
            if w.proc is None:
                continue
            try:
                w.proc.wait(timeout=max(0.1, t_end - time.time()))
            except subprocess.TimeoutExpired:
                w.proc.kill()
                w.proc.wait()
            if w.task is not None:
                self.events.append(f"worker {w.index}: stopped during {w.task.id}")
                w.task = None
        sel.close()
        if self.held:
            self.log(f"{sum(1 for kind, *_ in self.held if kind == 'done')} speculative prompts dropped: the "
                     "calibration did not decide before the end")
            self.held = []
        self.unstarted = [p.id for p in queue]

    def handle(self, w: Worker, m: Mapping[str, Any]) -> None:
        kind = m.get("type")
        if kind == "ready":
            w.ready, w.info = True, dict(m)
            if self.t_first_ready is None:
                self.t_first_ready = time.time()
            chk = m.get("warmup") or {}
            self.log(f"worker {w.index} ready on {m.get('affinity')} ({m.get('backend')}, {chk.get('role', 'full')}"
                     f" {chk.get('seconds', '')} s): import {m.get('import_s')} s, load {m.get('load_s')} s, "
                     f"compile {m.get('compile_s')} s, {m.get('rss_mb')} MB")
            if m.get("backend") != "cpu" or (m.get("platforms") or ["cpu"]) != ["cpu"]:
                self.stopping = self.setup_error = (f"worker {w.index} runs JAX on {m.get('backend')} "
                                                    f"{m.get('platforms')}, not on the CPU only")
                self.log(f"ABORT: {self.stopping}")
            elif self.mode is None and chk.get("role") == "calibrate":
                self.decide(w, m, chk)
        elif kind == "line":
            if w.spec:
                self.held.append(("line", w, dict(m), w.task))
            else:
                self.accept_line(w, m)
        elif kind == "done":
            p = w.task
            w.task = None
            w.busy_s += float(m.get("seconds") or 0.0)
            w.max_rss_kb = max(w.max_rss_kb, int(float(m.get("hwm_mb") or 0) * 1024))
            if w.spec:
                w.spec = False
                self.held.append(("done", w, dict(m), p))
                return
            dropped, w.dropped = w.dropped, False
            if dropped and p is not None:
                self.requeue.appendleft(p)             # its covered replies are in the cache already
                return
            self.finish(w, m, p)
        elif kind == "fatal":
            self.log(f"worker {w.index} could not start: {m.get('error')} (see {self.out}/logs/worker-{w.index:02d}.log)")
            w.restarts = self.args.restarts            # the same error again: no restart
            w.dead = True

    def finish(self, w: Worker, m: Mapping[str, Any], p) -> None:
        """A prompt's conversation came back (its replies are in the cache already)."""
        pid = m["prompt_id"]
        w.prompts += 1
        if m.get("record") is not None:
            self.records[pid] = m["record"]
            self.failed.pop(pid, None)
        else:
            self.failed[pid] = str(m.get("error"))
            self.log(f"{pid} failed on worker {w.index}: {m.get('error')}")
        self.prompt_rows.append({"prompt_id": pid, "worker": w.index, "seconds": m.get("seconds"),
                                 "cpu_s": m.get("cpu_s"), "turns": len(p.turns) if p is not None else None,
                                 "rss_mb": m.get("rss_mb"), "hwm_mb": m.get("hwm_mb")})
        self.progress(force=self.args.progress_s <= 0)

    def covered(self, m: Mapping[str, Any]) -> bool:
        """A reply's (length, span) usage is covered by the decided cache (``elastic-always``: any)."""
        if self.args.cache == "elastic-always":
            return True
        lengths = (self.mode or {}).get("lengths") if (self.mode or {}).get("mode") == "elastic" else {}
        full = int(self.calibration.get("reference_slots") or 10 ** 9)
        return usage_ok(m.get("usage"), lengths or {}, full)

    def accept_line(self, w: Worker, m: Mapping[str, Any]) -> None:
        """A finished reply: into the cache when the calibration covers every (length, span) it
        stepped on, else dropped (its prompt is handed out again when the task ends)."""
        if self.covered(m):
            self.add_line(m["line"])
            return
        w.dropped = True
        self.speculation["dropped_replies"] = self.speculation.get("dropped_replies", 0) + 1
        self.log(f"a reply of {m.get('prompt_id')} used lengths the calibration does not cover "
                 f"({m.get('usage')}): dropped, handed out again")

    def settle(self) -> None:
        """The calibration has decided: every held speculative reply goes into the cache when the
        decided lengths cover what it stepped on, else it is dropped and its prompt handed out
        again (its covered earlier turns stay in the cache); a task still in flight is checked
        reply by reply from now on."""
        held, self.held = self.held, []
        n_done = sum(1 for kind, *_ in held if kind == "done")
        n_lines = sum(1 for kind, *_ in held if kind == "line")
        before = self.speculation.get("dropped_replies", 0)
        dropped_tasks: set[int] = set()
        for kind, w, m, p in held:
            if kind == "line":
                flag = w.dropped
                w.dropped = False
                self.accept_line(w, m)
                if w.dropped:
                    dropped_tasks.add(id(p))
                w.dropped = flag
            elif id(p) in dropped_tasks:
                if p is not None:
                    self.requeue.appendleft(p)
            else:
                self.finish(w, m, p)
        in_flight = [w for w in self.workers if w.spec and w.task is not None]
        for w in in_flight:
            w.spec = False                              # its later replies are checked as they come
            if id(w.task) in dropped_tasks:
                w.dropped = True
        dropped = self.speculation.get("dropped_replies", 0) - before
        self.speculation.update(kept_replies=n_lines - dropped, dropped_replies=dropped, prompts_done=n_done,
                                in_flight=len(in_flight))
        if held or in_flight:
            self.log(f"speculative replies: {n_lines - dropped} kept, {dropped} dropped (handed out again); "
                     f"{n_done} prompts done, {len(in_flight)} in flight")

    def decide(self, w: Worker, m: Mapping[str, Any], chk: Mapping[str, Any]) -> None:
        """How every reply decodes, from worker 0's calibration (:func:`calibrate_elastic`)."""
        self.calibration = dict(chk)
        try:
            self._decide(m, chk)
            from bes.eval.proxy.common import write_json_atomic

            write_json_atomic(self.out / CALIBRATION_JSON, {"fingerprint": getattr(self, "fingerprint", None),
                                                           "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                           "result": dict(chk)})
        finally:
            self.settle()

    def _decide(self, m: Mapping[str, Any], chk: Mapping[str, Any]) -> None:
        lengths = {int(t): int(v) for t, v in (chk.get("lengths") or {}).items()}
        if chk.get("elastic_ok") and lengths:
            self.mode = {"mode": "elastic", "lengths": lengths}
            self.log(f"calibration: decoding steps give the export's rows bitwise on {sorted(lengths)} slots "
                     f"(filled spans up to {[lengths[t] for t in sorted(lengths)]}; {chk.get('checks')} checks on a "
                     f"{chk.get('probe_tokens')}-token prompt, failed {chk.get('failed') or 'none'}; longer: the "
                     f"export's {chk.get('reference_slots')} slots) in {chk.get('seconds')} s")
            return
        if self.args.cache == "elastic-always":
            self.mode = {"mode": "elastic", "lengths": None}
            self.log("WARNING calibration: the elastic cache does NOT give the export's rows on this machine; "
                     "--cache elastic-always: it is used all the same (this run's key is its own)")
            return
        # exact replies need the export's own cache: about K x 5,248 slots per worker, twice that
        # during a resampling gather; as many workers at once as the memory holds
        k, slots, sb = int(chk.get("particles") or 1), int(chk.get("reference_slots") or 0), int(chk.get("slot_bytes") or 0)
        need = float(m.get("rss_mb") or 1024) * 2 ** 20 + 2.0 * k * slots * sb
        avail = (mem_available_gb() or 0.0) * 2 ** 30 - 2 * 2 ** 30
        self.max_busy = max(1, int(avail // need)) if need > 0 and avail > 0 else 1
        self.mode = {"mode": "full"}
        self.log(f"WARNING calibration: no elastic length gives the export's rows on this machine "
                 f"(failed {chk.get('failed')}, end to end {chk.get('end_to_end')}): every reply decodes with the "
                 f"export's own cache (exact, 5-10 times slower), about {need / 2 ** 30:.1f} GB per worker: at most "
                 f"{self.max_busy} at once")


def _prompt_dict(p) -> dict[str, Any]:
    return {"id": p.id, "category": p.category, "turns": list(p.turns),
            "answer": list(p.answer) if isinstance(p.answer, tuple) else p.answer, "tol": p.tol,
            "constraints": list(p.constraints), "notes": p.notes, "split": p.split}


def _stats(xs: Sequence[float]) -> dict[str, Any]:
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 2), "median": round(statistics.median(xs), 2),
            "p90": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 2), "max": round(xs[-1], 2),
            "min": round(xs[0], 2)}


def reply_lines(prompts, records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``replies.jsonl``: per complete prompt its turns, replies, finish information and seconds."""
    out = []
    for p in prompts:
        r = records.get(p.id)
        if r is None:
            continue
        finish = []
        for t in r["turns"]:
            e = t.get("extra") or {}
            finish.append({"finish_reason": t.get("finish_reason"), "particle_finish": e.get("finish"),
                           "at_budget": e.get("at_budget"), "tokens": t.get("completion_tokens"),
                           "prompt_tokens": t.get("prompt_tokens"), "rule": e.get("rule"),
                           "n_survivors": e.get("n_survivors"), "n_finished": e.get("n_finished"),
                           "steps": e.get("steps"), "particles": e.get("particles"),
                           "library": e.get("library"), "violations": e.get("violations"),
                           "infeasible": e.get("infeasible"), "rerun": e.get("rerun"), "gen_key": t.get("gen_key")})
        secs = [t.get("latency_s") for t in r["turns"]]
        out.append({"id": p.id, "category": p.category, "split": p.split, "multi_turn": p.multi_turn,
                    "turns": [t["user"] for t in r["turns"]], "replies": [t["reply"] for t in r["turns"]],
                    "finish": finish, "seconds": secs, "seconds_total": round(sum(s or 0.0 for s in secs), 3)})
    return out


def write_bundle(out: Path, files: Sequence[Path]) -> Path:
    path = out / "testgen.tar.gz"
    tmp = path.with_name(path.name + ".tmp")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for f in files:
            if Path(f).exists():
                tar.add(str(f), arcname=str(Path(f).relative_to(out)))
    os.replace(tmp, path)
    return path


# ================================================================================ merge
def merge_into(src_out: Path, root: Path) -> int:
    """``--merge-into``: our run (``src_out``: an output directory of this script, e.g. fetched
    from the runtime) into another harness root."""
    from bes.eval.proxy import prompts as prompt_set
    from bes.eval.proxy import runner
    from bes.eval.proxy.common import GenSettings, append_jsonl, read_jsonl

    summary = _read_json(src_out / SUMMARY_JSON)
    if not summary:
        raise SystemExit(f"{src_out} holds no {SUMMARY_JSON}")
    src_root = Path(summary["root"])
    if not src_root.is_dir():
        src_root = src_out / "proxy"
    meta = _read_json(src_root / "outputs" / f"{summary['run_id']}.meta.json")
    if not meta:
        raise SystemExit(f"no outputs meta for run {summary['run_id']} under {src_root}")
    spec = runner.RunSpec(backend=meta["backend"], key=meta["backend_key"],
                          settings=GenSettings.from_dict(meta["settings"]), label=meta["label"], seed=int(meta["seed"]))
    if spec.run_id != summary["run_id"]:
        raise SystemExit(f"the meta rebuilds run {spec.run_id}, the summary says {summary['run_id']}")
    have = runner.load_cache(root, spec.key)
    lines = [ln for ln in read_jsonl(runner.cache_path(src_root, spec.key)) if "key" in ln and "reply" in ln]
    new = [ln for ln in lines if ln["key"] not in have]
    clash = [ln["key"] for ln in lines if ln["key"] in have and have[ln["key"]].get("reply") != ln.get("reply")]
    if new:
        append_jsonl(runner.cache_path(root, spec.key), new)
    path = runner.assemble(spec, root, prompt_set.load())
    n = len(read_jsonl(path))
    print(f"merged {len(new)} new cache lines ({len(lines) - len(new)} already there"
          f"{f', {len(clash)} with another reply kept as they were' if clash else ''}) into {root}; "
          f"{path}: {n} complete prompts")
    print(f"run id: {spec.run_id}")
    split = summary.get("split") if summary.get("split") in ("dev", "test", "all") else "all"
    print(f"pairs:  python -m bes.eval.proxy --root {root} pairs --x {spec.run_id} --y OPPONENT_RUN_ID "
          f"--name NAME --split {split}")
    return 0


# ================================================================================= main
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="scripts/gen_test_replies.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="export directory, run directory or snapshot directory "
                                              "(with --merge-into: an output directory of this script)")
    ap.add_argument("--out", default=None, help="output directory (default: ./testgen)")
    ap.add_argument("--root", default=None, help="harness root to write the run into (default OUT/proxy)")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--categories", default=None, help="comma-separated subset of categories")
    ap.add_argument("--ids", default=None, help="comma-separated prompt ids (any split)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of the split (harness order)")
    ap.add_argument("--seed", type=int, default=0, help="harness seed (the opponents' runs use 0)")
    ap.add_argument("--label", default=None, help="run label (default final-<the export's decode choice>)")
    ap.add_argument("--set", action="append", default=[], help="harness settings override key=value "
                                                                "(e.g. extra.particles=1), after the export's choice")
    ap.add_argument("--cache", default="elastic", choices=["elastic", "full", "elastic-always"],
                    help="elastic: the export's decoder with a key-value cache that grows with the reply "
                         "(the export's rows bitwise, calibrated by worker 0; the export's own cache where the "
                         "calibration fails); full: the export's own fixed cache; elastic-always: elastic even "
                         "where the calibration fails (replies may then differ from the fixed cache's at float "
                         "rounding, so the run gets a cache key of its own, <key>-elastic)")
    ap.add_argument("--cache-step", type=int, default=256, help="candidate lengths of the elastic cache below "
                                                                 "1,024 slots: multiples of this")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (default: allowed cores / --cores-per-worker)")
    ap.add_argument("--cores-per-worker", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="OpenMP/BLAS threads per worker (default: its cores)")
    ap.add_argument("--nice", type=int, default=0, help="niceness added to every worker")
    ap.add_argument("--minutes", type=float, default=None, help="no prompt is handed out after this many minutes")
    ap.add_argument("--grace-minutes", type=float, default=3.0, help="workers still busy this long after "
                                                                     "--minutes are stopped")
    ap.add_argument("--max-rss-gb", type=float, default=0.0, help="kill a worker above this resident memory")
    ap.add_argument("--restarts", type=int, default=2, help="restarts of a worker that died")
    ap.add_argument("--progress-s", type=float, default=30.0, help="seconds between progress lines "
                                                                   "(0: one per finished prompt)")
    ap.add_argument("--no-warmup", action="store_true", help="compile during the first reply instead")
    ap.add_argument("--no-speculate", action="store_true", help="the other workers wait for worker 0's "
                                                                "calibration instead of decoding meanwhile")
    ap.add_argument("--recalibrate", action="store_true", help="check the elastic cache again even when "
                                                               "OUT/calibration.json is of the same machine type")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA256SUMS check of the export")
    ap.add_argument("--make-export", action="store_true", help="a run or snapshot: export its newest weights "
                                                               "into OUT/export even when final/ is current")
    ap.add_argument("--library", default=None, help="library for --make-export (default: the run's)")
    ap.add_argument("--data-dir", default=None, help="held-out files for --make-export (default run.data_dir)")
    ap.add_argument("--no-download", action="store_true", help="--make-export: never download held-out files")
    ap.add_argument("--merge-into", default=None, metavar="ROOT", help="merge an output directory into ROOT")
    ap.add_argument("--start-timeout", type=float, default=900.0, help="seconds a worker may take to get ready")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-cores", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-parent", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--fail-ids", default="", help=argparse.SUPPRESS)       # tests: these prompts raise
    args = ap.parse_args(argv)
    if not args.worker and not args.source:
        ap.error("give the source (export, run or snapshot directory)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        _child_setup(sorted(parse_cpu_list(args.worker_cores)), args.nice, args.worker_parent)
        return worker_main()
    if args.merge_into:
        return merge_into(Path(args.source).resolve(), Path(args.merge_into).resolve())
    t_start = time.time()
    out = Path(args.out or "testgen").resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "testgen.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[testgen {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    stop: dict[str, str] = {}
    gen_box: dict[str, Generation] = {}

    def on_signal(signum, _frame):
        stop["why"] = f"signal {signal.Signals(signum).name}"
        if "g" in gen_box:
            gen_box["g"].stopping = stop["why"]

    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        return _generate(args, out, log, logf, stop, gen_box, t_start)
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)


def _generate(args, out: Path, log, logf, stop: dict, gen_box: dict, t_start: float) -> int:
    summary: dict[str, Any] = {"status": "failed", "out": str(out), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    bundle_files: list[Path] = []
    try:
        from bes import export as EX
        from bes.eval.proxy import __main__ as cli
        from bes.eval.proxy import prompts as prompt_set
        from bes.eval.proxy import runner

        src = Path(args.source).resolve()
        export_dir, provenance = resolve_source(src, out, args, log)
        t0 = time.time()
        ex = EX.load_export(export_dir, verify=not args.no_verify)
        t_load = time.time() - t0
        chosen = (ex.record.get("decode_choice") or {}).get("chosen") or "default"
        overrides = ex.default_overrides() + list(args.set)
        if ex.library is None:
            overrides.append("extra.use_library=false")
        # the bundle loader's key: the replies are export_generator's (elastic-always: maybe not)
        key = ex.key() if args.cache != "elastic-always" else f"{ex.key()}-elastic"
        spec = cli.build_spec("ours", "default", overrides, args.label or f"final-{chosen}", args.seed, None, None,
                              1, None, EXPORT_FACTORY, {"export": str(export_dir)}, key)
        exp_info = {"dir": str(export_dir), "step": ex.step, "source": ex.record.get("source"),
                    "weights_sha256": (ex.record.get("weights") or {}).get("sha256"),
                    "library": None if ex.library is None else (ex.record.get("library") or {}).get("digest"),
                    "decode_choice": ex.record.get("decode_choice"), "notes": ex.notes,
                    "verified": not args.no_verify, "load_s": round(t_load, 2), **provenance}
        del ex
        gc.collect()
        ps = prompt_set.load()
        if args.ids:
            want = [x.strip() for x in args.ids.split(",") if x.strip()]
            known = {p.id for p in ps}
            unknown = [x for x in want if x not in known]
            if unknown:
                raise SystemExit(f"unknown prompt ids {unknown}")
            todo = [p for p in prompt_set.select(ps, "all") if p.id in set(want)]
        else:
            todo = prompt_set.select(ps, args.split, args.categories.split(",") if args.categories else None)
        if args.limit is not None:
            todo = todo[:args.limit]
        root = Path(args.root).resolve() if args.root else out / "proxy"
        summary.update(run_id=spec.run_id, key=spec.key, root=str(root), split=args.split if not args.ids else "ids",
                       settings=spec.settings.as_dict(), settings_hash=spec.settings.hash(), overrides=overrides,
                       cache=args.cache, cache_step=args.cache_step, export=exp_info, total=len(todo))
        log(f"run {spec.run_id}: {len(todo)} prompts ({sum(len(p.turns) for p in todo)} turns) of "
            f"{summary['split']}; export {export_dir} (step {exp_info['step']}, {provenance.get('how')}); "
            f"overrides {overrides}")
        g = Generation(args, spec, export_dir, todo, ps, root, out, log, t_start)
        gen_box["g"] = g
        if stop:
            g.stopping = stop["why"]
        for p in todo:
            rec = g.complete(p)
            if rec is not None:
                g.records[p.id] = rec
        g.reused = len(g.records)
        if g.reused:
            log(f"{g.reused} prompts are complete in the cache {g.cpath}")
        g.unstarted = []
        if not g.stopping:
            g.run()
        g.progress(force=True)
        # outputs: every prompt of the set complete in the cache (as the harness writes them)
        path = runner.assemble(spec, root, ps, g.cached)
        records = {p.id: g.complete(p) for p in todo}
        records = {k: v for k, v in records.items() if v is not None}
        from bes.eval.proxy.common import write_json_atomic, write_jsonl_atomic

        write_jsonl_atomic(out / REPLIES_JSONL, reply_lines(todo, records))
        missing = [p.id for p in todo if p.id not in records]
        wall = time.time() - t_start
        gen_s = (time.time() - g.t_first_ready) if g.t_first_ready else 0.0
        workers = [{"index": w.index, "cores": w.cores, "restarts": w.restarts, "prompts": w.prompts,
                    "busy_s": round(w.busy_s, 1), "max_rss_mb": round(w.max_rss_kb / 1024, 1),
                    **{k: w.info.get(k) for k in ("pid", "affinity", "backend", "platforms", "jax_platforms", "cache",
                                                  "import_s", "load_s", "compile_s", "ready_s", "rss_mb", "warmup")}}
                   for w in g.workers]
        by_k: dict[str, list] = collections.defaultdict(list)
        for r in g.reply_rows:
            by_k[str(r.get("particles"))].append(r)
        cost = {"replies_per_min": round(g.new_lines / (gen_s / 60.0), 3) if gen_s > 0 and g.new_lines else None,
                "prompts_per_min": round(len(g.prompt_rows) / (gen_s / 60.0), 3) if gen_s > 0 and g.prompt_rows else None,
                "generation_s": round(gen_s, 1), "reply_seconds": _stats([r["seconds"] for r in g.reply_rows]),
                "reply_seconds_by_particles": {k: _stats([r["seconds"] for r in v]) for k, v in by_k.items()},
                "ms_per_step": _stats([1000.0 * r["seconds"] / r["steps"] for r in g.reply_rows
                                       if r.get("seconds") and r.get("steps")]),
                "tokens": _stats([r["tokens"] for r in g.reply_rows]),
                "steps": _stats([r["steps"] for r in g.reply_rows]),
                "prompt_seconds": _stats([r["seconds"] for r in g.prompt_rows]),
                "startup": {k: _stats([w.info.get(k) for w in g.workers if w.info.get(k) is not None])
                            for k in ("import_s", "load_s", "compile_s", "ready_s")},
                "worker_max_rss_mb": max((w.max_rss_kb / 1024 for w in g.workers), default=0.0),
                "mem_available_gb_end": mem_available_gb()}
        status = "complete" if not missing else ("partial" if records else "failed")
        summary.update(status=status, done=len(records), missing=len(missing), missing_ids=missing[:20],
                       failed=dict(g.failed), unstarted=len(getattr(g, "unstarted", [])),
                       reused=g.reused, new_replies=g.new_lines, wall_s=round(wall, 1), stopped=g.stopping or None,
                       dispatch_closed=g.dispatch_closed or None, workers=len(g.workers),
                       workers_ready=sum(1 for w in g.workers if w.info), decode_mode=g.mode,
                       calibration=g.calibration or None, speculation=g.speculation,
                       max_busy=g.max_busy if g.max_busy < 10 ** 9 else None,
                       cores_per_worker=args.cores_per_worker, threads=args.threads or args.cores_per_worker,
                       cost=cost, worker_details=workers, events=g.events[-50:],
                       files={"outputs": str(path), "meta": str(path.with_suffix(".meta.json")),
                              "cache": str(g.cpath), "replies": str(out / REPLIES_JSONL),
                              "summary": str(out / SUMMARY_JSON), "log": str(out / "testgen.log"),
                              "bundle": str(out / "testgen.tar.gz")},
                       per_prompt=g.prompt_rows, per_reply=g.reply_rows,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json_atomic(out / SUMMARY_JSON, summary)
        bundle_files = [out / REPLIES_JSONL, out / SUMMARY_JSON, out / "testgen.log", out / CALIBRATION_JSON, g.cpath,
                        path, path.with_suffix(".meta.json")] + sorted((out / "logs").glob("worker-*.log"))
        rc = 0 if records else 1                       # non-zero only when no prompt is complete
        if not records and g.workers and not any(w.info for w in g.workers):
            rc = 2                                     # no worker could start
        if g.setup_error:
            summary["error"] = g.setup_error
            rc = rc if records else 2
            write_json_atomic(out / SUMMARY_JSON, summary)
    except SystemExit as e:
        summary["error"] = str(e)
        log(f"ERROR: {e}")
        rc = 2
    except Exception as e:  # noqa: BLE001 - the summary line says what happened
        summary["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {summary['error']}\n{traceback.format_exc()}")
        rc = 2
    short = {k: summary.get(k) for k in ("status", "run_id", "key", "done", "total", "missing", "new_replies",
                                         "reused", "wall_s", "workers", "cores_per_worker", "stopped",
                                         "dispatch_closed", "error", "root", "out")}
    cost = summary.get("cost") or {}
    mode = summary.get("decode_mode") or {}
    short.update(decode=mode.get("mode"), lengths=sorted(int(t) for t in (mode.get("lengths") or {})),
                 failed=len(summary.get("failed") or {}), unstarted=summary.get("unstarted"),
                 replies_per_min=cost.get("replies_per_min"),
                 reply_s_mean=(cost.get("reply_seconds") or {}).get("mean"),
                 ready_s_median=((cost.get("startup") or {}).get("ready_s") or {}).get("median"),
                 worker_max_rss_mb=cost.get("worker_max_rss_mb"),
                 bundle=(summary.get("files") or {}).get("bundle"))
    log(f"finished: {summary.get('status')} ({summary.get('done')}/{summary.get('total')} prompts)")
    logf.write(f"{SUMMARY_TAG} {_jline(short)}\n")
    logf.close()
    if bundle_files:
        try:
            write_bundle(out, [f for f in bundle_files if Path(f).exists() and _inside(Path(f), out)])
        except OSError as e:
            short["bundle"] = f"not written: {type(e).__name__}: {e}"
    print(f"{SUMMARY_TAG} {_jline(short)}", flush=True)
    return rc


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
