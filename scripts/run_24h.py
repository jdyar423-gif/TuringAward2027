"""The single 24-hour run (``docs/PLAN.md`` sections 2 and 6): one budget of wall time for
installation checks, downloads, tokenisation, the library, the start-up benchmark, training,
post-training, evaluation and export.

    python scripts/run_24h.py --config configs/final.py --out runs/final
    python scripts/run_24h.py --out runs/final --resume              # after a crash or a kill
    python scripts/run_24h.py --out runs/final --move-deadline +45   # before a --resume on another machine
    python scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 12 --out runs/smoke

Stages, their plan and the shedding order: ``bes/train/stages.py``. Manifest, preflight, core
plan, background processes, the stage loop, signals, the library gate and the background tasks:
``bes/train/supervisor.py``. Execution plans, canary, host rate and horizon:
``bes/train/benchmark.py``. This file implements the stages on top of the project's modules:

0. ``preflight``: device and host report, disk check, core plan into the manifest.
1. ``background``: downloads (first files first), tokenisation, library build, GGUF downloads,
   the opponents' replies to the proxy prompts; supervised for the whole run.
2. ``benchmark``: real batches from a benchmark loader, plan search, canary, host rate, loader
   workers, horizon; written to the manifest and reused on every resume.
3. ``train``: ``bes.train.train.Trainer`` with the loader behind the library gate, checkpoints,
   the guard, the side-car; SIGTERM / SIGINT stop it cleanly.
4. ``sample`` (particle samples labelled by the exact verifiers), ``critic``, ``prefopt`` (one
   epoch of DPO; a rejection keeps the trained weights), ``select`` (decode settings on the dev
   half by exact scores), ``judge`` (judged comparison on the held-out half, partial when time
   runs out).
5. ``export``: ``bes.export.export_run`` when it exists, else the checkpoint, the configuration
   and the accounting report with a note saying so.

Exit status: 0 when every stage is done, 3 when interrupted (``--resume`` continues), 1 on a
failure of an essential stage.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bes.config import CriticCfg, LoaderCfg, PrefCfg, config_hash, load_config, to_dict  # noqa: E402
from bes.config import check_profile
from bes.train import stages as ST  # noqa: E402
from bes.train import supervisor as SV  # noqa: E402
from bes.train.ledger import atomic_write_json  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 3
#: Share of the preference-optimisation slice for the health samples (start and candidates).
HEALTH_TIME_FRAC = 0.25
#: The library decision of the decode selection: significance level of the sign test of the
#: factual questions (helped against hurt), the reply cap of its arms (a factual answer comes
#: early; the whole reply budget would triple the cost) and questions per alternation of the arms.
FACTUAL_ALPHA = 0.05
FACTUAL_MAX_TOKENS = 256
FACTUAL_CHUNK = 10


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Run:
    """State shared by the stages of one supervisor process."""

    def __init__(self, args: argparse.Namespace, paths: SV.RunPaths, man: dict) -> None:
        self.args = args
        self.paths = paths
        self.man = man
        self.cfg = SV.base_config(man)
        self.cp = SV.CorePlan.from_dict(man["core_plan"]) if man.get("core_plan") else SV.core_plan()
        self.bg: SV.ProcessSupervisor | None = None
        self.sup: SV.Supervisor | None = None
        self._tok = None
        self._sidecar: Any = None
        self._sidecar_ckpt: str | None = None
        self._sidecar_started = 0.0
        self._sidecar_last_s = 0.0
        #: (plan key, {shape: (fingerprint, executable)}) compiled by the benchmark of this process
        self._programs: tuple[str, dict] | None = None
        #: responses of samples.jsonl labelled again by the last _samples() (stale labels)
        self.relabelled = 0

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        if self.sup is not None:
            self.sup.log(msg)
        else:
            _say(msg)

    def reload(self) -> dict:
        self.man = SV.read_manifest(self.paths.out) or self.man
        return self.man

    def tok(self):
        if self._tok is None:
            from bes.tokenizer import load as load_tokenizer

            self._tok = load_tokenizer()
        return self._tok

    def effective(self):
        man = self.reload()
        ecfg = SV.effective_config(man)
        want = (man.get("effective") or {}).get("config_hash")
        if want and config_hash(ecfg) != want:
            raise RuntimeError(f"the effective configuration hash is {config_hash(ecfg)}, the manifest "
                               f"recorded {want}: the configuration code changed during the run")
        return ecfg

    def library_ready(self):
        """The built and self-checked library (the ready marker), else ``None`` (also while the
        marker's library is not on this disk: being rebuilt, or moved aside)."""
        if not self.cfg.data.use_library:
            return None
        path = self.library_path()
        if not path:
            return None
        from bes.library.retrieve import Library

        try:
            return Library.load(path)
        except (OSError, ValueError, KeyError) as e:
            self.log(f"library: the ready marker's library at {path} does not load ({type(e).__name__}: {e})")
            return None

    def library_path(self) -> str:
        """The ready marker's library directory when it is on this disk, else ``""``."""
        mk = SV.read_marker(self.paths, "library") or {}
        path = str(mk.get("path") or "")
        from bes.library.retrieve import library_marker

        return path if path and library_marker(path) else ""

    def _wait_for_library(self, ctx: SV.StageContext, poll_s: float = 10.0):
        """The library, waiting for the library task: a resumed training whose loader journal
        planned blocks with the library (``_journal_used_library``) cannot go on without the same
        library, and on a recycled machine the task builds it again first (the build is
        deterministic). Raises when the task failed or the stage runs out of time."""
        self.log("train: the training so far used library passages; waiting for the library task "
                 "to make the library ready on this machine")
        while True:
            ctx.check()
            lib = self.library_ready()
            if lib is not None:
                return lib
            st = ((self.bg.status() if self.bg else {}) or {}).get("library") or {}
            if st.get("state") == "failed":
                raise RuntimeError("the training journal used the library and the library task failed on "
                                   f"this machine: {st.get('error')}")
            if ctx.now() > ctx.hard_deadline - 60.0:
                # nothing more can be trained: the run goes on with the checkpoint it has (a failed
                # essential stage would end the process and its background tasks)
                raise SV.StageSkipped("the training journal used the library and it was not ready on this "
                                      "machine before the end of the training window: the newest "
                                      "checkpoint is the trained state")
            time.sleep(poll_s)

    def _wait_for_shards(self, ctx: SV.StageContext, root: Path, want: dict[str, list[list]],
                         poll_s: float = 5.0) -> None:
        """A resumed training reads the shards its loader journal lists; on a recycled machine
        they are not there until the downloads and the tokenisation have run again (their output
        does not depend on the machine or on the number of workers). Wait for them; raise when
        the tokenisation ended without them or the stage runs out of time."""
        total = sum(len(v) for v in want.values())
        told = 0.0
        while True:
            ctx.check()
            missing = _missing_shards(root, want)
            if not missing:
                if told:
                    self.log(f"train: all {total} shards of the journal are committed on this machine")
                return
            st = ((self.bg.status() if self.bg else {}) or {}).get("tokenize") or {}
            if st.get("state") in ("failed", "finished"):
                missing = _missing_shards(root, want)    # committed just before the task ended?
                if not missing:
                    return
                raise RuntimeError(f"the training journal used {total} shards, {len(missing)} of them are "
                                   f"not on this machine (e.g. {missing[0]}) and the tokenisation has "
                                   f"{st.get('state')}: {st.get('error')}")
            if ctx.now() > ctx.hard_deadline - 60.0:
                raise SV.StageSkipped(f"{len(missing)} of the {total} shards the training journal used were "
                                      "not tokenised on this machine before the end of the training window: "
                                      "the newest checkpoint is the trained state")
            if time.time() - told >= 60.0:
                told = time.time()
                self.log(f"train: waiting for the tokenisation on this machine: {len(missing)} of the "
                         f"{total} shards of the journal are not committed yet (e.g. {missing[0]})")
            time.sleep(poll_s)

    def _grafted_stream(self, man) -> dict:
        """Loader arguments of a run that continues another one under a new configuration
        (``scripts/graft_run.py``; the manifest's ``graft`` record): its data stream starts at
        progress ``p`` after ``tokens`` training tokens. The loader estimates the remaining
        horizon from all tokens of the run, and a source may be read ``max_epochs * (1 - p)``
        times in the part that is left (its reads before the graft are not in this journal).
        ``{}`` for an ordinary run."""
        from bes.config import MixCfg

        graft = man.get("graft") or {}
        if not graft:
            return {}
        out: dict = {"tokens_before": int(graft.get("tokens") or 0)}
        if graft.get("p") is not None:
            left = min(1.0, max(0.05, 1.0 - float(graft["p"])))
            out["mix"] = dataclasses.replace(MixCfg(), max_epochs=MixCfg().max_epochs * left)
        self.log(f"train: this run continues {graft.get('from_config')} from step {graft.get('step')} "
                 f"({out['tokens_before']:.3e} tokens, p {graft.get('p')}): the loader plans for the rest"
                 + (f", at most {out['mix'].max_epochs:.2f} reads of a source" if "mix" in out else ""))
        return out

    def _micro_split(self, ecfg):
        """Into how many pieces the training step cuts every micro-batch on the device
        (``bes.train.train.accumulate``: same update, smaller tensors, faster on the TPU v6e):
        one number, or ``{seq_len: pieces}``. It is not part of the configuration (the batches,
        the journal and the checkpoints do not change): the operator writes ``4`` or
        ``1024:4,2048:8`` into ``<run>/micro_split`` while the run is stopped; it is copied into
        the manifest (``micro_split``), which a snapshot carries to the next machine. 1 when
        nothing is set; a number that does not divide the rows per micro-batch of a shape of the
        schedule is not used for that shape."""
        from bes.train import schedule
        from bes.train.train import parse_micro_split, split_for

        text = str((self.reload() or {}).get("micro_split") or self.cfg.run.micro_split or "1")
        path = self.paths.out / "micro_split"
        if path.exists():
            asked = path.read_text().strip()
            try:
                parse_micro_split(asked)
            except ValueError:
                self.log(f"train: {path} holds neither a number nor seq_len:pieces pairs: ignored")
                asked = text
            if asked != text:
                SV.update_manifest(self.paths.out, micro_split=asked)
                text = asked
        want = parse_micro_split(text)
        shapes = schedule.shapes(ecfg)
        split = {}
        for a, b, t in shapes:
            k = split_for(want, t)
            if k < 1 or b % k:
                self.log(f"train: micro_split {k} does not divide the {b} rows per micro-batch at "
                         f"length {t}: those micro-batches are not cut")
                k = 1
            split[int(t)] = k
        if any(k > 1 for k in split.values()):
            self.log("train: micro-batches are cut into pieces on the device: " + ", ".join(
                f"{a} x {b} x {t} -> {a * split[t]} x {b // split[t]} x {t}" for a, b, t in shapes))
        return split

    def _catch_up_cores(self) -> tuple[int, ...]:
        """Cores for the tokenisation when the training journal lists shards this machine does
        not hold (a run continued on a recycled machine): the training waits for them, so the
        task gets the idle cores, up to 24 workers, instead of its share of the plan."""
        cp = self.cp
        try:
            want = _journal_shards(self.paths.train / LoaderCfg.journal_name)
            if not want or not _missing_shards(SV.tok_root(self.cfg, self.paths.out), want):
                return cp.tokenize
        except Exception:  # noqa: BLE001 - the train stage reports what is wrong with the data
            return cp.tokenize
        # not the library build's cores: a journal that used the library waits for it as well
        spare = [c for c in cp.cores if c not in cp.trainer and c not in cp.downloads and c not in cp.library]
        n = min(24, len(spare) - 4)
        return tuple(sorted(spare[-n:])) if n > len(cp.tokenize) else cp.tokenize

    def _busy(self, names: Sequence[str]) -> list[str]:
        """The background tasks among ``names`` that this process has not seen end yet."""
        bg = getattr(self, "bg", None)
        st = (bg.status() if bg is not None else {}) or {}
        return [n for n in names if (st.get(n) or {}).get("state") in ("waiting", "running", "backoff")]

    def _await_tasks(self, ctx: SV.StageContext, names: Sequence[str], until: float,
                     ready: Any = None, poll_s: float = 10.0) -> None:
        """A post-training stage on a machine whose background work is not done: a run continued
        on a recycled machine downloads, tokenises and builds again what a snapshot does not hold
        (the library and its ready marker, the shards, the GGUF files, the opponents' replies).
        Wait while a task of ``names`` is still running, until ``ready()`` holds or until
        ``until`` (ledger time); then the stage goes on with what there is. An uninterrupted run,
        whose tasks ended long ago, does not wait."""
        told = 0.0
        while self._busy(names) and not (ready is not None and ready()):
            if ctx.stop_requested() or ctx.now() >= until:
                return
            if time.time() - told >= 60.0:
                told = time.time()
                self.log(f"{ctx.name}: waiting for the background tasks {self._busy(names)} on this machine "
                         f"(at most {max(0.0, until - ctx.now()) / 60:.1f} min more)")
            time.sleep(poll_s)

    def tasks(self) -> list[SV.TaskSpec]:
        out, cp = self.paths.out, self.cp
        mk = SV.task_argv
        tok = self._catch_up_cores()
        if tok != cp.tokenize:
            self.log(f"background: the journal's shards are not on this machine: tokenising with "
                     f"{len(tok)} workers (the plan's share is {len(cp.tokenize)})")
        want = int(self.cfg.run.tokenize_workers)
        if want > len(tok):
            # run.tokenize_workers: cores from the top of the list that are neither the trainer's,
            # the downloads' nor the library task's (the opponents' share, idle without opponents)
            spare = [c for c in cp.cores if c not in cp.trainer and c not in cp.downloads and c not in cp.library]
            n = min(want, len(spare))
            if n > len(tok):
                tok = tuple(sorted(spare[-n:]))
                self.log(f"background: tokenising with {len(tok)} workers (run.tokenize_workers)")
        return [
            SV.TaskSpec("downloads", mk("downloads", out, 5, cp.downloads), nice=5,
                        cores=cp.downloads, threads=4, restarts=8, backoff_s=10.0),
            SV.TaskSpec("tokenize", mk("tokenize", out, 5, tok), nice=5, cores=tok,
                        threads=1, restarts=8, backoff_s=10.0),
            SV.TaskSpec("library", mk("library", out, 10, cp.library), nice=10, cores=cp.library,
                        threads=2, restarts=2, backoff_s=30.0),
            SV.TaskSpec("gguf", mk("gguf", out, 10, cp.downloads), nice=10, cores=cp.downloads,
                        threads=2, restarts=4, backoff_s=30.0),
            SV.TaskSpec("opponents", mk("opponents", out, 15, cp.opponents), after=("gguf",),
                        nice=15, cores=cp.opponents, threads=cp.opponent_threads, restarts=3,
                        backoff_s=30.0),
        ]

    def start_background(self) -> SV.ProcessSupervisor:
        if self.bg is None:
            self.bg = SV.ProcessSupervisor(self.paths.logs, self.paths.background, log=self.log)
            for spec in self.tasks():
                self.bg.add(spec)
            self.bg.start()
        return self.bg

    def close(self) -> None:
        if self._sidecar is not None and self._sidecar.poll() is None:
            self._sidecar.terminate()
        if self.bg is not None:
            self.bg.stop()

    def train_checkpoint(self) -> Path:
        """The trainer's final weights: its newest checkpoint, or the parameters-only one it
        wrote when the full final save failed (``train/ckpt_params``) if that holds a later
        step (on a tie the full checkpoint wins)."""
        from bes.decode import respond
        from bes.train import checkpoint as CK
        from bes.train.train import PARAMS_DIR

        best = None
        for rank, root in ((1, self.paths.train / "ckpt"), (0, self.paths.train / PARAMS_DIR)):
            info = CK.latest(root)
            if info is not None and (best is None or (info.step, rank) > best[0]):
                best = ((info.step, rank), info.path)
        return best[1] if best is not None else respond.find_checkpoint(self.paths.train)

    def latest_params(self, ecfg, prefer_post: bool = True):
        """``(params, consts, info, ckpt dir, source)``: the post-DPO checkpoint when accepted,
        else the trainer's final weights (:meth:`train_checkpoint`)."""
        from bes.decode import respond

        post = self.paths.post / "ckpt"
        if prefer_post and (self.man.get("prefopt") or {}).get("accepted") and post.exists():
            try:
                path = respond.find_checkpoint(post)
                params, consts, info = respond.load_checkpoint(path, ecfg.model)
                return params, consts, info, Path(path), "prefopt"
            except Exception as e:  # noqa: BLE001 - lost or broken: the trained weights, said loudly
                self.log(f"WARNING: the preference optimisation was accepted but {post} holds no loadable "
                         f"checkpoint ({type(e).__name__}: {e}); the trained weights are used")
        path = self.train_checkpoint()
        params, consts, info = respond.load_checkpoint(path, ecfg.model)
        return params, consts, info, Path(path), "train"

    # ================================================================= stage 0
    def stage_preflight(self, ctx: SV.StageContext) -> dict:
        host = SV.host_report(self.paths.out)
        dev = SV.device_report()
        warnings = SV.check_devices(dev, self.args.allow_cpu)
        cp = SV.core_plan()
        self.cp = cp
        estimate = SV.disk_need(self.cfg, tokenize_workers=len(cp.tokenize))
        need = estimate["total_gb"]
        # the data goes where the data paths point (relative to the code), not under --out
        locs = SV.data_locations(self.cfg, self.paths.out)
        drive = {k: str(v) for k, v in locs.items() if SV.on_drive(v)}
        if drive:
            msg = (f"data paths on a network drive: {drive}; run the code from a copy on the local disk "
                   "(RUNBOOK.md 1.1: downloads, shards and the library would fill the drive's quota)")
            if dev.get("backend") == "tpu":
                raise SV.PreflightError(msg)
            warnings.append(msg)
        free = {k: round(SV.free_gb(v), 1) for k, v in locs.items() if k in ("run", "data", "tok")}
        where = min(free, key=free.get)
        if free[where] < need:
            msg = (f"free disk {free[where]:.1f} GB under {locs[where]} ({where}) is below the "
                   f"estimated need of {need:.1f} GB ({estimate['rule']}; raw files deleted after "
                   f"tokenisation: {estimate['delete_raw_after_tokenize']})")
            if dev.get("backend") == "tpu":
                raise SV.PreflightError(msg + "; lower the caps (run.download_sources) or free disk")
            warnings.append(msg)
        host = {**host, "data_free_gb": free, "data_locations": {k: str(v) for k, v in locs.items()}}
        if dev.get("backend") == "tpu" and host["ram_gb"] and host["ram_gb"] < 32:
            warnings.append(f"only {host['ram_gb']:.0f} GB of host memory")
        for w in warnings:
            self.log(f"preflight WARNING: {w}")
        self.log(f"preflight: jax {dev['jax']} backend {dev['backend']} devices "
                 f"{[d['kind'] for d in dev['devices']]}; {host['cores']} cores, "
                 f"{host['ram_gb']} GB RAM, {host['disk_free_gb']} GB free disk, "
                 f"HF_TOKEN {'set' if host['hf_token'] else 'not set'}")
        init = None
        if self.cfg.run.init_from:
            # a continuation: the weights it starts from must load into this configuration now,
            # not after the data and the benchmark
            from bes.train import init_from as IF

            try:
                _, init = IF.load_params(self.cfg.run.init_from, self.cfg.model, self.cfg.run.data_dir,
                                         self.log)
            except IF.InitFromError as e:
                raise SV.PreflightError(str(e)) from e
            want = self.cfg.run.init_sha256
            if want and init["sha256"] != want:
                raise SV.PreflightError(f"init_from: weights16.npz of {init['export']} has SHA-256 "
                                        f"{init['sha256']}, the configuration expects {want}")
        self.man = SV.update_manifest(self.paths.out, host=host, devices=dev,
                                      core_plan=cp.to_dict(), preflight_warnings=warnings,
                                      disk_need_gb=round(need, 1), disk_need=estimate,
                                      **({"init_from": init} if init else {}))
        return {"backend": dev["backend"], "devices": [d["kind"] for d in dev["devices"]],
                "cores": host["cores"], "warnings": warnings}

    # ================================================================= stage 1
    def stage_background(self, ctx: SV.StageContext) -> dict:
        bg = self.start_background()
        return {"tasks": sorted(bg.specs)}

    # ================================================================= stage 2
    def _wait_for_data(self, ctx: SV.StageContext) -> None:
        """Until shards of the main web source exist (and, for a while, of a chat source)."""
        from bes.data import shards, sources

        root = SV.tok_root(self.cfg, self.paths.out) / "tok"
        plan = SV.download_plan(self.cfg)
        main = plan[0][0]
        chats = [n for n, _, _ in plan if sources.get(n).kind in ("chat", "rag", "math")]
        t_chat = ctx.started + 0.25 * ctx.slice_s
        while True:
            ctx.check()
            have_main = bool(shards.read_manifest(root / main)) if (root / main).exists() else False
            have_chat = any((root / c).exists() and shards.read_manifest(root / c) for c in chats)
            if have_main and (have_chat or not chats or ctx.now() > t_chat):
                return
            if ctx.now() > ctx.hard_deadline - 0.5 * ctx.slice_s:
                st = self.bg.status() if self.bg else {}
                raise RuntimeError(f"no training shards under {root} after "
                                   f"{ctx.now() - ctx.started:.0f} s (main source {main}: "
                                   f"{have_main}); background: "
                                   f"{ {k: (v['state'], v['error']) for k, v in st.items()} }")
            time.sleep(2.0)

    def stage_benchmark(self, ctx: SV.StageContext) -> dict:
        import jax

        from bes.data.loader import Loader
        from bes.train import benchmark as B
        from bes.train import schedule as S

        man = self.reload()
        if (man.get("benchmark") or {}).get("complete"):
            return {"reused": True, "plan": man["benchmark"]["plan"]}
        cfg = self.cfg
        self.start_background()
        self._wait_for_data(ctx)
        self.log("benchmark: training shards found; building the benchmark loader")
        workers = cfg.run.loader_workers or self.cp.loader_workers
        lcfg = dataclasses.replace(LoaderCfg(), workers=workers)
        journal = self.paths.bench / "loader_journal.jsonl"
        if journal.exists():
            journal.unlink()
        lens = B.seq_lens(cfg)
        st0 = dataclasses.replace(S.stage_at(0.0, cfg), library_frac_base=0.0, library_frac_chat=0.0)
        st1 = None
        if len(lens) > 1:
            st1 = dataclasses.replace(S.stage_at(cfg.data.late_from, cfg), library_frac_base=0.0,
                                      library_frac_chat=0.0)
        n0 = max(4, cfg.run.canary_steps)
        batches: dict[int, list] = {lens[0]: []}
        with Loader(cfg, self.tok(), SV.tok_root(cfg, self.paths.out), library=None,
                    n_workers=workers, seed=cfg.run.seed, loader_cfg=lcfg, journal=journal) as loader:
            for i in range(n0):
                batches[lens[0]].append(loader(i, st0))
            rate = B.host_rate(loader, st0, start=n0, n=max(4, 2 * lcfg.ring))
            nxt = int(rate["next_index"])
            if st1 is not None:
                batches[lens[1]] = [loader(nxt + i, st1) for i in range(3)]
        self.log(f"benchmark: loader {rate['rows_per_s']:.1f} rows/s with {workers} workers")
        backend = jax.default_backend()
        mlim = B.memory_limit_bytes(man.get("devices"))
        real_measure = B.make_measure(cfg, {t: b[:4] for t, b in batches.items()},
                                      steps=cfg.run.bench_steps,
                                      warmup=1 if cfg.run.bench_plans == "reduced" else 2,
                                      memory_limit=mlim,
                                      consts_switches=SV.parse_switches(cfg.run.switches),
                                      seed=cfg.run.seed)

        base_plan = B.Plan.of(cfg)
        canary_s = [0.0]                    # time kept for the canary, known after one measurement
        results = self.paths.bench / "results.jsonl"
        known = B.load_results(results)     # an earlier attempt's measurements (reused)
        ok_at: dict[str, set] = {}          # plan key -> lengths it ran at
        for key, t in known:
            ok_at.setdefault(key, set()).add(t)

        def measure(plan, t):
            ctx.check()                     # a stop request ends the benchmark between plans
            # until some plan runs at every length nothing is held back for the canary: without
            # a plan there is no canary (the search's essential candidates)
            have_plan = any(len(v) == len(lens) for v in ok_at.values())
            if plan != base_plan and have_plan and canary_s[0] and \
                    ctx.now() + canary_s[0] > ctx.hard_deadline:
                return B.Measurement(plan, t, False, "benchmark hard cap: the rest is kept for "
                                                     "the canary")
            m = real_measure(plan, t)
            if m.ok:
                ok_at.setdefault(plan.key, set()).add(t)
            if m.ok and not canary_s[0]:
                # safe + full runs (a ladder may add more) and the two loss compiles
                canary_s[0] = 3 * cfg.run.canary_steps * m.step_s + 2 * m.compile_s + 30.0
            return m

        soft = ctx.started + 0.65 * ctx.slice_s
        ms = B.search(cfg, measure, clock=ctx.now, soft_deadline=soft,
                      hard_deadline=ctx.hard_deadline, mode=cfg.run.bench_plans,
                      platform=backend, known=known,
                      record=lambda m: B.append_result(results, m), log=self.log)
        min_speed = cfg.run.sel_min_speed
        sel = B.select_plan(ms, lens, B.time_shares(cfg), min_speed)
        plan = sel["plan"]
        self.log(f"benchmark: chosen plan {plan.key} ({sel['expected_tps']:,.0f} tok/s expected)")
        if "sel_attn" in sel:
            d = sel["sel_attn"]
            self.log(f"benchmark: selective attention {'kept' if d['kept'] else 'switched off'}: "
                     f"{d['sel_plan']} {d['sel_tps'] or 0:,.0f} tok/s, without it {d['plain_plan']} "
                     f"{d['plain_tps'] or 0:,.0f} tok/s ({d['reason']})")
        programs = real_measure.programs         # the measured plans' compiled steps
        canary_batches = {t: batches[t] for t in lens}

        def run_canary(p):
            # every compiled shape of the run; the measurement's executables are reused
            return B.canary(cfg, p, canary_batches, steps=cfg.run.canary_steps, seed=cfg.run.seed,
                            log=self.log, programs=B.plan_programs(programs, p))

        # the canary always runs to the end: a correctness check is not cut for time
        warnings = []
        try:
            can = run_canary(plan)
            problem = None if can["parity"]["ok"] else f"parity failed for {plan.key}: {can['parity']}"
        except B.CanaryError as e:
            if plan.attn_impl != "splash":
                raise
            can, problem = None, f"canary failed for {plan.key}: {e}"
        if problem is not None:
            # the flash kernel and the dense attention disagree, or the flash plan does not train
            # at some shape: train with the dense attention
            self.log(f"benchmark: WARNING {problem}; falling back to the dense attention")
            warnings.append(problem)
            dense = None
            try:                    # the best measured dense plan (selective attention ruled)
                dense = B.select_plan([m for m in ms if m.plan.attn_impl == "xla"], lens,
                                      B.time_shares(cfg), min_speed)
                plan = dense["plan"]
            except B.BenchmarkError:
                plan = plan.flip_attn()
                extra = [real_measure(plan, t) for t in lens]
                for m in extra:
                    B.append_result(results, m)
                ms = list(ms) + extra
                if not all(m.ok for m in extra):
                    raise B.BenchmarkError(f"the dense fallback {plan.key} does not run: "
                                           f"{[m.reason for m in extra]}") from None
            sel = B.select_plan([m for m in ms if m.plan == plan], lens, B.time_shares(cfg), min_speed)
            if dense is not None and "sel_attn" in dense:
                sel["sel_attn"] = dense["sel_attn"]
            can = run_canary(plan)            # a CanaryError here: the run cannot train
        # the chosen plan's compiled steps go to the trainer of this process; the rest is freed
        self._programs = (plan.key, B.plan_programs(programs, plan))
        programs.clear()
        if not can["determinism"]["ok"]:
            self.log(f"benchmark: WARNING the same batch gave two losses ({can['determinism']}): "
                     "restarts will not reproduce the trajectory bit for bit")
            warnings.append(f"non-deterministic loss: {can['determinism']}")
        m0 = next(m for m in ms if m.plan.key == plan.key and m.seq_len == lens[0] and m.ok)
        a, b, t = m0.shape
        needed = a * b / m0.step_s
        if cfg.run.loader_workers:
            w_new = cfg.run.loader_workers
        else:
            w_new = B.loader_workers_needed(rate["rows_per_s"], workers, needed, self.cp.loader_max)
        loader_tps = rate["rows_per_s"] * w_new / max(1, workers) * lens[0]
        n_shapes = len(S.shapes(plan.apply(cfg)))
        setup_s = n_shapes * max(m.compile_s for m in ms if m.plan.key == plan.key and m.ok)
        train_s = max(0.0, ctx.train_end - ctx.now() - setup_s)
        hz = B.horizon(sel["expected_tps"], loader_tps, train_s)
        bench = {"complete": True, "plan": plan.to_dict(), "plan_key": plan.key,
                 "plan_overrides": plan.overrides(), "per_seq_len": sel["per_seq_len"],
                 "expected_tps": sel["expected_tps"], "candidates": sel["candidates"],
                 "measurements": [m.to_dict() for m in ms],
                 "canary": {k: v for k, v in can.items() if k != "ladder"},
                 "ladder": can["ladder"],
                 "host": {"rows_per_s": rate["rows_per_s"], "workers": workers,
                          "needed_rows_per_s": needed, "loader_workers": w_new,
                          "loader_tokens_per_s": loader_tps},
                 "loader_workers": w_new, "tokens_per_s": hz["tokens_per_s"],
                 "train_seconds_planned": train_s, "horizon_tokens": hz["horizon_tokens"],
                 "backend": backend, "memory_limit_bytes": mlim, "warnings": warnings,
                 "time": time.time()}
        if "sel_attn" in sel:                 # the selective-attention decision and its numbers
            bench["sel_attn"] = sel["sel_attn"]
        if can["ladder"]["rung"] != 0:
            self.log(f"benchmark: FALLBACK LADDER rung {can['ladder']['rung']}: "
                     f"{can['ladder']['groups']} switched off")
        man = SV.update_manifest(self.paths.out, benchmark=bench)
        ecfg = SV.effective_config(man)
        self.man = SV.update_manifest(self.paths.out, effective={
            "overrides": SV.run_overrides(man), "config_hash": config_hash(ecfg)})
        self.log(f"benchmark: horizon {hz['horizon_tokens']:.3e} tokens "
                 f"({hz['tokens_per_s']:,.0f} tok/s x {train_s:.0f} s); loader workers {w_new}")
        return {"plan": plan.key, "expected_tps": sel["expected_tps"],
                "horizon_tokens": hz["horizon_tokens"], "ladder_rung": can["ladder"]["rung"],
                "loader_workers": w_new}

    # ================================================================= stage 3
    def _sidecar_hook(self, state: Any, step: int) -> dict:
        """The trainer's eval hook: start a side-car on the newest checkpoint (never blocks)."""
        from bes.train import checkpoint as CK

        if not self.cfg.run.sidecar:
            return {"sidecar": "off"}
        now = time.time()
        if self._sidecar is not None and self._sidecar.poll() is None:
            if now - self._sidecar_started > max(60.0, self.cfg.run.eval_every_min * 60.0):
                self._sidecar.kill()
                return {"sidecar": "killed: ran longer than the evaluation interval"}
            return {"sidecar": "previous report still running"}
        if self._sidecar is not None:
            self._sidecar_last_s = now - self._sidecar_started
        left = self.sup.train_end - self.sup.ledger.now() if self.sup else 0.0
        if left < max(120.0, 3.0 * self._sidecar_last_s):
            return {"sidecar": f"skipped: late ({left:.0f} s of training left)"}
        info = CK.latest(self.paths.train / "ckpt")
        if info is None or str(info.path) == self._sidecar_ckpt:
            return {"sidecar": "no new checkpoint"}
        self._sidecar_ckpt = str(info.path)
        argv = SV.task_argv("sidecar", self.paths.out, 19, self.cp.sidecar, ["--ckpt", str(info.path)])
        env = dict(os.environ)
        env.update({"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "BES_SUPERVISOR_PID": str(os.getpid())})
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_sidecar_log", None) is not None:
            self._sidecar_log.close()
        self._sidecar_log = open(self.paths.logs / "sidecar.log", "ab")
        import subprocess

        self._sidecar = subprocess.Popen(list(argv), stdout=self._sidecar_log,
                                         stderr=subprocess.STDOUT, env=env, cwd=str(REPO),
                                         start_new_session=True)
        self._sidecar_started = now
        return {"sidecar": f"started on {info.name}"}

    def stage_train(self, ctx: SV.StageContext) -> dict:
        from bes.data.loader import Loader
        from bes.model import model as M
        from bes.roles import role_of
        from bes.train import checkpoint as CK
        from bes.train.train import JsonlLog, Trainer

        man = self.reload()
        bench = man.get("benchmark") or {}
        if not bench.get("complete"):
            raise SV.StageSkipped("no benchmark result in the manifest: without an execution plan "
                                  "and a horizon there is no training")
        ecfg = self.effective()
        if self.args.restart_training:
            if ctx.now() - self.sup.ledger.t0 > ecfg.run.restart_window_hours * 3600.0:
                raise RuntimeError(f"--restart-training refused: {(ctx.now() - self.sup.ledger.t0) / 3600:.2f} h "
                                   f"into the run, the window is {ecfg.run.restart_window_hours} h "
                                   "(after it only rollbacks to checkpoints are allowed)")
            if self.paths.train.exists():
                aside = self.paths.out / f"train.discarded-{int(time.time())}"
                os.replace(self.paths.train, aside)
                self.log(f"train: restart from scratch; the previous training moved to {aside}")
            self.args.restart_training = False
        lib = self.library_ready()
        if lib is None and self.cfg.data.use_library and \
                _journal_used_library(self.paths.train / LoaderCfg.journal_name):
            lib = self._wait_for_library(ctx)
        want = _journal_shards(self.paths.train / LoaderCfg.journal_name)
        if want:
            self._wait_for_shards(ctx, SV.tok_root(ecfg, self.paths.out), want)
        workers = int(bench.get("loader_workers") or self.cp.loader_workers)
        lcfg = dataclasses.replace(LoaderCfg(), workers=workers,
                                   horizon_tokens=float(bench["horizon_tokens"]) or LoaderCfg.horizon_tokens)
        self.paths.train.mkdir(parents=True, exist_ok=True)
        base_log = JsonlLog(self.paths.train / "train_log.jsonl", verbose=False)
        loader = Loader(ecfg, self.tok(), SV.tok_root(ecfg, self.paths.out), library=lib,
                        n_workers=workers, seed=ecfg.run.seed, loader_cfg=lcfg,
                        journal=self.paths.train / LoaderCfg.journal_name, **self._grafted_stream(man))
        gate = SV.LibraryGate(loader, self.library_ready, active=lib is not None, log=self.log)
        n_rec = [0]

        def log_fn(rec: dict) -> None:
            if rec.get("event") == "step":
                n_rec[0] += 1
                try:
                    rec = dict(rec, loader_block=loader.stats()["block"], library=gate.active)
                except Exception as e:  # noqa: BLE001 - logging only
                    rec = dict(rec, loader_block={"error": repr(e)})
                if n_rec[0] % 10 == 1:
                    self.log(f"train: step {rec['step']} p {rec['p']:.3f} loss {rec['loss']:.4f} "
                             f"T {rec['seq_len']} tok/s {rec.get('tok_per_s', 0):,.0f}")
            else:
                self.log(f"train: {rec.get('event')} " + " ".join(
                    f"{k}={rec[k]}" for k in ("step", "reason", "path", "p", "result") if k in rec))
            base_log(rec)

        # the benchmark's compiled steps of the chosen plan (same process): used when the
        # trainer's own lowering is the same program (Trainer checks the fingerprints)
        programs = None
        if self._programs is not None and self._programs[0] == bench.get("plan_key"):
            programs = self._programs[1]
        split = self._micro_split(ecfg)
        init_fn = M.init_params
        if ecfg.run.init_from:
            from bes.train import init_from as IF

            init_fn = IF.make_init_fn(ecfg.run.init_from, ecfg.run.data_dir, self.log)
        trainer = Trainer(ecfg, loss_fn=M.loss_fn, init_fn=init_fn, data=gate,
                          role_fn=role_of, consts_fn=SV.make_consts_fn(ecfg.run.switches),
                          out_dir=self.paths.train, deadline=self.sup.deadline,
                          eval_fn=self._sidecar_hook, log_fn=log_fn, async_ckpt=True, resume=True,
                          programs=None if any(k > 1 for k in split.values()) else programs,
                          micro_split=split, train_end=self.sup.train_end)
        remove = self.sup.on_stop(trainer.request_stop)
        try:
            if self.sup.stop_requested:
                raise SV.Interrupted("stop requested before training started")
            t0 = time.time()
            trainer.setup()
            self._programs = None             # the trainer holds what it uses
            reused = sum(1 for c in trainer.compile_info.values() if c["reused"])
            self.log(f"train: set-up (compile {len(trainer.shapes)} shapes, {reused} reused from "
                     f"the benchmark, restore) took {time.time() - t0:.1f} s; step {trainer.step}")
            summary = trainer.run()           # CheckpointError: no final state on disk -> fails
        finally:
            remove()
            loader.close()
        if summary["reason"] == "requested":
            raise SV.Interrupted(f"training stopped at step {summary['step']} on request "
                                 f"(final checkpoint {summary['checkpoint']})")
        if summary["final_kind"] == "params":
            self.log(f"train: WARNING only the parameters of the final state were saved "
                     f"({summary['checkpoint']}); the later stages use them")
        last = summary["checkpoint"]          # the checkpoint holding the final state
        if last is None:
            info = CK.latest(self.paths.train / "ckpt")
            last = None if info is None else str(info.path)
        return {"reason": summary["reason"], "step": summary["step"], "p_end": summary["p_end"],
                "tokens": summary["ledger"]["tokens"], "library_active": gate.active,
                "library_at_index": gate.activated_at_index,
                "checkpoint": None if last is None else Path(last).name,
                "final_kind": summary["final_kind"], "skipped_updates": summary["skipped_updates"],
                "compile_reused": reused, "ledger_seconds": summary["ledger"]["seconds"]}

    # ================================================================= stage 4
    def _prompt_item(self, name: str, msgs: list, rng, item_id: str, heldout: bool) -> dict | None:
        """A sampling prompt from one conversation (its first user turn answered by the
        assistant): ``kind`` math (GSM8K, with its reference answer), constraint (explicit, or a
        verified hindsight constraint for half of the plain chat prompts) or chat; ``None`` when
        the conversation does not serve."""
        from bes.data import hindsight
        from bes.eval import exact_score
        from bes.exact import constraints as C

        idx = next((j for j, m in enumerate(msgs) if m["role"] == "assistant"), None)
        if idx is None or idx == 0 or msgs[idx - 1]["role"] != "user":
            return None
        head = [m for m in msgs[:idx] if m["role"] in ("system", "user")]
        if not isinstance(head[-1]["content"], str):
            return None
        reply = hindsight.response_text(msgs[idx]["content"], self.tok()) or ""
        if exact_score.canned_refusal(reply):
            # the data refuses this request: no preference is sampled on it (a refusal counts as
            # a failed reply in the labels, and that must only teach about harmless requests)
            return None
        item = {"id": item_id, "source": name, "heldout": bool(heldout),
                "messages": [{"role": m["role"], "content": str(m["content"])} for m in head],
                "reference": None, "kind": "chat", "reply": msgs[idx]["content"]}
        user = head[-1]["content"]
        if name == "gsm8k":
            got = exact_score.extract_answer(reply)
            if got["value"] is None:
                return None
            v = got["value"]
            item["reference"] = str(v.numerator) if v.denominator == 1 else repr(float(v))
            item["kind"] = "math"
        elif not C.compile(user).empty:
            item["kind"] = "constraint"
        elif rng.random() < 0.5:
            res = hindsight.relabel(head + [msgs[idx]], rng, self.tok())
            if res is not None:
                item["messages"][-1] = {"role": "user", "content": res[-2]["content"]}
                item["kind"] = "constraint"
        return item

    def _sample_prompts(self, limit: int) -> list[dict]:
        """Prompts for preference sampling from the downloaded chat and math sources (see
        :meth:`_prompt_item`). They come first from the loader's held-out split
        (``bes.data.loader.heldout_samples``: conversations the training stream never reads);
        a source whose held-out conversations do not fill its share is topped up from the start
        of its newest downloaded file, rows that are in the training universe (``heldout`` tells
        which is which)."""
        import hashlib

        import numpy as np

        from bes.data import download, sources
        from bes.data.loader import heldout_samples
        from bes.data.samples import tokens_to_messages

        names = [n for n, _, _ in SV.download_plan(self.cfg)
                 if sources.get(n).kind in ("chat", "math") and sources.get(n).fmt == "messages"]
        names = list(dict.fromkeys(names))
        per = max(4, limit // max(1, len(names)))
        pools: list[list[dict]] = []
        rng = np.random.default_rng(self.cfg.run.seed + 17)
        root = SV.tok_root(self.cfg, self.paths.out)
        for name in names:
            src = sources.get(name)
            out: list[dict] = []
            try:
                held = heldout_samples(root, name, per, seed=self.cfg.run.seed + 17,
                                       data=self.cfg.data)
            except Exception as e:  # noqa: BLE001 - no shards yet: the raw rows only
                self.log(f"sample: no held-out conversations of {name}: {type(e).__name__}: {e}")
                held = []
            for smp in held:
                try:
                    msgs = tokens_to_messages(smp.ids, self.tok())
                except ValueError:
                    continue
                key = hashlib.sha1(np.asarray(smp.ids, np.int32).tobytes()).hexdigest()[:16]
                item = self._prompt_item(name, msgs, rng, f"{name}/heldout/{key}", heldout=True)
                if item is not None:
                    out.append(item)
            n_held = len(out)
            try:
                files = download.completed_files(name, self.cfg.run.data_dir) if len(out) < per else []
            except Exception:  # noqa: BLE001 - nothing downloaded
                files = []
            if files:
                for batch in download.iter_records(files[-1].local, list(src.columns), batch_rows=256):
                    for rec in batch:
                        try:
                            msgs = sources.adapt(src, rec)
                        except Exception:  # noqa: BLE001 - a bad row
                            continue
                        if not msgs:
                            continue
                        item = self._prompt_item(name, msgs, rng,
                                                 f"{name}/{files[-1].index}/{len(out) - n_held}",
                                                 heldout=False)
                        if item is not None:
                            out.append(item)
                        if len(out) >= per:
                            break
                    if len(out) >= per:
                        break
            if out:
                self.log(f"sample: {name}: {n_held} held-out prompts, {len(out) - n_held} from the "
                         "training rows")
                pools.append(out)
        mixed = []
        for i in range(max((len(p) for p in pools), default=0)):
            mixed += [p[i] for p in pools if i < len(p)]
        return mixed[:limit]

    def stage_sample(self, ctx: SV.StageContext) -> dict:
        from bes.config import END
        from bes.decode import respond, smc
        from bes.decode import sampling as SMP
        from bes.decode.lm import OurLM
        from bes.model import generate as G

        ecfg = self.effective()
        if self._busy(("library", "tokenize")):    # the passages and the held-out conversations
            self._await_tasks(ctx, ("library", "tokenize"), ctx.started + 0.3 * ctx.slice_s)
        params, consts, info, _, _ = self.latest_params(ecfg, prefer_post=False)
        lib = self.library_ready() if ecfg.decode.use_library else None
        tok = self.tok()
        resp = respond.Responder(lambda n: None, tok, ecfg, library=lib)
        r = respond.resolve(ecfg.decode, None)
        d = r.decode
        dec = G.jitted(ecfg.model)
        out_path = self.paths.post / "samples.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
        prompts = self._sample_prompts(4000 if ecfg.model.d_model >= 512 else 64)
        if not prompts:
            raise SV.StageSkipped("no chat prompts on disk (no chat source downloaded)")
        n_new = n_pairs = 0
        budget = resp.context - d.max_new_tokens
        t0 = time.time()
        with open(out_path, "a") as f:
            for k, item in enumerate(prompts):
                if item["id"] in done:
                    continue
                if ctx.should_stop():
                    break
                try:
                    msgs = respond.clean_messages(item["messages"])
                except ValueError:
                    continue
                user = msgs[-1]["content"]
                spec, _ = resp.compile_spec(user)
                # as Responder.reply retrieves: the earlier user turns are the context
                history = [m["content"] for m in msgs[:-1] if m["role"] == "user"]
                passage = resp.retrieve(user, history) if d.use_library else None
                try:
                    prompt, _fit = respond.fit_prompt(resp.chat, msgs,
                                                      None if passage is None else passage.text, budget)
                except ValueError:
                    continue
                if len(prompt) + 64 > min(PrefCfg.seq_len, ecfg.model.max_seq) + 1:
                    continue        # a DPO row (prompt + response + END <= T + 1) has no room for a reply
                try:
                    program = resp.program(spec, r)
                except Exception:  # noqa: BLE001 - structural potentials only
                    program = resp.program(None, r)
                # the run's decode settings, explicitly: the self-evidence weight here, the
                # temperature through the settings (smc.decode hands it to the model)
                lm = OurLM(params, consts, ecfg.model, tok, max_steps=4 * d.max_new_tokens, decoder=dec,
                           self_evidence_weight=d.self_evidence_weight)
                res = smc.decode(lm, prompt, n_particles=d.particles, max_new_tokens=d.max_new_tokens,
                                 seed=1000 + k, settings=SMP.SamplingSettings.from_cfg(d),
                                 program=program, tok=tok, ess_frac=d.ess_frac,
                                 resample_at=d.resample_at, unit_max_tokens=d.unit_max_tokens,
                                 no_repeat_ngram=d.no_repeat_ngram, max_copy_run=d.max_copy_run)
                # a particle that ended with its whole budget used was cut by the budget (the
                # budget-aware potentials force END into its last slot): labelled with the finish
                # Responder.reply reports for it ("length"), as the exact scores of select and judge
                parts = [_labelled_part(user, [int(x) for x in p.tokens if int(x) != END],
                                        "length" if p.finish == "stop" and len(p.tokens) >= d.max_new_tokens
                                        else p.finish, item["reference"], "particle", resp.chat.display)
                         for p in res.particles]
                ref = _reference_response(item, tok)
                if ref is not None:            # the data set's own reply, labelled the same way
                    parts.append(_labelled_part(user, ref["tokens"], "stop", item["reference"],
                                                "reference", resp.chat.display))
                scores = [q["score"] for q in parts]
                n_pairs += min(sum(1 for s in scores if s > min(scores)),
                               sum(1 for s in scores if s < max(scores)))
                rec = {"id": item["id"], "source": item["source"], "kind": item["kind"],
                       "heldout": item["heldout"],
                       "user": user, "reference": item["reference"], "seed": 1000 + k,
                       "prompt": [int(x) for x in prompt], "particles": parts,
                       "library": None if passage is None else passage.title}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
                n_new += 1
        ctx.check()
        n_total = len(done) + n_new
        rate = n_new / max(1e-9, time.time() - t0)
        self.log(f"sample: {n_new} new prompts ({n_total} in all), about {n_pairs} pairs, "
                 f"{rate * 60:.1f} prompts/min")
        return {"prompts_new": n_new, "prompts": n_total, "pairs_new_upper": n_pairs,
                "pool": len(prompts), "partial": n_total < len(prompts), "checkpoint": info["path"]}

    def _samples(self) -> list[dict]:
        """The sampling records, every response's label checked against the text of its own
        tokens (:func:`_scored_samples`; a stale label is computed again and counted in
        ``self.relabelled``)."""
        from bes.data.chat_format import display_text

        path = self.paths.post / "samples.jsonl"
        out = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                         # a torn last line
        tok = self.tok()
        out, self.relabelled = _scored_samples(out, lambda ids: display_text(ids, tok))
        if self.relabelled:
            self.log(f"samples: {self.relabelled} responses labelled again from their tokens (their "
                     "stored label was not the label of their own text)")
        return out

    def stage_critic(self, ctx: SV.StageContext) -> dict:
        import numpy as np

        from bes.data import prefs
        from bes.train import critic_fit

        samples = [s for s in self._samples() if len({p["score"] for p in s["particles"]}) > 1]
        if len(samples) < 2:
            raise SV.StageSkipped(f"{len(samples)} prompts with differing labels: nothing to rank")
        ecfg = self.effective()
        params, consts, _, _, _ = self.latest_params(ecfg, prefer_post=False)
        rows, groups, quality = [], [], []
        for gi, s in enumerate(samples):
            for p in s["particles"]:
                rows.append(prefs.row_tokens(s["prompt"], p["tokens"]))
                groups.append(gi)
                quality.append(float(p["score"]))
        ccfg = dataclasses.replace(CriticCfg(), seq_len=min(CriticCfg.seq_len, ecfg.model.max_seq),
                                   steps=CriticCfg.steps if len(rows) > 64 else 50)
        new_params, rep = critic_fit.run_critic_fit(params, consts, rows, ecfg, groups=np.asarray(groups),
                                                    quality=np.asarray(quality), ccfg=ccfg, tok=self.tok())
        crit = {k: np.asarray(v) for k, v in new_params["critic"].items()}
        np.savez(self.paths.post / "critic.npz", **crit)
        keep = {k: rep[k] for k in ("rho_heldout", "rho_train", "use_critic", "n_heldout_groups",
                                    "n_train_groups", "loss_start", "loss_end", "n_invalid_rows")
                if k in rep}
        keep["relabelled"] = self.relabelled
        atomic_write_json(self.paths.post / "critic_report.json", SV._jsonable(rep))
        self.log(f"critic: held-out rho {rep.get('rho_heldout')}, gate passed: {rep.get('use_critic')} "
                 "(the decoder has no critic path; recorded only)")
        return keep

    def _sample_stats_fn(self, ecfg, consts, prompts: list[list[int]], max_new: int,
                         budget_s: float | None = None, min_prompts: int = 8):
        """``fn(params) -> prefopt.response_stats``: one plain sample (the run's temperature and
        top-p, no potentials) per held-out prompt, prompt ``i`` with seed ``7 + i`` for every
        parameter set, so the start's and each candidate's statistics are paired prompt by
        prompt. The first call (the start) samples the prompts in order until ``budget_s``
        seconds are used (at least ``min_prompts``); every later call samples exactly those."""
        from bes.decode import sampling as SMP
        from bes.decode import smc
        from bes.decode.lm import OurLM
        from bes.model import generate as G
        from bes.train.prefopt import response_stats

        dec = G.jitted(ecfg.model)
        tok = self.tok()
        settings = SMP.SamplingSettings.from_cfg(ecfg.decode)
        fixed: dict[str, int | None] = {"n": None}

        def fn(params) -> dict:
            outs = []
            t0 = time.time()
            n = len(prompts) if fixed["n"] is None else fixed["n"]
            for i, pr in enumerate(prompts[:n]):
                if (fixed["n"] is None and budget_s is not None and i >= min(min_prompts, n)
                        and time.time() - t0 > budget_s):
                    break
                lm = OurLM(params, consts, ecfg.model, tok, max_steps=4 * max_new, decoder=dec,
                           self_evidence_weight=ecfg.decode.self_evidence_weight)
                res = smc.decode(lm, pr, n_particles=1, max_new_tokens=max_new, seed=7 + i,
                                 settings=settings, tok=tok)
                outs.append(res.particles[0].tokens)
            if fixed["n"] is None:
                fixed["n"] = len(outs)
            st = response_stats(outs, max_new=max_new)
            st["seconds"] = round(time.time() - t0, 1)
            return st

        return fn

    def stage_prefopt(self, ctx: SV.StageContext) -> dict:
        import numpy as np

        from bes.data import prefs
        from bes.exact.features import exact_features
        from bes.train import checkpoint as CK
        from bes.train import prefopt

        ecfg = self.effective()
        params, consts, info, _, _ = self.latest_params(ecfg, prefer_post=False)
        t = min(PrefCfg.seq_len, ecfg.model.max_seq)
        ppm = max(1, min(PrefCfg.pairs_per_micro, ecfg.optim.micro_seqs * ecfg.data.seq_len // (2 * t)))
        # the micro-batch of the execution plan bounds the rows of one DPO micro-batch; when it
        # halves them (optim.micro_seqs = 16 at 1,024 tokens: 8 pairs), twice the micro-batches
        # keep the pairs per optimiser step, and so the step the learning rates were chosen for
        n_micro = PrefCfg.n_micro * min(2, max(1, PrefCfg.pairs_per_micro // ppm))
        # PairBatches keeps the feature batches between passes only when all of them fit; with
        # UltraFeedback that is ~20 GB, so a 1 GB cache meant computing the exact features of every
        # row on one host core again in each pass (reference, each candidate, each held-out scoring)
        cache_gb = max(1.0, min(32.0, 0.2 * SV._meminfo().get("MemAvailable", 0.0)))
        # the rows' exact features are host work (19 ms a pair on one core in the 24-hour run):
        # the cores the background tasks no longer use compute them before the first pass
        n_cores = len(self.cp.cores) if getattr(self, "cp", None) is not None else (os.cpu_count() or 1)
        fw = min(24, n_cores - 8) if n_cores >= 12 else 0
        if os.environ.get("BES_PREF_WORKERS", "").strip().isdigit():     # rehearsals on small hosts
            fw = int(os.environ["BES_PREF_WORKERS"])
        pref = dataclasses.replace(PrefCfg(), seq_len=t, pairs_per_micro=ppm, n_micro=n_micro,
                                   cache_gb=cache_gb, feature_workers=fw,
                                   feature_timeout_s=max(60.0, 0.3 * ctx.left()))
        sampled = []
        for s in self._samples():               # labels checked against their tokens' text
            toks = [p["tokens"] for p in s["particles"]]
            sampled += prefs.pairs_from_scores(s["prompt"], toks, [p["score"] for p in s["particles"]],
                                               source="sampled")
        relabelled = self.relabelled
        groups: dict[str, list] = {"sampled": sampled}
        ratio: dict[str, float] = {"sampled": 0.0}
        if self._busy(("tokenize",)):              # UltraFeedback is the last source it tokenises
            self._await_tasks(ctx, ("tokenize",), ctx.started + 0.25 * ctx.slice_s)
        uf = SV.tok_root(ecfg, self.paths.out) / "tok" / "ultrafeedback"
        if uf.exists():
            pairs, counts = prefs.pairs_from_shards(uf)
            groups["ultrafeedback"] = pairs
            ratio["ultrafeedback"] = PrefCfg.max_len_ratio
        # the larger held-out share is for tiny pair sets only: with UltraFeedback in the groups,
        # a quarter of every pair would be held out of the epoch because few were sampled
        n_all = sum(len(g) for g in groups.values())
        split = prefs.prepare(groups, t, heldout_frac=pref.heldout_frac if n_all >= 200 else 0.25,
                              seed=pref.seed, max_len_ratio=ratio)
        if not split.heldout and len({bytes(q.prompt) for q in split.train}) >= 2:
            # a tiny set whose hash split held nothing out: hold out the pairs of one prompt
            first = bytes(split.train[0].prompt)
            split = prefs.PrefSplit([q for q in split.train if bytes(q.prompt) != first],
                                    [q for q in split.train if bytes(q.prompt) == first],
                                    dict(split.stats, note="one prompt held out by hand"))
        if not split.train or not split.heldout:
            raise SV.StageSkipped(f"not enough preference pairs (train {len(split.train)}, held out "
                                  f"{len(split.heldout)}; filter counts {split.stats})")
        tps = float((self.man.get("benchmark") or {}).get("tokens_per_s") or 0.0)
        lrs = tuple(float(x) for x in str(ecfg.run.prefopt_lrs).replace(",", " ").split()) or (1e-6, 3e-6)
        full_split = split

        def fit_slice(sp, workers: int):
            """``sp`` cut to the pairs that fit what is left of the slice: per pair one
            reference pass and one epoch per learning rate on the chip (measured in the
            24-hour run: an epoch costs 1.05 x the training rate), the host's feature work,
            and the health sampling's share of the slice."""
            if tps <= 0:
                return sp
            host = 0.019 / max(1.0, 0.8 * workers)
            per_pair = 2 * t * (1.05 * len(lrs) + 0.4) / tps + host
            usable = (0.9 - HEALTH_TIME_FRAC) * ctx.left() - 30.0
            n_max = max(pref.pairs_per_micro * pref.n_micro, int(usable / per_pair))
            if len(sp.train) <= n_max:
                return sp
            self.log(f"prefopt: {len(sp.train)} training pairs, {n_max} fit in the slice "
                     f"({workers} feature processes)")
            order = prefs.shuffled_order(len(sp.train), pref.seed)[:n_max]
            return prefs.PrefSplit([sp.train[i] for i in sorted(order)], sp.heldout, sp.stats)

        split = fit_slice(full_split, pref.feature_workers)
        held_rows = []
        for p in split.heldout[:64]:
            rows = prefs.pair_rows(p, t, pref.mask_forced)
            held_rows.append({k: v[0] for k, v in rows.items()})       # the chosen response
        tok = self.tok()
        batches = []
        for i in range(0, len(held_rows), 8):
            chunk = held_rows[i:i + 8]
            b = {k: np.stack([r[k] for r in chunk]) for k in ("tokens", "w", "seg")}
            b.update(exact_features(b["tokens"][:, :t], b["seg"], tok))
            batches.append(b)
        heldout_loss = prefopt.make_heldout_loss_fn(batches, consts, ecfg.model)
        # health samples: up to health_prompts distinct held-out prompts at the run's own reply
        # limit (a shorter limit cuts most chat replies and hides a length change); the start's
        # sampling may take HEALTH_TIME_FRAC of the slice shared by the start and every candidate
        prompts, seen = [], set()
        for p in split.heldout:
            if bytes(p.prompt) not in seen and len(prompts) < pref.health_prompts:
                seen.add(bytes(p.prompt))
                prompts.append(list(map(int, p.prompt)) + [_assistant()])
        stats_fn = self._sample_stats_fn(ecfg, consts, prompts, ecfg.decode.max_new_tokens,
                                         budget_s=HEALTH_TIME_FRAC * ctx.left() / (1 + len(lrs)))
        try:
            new_params, rep = prefopt.run_prefopt(params, split, ecfg, lrs=lrs, heldout_loss_fn=heldout_loss,
                                                  sample_stats_fn=stats_fn, pref=pref, consts=consts,
                                                  tok=tok, log=self.log)
        except prefopt.FeatureWarmError as e:
            # the pool did not work on this host: this process computes the features, for the
            # pairs that then still fit
            self.log(f"prefopt: WARNING the parallel feature computation failed ({e}); going on in one process")
            pref = dataclasses.replace(pref, feature_workers=0)
            split = fit_slice(full_split, 0)
            new_params, rep = prefopt.run_prefopt(params, split, ecfg, lrs=lrs, heldout_loss_fn=heldout_loss,
                                                  sample_stats_fn=stats_fn, pref=pref, consts=consts,
                                                  tok=tok, log=self.log)
        # a stop requested during the DPO (run_prefopt has no checks) is honoured at the next
        # stage boundary: the finished result is saved and recorded first, never thrown away
        out = {"accepted": bool(rep["accepted"]), "lr": rep.get("lr"), "reason": rep.get("reason"),
               "n_train_pairs": rep.get("n_train_pairs"), "n_heldout_pairs": rep.get("n_heldout_pairs"),
               "pairs_sampled": len(sampled), "pairs_by_source": split.stats,
               "labels_recomputed": relabelled, "from_checkpoint": info["path"]}
        atomic_write_json(self.paths.post / "prefopt_report.json", SV._jsonable(rep))
        if rep["accepted"]:
            root = self.paths.post / "ckpt"
            CK.save(root, {"state": {"params": new_params, "consts": consts}}, step=int(info["step"] or 0),
                    config_hash=config_hash(ecfg), meta={"source": "prefopt", "lr": rep.get("lr")}, keep=1)
            self.log(f"prefopt: accepted (lr {rep.get('lr')}); post-DPO weights saved")
        else:
            self.log(f"prefopt: not accepted, the trained weights are kept: {rep.get('reason')}")
        self.man = SV.update_manifest(self.paths.out, prefopt=out)
        return out

    def _generator(self, ecfg, ckpt: Path):
        """One generator (model loaded once, one compiled decoder shared by every reply)."""
        from bes.decode import respond

        gen = respond.our_generator(str(ckpt), str(self.man["config_abs"]), self.library_path(),
                                    SV.generator_overrides(self.man))
        orig = gen.responder.lm_factory
        shared: dict[str, Any] = {}

        def lm_factory(n: int):
            lm = orig(n)
            if "dec" in shared:
                lm.dec = shared["dec"]
            else:
                shared["dec"] = lm.dec
            return lm

        gen.responder.lm_factory = lm_factory
        return gen

    def stage_select(self, ctx: SV.StageContext) -> dict:
        from bes.eval.proxy import __main__ as cli
        from bes.eval.proxy import prompts as prompt_set
        from bes.eval.proxy import runner
        from bes.eval.proxy.run_ours import CallableBackend

        ecfg = self.effective()
        if ecfg.decode.use_library and self._busy(("library",)):
            # without the library on this disk the rule below records "no library was built"
            self._await_tasks(ctx, ("library",), ctx.started + 0.4 * ctx.slice_s,
                              ready=lambda: bool(self.library_path()))
        _, _, info, ckpt, source = self.latest_params(ecfg)
        gen = self._generator(ecfg, ckpt)
        backend = CallableBackend(gen)
        root = SV.proxy_root(ecfg, self.paths.out)
        ps = prompt_set.load()
        todo = prompt_set.select(ps, "dev", None, ecfg.run.eval_limit or None)
        base = _decode_settings(ecfg)
        has_lib = bool(self.library_path()) and ecfg.decode.use_library
        variants = [("default", base)]
        if ecfg.decode.particles > 1:
            variants.append(("k1", base + ["extra.particles=1"]))
        if has_lib:
            variants.append(("nolib", base + ["extra.use_library=false"]))
        variants.append(("noband", base + ["extra.band=false"]))
        # where the gate fires on no turn, the library-off arm gives the default's very reply (same
        # prompt, same seed): it is generated only on the prompts where the gate fires (6 of the 200
        # dev prompts) and takes the default's score elsewhere
        responder = getattr(gen, "responder", None)
        fired = {p.id for p in todo if has_lib and responder is not None and any(
            responder.retrieve(u, list(p.turns[:t])) is not None for t, u in enumerate(p.turns))}
        # the factual library measurement (_library_decision) gets its share after the variants
        fact_s = float(ecfg.run.factual_frac) * max(0.0, ctx.deadline - ctx.now()) if has_lib else 0.0
        results = {}
        for i, (name, ov) in enumerate(variants):
            left = ctx.deadline - fact_s - ctx.now()
            minutes = left / 60.0 / (len(variants) - i)
            if minutes <= 0.05 or ctx.stop_requested():
                results[name] = {"skipped": "no time left"}
                continue
            spec = cli.build_spec("ours", "default", ov, f"select-{name}", 0, None, None, 1, None,
                                  "bes.decode.respond:our_generator", {}, gen.key)
            part = [p for p in todo if p.id in fired] if name == "nolib" else todo
            res = runner.run(part, spec, root, workers=0, minutes=minutes, all_prompts=ps,
                             backend=backend, log=self.log)
            rep = cli.exact_report(root, res.run_id, "dev")
            per = _per_prompt_scores(root, res.run_id, "dev")
            results[name] = {"run_id": res.run_id, "done": res.done, "summary_math": rep.get("math"),
                             "summary_constraints": rep.get("constraints"),
                             "summary_degenerate": rep.get("degenerate"), "per_prompt": per,
                             "overrides": ov}
        ctx.check()
        if "per_prompt" in results.get("nolib", {}) and "per_prompt" in results.get("default", {}):
            same = {p: s for p, s in results["default"]["per_prompt"].items() if p not in fired}
            results["nolib"]["per_prompt"] = {**same, **results["nolib"]["per_prompt"]}
            results["nolib"]["from_default"] = len(same)
        ran = {k: v for k, v in results.items() if "per_prompt" in v}
        common = set.intersection(*(set(v["per_prompt"]) for v in ran.values())) if ran else set()
        scores = {k: (sum(v["per_prompt"][p] for p in common) / len(common) if common else None)
                  for k, v in ran.items()}
        # the decoding settings other than the library: the default stays unless a variant is
        # better than it on significantly more of the common prompts than worse (exact two-sided
        # sign test below FACTUAL_ALPHA, as the library); among those the highest mean exact score.
        # The library is decided on its own evidence (it cannot win by a tie)
        from bes.eval.proxy.score import paired_sign_test

        cands = {k: v for k, v in scores.items() if k != "nolib" and v is not None}
        vs_default = {k: paired_sign_test({p: ran[k]["per_prompt"][p] for p in common},
                                          {p: ran["default"]["per_prompt"][p] for p in common})
                      for k in cands if k != "default" and "default" in ran}
        better = {k: cands[k] for k, t in vs_default.items()
                  if t["a_better"] > t["b_better"] and t["p"] < FACTUAL_ALPHA}
        best = max(better, key=lambda k: better[k]) if common and better else "default"
        chosen = list(dict(variants)[best])
        lib = self._library_decision(ctx, ecfg, gen, root, base, scores, len(common))
        if has_lib and not lib["ship"]:
            chosen = [o for o in chosen if not o.startswith("extra.use_library=")] + ["extra.use_library=false"]
            best = "nolib" if best == "default" else f"{best}+nolib"
        out = {"chosen": best, "overrides": chosen, "scores": scores, "n_common": len(common),
               "n_dev": len(todo), "checkpoint": str(ckpt), "weights": source,
               "runs": {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"} for k, v in results.items()},
               "library": lib,
               "vs_default": vs_default,
               "rule": "decoding settings: the default unless a variant's exact score (math correct, "
                       "constraints satisfied, not degenerate) is higher on significantly more of the dev "
                       f"prompts every variant answered than lower (sign test p < {FACTUAL_ALPHA}); then "
                       "the highest mean among those; the library: the rule of library.rule (not shipped: "
                       "switched off)"}
        self.man = SV.update_manifest(self.paths.out, decode=out)
        self.log(f"select: decode variant {best} (scores {scores} on {len(common)} common prompts); "
                 f"library {'shipped' if lib['ship'] else 'not shipped'}: {lib['reason']}")
        return {**{k: out[k] for k in ("chosen", "scores", "n_common", "n_dev")},
                "library": {k: lib.get(k) for k in ("ship", "reason", "factual", "factual_error", "dev")}}

    def _library_decision(self, ctx: SV.StageContext, ecfg, gen, root: Path, base: list[str],
                          scores: Mapping[str, Any], n_common: int) -> dict:
        """Whether the library ships (plan section 2, ``docs/EVAL_PROTOCOL.md`` sections 8 and
        10): only when the factual questions answered with the library on and off
        (``bes.eval.proxy.factual``, :meth:`_factual`) show a significant gain (more helped than
        hurt, exact two-sided sign test below :data:`FACTUAL_ALPHA`) and the dev prompts show
        no loss (mean exact score with the library on at least as high as off). No measurement,
        no gain: not shipped."""
        dec: dict[str, Any] = {"rule": f"ship the library only if the factual questions show a significant "
                                       f"gain (helped > hurt, exact two-sided sign test p < {FACTUAL_ALPHA}) "
                                       "and the dev prompts show no loss (mean exact score with the library "
                                       "on >= off over the prompts every variant answered)",
                               "ship": False, "factual": None, "dev": None}
        if not self.library_path():
            return {**dec, "reason": "no library was built"}
        if not ecfg.decode.use_library:
            return {**dec, "reason": "decode.use_library is off"}
        on, off = scores.get("default"), scores.get("nolib")
        dec["dev"] = {"on": on, "off": off, "n_common": n_common}
        summary = None
        try:
            summary, extra = self._factual(ctx, ecfg, gen, root, base)
        except Exception as e:  # noqa: BLE001 - no measurement: no evidence of a gain
            dec["factual_error"] = f"{type(e).__name__}: {e}"
            self.log(f"select: the factual library measurement did not run: {dec['factual_error']}")
        if summary is not None:
            o = summary["overall"]
            dec["factual"] = {"n_questions": summary["n_questions"], "n_paired": summary["n_paired"],
                              "fired": summary["fired"]["n"], "helped": o["helped"], "hurt": o["hurt"],
                              "p": o["sign_test_p"], "lib_acc": o["lib_acc"], "nolib_acc": o["nolib_acc"],
                              "stopped_early": summary["stopped_early"],
                              "not_fired_identical": summary["not_fired_identical"],
                              "not_fired": summary["not_fired"]["n"],
                              "by_set": {s: {k: v["all"][k] for k in ("n", "helped", "hurt", "sign_test_p")}
                                         for s, v in summary["by_set"].items()},
                              "lib_run": summary["lib_run"], "nolib_run": summary["nolib_run"], **extra}
        gain = summary is not None and summary["overall"]["helped"] > summary["overall"]["hurt"] \
            and summary["overall"]["sign_test_p"] < FACTUAL_ALPHA
        no_loss = on is not None and off is not None and on >= off
        dec["ship"] = bool(gain and no_loss)
        why = []
        if summary is None:
            why.append("no factual measurement")
        elif not gain:
            o = summary["overall"]
            why.append(f"no significant factual gain (helped {o['helped']}, hurt {o['hurt']}, p {o['sign_test_p']})")
        if on is None or off is None:
            why.append("the dev variants with and without the library did not both run")
        elif not no_loss:
            why.append(f"dev exact score lower with the library ({on:.4f} < {off:.4f})")
        dec["reason"] = "; ".join(why) if why else (
            f"factual gain helped {summary['overall']['helped']} against hurt {summary['overall']['hurt']} "
            f"(p {summary['overall']['sign_test_p']}), dev exact score {on:.4f} with the library, {off:.4f} without")
        return dec

    def _factual(self, ctx: SV.StageContext, ecfg, gen, root: Path, base: list[str]) -> tuple[dict, dict]:
        """``python -m bes.eval.proxy factual`` in this process for the decode selection: the
        fixed question set (``run.factual_per_set`` per evaluation split and head-entity probes;
        ``run.eval_limit`` replaces it), both arms of the run's own generator with the run's
        decode settings and replies capped at :data:`FACTUAL_MAX_TOKENS` tokens, until the
        stage's deadline. The questions on which the gate fires come first (elsewhere both arms
        give the same reply, so they cannot tip the sign test). Returns the report's summary and
        what the record adds (question file, notes)."""
        from bes.eval.proxy import __main__ as cli
        from bes.eval.proxy import factual as F
        from bes.eval.proxy.run_ours import CallableBackend

        if (ctx.deadline - ctx.now()) / 60.0 <= 0.05 or ctx.stop_requested():
            raise RuntimeError("no time left in the decode selection's slice")
        n = int(ecfg.run.eval_limit or ecfg.run.factual_per_set)
        lib_dir, cache = self.library_path(), Path(ecfg.run.data_dir) / "library"
        notes = []
        try:
            path, qs = F.prepare(root, per_set=n, n_probes=n, seed=ecfg.run.seed, cache_dir=cache,
                                 library_dir=lib_dir, log=self.log)
        except ValueError as e:                 # the probes need the library's title importance
            path, qs = F.prepare(root, per_set=n, n_probes=0, seed=ecfg.run.seed,
                                 sets=tuple(s for s in F.SETS if s != "probe"), cache_dir=cache,
                                 library_dir=lib_dir, log=self.log)
            notes.append(f"no head-entity probes: {e}")
        lib = self.library_ready()
        fired = {q.id for q in qs if lib is not None and lib.retrieve(q.q) is not None}
        qs = sorted(qs, key=lambda q: q.id not in fired)            # stable: fired questions first
        # elsewhere both arms give the same reply and cannot tip the sign test: ten of them keep
        # the report's check that they are identical, the rest of the slice goes to later stages
        qs = qs[:len(fired) + 10]
        ov = [o for o in base if not o.startswith(("extra.use_library=", "max_tokens="))]
        ov.append(f"max_tokens={min(ecfg.decode.max_new_tokens, FACTUAL_MAX_TOKENS)}")
        specs = [cli.build_spec("ours", preset, ov, f"select-factual-{arm}", 0, None, None, 1, None,
                                "bes.decode.respond:our_generator", {}, gen.key)
                 for preset, arm in (("default", "lib"), ("nolib", "nolib"))]
        minutes = (ctx.deadline - ctx.now()) / 60.0
        if minutes <= 0.05 or ctx.stop_requested():
            raise RuntimeError("no time left in the decode selection's slice after building the questions")
        summary = F.run_factual(qs, specs[0], specs[1], root, backend=CallableBackend(gen), minutes=minutes,
                                chunk=FACTUAL_CHUNK, name=f"{self.paths.out.name}-factual", log=self.log)
        return summary, {"questions": str(path), "gate_fires_on": len(fired), "order": "gate-fired questions first",
                         "max_tokens": min(ecfg.decode.max_new_tokens, FACTUAL_MAX_TOKENS),
                         "report": str(root / "reports" / "factual" / f"{self.paths.out.name}-factual.md"),
                         "notes": notes}

    def stage_judge(self, ctx: SV.StageContext) -> dict:
        from bes.eval.proxy import __main__ as cli
        from bes.eval.proxy import make_ab, prompts as prompt_set, runner
        from bes.eval.proxy.judge import run_judge
        from bes.eval.proxy.run_ours import CallableBackend

        ecfg = self.effective()
        _, _, info, ckpt, source = self.latest_params(ecfg)
        root = SV.proxy_root(ecfg, self.paths.out)
        ps = prompt_set.load()
        todo = prompt_set.select(ps, "test", None, ecfg.run.eval_limit or None)

        def test_runs() -> dict:
            mk = SV.read_marker(self.paths, "opponents") or SV.read_marker(self.paths, "opponents_progress") or {}
            return {k: v for k, v in (mk.get("runs") or {}).items() if k.endswith("/test") and v.get("done")}

        if self._busy(("gguf", "opponents")):     # replies of a recycled machine come again first
            n_arms = len(ecfg.run.eval_opponents.replace(",", " ").split())
            self._await_tasks(ctx, ("gguf", "opponents"), ctx.started + 0.5 * ctx.slice_s,
                              ready=lambda: len(test_runs()) >= n_arms)
        opp_runs = test_runs()
        if not opp_runs:
            raise SV.StageSkipped("no opponent replies on the test split yet (background task "
                                  f"'opponents': {(self.bg.status() if self.bg else {}).get('opponents')})")
        dec = (self.man.get("decode") or {})
        ov = list(dec.get("overrides") or _decode_settings(ecfg))
        gen = self._generator(ecfg, ckpt)
        spec = cli.build_spec("ours", "default", ov, f"final-{dec.get('chosen', 'default')}", 0, None,
                              None, 1, None, "bes.decode.respond:our_generator", {}, gen.key)
        judge = None
        out: dict[str, Any] = {"n_test": len(todo), "weights": source, "opponents": {}, "rounds": 0}
        # rounds of generation (45% of what is left) and judging, until the slice is used: a judge
        # faster than the generation no longer leaves the end of the slice idle (the pair sets grow,
        # published pairs and verdicts are kept, the harness cache resumes the generation)
        while True:
            res = runner.run(todo, spec, root, workers=0, minutes=max(0.05, 0.45 * ctx.left() / 60.0),
                             all_prompts=ps, backend=CallableBackend(gen), log=self.log)
            x_records, _ = runner.load_outputs(root, res.run_id)
            ctx.check()
            if judge is None:
                judge = cli.make_judge(ecfg.run.eval_judge, self.cp.judge_threads, None, None, True)
            out.update(ours=res.run_id, ours_done=res.done, rounds=out["rounds"] + 1)
            for i, (arm, run) in enumerate(sorted(opp_runs.items())):
                name = arm.split("/")[0].replace("=", "-")
                y_records, _ = runner.load_outputs(root, run["run_id"])
                pairset = f"{self.paths.out.name}-ours-vs-{name}-test"
                try:
                    make_ab.make_pairs(todo, x_records, y_records, res.run_id, run["run_id"], pairset, 0, root)
                except ValueError as e:
                    out["opponents"][arm] = {"skipped": str(e)}
                    continue
                pairs, _ = make_ab.load_pairs(root, pairset)
                minutes = (ctx.deadline - ctx.now()) / 60.0 / (len(opp_runs) - i)
                if minutes > 0.02 and not ctx.stop_requested():
                    run_judge(pairs, judge, pairset, root, minutes, log=self.log)
                rows = {}
                for src in cli.sources_of(root, pairset):
                    s = cli.score_report(root, pairset, src)["overall"]
                    rows[src] = {"n": s["n"], "score": s.get("score"), "ci95": s.get("score_ci95"),
                                 "win_rate_decisive": s.get("win_rate_decisive"),
                                 "partial": s["n"] < len(todo)}
                out["opponents"][arm] = {"pairset": pairset, "pairs": len(pairs), "scores": rows,
                                         "partial": len(pairs) < len(todo) or any(r["partial"] for r in rows.values())}
            if (res.new_generations == 0 or res.done >= len(todo) or ctx.left() < 120.0
                    or ctx.stop_requested()):
                break
        self.man = SV.update_manifest(self.paths.out, judged=out)
        self.log(f"judge: {json.dumps(out['opponents'])[:400]}")
        return out

    # ================================================================= stage 5
    def stage_export(self, ctx: SV.StageContext) -> dict:
        man = self.reload()
        final = self.paths.final
        final.mkdir(parents=True, exist_ok=True)
        try:
            ecfg = self.effective() if (man.get("benchmark") or {}).get("complete") else self.cfg
        except Exception:  # noqa: BLE001
            ecfg = self.cfg
        try:
            _, _, info, ckpt, source = self.latest_params(ecfg)
        except (FileNotFoundError, ValueError) as e:
            note = f"no checkpoint to export: {e}"
            if any(final.iterdir()):            # a note inside an earlier export breaks its SHA256SUMS
                self.log(f"export: {note}; {final} keeps the earlier export it holds")
            else:
                (final / "EXPORT_NOTE.txt").write_text(note + "\n")
            return {"exported": False, "note": note}
        if ((man.get("decode") or {}).get("library") or {}).get("ship") is True and not self.library_path() \
                and self._busy(("library",)):
            # the decode selection shipped the library and this machine is building it again (a
            # snapshot holds neither the library nor its marker): an export now would leave it out
            self._await_tasks(ctx, ("library",), ctx.run_deadline - ctx.slice_s,
                              ready=lambda: bool(self.library_path()))
            if not self.library_path():
                self.log("export: WARNING the decode selection shipped the library, but it is not ready "
                         "on this machine: exported without it")
        reports = {"stages": str(self.paths.stages), "manifest": str(self.paths.manifest),
                   "health": str(self.paths.health), "post": str(self.paths.post),
                   "proxy": str(SV.proxy_root(ecfg, self.paths.out) / "reports")}
        try:
            from bes import export as EX  # the tools track's exporter
            fn = getattr(EX, "export_run")
        except (ImportError, AttributeError) as e:
            fn, why = None, f"{type(e).__name__}: {e}"
        decode = man.get("decode")
        if self.library_path() and "ship" not in ((decode or {}).get("library") or {}):
            if ecfg.run.ship_library == "always":
                decode = {**(decode or {}), "library": {"ship": True, "rule": "run.ship_library",
                                                        "reason": "the library is part of the model "
                                                                  "(run.ship_library = always)"}}
            else:
                # no decode selection (it failed, was shed or has not run): no measurement, no gain
                decode = {**(decode or {}), "library": {"ship": False, "reason": "the decode selection made no "
                                                        "library decision (no measurement, no gain)"}}
        if fn is not None:
            avail = {"checkpoint": str(ckpt), "config": str(man["config"]), "overrides": SV.run_overrides(man),
                     "horizon_tokens": SV.horizon_of(man),
                     "library": self.library_path() or None, "final_dir": str(final), "reports": reports,
                     "decode": decode, "cfg": ecfg}
            params = inspect.signature(fn).parameters
            kw = {k: v for k, v in avail.items() if k in params}
            try:
                res = fn(self.paths.out, **kw)
            except Exception as e:  # noqa: BLE001 - the final weights leave the run all the same
                why = f"export_run failed: {type(e).__name__}: {e}"
                self.log(f"export: {why}; writing the fallback export of the {source} weights instead")
            else:
                return {"exported": True, "by": "bes.export.export_run", "checkpoint": str(ckpt),
                        "weights": source, "result": SV._jsonable(res) if isinstance(res, Mapping) else str(res)}
        # fallback: the parameters (no optimiser state; atomic, readable by
        # bes.decode.respond.load_checkpoint), the configuration, the accounting and a note
        from bes.train import checkpoint as CK

        if any(final.iterdir()):
            # an earlier export (an emergency one, of older weights) must neither stay mixed with
            # these files nor be packaged and fetched in their place
            aside = final.with_name(f"{final.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
            os.replace(final, aside)
            final.mkdir()
            self.log(f"export: the earlier export moved to {aside}")
        params, consts, info, _, _ = self.latest_params(ecfg)
        root = final / "ckpt"
        saved = CK.save(root, {"state": {"params": params, "consts": consts}},
                        step=int(info.get("step") or 0), config_hash=config_hash(ecfg),
                        meta={"weights": source, "from": str(ckpt)}, keep=1)
        dst = saved.path
        # a library the decode selection did not keep is neither shipped nor counted
        shipped = "" if ((decode or {}).get("library") or {}).get("ship") is False \
            else self.library_path()
        acc = _accounting(ecfg, dst, shipped)
        atomic_write_json(final / "config.json", SV._jsonable({
            "config": man["config"], "overrides": SV.run_overrides(man),
            "horizon_tokens": SV.horizon_of(man), "config_hash": config_hash(ecfg),
            "cfg": to_dict(ecfg), "decode_choice": man.get("decode"), "weights": source,
            "library": shipped or None}))
        atomic_write_json(final / "accounting.json", SV._jsonable(acc))
        note = (f"bes/export.py (export_run) did not export ({why}); this directory holds the "
                f"checkpoint ({source} weights), the configuration and the accounting report only. "
                f"The tokenizer is assets/tokenizer/ of the repository; the library is "
                f"{shipped or 'none'}; reports: {reports}.")
        (final / "EXPORT_NOTE.txt").write_text(note + "\n")
        self.log(f"export: {note}")
        return {"exported": True, "by": "fallback", "checkpoint": str(dst), "weights": source,
                "note": note, "total_numbers": acc.get("total"), "active_step": acc.get("active_step")}


def _assistant() -> int:
    from bes.config import ASSISTANT
    return ASSISTANT


def post_plan_run(run_cfg, out: Path, log=None):
    """The run settings the stage plan is made from. They are the configuration's, unless the
    operator changed the post-training plan of a run that is under way: the minutes of every
    post-training stage and of the buffer, written into ``<run>/post_plan`` while the run is
    stopped (``"sample=3 critic=0 prefopt=60 select=25 judge=0 export=5 buffer=15"``, the syntax
    of ``run.post_minutes``; the reserve after training is their sum). The text is copied into
    the manifest (``post_plan``), which a snapshot carries to the next machine. The
    configuration and its hash do not change: only the stage plan and the end of training do
    (the schedule is ``T / (T + remaining)``, so it follows the new end by itself)."""
    man = SV.read_manifest(out) or {}
    text = str(man.get("post_plan") or "")
    path = Path(out) / "post_plan"
    if path.exists():
        asked = " ".join(path.read_text().split())
        if asked != text:
            try:
                ST.post_minutes(dataclasses.replace(run_cfg, post_minutes=asked))
            except ValueError as e:
                if log:
                    log(f"post_plan: {path} is not a plan ({e}): ignored")
            else:
                SV.update_manifest(out, post_plan=asked)
                text = asked
    if not text:
        return run_cfg
    minutes = ST.post_minutes(dataclasses.replace(run_cfg, post_minutes=text))
    total = sum(minutes.values())
    if log:
        log(f"post_plan: the post-training plan of this run is {text} ({total:.0f} minutes after "
            f"training; the configuration's reserve was {run_cfg.post_reserve_hours * 60:.0f})")
    return dataclasses.replace(run_cfg, post_minutes=text, post_reserve_hours=total / 60.0)


def fast_switches_run(out: Path, log=None, default: str = "") -> str:
    """Apply the speed switches of a run (``bes.model.fast``): the kernel and the hand-written
    backward passes its programs are traced with. They are not configuration (same values to
    float rounding; the batches, the journal, the checkpoints and the hash do not change): the
    operator writes e.g. ``kernel=onetile:save relu2=1 ce=1`` into ``<run>/fast`` while the run
    is stopped; the text is copied into the manifest (``fast``), which a snapshot carries to the
    next machine. Called once, before any stage traces a program; returns the text in force
    (``""``: nothing was asked, the switches keep their defaults)."""
    from bes.model import fast

    man = SV.read_manifest(out) or {}
    text = str(man.get("fast") or "")
    if not text and default:
        # the configuration's switches (run.fast) for a run whose operator set none
        try:
            fast.parse(default)
        except ValueError as e:
            if log:
                log(f"fast: run.fast is not a setting ({e}): ignored")
        else:
            text = " ".join(default.split())
            SV.update_manifest(out, fast=text)
    path = Path(out) / "fast"
    if path.exists():
        asked = " ".join(path.read_text().split())
        if asked != text:
            try:
                fast.parse(asked)
            except ValueError as e:
                if log:
                    log(f"fast: {path} is not a setting ({e}): ignored")
            else:
                SV.update_manifest(out, fast=asked)
                text = asked
    if not text:
        return ""
    fast.apply(fast.parse(text))
    if log:
        log(f"fast: the speed switches of this run are {text} (now on: {fast.describe()})")
    return text


def _journal_shards(path: Path) -> dict[str, list[list]]:
    """Every shard the loader journal at ``path`` has shown to a block, as the journal lists
    them: ``{source: [[id, n_segments, n_tokens, ...], ...]}`` (a loader reopening the journal
    reads these very shards from disk: ``bes.data.loader``)."""
    out: dict[str, list[list]] = {}
    if not path.exists():
        return out
    with open(path, errors="replace") as f:
        for line in f:
            if '"new_shards"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue                         # a torn last line
            if rec.get("type") != "block":
                continue
            for name, entries in (rec.get("new_shards") or {}).items():
                out.setdefault(name, []).extend(entries)
    return out


def _missing_shards(root: Path, want: dict[str, list[list]]) -> list[str]:
    """Which shards of ``want`` (:func:`_journal_shards`) are not committed under ``root/tok``
    (``"source/id"``). Raises when a committed shard has another size than the journal's: the
    data on this machine is not the data the run was trained on."""
    from bes.data import shards as SH

    missing = []
    for name, entries in want.items():
        directory = Path(root) / "tok" / name
        man = SH.read_manifest(directory) if directory.exists() else {}
        for e in entries:
            m = man.get(int(e[0]))
            if m is None:
                missing.append(f"{name}/{int(e[0])}")
            elif int(m.get("n_segments", -1)) != int(e[1]) or int(m.get("n_tokens", -1)) != int(e[2]):
                raise RuntimeError(f"shard {name}/{int(e[0])}: this machine tokenised {m.get('n_segments')} "
                                   f"segments and {m.get('n_tokens')} tokens, the training journal used "
                                   f"{e[1]} and {e[2]}: the data differs, the training cannot be continued "
                                   "on it")
    return missing


def _journal_used_library(path: Path) -> bool:
    """The loader journal at ``path`` planned a block with a library (a loader reopening it
    must have the same library: ``bes.data.loader``)."""
    if not path.exists():
        return False
    with open(path, errors="replace") as f:
        for line in f:
            if '"library"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue                         # a torn last line
            if rec.get("type") == "block" and rec.get("library") is not None:
                return True
    return False


def _reference_response(item: Mapping[str, Any], tok) -> dict | None:
    """The data set's own reply to a sampling prompt as ``{"tokens", "text"}`` (assistant
    content without the role token and ``END``; ``text`` is the whole display text of those
    tokens), or ``None``: a hindsight constraint was verified on it, a GSM8K reply carries the
    reference answer."""
    from bes.data.chat_format import display_text, render_turn

    content = item.get("reply")
    if not content:
        return None
    try:
        ids, _ = render_turn("assistant", content, tok)
    except (ValueError, TypeError):
        return None
    body = [int(x) for x in ids[1:-1]]
    return {"tokens": body, "text": display_text(body, tok).strip()} if body else None


#: characters of a sampled response kept in ``samples.jsonl`` for reading (the label is computed
#: on the whole text of its tokens, never on this cut)
SAMPLE_TEXT_CHARS = 2000


def _text_sha1(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _labelled_part(user: str, tokens: Sequence[int], finish: str, reference: str | None, origin: str,
                   display) -> dict:
    """One response of a sampling record: ``tokens`` (what a preference pair is built from)
    and the exact label (``SV.label_sample``) of their whole display text ``display(tokens)``;
    ``text`` keeps the first :data:`SAMPLE_TEXT_CHARS` characters for reading and
    ``text_sha1`` identifies the text that was scored (:func:`_scored_samples` checks it)."""
    text = display(list(tokens)).strip()
    lab = SV.label_sample(user, text, finish, reference)
    return {"tokens": [int(x) for x in tokens], "text": text[:SAMPLE_TEXT_CHARS], "text_chars": len(text),
            "text_sha1": _text_sha1(text), "finish": finish, "origin": origin, **lab}


def _scored_samples(samples: Sequence[Mapping[str, Any]], display) -> tuple[list[dict], int]:
    """``samples`` with every response's label guaranteed to be the label of the text of its
    own tokens: a response whose ``text_sha1`` is missing (a record written before labels were
    tied to the tokens) or differs from ``display(tokens)`` is labelled again from its tokens.
    Returns ``(samples, number of responses labelled again)``."""
    out, n = [], 0
    for s in samples:
        parts = []
        for p in s["particles"]:
            text = display(list(p["tokens"])).strip()
            if p.get("text_sha1") != _text_sha1(text):
                p = _labelled_part(s["user"], p["tokens"], p.get("finish", "stop"), s.get("reference"),
                                   p.get("origin", "particle"), display)
                n += 1
            parts.append(p)
        out.append({**s, "particles": parts})
    return out, n


def _decode_settings(ecfg) -> list[str]:
    """Harness settings that make an arm of ours decode with the run's own ``DecodeCfg``
    (the harness presets take the defaults of ``DecodeCfg``)."""
    d = ecfg.decode
    return [f"extra.particles={d.particles}", f"extra.use_library={str(d.use_library).lower()}",
            f"max_tokens={d.max_new_tokens}", f"temperature={d.temperature!r}", f"top_p={d.top_p!r}"]


def _per_prompt_scores(root: Path, run_id: str, split: str) -> dict[str, float]:
    """Per-prompt exact score in ``[0, 1]`` from the exact report: the mean of math correct,
    constraints satisfied and not degenerate (whichever apply)."""
    path = root / "reports" / "exact" / f"{run_id.replace('/', '__')}-{split}.json"
    data = json.loads(path.read_text()) if path.exists() else {"per_prompt": []}
    out = {}
    for row in data.get("per_prompt", []):
        parts = []
        if row.get("math_correct") is not None:
            parts.append(1.0 if row["math_correct"] else 0.0)
        if row.get("constraints_ok") is not None:
            parts.append(1.0 if row["constraints_ok"] else 0.0)
        deg = any(t.get("degenerate", {}).get("degenerate") for t in row.get("turns", []))
        parts.append(0.0 if deg else 1.0)
        out[row["prompt_id"]] = sum(parts) / len(parts)
    return out


def _accounting(ecfg, ckpt: Path, library: str) -> dict:
    """``bes.budget.count`` of the exported parameters plus the library, and the read-set audit
    (``bes.readset.ReadLogger``) over one probe sequence."""
    import numpy as np

    from bes import budget
    from bes.decode import respond

    params, consts, _ = respond.load_checkpoint(ckpt, ecfg.model)
    assets = None
    if library:
        try:
            from bes.library.retrieve import Library

            lib = Library.load(library)
            acc = lib.accounting()
            # the library's own worst-case read bound of one retrieval (index included)
            assets = budget.Assets(library_numbers=int(acc["total_numbers"]) - int(acc["index_numbers"]),
                                   index_numbers=int(acc["index_numbers"]),
                                   library_read=int(lib.read_bound()["total"]), index_read=0)
        except Exception:  # noqa: BLE001 - no library assets in the count
            assets = None
    b = budget.count(params, ecfg.model, assets)
    out = {"total": b.total, "active_step": b.active_step, "within_caps": b.within_caps(),
           "ledger": {k: list(v) for k, v in b.ledger().items()}}
    try:
        from bes.exact.features import exact_features
        from bes.readset import ReadLogger
        from bes.tokenizer import load as load_tokenizer

        tok = load_tokenizer()
        text = "The library opens at nine. It closes at five on weekdays and at noon on Saturdays. " * 4
        ids = np.asarray([[_bos()] + tok.encode(text)], np.int32)[:, :128]
        seg = np.zeros_like(ids)
        feats = exact_features(ids, seg, tok)
        rl = ReadLogger(params, ecfg.model, consts)
        rl.log_sequence({"tokens": ids[0], "seg": seg[0], **{k: v[0] for k, v in feats.items()}})
        out["readset"] = rl.summary()
    except Exception as e:  # noqa: BLE001 - the audit is a report, not a gate
        out["readset"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _bos() -> int:
    from bes.config import BOS
    return BOS


# ==================================================================================== main
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configs/<name>.py (a new run)")
    ap.add_argument("--out", default=None, help="run directory (default: run.out_dir of the config)")
    ap.add_argument("--override", action="append", default=[], help="section.field=value")
    ap.add_argument("--resume", action="store_true", help="continue the run in --out (same budget)")
    ap.add_argument("--allow-cpu", action="store_true", help="accept a CPU-only host (rehearsals)")
    ap.add_argument("--budget-minutes", type=float, default=None,
                    help="the run's whole budget in minutes (sets run.deadline_hours)")
    ap.add_argument("--restart-training", action="store_true",
                    help="discard the training so far and start it again (inside the restart window)")
    ap.add_argument("--move-deadline", default=None, metavar="WHEN",
                    help="move the deadline of the run in --out and exit, without running anything: "
                         "seconds since the epoch, or +MINUTES / -MINUTES relative to the present "
                         "deadline (a run that continues on another machine: RUNBOOK.md section 5)")
    return ap.parse_args(argv)


def parse_deadline(when: str, present: float) -> float:
    """``--move-deadline``: an absolute time in seconds since the epoch, or ``+M`` / ``-M``
    minutes from the present deadline."""
    text = str(when).strip()
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"--move-deadline {when!r}: give seconds since the epoch, or +MINUTES / -MINUTES") from None
    return present + 60.0 * value if text[:1] in "+-" else value


def configure_jax(paths: SV.RunPaths, log=_say) -> dict:
    """Enable JAX's persistent compilation cache under ``<run>/jax_cache``: a restarted
    supervisor (and the trainer after the benchmark, whose program it is) loads the compiled
    training steps instead of compiling them again. Guarded: when an option is unknown to this
    JAX or the backend cannot use the cache, that is logged and the run goes on without it.
    Returns ``{"dir", "enabled", "reason"}`` (``enabled`` is None when it could not be checked)."""
    out: dict[str, Any] = {"dir": str(paths.jax_cache), "enabled": False, "reason": ""}
    try:
        import jax

        paths.jax_cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(paths.jax_cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception as e:  # noqa: BLE001 - an unknown option or an unwritable directory
        out["reason"] = f"{type(e).__name__}: {e}"
        log(f"run_24h: the persistent compilation cache is not enabled ({out['reason']}); "
            "going on without it")
        try:
            jax.config.update("jax_compilation_cache_dir", None)
        except Exception:  # noqa: BLE001
            pass
        return out
    try:
        from jax._src import compilation_cache as cc      # the backend check JAX itself uses

        out["enabled"] = bool(cc.is_cache_used(jax.devices()[0].client))
        if not out["enabled"]:
            out["reason"] = f"backend {jax.default_backend()} does not use the persistent cache"
            log(f"run_24h: {out['reason']}; going on without it")
    except Exception as e:  # noqa: BLE001 - the check is internal API: report, keep the setting
        out.update(enabled=None, reason=f"could not check the cache: {type(e).__name__}: {e}")
        log(f"run_24h: persistent compilation cache set to {paths.jax_cache}; {out['reason']}")
    return out


def new_manifest(args: argparse.Namespace, out: Path) -> dict:
    if not args.config:
        raise SystemExit("a new run needs --config")
    overrides = list(args.override)
    if args.budget_minutes is not None:
        overrides.append(f"run.deadline_hours={args.budget_minutes / 60.0!r}")
    cfg = load_config(args.config, overrides)
    check_profile(cfg.data)                   # BES_PROFILE must be the configuration's profile
    t0 = time.time()
    budget_s = cfg.run.deadline_hours * 3600.0
    if budget_s <= cfg.run.post_reserve_hours * 3600.0:
        raise SystemExit(f"the budget ({budget_s:.0f} s) does not exceed the post-training reserve "
                         f"({cfg.run.post_reserve_hours * 3600:.0f} s)")
    return {"format": SV.FORMAT, "config": str(args.config),
            "config_abs": str(Path(args.config).resolve()), "overrides": overrides,
            "budget": {"t0": t0, "deadline": t0 + budget_s, "budget_s": budget_s},
            "argv": list(sys.argv), "created": t0, "code": SV.code_commit(),
            "config_hash": config_hash(cfg), "resumes": []}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out is None:
        if args.config is None:
            raise SystemExit("give --out (and --config for a new run)")
        args.out = load_config(args.config).run.out_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = SV.RunPaths(out)
    lock = SV.RunLock(paths.lock).acquire()
    try:
        man = SV.read_manifest(out)
        if args.move_deadline is not None:
            if man is None:
                raise SystemExit(f"--move-deadline: no run in {out} (no manifest.json)")
            try:
                res = SV.move_deadline(out, parse_deadline(args.move_deadline, float(man["budget"]["deadline"])),
                                       reason="--move-deadline " + str(args.move_deadline))
            except ValueError as e:
                raise SystemExit(f"--move-deadline: {e}") from None
            _say(f"run_24h: deadline of {out} moved by {res['moved_s'] / 60:+.1f} min to "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(res['to']))} ({res['left_s'] / 3600:.2f} h "
                 f"from now; training until {time.strftime('%H:%M:%S', time.localtime(res['train_end']))}); "
                 f"rewritten: {', '.join(res['files'])}")
            return EXIT_OK
        if man is not None and not args.resume:
            raise SystemExit(f"{out} already holds a run (manifest.json); continue it with --resume "
                             "or choose another --out")
        if man is None:
            if args.resume:
                raise SystemExit(f"--resume: no run in {out} (no manifest.json)")
            man = new_manifest(args, out)
            SV.write_manifest(out, man)
        else:
            if args.config and Path(args.config).resolve() != Path(man["config_abs"]):
                raise SystemExit(f"--resume: the run uses {man['config']}, not {args.config}")
            if args.budget_minutes is not None and abs(
                    args.budget_minutes * 60.0 - float(man["budget"]["budget_s"])) > 1e-6:
                raise SystemExit("--resume: the budget of a run is fixed when it starts")
            if args.override and list(args.override) != [o for o in man["overrides"]
                                                         if not o.startswith("run.deadline_hours=")]:
                raise SystemExit("--resume: overrides are fixed when the run starts")
            man.setdefault("resumes", []).append(time.time())
            SV.write_manifest(out, man)
        man = SV.update_manifest(out, jax_cache=configure_jax(paths))
        run = Run(args, paths, man)
        cfg = run.cfg
        budget = man["budget"]
        plan_run = post_plan_run(cfg.run, out, run.log)
        fast_switches_run(out, run.log, cfg.run.fast)    # before any stage traces a program
        specs = ST.stage_specs(plan_run, float(budget["budget_s"]))
        fns = {"preflight": run.stage_preflight, "background": run.stage_background,
               "benchmark": run.stage_benchmark, "train": run.stage_train,
               "sample": run.stage_sample, "critic": run.stage_critic,
               "prefopt": run.stage_prefopt, "select": run.stage_select,
               "judge": run.stage_judge, "export": run.stage_export}
        sup = SV.Supervisor(out, specs, fns, t0=float(budget["t0"]), deadline=float(budget["deadline"]),
                            post_reserve_s=plan_run.post_reserve_hours * 3600.0)
        run.sup = sup
        sup.install_signal_handlers()
        if man.get("devices"):                  # a resumed run: check this machine again
            warnings = SV.check_devices(SV.device_report(), args.allow_cpu)
            for w in warnings:
                sup.log(f"preflight WARNING: {w}")
        if sup.ledger.done("background"):
            run.start_background()
        try:
            res = sup.run()
        finally:
            run.close()
        sup.log("stage table:\n" + sup.ledger.format_table(specs))
        atomic_write_json(out / "stage_table.json", sup.ledger.table(specs))
        return EXIT_OK if res["status"] == "complete" else EXIT_INTERRUPTED
    except SV.FatalStageError as e:
        _say(f"run_24h: {e}")
        return EXIT_FAILED
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
