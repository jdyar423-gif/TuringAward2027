"""The single 24-hour run (``docs/PLAN.md`` sections 2 and 6): one budget of wall time for
installation checks, downloads, tokenisation, the library, the start-up benchmark, training,
post-training, evaluation and export.

    python scripts/run_24h.py --config configs/final.py --out runs/final
    python scripts/run_24h.py --out runs/final --resume              # after a crash or a kill
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
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bes.config import CriticCfg, LoaderCfg, PrefCfg, config_hash, load_config, to_dict  # noqa: E402
from bes.train import stages as ST  # noqa: E402
from bes.train import supervisor as SV  # noqa: E402
from bes.train.ledger import atomic_write_json  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 3


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
        """The built and self-checked library (the ready marker), else ``None``."""
        if not self.cfg.data.use_library:
            return None
        mk = SV.read_marker(self.paths, "library")
        if not mk or not mk.get("path"):
            return None
        from bes.library.retrieve import Library

        return Library.load(str(mk["path"]))

    def library_path(self) -> str:
        mk = SV.read_marker(self.paths, "library") or {}
        return str(mk.get("path") or "")

    def tasks(self) -> list[SV.TaskSpec]:
        out, cp = self.paths.out, self.cp
        mk = SV.task_argv
        return [
            SV.TaskSpec("downloads", mk("downloads", out, 5, cp.downloads), nice=5,
                        cores=cp.downloads, threads=4, restarts=8, backoff_s=10.0),
            SV.TaskSpec("tokenize", mk("tokenize", out, 5, cp.tokenize), nice=5, cores=cp.tokenize,
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

    def latest_params(self, ecfg, prefer_post: bool = True):
        """``(params, consts, info, ckpt dir, source)``: the post-DPO checkpoint when accepted,
        else the trainer's newest one."""
        from bes.decode import respond

        post = self.paths.post / "ckpt"
        if prefer_post and (self.man.get("prefopt") or {}).get("accepted") and post.exists():
            path, source = respond.find_checkpoint(post), "prefopt"
        else:
            path, source = respond.find_checkpoint(self.paths.train), "train"
        params, consts, info = respond.load_checkpoint(path, ecfg.model)
        return params, consts, info, Path(path), source

    # ================================================================= stage 0
    def stage_preflight(self, ctx: SV.StageContext) -> dict:
        host = SV.host_report(self.paths.out)
        dev = SV.device_report()
        warnings = SV.check_devices(dev, self.args.allow_cpu)
        need = SV.disk_need_gb(self.cfg)
        if host["disk_free_gb"] < need:
            msg = (f"free disk {host['disk_free_gb']:.1f} GB under {self.paths.out} is below the "
                   f"estimated need of {need:.1f} GB (downloads at the registry caps plus shards)")
            if dev.get("backend") == "tpu":
                raise SV.PreflightError(msg + "; lower the caps (run.download_sources) or free disk")
            warnings.append(msg)
        if dev.get("backend") == "tpu" and host["ram_gb"] and host["ram_gb"] < 32:
            warnings.append(f"only {host['ram_gb']:.0f} GB of host memory")
        cp = SV.core_plan()
        self.cp = cp
        for w in warnings:
            self.log(f"preflight WARNING: {w}")
        self.log(f"preflight: jax {dev['jax']} backend {dev['backend']} devices "
                 f"{[d['kind'] for d in dev['devices']]}; {host['cores']} cores, "
                 f"{host['ram_gb']} GB RAM, {host['disk_free_gb']} GB free disk, "
                 f"HF_TOKEN {'set' if host['hf_token'] else 'not set'}")
        self.man = SV.update_manifest(self.paths.out, host=host, devices=dev,
                                      core_plan=cp.to_dict(), preflight_warnings=warnings,
                                      disk_need_gb=round(need, 1))
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

        def measure(plan, t):
            ctx.check()                     # a stop request ends the benchmark between plans
            if plan != base_plan and canary_s[0] and ctx.now() + canary_s[0] > ctx.hard_deadline:
                return B.Measurement(plan, t, False, "benchmark hard cap: the rest is kept for "
                                                     "the canary")
            m = real_measure(plan, t)
            if m.ok and not canary_s[0]:
                # safe + full runs (a ladder may add more) and the two loss compiles
                canary_s[0] = 3 * cfg.run.canary_steps * m.step_s + 2 * m.compile_s + 30.0
            return m

        results = self.paths.bench / "results.jsonl"
        soft = ctx.started + 0.65 * ctx.slice_s
        ms = B.search(cfg, measure, clock=ctx.now, soft_deadline=soft,
                      hard_deadline=ctx.hard_deadline, mode=cfg.run.bench_plans,
                      platform=backend, known=B.load_results(results),
                      record=lambda m: B.append_result(results, m), log=self.log)
        sel = B.select_plan(ms, lens, B.time_shares(cfg))
        plan = sel["plan"]
        self.log(f"benchmark: chosen plan {plan.key} ({sel['expected_tps']:,.0f} tok/s expected)")
        # the canary always runs to the end: a correctness check is not cut for time
        can = B.canary(cfg, plan, batches[lens[0]], steps=cfg.run.canary_steps, seed=cfg.run.seed,
                       log=self.log)
        warnings = []
        if not can["parity"]["ok"]:
            # the flash kernel and the dense attention disagree: train with the dense attention
            self.log(f"benchmark: PARITY FAILED for {plan.key} ({can['parity']}); falling back to "
                     "the dense attention")
            warnings.append(f"parity failed for {plan.key}: {can['parity']}")
            dense = next((c["plan"] for c in sel["candidates"] if c["plan"].startswith("xla-")), None)
            if dense is not None:
                plan = next(m.plan for m in ms if m.plan.key == dense)
            else:
                plan = plan.flip_attn()
                extra = [real_measure(plan, t) for t in lens]
                for m in extra:
                    B.append_result(results, m)
                ms = list(ms) + extra
                if not all(m.ok for m in extra):
                    raise B.BenchmarkError(f"the dense fallback {plan.key} does not run: "
                                           f"{[m.reason for m in extra]}")
            sel = B.select_plan([m for m in ms if m.plan == plan], lens, B.time_shares(cfg))
            can = B.canary(cfg, plan, batches[lens[0]], steps=cfg.run.canary_steps,
                           seed=cfg.run.seed, log=self.log)
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
        workers = int(bench.get("loader_workers") or self.cp.loader_workers)
        lcfg = dataclasses.replace(LoaderCfg(), workers=workers,
                                   horizon_tokens=float(bench["horizon_tokens"]) or LoaderCfg.horizon_tokens)
        self.paths.train.mkdir(parents=True, exist_ok=True)
        base_log = JsonlLog(self.paths.train / "train_log.jsonl", verbose=False)
        loader = Loader(ecfg, self.tok(), SV.tok_root(ecfg, self.paths.out), library=lib,
                        n_workers=workers, seed=ecfg.run.seed, loader_cfg=lcfg,
                        journal=self.paths.train / LoaderCfg.journal_name)
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

        trainer = Trainer(ecfg, loss_fn=M.loss_fn, init_fn=M.init_params, data=gate,
                          role_fn=role_of, consts_fn=SV.make_consts_fn(ecfg.run.switches),
                          out_dir=self.paths.train, deadline=self.sup.deadline,
                          eval_fn=self._sidecar_hook, log_fn=log_fn, async_ckpt=True, resume=True)
        remove = self.sup.on_stop(trainer.request_stop)
        try:
            if self.sup.stop_requested:
                raise SV.Interrupted("stop requested before training started")
            t0 = time.time()
            trainer.setup()
            self.log(f"train: set-up (compile {len(trainer.shapes)} shapes, restore) took "
                     f"{time.time() - t0:.1f} s; step {trainer.step}")
            summary = trainer.run()
        finally:
            remove()
            loader.close()
        if summary["reason"] == "requested":
            raise SV.Interrupted(f"training stopped at step {summary['step']} on request "
                                 f"(final checkpoint {summary['final']})")
        last = CK.latest(self.paths.train / "ckpt")
        return {"reason": summary["reason"], "step": summary["step"], "p_end": summary["p_end"],
                "tokens": summary["ledger"]["tokens"], "library_active": gate.active,
                "library_at_index": gate.activated_at_index,
                "checkpoint": None if last is None else last.name,
                "ledger_seconds": summary["ledger"]["seconds"]}

    # ================================================================= stage 4
    def _sample_prompts(self, limit: int) -> list[dict]:
        """Prompts for preference sampling from the downloaded chat sources: explicit-constraint
        prompts, GSM8K prompts with their reference answers, chat prompts (half of them given a
        verified hindsight constraint). Rows are taken from the start of each source's newest
        downloaded file (they are also in the training universe: the loader has no held-out
        split)."""
        import numpy as np

        from bes.data import download, hindsight, sources
        from bes.eval import exact_score
        from bes.exact import constraints as C

        names = [n for n, _, _ in SV.download_plan(self.cfg)
                 if sources.get(n).kind in ("chat", "math") and sources.get(n).fmt == "messages"]
        names = list(dict.fromkeys(names))
        per = max(4, limit // max(1, len(names)))
        pools: list[list[dict]] = []
        rng = np.random.default_rng(self.cfg.run.seed + 17)
        for name in names:
            src = sources.get(name)
            try:
                files = download.completed_files(name, self.cfg.run.data_dir)
            except Exception:  # noqa: BLE001 - nothing downloaded
                continue
            if not files:
                continue
            out: list[dict] = []
            for batch in download.iter_records(files[-1].local, list(src.columns), batch_rows=256):
                for i, rec in enumerate(batch):
                    try:
                        msgs = sources.adapt(src, rec)
                    except Exception:  # noqa: BLE001 - a bad row
                        continue
                    if not msgs:
                        continue
                    idx = next((j for j, m in enumerate(msgs) if m["role"] == "assistant"), None)
                    if idx is None or idx == 0 or msgs[idx - 1]["role"] != "user":
                        continue
                    head = [m for m in msgs[:idx] if m["role"] in ("system", "user")]
                    if not isinstance(head[-1]["content"], str):
                        continue
                    reply = hindsight.response_text(msgs[idx]["content"], self.tok()) or ""
                    item = {"id": f"{name}/{files[-1].index}/{len(out)}", "source": name,
                            "messages": [{"role": m["role"], "content": str(m["content"])} for m in head],
                            "reference": None, "kind": "chat",
                            "reply": msgs[idx]["content"]}
                    user = head[-1]["content"]
                    if name == "gsm8k":
                        got = exact_score.extract_answer(reply)
                        if got["value"] is None:
                            continue
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
                    out.append(item)
                    if len(out) >= per:
                        break
                if len(out) >= per:
                    break
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
                passage = resp.retrieve(user) if d.use_library else None
                try:
                    prompt, _fit = respond.fit_prompt(resp.chat, msgs,
                                                      None if passage is None else passage.text, budget)
                except ValueError:
                    continue
                try:
                    program = resp.program(spec, r)
                except Exception:  # noqa: BLE001 - structural potentials only
                    program = resp.program(None, r)
                lm = OurLM(params, consts, ecfg.model, tok, max_steps=4 * d.max_new_tokens, decoder=dec)
                res = smc.decode(lm, prompt, n_particles=d.particles, max_new_tokens=d.max_new_tokens,
                                 seed=1000 + k, settings=SMP.SamplingSettings.from_cfg(d),
                                 program=program, tok=tok, ess_frac=d.ess_frac,
                                 resample_at=d.resample_at, unit_max_tokens=d.unit_max_tokens,
                                 no_repeat_ngram=d.no_repeat_ngram, max_copy_run=d.max_copy_run)
                parts = []
                for p in res.particles:
                    text = resp.chat.display(p.tokens).strip()
                    lab = SV.label_sample(user, text, p.finish, item["reference"])
                    parts.append({"tokens": [int(x) for x in p.tokens if int(x) != END],
                                  "text": text[:2000], "finish": p.finish, "origin": "particle",
                                  **lab})
                ref = _reference_response(item, tok)
                if ref is not None:            # the data set's own reply, labelled the same way
                    lab = SV.label_sample(user, ref["text"], "stop", item["reference"])
                    parts.append({**ref, "finish": "stop", "origin": "reference", **lab})
                scores = [q["score"] for q in parts]
                n_pairs += min(sum(1 for s in scores if s > min(scores)),
                               sum(1 for s in scores if s < max(scores)))
                rec = {"id": item["id"], "source": item["source"], "kind": item["kind"],
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
        path = self.paths.post / "samples.jsonl"
        out = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                         # a torn last line
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
        atomic_write_json(self.paths.post / "critic_report.json", SV._jsonable(rep))
        self.log(f"critic: held-out rho {rep.get('rho_heldout')}, gate passed: {rep.get('use_critic')} "
                 "(the decoder has no critic path; recorded only)")
        return keep

    def _sample_stats_fn(self, ecfg, consts, prompts: list[list[int]], max_new: int):
        """``fn(params) -> {"mean_len", "rep4"}``: one plain sample (the run's temperature and
        top-p, no potentials) per held-out prompt."""
        from bes.decode import sampling as SMP
        from bes.decode import smc
        from bes.decode.lm import OurLM
        from bes.model import generate as G
        from bes.train.prefopt import response_stats

        dec = G.jitted(ecfg.model)
        tok = self.tok()
        settings = SMP.SamplingSettings.from_cfg(ecfg.decode)

        def fn(params) -> dict:
            outs = []
            for i, pr in enumerate(prompts):
                lm = OurLM(params, consts, ecfg.model, tok, max_steps=4 * max_new, decoder=dec)
                res = smc.decode(lm, pr, n_particles=1, max_new_tokens=max_new, seed=7 + i,
                                 settings=settings, tok=tok)
                outs.append(res.particles[0].tokens)
            return response_stats(outs)

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
        pref = dataclasses.replace(PrefCfg(), seq_len=t, pairs_per_micro=ppm,
                                   cache_gb=min(PrefCfg.cache_gb, 1.0))
        sampled = []
        for s in self._samples():
            toks = [p["tokens"] for p in s["particles"]]
            sampled += prefs.pairs_from_scores(s["prompt"], toks, [p["score"] for p in s["particles"]],
                                               source="sampled")
        groups: dict[str, list] = {"sampled": sampled}
        ratio: dict[str, float] = {"sampled": 0.0}
        uf = SV.tok_root(ecfg, self.paths.out) / "tok" / "ultrafeedback"
        if uf.exists():
            pairs, counts = prefs.pairs_from_shards(uf)
            groups["ultrafeedback"] = pairs
            ratio["ultrafeedback"] = PrefCfg.max_len_ratio
        split = prefs.prepare(groups, t, heldout_frac=pref.heldout_frac if len(sampled) >= 200 else 0.25,
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
        lrs = (1e-6, 3e-6)
        if tps > 0:
            per_pair = 2 * t * (len(lrs) + 0.5) * 1.6 / tps
            n_max = max(pref.pairs_per_micro * pref.n_micro, int(0.6 * ctx.left() / per_pair))
            if len(split.train) > n_max:
                self.log(f"prefopt: {len(split.train)} training pairs, {n_max} fit in the slice")
                order = prefs.shuffled_order(len(split.train), pref.seed)[:n_max]
                split = prefs.PrefSplit([split.train[i] for i in sorted(order)], split.heldout,
                                        split.stats)
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
        prompts = [list(map(int, p.prompt)) + [_assistant()] for p in split.heldout[:8]]
        stats_fn = self._sample_stats_fn(ecfg, consts, prompts, min(ecfg.decode.max_new_tokens, 128))
        new_params, rep = prefopt.run_prefopt(params, split, ecfg, lrs=lrs, heldout_loss_fn=heldout_loss,
                                              sample_stats_fn=stats_fn, pref=pref, consts=consts,
                                              tok=tok, log=self.log)
        ctx.check()
        out = {"accepted": bool(rep["accepted"]), "lr": rep.get("lr"), "reason": rep.get("reason"),
               "n_train_pairs": rep.get("n_train_pairs"), "n_heldout_pairs": rep.get("n_heldout_pairs"),
               "pairs_sampled": len(sampled), "pairs_by_source": split.stats,
               "from_checkpoint": info["path"]}
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
        _, _, info, ckpt, source = self.latest_params(ecfg)
        gen = self._generator(ecfg, ckpt)
        backend = CallableBackend(gen)
        root = SV.proxy_root(ecfg, self.paths.out)
        ps = prompt_set.load()
        todo = prompt_set.select(ps, "dev", None, ecfg.run.eval_limit or None)
        base = _decode_settings(ecfg)
        variants = [("default", base)]
        if ecfg.decode.particles > 1:
            variants.append(("k1", base + ["extra.particles=1"]))
        if self.library_path() and ecfg.decode.use_library:
            variants.append(("nolib", base + ["extra.use_library=false"]))
        variants.append(("noband", base + ["extra.band=false"]))
        results = {}
        for i, (name, ov) in enumerate(variants):
            left = ctx.deadline - ctx.now()
            minutes = left / 60.0 / (len(variants) - i)
            if minutes <= 0.05 or ctx.stop_requested():
                results[name] = {"skipped": "no time left"}
                continue
            spec = cli.build_spec("ours", "default", ov, f"select-{name}", 0, None, None, 1, None,
                                  "bes.decode.respond:our_generator", {}, gen.key)
            res = runner.run(todo, spec, root, workers=0, minutes=minutes, all_prompts=ps,
                             backend=backend, log=self.log)
            rep = cli.exact_report(root, res.run_id, "dev")
            per = _per_prompt_scores(root, res.run_id, "dev")
            results[name] = {"run_id": res.run_id, "done": res.done, "summary_math": rep.get("math"),
                             "summary_constraints": rep.get("constraints"),
                             "summary_degenerate": rep.get("degenerate"), "per_prompt": per,
                             "overrides": ov}
        ctx.check()
        ran = {k: v for k, v in results.items() if "per_prompt" in v}
        common = set.intersection(*(set(v["per_prompt"]) for v in ran.values())) if ran else set()
        scores = {k: (sum(v["per_prompt"][p] for p in common) / len(common) if common else None)
                  for k, v in ran.items()}
        best = "default"
        if common:
            best = max(scores, key=lambda k: (scores[k], k == "default"))
        chosen = dict(variants)[best]
        out = {"chosen": best, "overrides": chosen, "scores": scores, "n_common": len(common),
               "n_dev": len(todo), "checkpoint": str(ckpt), "weights": source,
               "runs": {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"} for k, v in results.items()},
               "rule": "highest mean exact score (math correct, constraints satisfied, not degenerate) "
                       "over the dev prompts every variant answered; ties keep the default"}
        self.man = SV.update_manifest(self.paths.out, decode=out)
        self.log(f"select: decode variant {best} (scores {scores} on {len(common)} common prompts)")
        return {k: out[k] for k in ("chosen", "scores", "n_common", "n_dev")}

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
        mk = SV.read_marker(self.paths, "opponents") or SV.read_marker(self.paths, "opponents_progress") or {}
        opp_runs = {k: v for k, v in (mk.get("runs") or {}).items() if k.endswith("/test") and v.get("done")}
        if not opp_runs:
            raise SV.StageSkipped("no opponent replies on the test split yet (background task "
                                  f"'opponents': {(self.bg.status() if self.bg else {}).get('opponents')})")
        dec = (self.man.get("decode") or {})
        ov = list(dec.get("overrides") or _decode_settings(ecfg))
        gen = self._generator(ecfg, ckpt)
        spec = cli.build_spec("ours", "default", ov, f"final-{dec.get('chosen', 'default')}", 0, None,
                              None, 1, None, "bes.decode.respond:our_generator", {}, gen.key)
        res = runner.run(todo, spec, root, workers=0, minutes=max(0.05, 0.45 * ctx.left() / 60.0),
                         all_prompts=ps, backend=CallableBackend(gen), log=self.log)
        x_records, _ = runner.load_outputs(root, res.run_id)
        ctx.check()
        judge = cli.make_judge(ecfg.run.eval_judge, self.cp.judge_threads, None, None, True)
        out: dict[str, Any] = {"ours": res.run_id, "ours_done": res.done, "n_test": len(todo),
                               "weights": source, "opponents": {}}
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
            (final / "EXPORT_NOTE.txt").write_text(note + "\n")
            return {"exported": False, "note": note}
        reports = {"stages": str(self.paths.stages), "manifest": str(self.paths.manifest),
                   "health": str(self.paths.health), "post": str(self.paths.post),
                   "proxy": str(SV.proxy_root(ecfg, self.paths.out) / "reports")}
        try:
            from bes import export as EX  # the tools track's exporter
            fn = getattr(EX, "export_run")
        except (ImportError, AttributeError) as e:
            fn, why = None, f"{type(e).__name__}: {e}"
        if fn is not None:
            avail = {"checkpoint": str(ckpt), "config": str(man["config"]), "overrides": SV.run_overrides(man),
                     "horizon_tokens": SV.horizon_of(man),
                     "library": self.library_path() or None, "final_dir": str(final), "reports": reports,
                     "decode": man.get("decode"), "cfg": ecfg}
            params = inspect.signature(fn).parameters
            kw = {k: v for k, v in avail.items() if k in params}
            res = fn(self.paths.out, **kw)
            return {"exported": True, "by": "bes.export.export_run", "checkpoint": str(ckpt),
                    "weights": source, "result": SV._jsonable(res) if isinstance(res, Mapping) else str(res)}
        # fallback: the parameters (no optimiser state; atomic, readable by
        # bes.decode.respond.load_checkpoint), the configuration, the accounting and a note
        from bes.train import checkpoint as CK

        params, consts, info, _, _ = self.latest_params(ecfg)
        root = final / "ckpt"
        if root.exists():
            shutil.rmtree(root)                 # an earlier (emergency) export is replaced
        saved = CK.save(root, {"state": {"params": params, "consts": consts}},
                        step=int(info.get("step") or 0), config_hash=config_hash(ecfg),
                        meta={"weights": source, "from": str(ckpt)}, keep=1)
        dst = saved.path
        acc = _accounting(ecfg, dst, self.library_path())
        atomic_write_json(final / "config.json", SV._jsonable({
            "config": man["config"], "overrides": SV.run_overrides(man),
            "horizon_tokens": SV.horizon_of(man), "config_hash": config_hash(ecfg),
            "cfg": to_dict(ecfg), "decode_choice": man.get("decode"), "weights": source,
            "library": self.library_path() or None}))
        atomic_write_json(final / "accounting.json", SV._jsonable(acc))
        note = (f"bes/export.py (export_run) is not available ({why}); this directory holds the "
                f"checkpoint ({source} weights), the configuration and the accounting report only. "
                f"The tokenizer is assets/tokenizer/ of the repository; the library is "
                f"{self.library_path() or 'none'}; reports: {reports}.")
        (final / "EXPORT_NOTE.txt").write_text(note + "\n")
        self.log(f"export: {note}")
        return {"exported": True, "by": "fallback", "checkpoint": str(dst), "weights": source,
                "note": note, "total_numbers": acc.get("total"), "active_step": acc.get("active_step")}


def _assistant() -> int:
    from bes.config import ASSISTANT
    return ASSISTANT


def _reference_response(item: Mapping[str, Any], tok) -> dict | None:
    """The data set's own reply to a sampling prompt as ``{"tokens", "text"}`` (assistant
    content without the role token and ``END``), or ``None``: a hindsight constraint was
    verified on it, a GSM8K reply carries the reference answer."""
    from bes.data.chat_format import display_text, render_turn

    content = item.get("reply")
    if not content:
        return None
    try:
        ids, _ = render_turn("assistant", content, tok)
    except (ValueError, TypeError):
        return None
    body = [int(x) for x in ids[1:-1]]
    return {"tokens": body, "text": display_text(body, tok).strip()[:2000]} if body else None


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
    return ap.parse_args(argv)


def configure_jax(paths: SV.RunPaths) -> None:
    import jax

    paths.jax_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(paths.jax_cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def new_manifest(args: argparse.Namespace, out: Path) -> dict:
    if not args.config:
        raise SystemExit("a new run needs --config")
    overrides = list(args.override)
    if args.budget_minutes is not None:
        overrides.append(f"run.deadline_hours={args.budget_minutes / 60.0!r}")
    cfg = load_config(args.config, overrides)
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
        configure_jax(paths)
        run = Run(args, paths, man)
        cfg = run.cfg
        budget = man["budget"]
        specs = ST.stage_specs(cfg.run, float(budget["budget_s"]))
        fns = {"preflight": run.stage_preflight, "background": run.stage_background,
               "benchmark": run.stage_benchmark, "train": run.stage_train,
               "sample": run.stage_sample, "critic": run.stage_critic,
               "prefopt": run.stage_prefopt, "select": run.stage_select,
               "judge": run.stage_judge, "export": run.stage_export}
        sup = SV.Supervisor(out, specs, fns, t0=float(budget["t0"]), deadline=float(budget["deadline"]),
                            post_reserve_s=cfg.run.post_reserve_hours * 3600.0)
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
