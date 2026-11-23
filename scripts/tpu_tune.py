"""One measurement on the idle accelerator before the 24-hour run: which same-mathematics
settings train fastest on this chip, and how fast the post-training programs are.

    python scripts/tpu_tune.py --config configs/final.py --out /content/bes_cli/tune.json

Everything here uses synthetic batches and freshly initialised parameters of the configured
shape, so it needs no data on disk and can run right after the set-up, before the run exists
(nothing else may hold the device). It takes about five minutes on a TPU v6e and prints

* **training**: one ``value_and_grad`` of the loss per micro-batch for the candidates that
  change no value (``docs/TPU_NOTES.md``): ``model.remat_keep`` ("" or "kernel"),
  ``model.loss_chunk`` (the configured one or one chunk per micro-batch), ``optim.micro_seqs``
  (the configured rows or half of them) and, at the long sequence length, the splash tiles
  (the configured ones or one tile per sequence). A candidate replaces the configured value
  only when it is at least ``--margin`` (1.5%) faster;
* **decoding**: milliseconds per synchronous decode step of ``bes.decode.lm.OurLM`` (device
  step plus the host's exact features) for 6 and 24 rows, and one ``bes.decode.smc.decode`` of
  a chat prompt with the run's decode settings: the post-training stages (sampling, decode
  selection, the judged comparison) are bounded by this, and it tells whether more prompts per
  device step would pay;
* **preference optimisation**: the reference log-probabilities and two optimiser steps of
  ``bes.train.prefopt`` on synthetic pairs of the stage's shape (compile and step time).

The last line is ``RECOMMEND <json>``: ``{"overrides": [...], "tokens_per_s": {...}, ...}``;
``--out`` writes the whole report. Pass the overrides to the run (``--override``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def synth_batch(mcfg, rows: int, t: int, seed: int = 0) -> dict:
    """A micro-batch ``[rows, t]`` with the loader's fields and dtypes (random ordinary tokens,
    one segment per row starting with BOS, sparse candidate slots)."""
    import numpy as np

    from bes.config import CAND_FEATS, CAND_SLOTS, N_CLOCKS, REG_FIELDS, special_id

    rng = np.random.default_rng(seed)
    v = int(mcfg.vocab_size)
    tokens = rng.integers(10, v - 16, size=(rows, t + 1), dtype=np.int32)
    tokens[:, 0] = special_id("<|bos|>", v)
    reg = np.stack([rng.integers(0, card, size=(rows, t)) for _, card in REG_FIELDS], -1).astype(np.int16)
    cand_tok = rng.integers(10, v - 16, size=(rows, t, CAND_SLOTS), dtype=np.int32)
    empty = rng.random((rows, t, CAND_SLOTS)) < 0.9
    cand_tok[empty] = -1
    feats = []
    for name, card in CAND_FEATS:
        f = rng.integers(1 if name == "kind" else 0, min(card, 5), size=(rows, t, CAND_SLOTS))
        f[empty] = 0
        feats.append(f)
    return {"tokens": tokens, "w": np.ones((rows, t), np.float32), "seg": np.zeros((rows, t), np.int32),
            "reg": reg, "clock": rng.integers(0, 100, size=(rows, t, N_CLOCKS), dtype=np.int32),
            "cand_tok": cand_tok, "cand_feat": np.stack(feats, -1).astype(np.int8)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="section.field=value")
    ap.add_argument("--out", default=None, help="write the report (JSON) here")
    ap.add_argument("--margin", type=float, default=0.015, help="a candidate must be this much faster")
    ap.add_argument("--calls", type=int, default=6, help="timed calls per training candidate")
    ap.add_argument("--decode-steps", type=int, default=48)
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: train, decode, dpo")
    args = ap.parse_args(argv)
    skip = {s for s in args.skip.split(",") if s}

    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes.config import load_config
    from bes.model import model as M
    from bes.train import supervisor as SV

    cfg = load_config(args.config, list(args.override))
    mc = cfg.model
    t1, t2 = int(cfg.data.seq_len), int(cfg.data.seq_len_late)
    rows1 = int(cfg.optim.micro_seqs)
    dev = jax.devices()[0].device_kind
    say = lambda msg: print(msg, flush=True)                                   # noqa: E731
    say(f"tpu_tune: {args.config} {args.override}, device {dev}, base micro-batch {rows1}x{t1}, long {t2}")
    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    report: dict[str, Any] = {"config": args.config, "overrides_in": list(args.override), "device": dev,
                              "train": [], "decode": {}, "dpo": {}, "time": time.time()}
    t_start = time.time()

    # ------------------------------------------------------------------ training candidates
    def timed(label: str, mcfg, rows: int, t: int) -> float | None:
        batch = jax.device_put(synth_batch(mcfg, rows, t))
        f = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
        try:
            t0 = time.perf_counter(); out = f(params, batch); jax.block_until_ready(out)
            comp = time.perf_counter() - t0
            out = f(params, batch); jax.block_until_ready(out)                  # warm
            t0 = time.perf_counter()
            for _ in range(args.calls):
                out = f(params, batch)
            jax.block_until_ready(out)
            ms = (time.perf_counter() - t0) / args.calls * 1e3
        except Exception as e:  # noqa: BLE001 - a candidate that does not run is not chosen
            say(f"  {label:44s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            report["train"].append({"label": label, "rows": rows, "t": t, "ok": False,
                                    "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"})
            return None
        tps = rows * t / ms * 1e3
        say(f"  {label:44s} {ms:8.1f} ms  {tps:10,.0f} tok/s  loss {float(out[0]):.5f}  (compile {comp:.0f} s)")
        report["train"].append({"label": label, "rows": rows, "t": t, "ok": True, "ms": ms,
                                "tokens_per_s": tps, "loss": float(out[0]), "compile_s": comp})
        return tps

    chosen: dict[str, Any] = {}
    tps_out: dict[str, float] = {}
    if "train" not in skip:
        say(f"training, T={t1}:")
        if dev.lower().startswith("tpu"):
            timed("warm-up (discarded)", mc, rows1, t1)
        best = timed("configured", mc, rows1, t1)
        cur = mc
        if best is None:
            raise SystemExit("the configured micro-batch does not run")

        def consider(label: str, key: str, value: Any, cand, rows: int, t: int, base: float) -> tuple[Any, float]:
            got = timed(label, cand, rows, t)
            if got is not None and got >= base * (1.0 + args.margin):
                chosen[key] = value
                return cand, got
            return None, base

        if mc.remat and not mc.remat_keep:
            c, best2 = consider("remat_keep=kernel", "model.remat_keep", "kernel",
                                dataclasses.replace(cur, remat_keep="kernel"), rows1, t1, best)
            cur, best = (c, best2) if c is not None else (cur, best)
        if int(cur.loss_chunk) < rows1 * t1:
            c, best2 = consider(f"loss_chunk={rows1 * t1}", "model.loss_chunk", rows1 * t1,
                                dataclasses.replace(cur, loss_chunk=rows1 * t1), rows1, t1, best)
            cur, best = (c, best2) if c is not None else (cur, best)
        rows = rows1
        if rows1 % 2 == 0 and rows1 >= 4:
            got = timed(f"micro_seqs={rows1 // 2}", cur, rows1 // 2, t1)
            if got is not None and got >= best * (1.0 + args.margin):
                chosen["optim.micro_seqs"] = rows1 // 2
                rows, best = rows1 // 2, got
        tps_out[str(t1)] = best
        if t2 != t1:
            rows_l = max(1, rows * t1 // t2)
            say(f"training, T={t2}:")
            best_l = timed("as chosen above", cur, rows_l, t2)
            if best_l is not None:
                if max(int(cur.splash_block_q), int(cur.splash_block_kv)) < t2 and cur.attn_impl == "splash":
                    c, got = consider(f"splash tiles {t2}/{t2}", "model.splash_block_q", t2,
                                      dataclasses.replace(cur, splash_block_q=t2, splash_block_kv=t2),
                                      rows_l, t2, best_l)
                    if c is not None:
                        chosen["model.splash_block_kv"] = t2
                        cur, best_l = c, got
                        # the short length must not lose: tiles are clipped to the sequence there
                        again = timed(f"T={t1} with those tiles", cur, rows, t1)
                        if again is None or again < tps_out[str(t1)] * (1.0 - args.margin):
                            chosen.pop("model.splash_block_q"), chosen.pop("model.splash_block_kv")
                            cur = dataclasses.replace(cur, splash_block_q=mc.splash_block_q,
                                                      splash_block_kv=mc.splash_block_kv)
                tps_out[str(t2)] = best_l
        mc_final = cur
    else:
        mc_final = mc
    report["train_seconds"] = time.time() - t_start

    # ------------------------------------------------------------------ decoding
    if "decode" not in skip:
        t_dec = time.time()
        try:
            from bes.decode import respond, sampling as SMP, smc
            from bes.decode.lm import OurLM
            from bes.model import generate as G
            from bes.tokenizer import load as load_tokenizer

            tok = load_tokenizer()
            ecfg = dataclasses.replace(cfg, model=mc_final)
            resp = respond.Responder(lambda n: None, tok, ecfg, library=None)
            r = respond.resolve(ecfg.decode, None)
            d = r.decode
            msgs = respond.clean_messages([{"role": "user", "content": "Explain in three short sentences why the "
                                                                       "sea is salty, then give one example."}])
            prompt, _ = respond.fit_prompt(resp.chat, msgs, None, resp.context - d.max_new_tokens)
            dec = G.jitted(mc_final)
            say(f"decoding (prompt of {len(prompt)} tokens, cache for {d.max_new_tokens} new tokens):")
            rng = np.random.default_rng(0)
            for k in (int(d.particles), 4 * int(d.particles)):
                lm = OurLM(params, consts, mc_final, tok, max_steps=4 * d.max_new_tokens, decoder=dec,
                           self_evidence_weight=d.self_evidence_weight)
                t0 = time.perf_counter(); lm.prefill(prompt, k); pre = time.perf_counter() - t0
                toks = rng.integers(10, mc_final.vocab_size - 16, size=(args.decode_steps + 2, k))
                act = np.ones(k, bool)
                t0 = time.perf_counter(); lm.step(toks[0], act); first = time.perf_counter() - t0
                lm.step(toks[1], act)
                t0 = time.perf_counter()
                for i in range(args.decode_steps):
                    lm.step(toks[2 + i], act)
                ms = (time.perf_counter() - t0) / args.decode_steps * 1e3
                lm.close()
                say(f"  {k:3d} rows: {ms:7.1f} ms per step ({k / ms * 1e3:7.0f} tokens/s), prefill and its "
                    f"compile {pre:.1f} s, first step {first:.1f} s")
                report["decode"][f"rows_{k}"] = {"ms_per_step": ms, "prefill_s": pre, "first_step_s": first}
            lm = OurLM(params, consts, mc_final, tok, max_steps=4 * d.max_new_tokens, decoder=dec,
                       self_evidence_weight=d.self_evidence_weight)
            t0 = time.perf_counter()
            res = smc.decode(lm, prompt, n_particles=d.particles, max_new_tokens=min(96, d.max_new_tokens),
                             seed=1, settings=SMP.SamplingSettings.from_cfg(d), program=resp.program(None, r),
                             tok=tok, ess_frac=d.ess_frac, resample_at=d.resample_at,
                             unit_max_tokens=d.unit_max_tokens, no_repeat_ngram=d.no_repeat_ngram,
                             max_copy_run=d.max_copy_run)
            sec = time.perf_counter() - t0
            n_tok = max(len(p.tokens) for p in res.particles)
            say(f"  smc.decode, {d.particles} particles, {n_tok} tokens: {sec:.1f} s = {sec / max(1, n_tok) * 1e3:.1f} "
                f"ms per token position (untrained weights: no early end)")
            report["decode"]["smc"] = {"seconds": sec, "tokens": n_tok, "particles": int(d.particles),
                                       "ms_per_position": sec / max(1, n_tok) * 1e3}
        except Exception as e:  # noqa: BLE001 - a probe: the report says what failed
            say(f"  decoding FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            report["decode"]["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        report["decode_seconds"] = time.time() - t_dec

    # ------------------------------------------------------------------ preference optimisation
    if "dpo" not in skip:
        t_dpo = time.time()
        try:
            from bes.config import BOS, END, USER, PrefCfg
            from bes.data import prefs as P
            from bes.tokenizer import load as load_tokenizer
            from bes.train import prefopt as D

            ecfg = dataclasses.replace(cfg, model=mc_final)
            t = min(PrefCfg.seq_len, mc_final.max_seq)
            ppm = max(1, min(PrefCfg.pairs_per_micro, ecfg.optim.micro_seqs * ecfg.data.seq_len // (2 * t)))
            pref = dataclasses.replace(PrefCfg(), seq_len=t, pairs_per_micro=ppm, cache_gb=1.0)
            rng = np.random.default_rng(3)
            n_pairs = 2 * pref.pairs_per_micro * pref.n_micro
            hi = mc_final.vocab_size - 16

            def text(n):
                return rng.integers(10, hi, size=n).tolist()
            pairs = [P.make_pair([BOS, USER] + text(int(rng.integers(8, min(120, t // 4)))) + [END],
                                 text(int(rng.integers(20, t // 2))), text(int(rng.integers(20, t // 2))))
                     for _ in range(n_pairs)]
            data = P.PairBatches(pairs, pref.seq_len, pref.pairs_per_micro, pref.n_micro,
                                 tok=load_tokenizer(), cache_bytes=1 << 30)
            say(f"preference optimisation ({n_pairs} synthetic pairs, batches {data.shape}):")
            t0 = time.perf_counter(); D.compute_ref_logps(params, consts, data, mc_final)
            ref_s = time.perf_counter() - t0
            t0 = time.perf_counter(); D.compute_ref_logps(params, consts, data, mc_final)
            ref2_s = time.perf_counter() - t0
            say(f"  reference log-probabilities: {ref_s:.1f} s with the compile, {ref2_s:.2f} s again "
                f"({2 * n_pairs / ref2_s:.0f} rows/s)")
            t0 = time.perf_counter()
            _, rep = D.train_epoch(params, consts, data, ecfg, 1e-6, pref)
            ep_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, rep = D.train_epoch(params, consts, data, ecfg, 1e-6, pref)
            ep2_s = time.perf_counter() - t0
            say(f"  one epoch of {rep['steps']} steps: {ep_s:.1f} s with the compile, {ep2_s:.2f} s again "
                f"({ep2_s / max(1, rep['steps']):.2f} s per step of {pref.pairs_per_micro * pref.n_micro} pairs), "
                f"loss {rep.get('first_loss')}, finite {rep.get('finite')}")
            report["dpo"] = {"ref_first_s": ref_s, "ref_s": ref2_s, "epoch_first_s": ep_s, "epoch_s": ep2_s,
                             "steps": rep["steps"], "pairs_per_step": pref.pairs_per_micro * pref.n_micro,
                             "finite": rep.get("finite"), "first_loss": rep.get("first_loss")}
        except Exception as e:  # noqa: BLE001
            say(f"  preference optimisation FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            report["dpo"]["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        report["dpo_seconds"] = time.time() - t_dpo

    overrides = [f"{k}={v}" for k, v in chosen.items()]
    rec = {"overrides": overrides, "tokens_per_s": tps_out, "seconds": round(time.time() - t_start, 1)}
    report["recommend"] = rec
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    say("RECOMMEND " + json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
