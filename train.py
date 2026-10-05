"""Train an LM on WikiText-2 under a hard FLOP budget, then report val/test BPB.

Example: python train.py --name base --d 256 --n_layer 4 --budget 1.28e14
"""
import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from flops import (ADAM_FLOPS_PER_PARAM, analytic_matmul_flops_per_token, elementwise_flops_per_token,
                   measure_matmul_flops, muon_flops, sparse_table_flops)
from model import LM, Config, ngram_ids
from optim import Muon, SparseRowAdam

ROOT = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2)


def load_data(vocab):
    d = os.path.join(ROOT, "data", f"bpe{vocab}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    toks = {s: torch.from_numpy(np.fromfile(os.path.join(d, f"{s}.bin"), dtype=np.uint16).astype(np.int64))
            for s in ("train", "valid", "test")}
    return meta, toks


@torch.no_grad()
def token_nll(model, toks, ng, T, stride, bs=16):
    """Per-target NLL in nats for toks[1:], sliding windows of length T, new tokens per window = stride."""
    was = model.training
    model.eval()
    N = toks.numel()
    T = min(T, N - 1)
    nll = torch.empty(N - 1)
    starts = list(range(0, N - 1 - T + 1, stride))
    if starts[-1] + T < N - 1:
        starts.append(N - 1 - T)
    covered = 0
    for i in range(0, len(starts), bs):
        sb = starts[i:i + bs]
        x = torch.stack([toks[s:s + T] for s in sb])
        y = torch.stack([toks[s + 1:s + T + 1] for s in sb])
        g = torch.stack([ng[s:s + T] for s in sb]) if ng is not None else None
        loss = model(x, y, g, reduction="none").view(len(sb), T)
        for j, s in enumerate(sb):
            nll[covered:s + T] = loss[j, covered - s:]
            covered = s + T
    model.train(was)
    assert covered == N - 1
    return nll


def lr_mult(step, total, warmup, cooldown, floor):
    if step < warmup:
        return (step + 1) / warmup
    cd_start = int(total * (1 - cooldown))
    if step < cd_start:
        return 1.0
    frac = (step - cd_start) / max(1, total - cd_start)
    return 1.0 - (1.0 - floor) * frac


def split_params(model):
    muon, dense, sparse = [], {}, []
    for n, p in model.named_parameters():
        if n.startswith("ngram"):
            sparse.append(p)
        elif n.startswith("blocks") and p.ndim == 2 and "gate" not in n:
            muon.append(p)
        elif n.startswith("emb") or n.startswith("ve."):
            dense.setdefault("emb", []).append(p)
        elif n.startswith("head"):
            dense.setdefault("head", []).append(p)
        else:
            dense.setdefault("scalar", []).append(p)
    return muon, dense, sparse


class Sampler:
    """'epoch': every epoch, non-overlapping windows with a random phase, shuffled. 'random': iid offsets."""

    def __init__(self, N, T, B, mode, gen):
        self.N, self.T, self.B, self.mode, self.gen = N, T, B, mode, gen
        self.queue = torch.empty(0, dtype=torch.long)

    def next(self):
        if self.mode == "random":
            return torch.randint(0, self.N - self.T - 1, (self.B,), generator=self.gen)
        while self.queue.numel() < self.B:
            phase = int(torch.randint(0, self.T, (1,), generator=self.gen))
            starts = torch.arange(phase, self.N - self.T - 1, self.T)
            starts = starts[torch.randperm(starts.numel(), generator=self.gen)]
            self.queue = torch.cat([self.queue, starts])
        out, self.queue = self.queue[:self.B], self.queue[self.B:]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--budget", type=float, default=1.28e14)
    ap.add_argument("--reserve", type=float, default=0.0, help="FLOPs reserved for test-time training")
    ap.add_argument("--vocab", type=int, default=4096)
    for f in dataclasses.fields(Config):
        if f.name == "vocab":
            continue
        t = type(f.default)
        if t is bool:
            ap.add_argument(f"--{f.name}", type=int, default=int(f.default))
        elif t is tuple:
            ap.add_argument(f"--{f.name}", type=str, default=",".join(map(str, f.default)))
        else:
            ap.add_argument(f"--{f.name}", type=t, default=f.default)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--sampler", default="epoch")
    ap.add_argument("--lr_muon", type=float, default=0.03)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--momentum_warmup", type=float, default=0.1, help="fraction of steps to warm 0.85->m")
    ap.add_argument("--ns_steps", type=int, default=5)
    ap.add_argument("--polar", type=int, default=1)
    ap.add_argument("--normuon", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lr_emb", type=float, default=0.05)
    ap.add_argument("--lr_ngram", type=float, default=0.1)
    ap.add_argument("--lr_head", type=float, default=0.008)
    ap.add_argument("--lr_scalar", type=float, default=0.02)
    ap.add_argument("--beta1", type=float, default=0.8)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cooldown", type=float, default=0.6)
    ap.add_argument("--lr_floor", type=float, default=0.1)
    ap.add_argument("--ema_frac", type=float, default=0.0, help="EMA over the last fraction of steps (0=off)")
    ap.add_argument("--ema_mix", type=float, default=0.6, help="eval weights = (1-mix)*theta + mix*EMA")
    ap.add_argument("--eval_seq", type=int, default=0)
    ap.add_argument("--eval_stride", type=int, default=0)
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of train articles (at the end) excluded from NN training (for gate fitting)")
    ap.add_argument("--countmix", type=int, default=0, help="train NN jointly through the count-expert mixture")
    ap.add_argument("--cm_K", type=int, default=6)
    ap.add_argument("--cm_Kd", type=int, default=4)
    ap.add_argument("--cm_aux", type=float, default=0.0, help="weight of plain NN cross-entropy alongside")
    ap.add_argument("--lr_gate", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=25)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    meta, toks = load_data(a.vocab)
    kw = {}
    for f in dataclasses.fields(Config):
        if f.name == "vocab":
            continue
        v = getattr(a, f.name)
        if isinstance(f.default, bool):
            v = bool(v)
        elif isinstance(f.default, tuple):
            v = tuple(int(x) for x in v.split(",") if x)
        kw[f.name] = v
    cfg = Config(vocab=meta["vocab_size"], **kw)
    model = LM(cfg)
    ng = {s: ngram_ids(toks[s], cfg.ngram_orders, cfg.ngram_buckets) for s in toks}

    muon_p, dense, sparse_p = split_params(model)
    lrs = dict(emb=a.lr_emb, head=a.lr_head, scalar=a.lr_scalar)
    adam = torch.optim.Adam([dict(params=v, lr=lrs[k], base_lr=lrs[k]) for k, v in dense.items()],
                            betas=(a.beta1, a.beta2), eps=1e-10)
    mu = Muon(muon_p, lr=a.lr_muon, momentum=a.muon_momentum, ns_steps=a.ns_steps, wd=a.wd,
              polar=bool(a.polar), normuon=bool(a.normuon))
    mu.param_groups[0]["base_lr"] = a.lr_muon
    opts = [mu, adam]
    if sparse_p:
        sp = SparseRowAdam(sparse_p, lr=a.lr_ngram, beta2=a.beta2)
        sp.param_groups[0]["base_lr"] = a.lr_ngram
        opts.append(sp)

    B, T = a.batch, a.seq
    n_dense = sum(p.numel() for v in dense.values() for p in v)
    n_all = sum(p.numel() for p in model.parameters())
    mm = measure_matmul_flops(model, B, T)
    ew = elementwise_flops_per_token(cfg, T) * B * T
    op = sum(muon_flops(p.shape, a.ns_steps) for p in muon_p) + ADAM_FLOPS_PER_PARAM * n_dense \
        + sparse_table_flops(cfg, B * T)
    step_flops = mm + ew + op
    ema_cost = 3 * n_all
    train_budget = a.budget - a.reserve
    steps = int(train_budget // step_flops)
    while steps * step_flops + math.ceil(steps * a.ema_frac) * ema_cost > train_budget:
        steps -= 1
    ema_start = steps - math.ceil(steps * a.ema_frac) if a.ema_frac else steps
    info = dict(name=a.name, config=dataclasses.asdict(cfg), args=vars(a), dataset=meta["dataset"],
                params_muon=sum(p.numel() for p in muon_p), params_dense_other=n_dense,
                params_sparse_tables=sum(p.numel() for p in sparse_p), params_total=n_all,
                flops_per_step=dict(matmul_attn=mm, elementwise=ew, optimizer=op, total=step_flops),
                analytic_fwd_matmul_per_token=analytic_matmul_flops_per_token(cfg, T),
                measured_train_matmul_per_token=mm / (B * T),
                steps=steps, tokens=steps * B * T, epochs=steps * B * T / meta["train_tokens"])
    print(json.dumps({k: v for k, v in info.items() if k not in ("config", "args")}, indent=1), flush=True)

    train = toks["train"]
    n_train = train.numel()
    if a.holdout:
        from countmem import doc_ids, load_tok_bytes
        docs = doc_ids(train.numpy(), load_tok_bytes(os.path.join(ROOT, "data", f"bpe{a.vocab}")))
        cut_doc = int(round((docs.max() + 1) * (1 - a.holdout)))
        n_train = int(np.searchsorted(docs, cut_doc))
        info.update(holdout_start=n_train, holdout_tokens=train.numel() - n_train)
        print(f"holdout: training on first {n_train} tokens, holding out {train.numel() - n_train}")
    gen = torch.Generator().manual_seed(a.seed)
    sampler = Sampler(n_train, T, B, a.sampler, gen)
    gate = None
    if a.countmix:
        from countmem import CountMemory, doc_ids, features, load_tok_bytes
        from mixture import Gate
        tc = time.time()
        docs_tr = doc_ids(train.numpy(), load_tok_bytes(os.path.join(ROOT, "data", f"bpe{a.vocab}")))
        cm = CountMemory(train.numpy(), docs_tr, K=a.cm_K, Kd=a.cm_Kd)
        Fc, Pc, Ac = (torch.from_numpy(v) for v in features(cm.stats(train.numpy(), docs_tr, is_train=True)))
        del cm
        gate = Gate(Fc.size(1) + 2, Pc.size(1) + 1, hidden=64)
        gate.mu.copy_(torch.cat([Fc.mean(0), torch.tensor([4.0, -2.0])]))
        gate.sd.copy_(torch.cat([Fc.std(0).clamp_min(1e-3), torch.tensor([2.0, 1.5])]))
        gate_opt = torch.optim.Adam(gate.parameters(), lr=a.lr_gate)
        gate_opt.param_groups[0]["base_lr"] = a.lr_gate
        opts.append(gate_opt)
        gf = (3 * gate.flops_per_token() + 3 * 6 * cfg.vocab + 30 * Pc.size(1)) * B * T \
            + ADAM_FLOPS_PER_PARAM * sum(p.numel() for p in gate.parameters())
        step_flops += gf
        steps = int(train_budget // step_flops)
        while steps * step_flops + math.ceil(steps * a.ema_frac) * ema_cost > train_budget:
            steps -= 1
        ema_start = steps - math.ceil(steps * a.ema_frac) if a.ema_frac else steps
        info.update(steps=steps, tokens=steps * B * T, epochs=steps * B * T / meta["train_tokens"],
                    countmix_flops_per_step=gf)
        print(f"countmix: features {tuple(Fc.shape)} experts {Pc.size(1)} built in {time.time() - tc:.0f}s; "
              f"steps now {steps}", flush=True)
    ar = torch.arange(T)
    ema = None
    t0, used, lema = time.time(), 0.0, None
    for step in range(steps):
        m = lr_mult(step, steps, a.warmup, a.cooldown, a.lr_floor)
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * m
        mw = max(1, int(a.momentum_warmup * steps))
        mu.param_groups[0]["momentum"] = a.muon_momentum - (a.muon_momentum - 0.85) * max(0.0, 1 - step / mw)
        pos = sampler.next()[:, None] + ar
        x, y = train[pos], train[pos + 1]
        g = ng["train"][pos] if ng["train"] is not None else None
        if gate is None:
            loss = model(x, y, g)
        else:
            lp = torch.log_softmax(model.logits(model.hidden(x, g)).float(), -1)
            lpy = lp.gather(-1, y[..., None])[..., 0]
            with torch.no_grad():
                ent = -(lp.exp() * lp).sum(-1)
                mx = lp.max(-1).values
            feat = torch.cat([Fc[pos], ent[..., None], mx[..., None]], -1)
            avail = torch.cat([torch.ones(B, T, 1, dtype=torch.bool), Ac[pos]], -1)
            logp = torch.cat([lpy[..., None], torch.log(Pc[pos].clamp_min(1e-30))], -1)
            mix = torch.logsumexp(gate.logw(feat, avail) + logp, -1)
            loss = -mix.mean()
            if a.cm_aux:
                loss = loss - a.cm_aux * lpy.mean()
        loss.backward()
        for opt in opts:
            opt.step()
        model.zero_grad(set_to_none=True)
        used += step_flops
        if step >= ema_start:
            with torch.no_grad():
                if ema is None:
                    ema = [p.detach().clone() for p in model.parameters()]
                else:
                    beta = 1 - 1 / (step - ema_start + 1)  # uniform average over the tail window
                    for e, p in zip(ema, model.parameters()):
                        e.lerp_(p.detach(), 1 - beta)
            used += ema_cost
        li = loss.item()
        lema = li if lema is None else 0.95 * lema + 0.05 * li
        if step % a.log_every == 0 or step == steps - 1:
            el = time.time() - t0
            print(f"step {step}/{steps} loss {li:.4f} ema {lema:.4f} lr {m:.3f} "
                  f"flops {used:.3e} ({used / a.budget:.1%}) {el:.0f}s {used / el / 1e9:.1f} GFLOP/s", flush=True)
    assert used <= train_budget + 1
    info.update(train_flops=used, train_time_s=time.time() - t0, final_train_loss_ema=lema)
    if ema is not None:
        with torch.no_grad():
            for e, p in zip(ema, model.parameters()):
                p.lerp_(e, a.ema_mix)

    es = a.eval_seq or T
    st = a.eval_stride or es // 2
    res = {}
    for s in (("valid",) if a.no_test else ("valid", "test")):
        nll = token_nll(model, toks[s], ng[s], es, st)
        res[s] = dict(bpb=nll.sum().item() / LN2 / meta[f"{s}_bytes"], token_loss=nll.mean().item())
        print(f"{s}: bpb {res[s]['bpb']:.4f} token_loss {res[s]['token_loss']:.4f}", flush=True)
    info.update(eval=res, eval_seq=es, eval_stride=st)
    out = os.path.join(ROOT, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(info, open(os.path.join(out, "result.json"), "w"), indent=1)
    torch.save({"cfg": dataclasses.asdict(cfg), "args": vars(a), "state": model.state_dict(),
                "gate": gate.state_dict() if gate is not None else None}, os.path.join(out, "model.pt"))


if __name__ == "__main__":
    main()
