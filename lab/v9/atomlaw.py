"""One law for every FFN-side arm of the lab:  dL = -c * E_{x ~ f}[ ln( (s + rho_x) / s0 ) ],
s0 = shared neurons per token type of the base (F/V = 192/1040 per block), s = shared neurons per token type of the arm,
rho_x = neurons of the arm that only token x's group reads, divided by the number of token types in that group
(class experts: width / tokens per class, frequency-balanced classes ~ V/K tokens each; private neurons: rho_x itself).
c is fitted per checkpoint on ALL arms by least squares through the origin; the table shows prediction against measurement."""
import numpy as np, os, json
SP = os.environ.get("LAB_DATA", os.path.dirname(os.path.abspath(__file__)))   # token files and runs/
V, F = 1040, 192
web = np.memmap(os.path.join(SP, "web1k.u16"), dtype=np.uint16, mode="r")[:40_000_000]
chat = np.memmap(os.path.join(SP, "chat1k.u16"), dtype=np.uint16, mode="r")[:10_000_000]
f = 0.8 * np.bincount(web, minlength=V) / len(web) + 0.2 * np.bincount(chat, minlength=V) / len(chat); f /= f.sum()
s0 = F / V
def alloc(mean, gamma, cap):
    lo, hi = 0.0, 1e12
    for _ in range(80):
        mid = (lo + hi) / 2
        r = np.minimum(np.floor(mid * f ** gamma + 0.5), cap)
        lo, hi = (mid, hi) if r.sum() < mean * V else (lo, mid)
    return np.minimum(np.floor(hi * f ** gamma + 0.5), cap)
def x_of(s, rho):                               # E_f ln((s + rho)/s0)
    return float((f * np.log((s + rho) / s0)).sum())
arms = {  # name: (shared neurons per type, private-or-class neurons per type)
    "k_f384 (dense x2)": (384 / V, np.zeros(V)),
    "k_f768 (dense x4)": (768 / V, np.zeros(V)),
    "k_exp8_bal (8 class experts)": (0.0, np.full(V, 8 * 192 / V)),
    "k_exp8_tree (8 class experts)": (0.0, np.full(V, 8 * 192 / V)),
    "q_ffn32 (32 class experts)": (0.0, np.full(V, 32 * 192 / V)),
    "x_lex2_lr (2 private, uniform)": (s0, np.full(V, 2.0)),
    "x_lex2n_lr (2 private, normalised keys)": (s0, np.full(V, 2.0)),
    "x_lexg2_lr (private, by frequency, cap 32)": (s0, alloc(2, 1.0, 32)),
    "x_lexg2c8_lr (private, by frequency, cap 8)": (s0, alloc(2, 1.0, 8)),
    "x_lexg2c4_lr (private, by frequency, cap 4)": (s0, alloc(2, 1.0, 4)),
}
def evals(n):
    out = {}
    try:
        for ln in open(os.path.join(SP, "runs", n + ".log")):
            if ln.startswith("{") and '"mix"' in ln:
                d = json.loads(ln); out[d["tokens"]] = d["mix"]
    except FileNotFoundError:
        pass
    return out
base = evals("k_base")
cks = sorted(base)
rows = {}
for name, (s, rho) in arms.items():
    e = evals(name.split()[0])
    rows[name] = (x_of(s, rho), [e[t] - base[t] if t in e else None for t in cks])
print("checkpoints (tokens):", cks)
for j, t in enumerate(cks):
    xs = np.array([r[0] for r in rows.values() if r[1][j] is not None]); ys = np.array([r[1][j] for r in rows.values() if r[1][j] is not None])
    c = -float((xs * ys).sum() / (xs * xs).sum())
    res = ys + c * xs
    print(f"\n{t/1e6:.1f}M tokens: c = {c:.4f}  ({len(xs)} arms, rms residual {np.sqrt((res**2).mean()):.4f}, max |residual| {np.abs(res).max():.4f})")
    for name, (x, d) in rows.items():
        if d[j] is not None:
            print(f"   {name:45s} E ln(neurons per type ratio) {x:5.2f}   predicted {-c*x:+.3f}   measured {d[j]:+.3f}")
