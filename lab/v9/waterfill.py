"""Water-filling model of token pages against the lab measurements.

Model: a token x seen with frequency f_x that owns rho_x private neurons gains  c * log(1 + rho_x / rho0)  nats at each
of its positions, so the total gain is  G = sum_x f_x c log(1 + rho_x / rho0).  Under a storage constraint
sum_x rho_x = R the maximiser is  rho_x = max(0, f_x * (R + n_act * rho0) / F_act - rho0)  (water-filling), optionally capped.
Fit (c, rho0) on two lab arms at 6.6M tokens and predict the third; then evaluate other allocations.
Lab arms (same storage, mean 2 neurons per token type, V = 1040, both blocks):
    uniform rho = 2: -0.169      frequency-proportional, cap 8: -0.221      frequency-proportional, cap 32: -0.226
"""
import numpy as np, os
SP = os.environ.get("LAB_DATA", os.path.dirname(os.path.abspath(__file__)))   # token files and runs/
V = 1040
web = np.memmap(os.path.join(SP, "web1k.u16"), dtype=np.uint16, mode="r")[:40_000_000]
chat = np.memmap(os.path.join(SP, "chat1k.u16"), dtype=np.uint16, mode="r")[:10_000_000]
f = 0.8 * np.bincount(web, minlength=V) / len(web) + 0.2 * np.bincount(chat, minlength=V) / len(chat)
f = f / f.sum()

def alloc(mean, gamma, cap):                       # the lab's rule (lab12.py): round(scale * f^gamma), capped, total = mean * V
    lo, hi = 0.0, 1e12
    for _ in range(80):
        mid = (lo + hi) / 2
        r = np.minimum(np.floor(mid * f ** gamma + 0.5), cap)
        lo, hi = (mid, hi) if r.sum() < mean * V else (lo, mid)
    return np.minimum(np.floor(hi * f ** gamma + 0.5), cap)

def gain(rho, c, rho0):
    return float((f * c * np.log1p(rho / rho0)).sum())

arms = {"uniform2": (np.full(V, 2.0), 0.169), "prop_cap8": (alloc(2, 1.0, 8), 0.221), "prop_cap32": (alloc(2, 1.0, 32), 0.226)}
for k, (r, g) in arms.items():
    print(f"{k:10s} measured {g:.3f}  neurons: total {int(r.sum())}, tokens with 0: {int((r == 0).sum())}, at cap: {int((r == r.max()).sum())}, mean read {float((f * r).sum()):.2f}")
# fit on uniform2 and prop_cap32 by a grid over rho0, c from the first arm
best = None
for rho0 in np.exp(np.linspace(np.log(0.05), np.log(50), 400)):
    c = 0.169 / gain(arms["uniform2"][0], 1.0, rho0)
    err = abs(gain(arms["prop_cap32"][0], c, rho0) - 0.226)
    if best is None or err < best[0]:
        best = (err, rho0, c)
_, rho0, c = best
print(f"fit on uniform2 + prop_cap32: rho0 = {rho0:.2f} neurons, c = {c:.4f} nats;  prediction for prop_cap8: {gain(arms['prop_cap8'][0], c, rho0):.3f} (measured 0.221)")
def waterfill(R, cap=np.inf):
    act = np.ones(V, bool)
    for _ in range(200):
        lam_inv = (R + act.sum() * rho0) / f[act].sum()
        r = np.where(act, f * lam_inv - rho0, 0.0)
        new = r > 0
        if (new == act).all(): break
        act = new
    return np.minimum(np.maximum(r, 0), cap)
for name, r in (("water-filling, same storage (continuous)", waterfill(2 * V)), ("uniform 8 (4x storage)", np.full(V, 8.0)),
                ("uniform 1", np.full(V, 1.0)), ("prop cap 4 (lab arm pending)", alloc(2, 1.0, 4)), ("tiers 8/4/2 by frequency rank (mean 4)", None)):
    if r is None:
        order = np.argsort(-f); r = np.zeros(V); n = V // 4
        r[order[:n]] = 8; r[order[n:2 * n]] = 4; r[order[2 * n:]] = 2
    print(f"{name:45s} total {r.sum():7.0f}  predicted gain {gain(r, c, rho0):.3f}  mean read {float((f * r).sum()):.2f}")
