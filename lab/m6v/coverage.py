import sys, numpy as np
sys.path.insert(0, "/home/user/turingaward2027")
from bes.eval.proxy import prompts as prompt_set
from bes.library.retrieve import Library
from bes.library.qa import question_key
lib = Library.load("/home/user/turingaward2027/data_cache/library/answers_v1")
ps = prompt_set.select(prompt_set.load(), "dev")
def tops(q, exclude=()):
    ents, part = lib.partial(q, exclude=exclude)
    out = {}
    for name, mod in (("100%", 1), ("50%", 2), ("25%", 4), ("12.5%", 8), ("6.25%", 16), ("3.1%", 32)):
        m = part[(ents % mod) == 0] if mod > 1 else part
        out[name] = float(m[0]) if m.size else 0.0
    return out, int(ents.size)
rows = []
for p in ps:
    q = p.turns[0]
    if len(q) > 1000:
        rows.append((p.category, {k: 0.0 for k in ("100%", "50%", "25%", "12.5%", "6.25%", "3.1%")}, 0)); continue
    t, n = tops(q)
    rows.append((p.category, t, n))
def show(tag, rows):
    print(tag, "n =", len(rows))
    for name in ("100%", "50%", "25%", "12.5%", "6.25%", "3.1%"):
        v = np.array([r[1][name] for r in rows])
        print(f"  library {name:6s}: mean top partial {v.mean():.3f} | share >=0.8: {np.mean(v>=0.8):.2f}  >=0.6: {np.mean(v>=0.6):.2f}  >=0.4: {np.mean(v>=0.4):.2f}  none: {np.mean(v==0):.2f}")
show("DEV prompts (first turn)", rows)
for cat in ("chat", "code", "instruct", "math", "writing"):
    v = np.array([r[1]["100%"] for r in rows if r[0] == cat])
    print(f"  {cat:9s} n={v.size} mean {v.mean():.3f}  >=0.6: {np.mean(v>=0.6):.2f}  >=0.4: {np.mean(v>=0.4):.2f}")
rng = np.random.default_rng(7)
rows2 = []
while len(rows2) < 300:
    e = int(rng.integers(lib.n_entries))
    q = lib.question(e)
    if not (20 <= len(q) <= 300):
        continue
    t, n = tops(q, exclude=[e])
    rows2.append(("lib", t, n))
show("random library questions, leave-one-out", rows2)
print("median candidates per query: dev", int(np.median([r[2] for r in rows])), " lib", int(np.median([r[2] for r in rows2])))
