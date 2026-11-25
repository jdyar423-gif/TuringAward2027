"""Fourth pass (REVIEW_2 items 3, 18, 19, 22): exact two-sided sign tests and Wilson intervals for the
counts the abstract and Table 7 quote, from the W / L / T of lead3_work/readers.json (the repository's key
check and tally over study/proxy/ratings, reproduced by REVIEW_2), the raw verdict counts of the
first Opus reading of three s2 sets, and which reading files each development set has.
Arguments: scratchpad dir, output json.
"""
import collections
import json
import math
import sys
from pathlib import Path

N, OUT = Path(sys.argv[1]), Path(sys.argv[2])
Z = 1.959963984540054


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(x, n):
    p = x / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z / (1 + Z * Z / n) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return c - h, c + h


rd = json.load(open(N / "paper" / "lead3_work" / "readers.json"))
out = {"counts": {}, "raw_first_reading": {}, "reading_files": {}}
want = {("s2-default-vs-falcon", "judge-opus"), ("s2-default-vs-falcon", "judge-qwen3-4b-2507-3605803b"),
        ("s2-default-vs-mobilellm", "judge-opus"), ("s2-default-vs-mobilellm", "judge-qwen3-4b-2507-3605803b"),
        ("s2-default-vs-mobilellm", "judge-gemma4-e4b-85a896a0"), ("s2-default-vs-falcon", "judge-gemma4-e4b-85a896a0")}


def find(obj, path=()):
    """Yield (path, dict) for every dict that holds wins/losses/ties counts."""
    if isinstance(obj, dict):
        keys = set(obj)
        if {"wins", "losses", "ties"} <= keys or {"W", "L", "T"} <= keys:
            yield path, obj
        for k, v in obj.items():
            yield from find(v, path + (k,))


for path, d in find(rd):
    w = d.get("wins", d.get("W"))
    l = d.get("losses", d.get("L"))
    t = d.get("ties", d.get("T"))
    if not all(isinstance(x, int) for x in (w, l, t)):
        continue
    n = w + l + t
    if n == 0:
        continue
    key = "/".join(map(str, path))
    lo, hi = wilson(w + t / 2, n)
    out["counts"][key] = {"W": w, "L": l, "T": t, "n": n, "sign_p": round(sign_p(w, l), 6),
                          "score": round((w + t / 2) / n, 4), "wilson": [round(lo, 6), round(hi, 6)]}
for s in ("s2-ngram3-vs-default", "s2-default-vs-mobilellm", "s2-default-vs-falcon"):
    c = collections.Counter(json.loads(x)["verdict"] for x in open(N / "study/proxy/ratings" / s / "opus-1.jsonl"))
    out["raw_first_reading"][s] = dict(c)
for d in sorted((N / "study/proxy/ratings").iterdir()):
    if d.name.startswith("final-"):
        continue
    out["reading_files"][d.name] = sorted(p.name for p in d.iterdir())
OUT.write_text(json.dumps(out, indent=1))
for k, v in out["counts"].items():
    if any(s in k for s in ("s2-default", "s1-default", "ref-")):
        print(k, v)
print(json.dumps(out["raw_first_reading"]))
print(json.dumps(out["reading_files"]))
