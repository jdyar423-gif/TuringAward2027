"""Paired comparisons, reader agreement and kappa, recomputed from the per-prompt scores (read-only)."""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from bes.eval.proxy import score

d = json.load(open(sys.argv[1]))
SEEN = {"chat-002", "chat-017", "math-034", "instruct-034", "instruct-062", "writing-100", "code-026",
        "math-001", "code-002", "writing-001", "instruct-001"}


def pp(k):
    return d[k]["per_prompt"]


def paired(a, b, excl=False):
    A, B = pp(a), pp(b)
    if excl:
        A = {k: v for k, v in A.items() if k not in SEEN}
        B = {k: v for k, v in B.items() if k not in SEEN}
    r = score.paired_sign_test(A, B)
    common = sorted(set(A) & set(B))
    md = Fraction(sum(Fraction(A[k]) - Fraction(B[k]) for k in common), len(common))
    r["mean_exact"] = str(md)
    r["p_full"] = score.sign_test(r["a_better"], r["b_better"])
    return r


pairs = [
    ("final-prefopt-vs-falcon|judge-opus", "final-prefopt-vs-falcon-k6lp|judge-opus"),
    ("final-prefopt-vs-falcon|judge-opus", "final-prefopt-vs-falcon-k6ds|judge-opus"),
    ("final-prefopt-vs-falcon|judge-opus", "final-k1-vs-falcon|judge-opus"),
    ("final-prefopt-vs-mobilellm|judge-opus", "final-k1-vs-mobilellm|judge-opus"),
    ("final-prefopt-vs-falcon|judge-opus", "final-prefopt-vs-falcon|judge-opus-neutral"),
    ("final-prefopt-vs-mobilellm|judge-opus", "final-prefopt-vs-mobilellm|judge-opus-neutral"),
    ("final-prefopt-vs-falcon-k6lp|judge-opus", "final-prefopt-vs-falcon-k6ds|judge-opus"),
]
for a, b in pairs:
    print(a, "VS", b)
    print("   200:", paired(a, b))
    print("   189:", paired(a, b, True))


def cls(s):
    return {1.0: "win", 0.0: "loss", 0.5: "tie"}[s]


def agreement(a, b):
    A, B = pp(a), pp(b)
    common = sorted(set(A) & set(B))
    both = [(cls(A[k]), cls(B[k])) for k in common]
    dec_both = [(x, y) for x, y in both if x != "tie" and y != "tie"]
    agree = sum(1 for x, y in dec_both if x == y)
    opposite = [(k) for k in common if {cls(A[k]), cls(B[k])} == {"win", "loss"}]
    # Cohen's kappa over 3 categories
    n = len(both)
    po = sum(1 for x, y in both if x == y) / n
    ca, cb = Counter(x for x, _ in both), Counter(y for _, y in both)
    pe = sum(ca[c] * cb[c] for c in ("win", "loss", "tie")) / n / n
    kappa = (po - pe) / (1 - pe)
    tab = Counter(both)
    return {"n": n, "both_decide": len(dec_both), "agree_where_both_decide": agree,
            "decided_opposite": len(opposite), "opposite_ids": opposite, "kappa3": round(kappa, 4),
            "po": po, "table": {f"{x}/{y}": v for (x, y), v in sorted(tab.items())}}


for a, b in [("final-prefopt-vs-falcon|judge-opus", "final-prefopt-vs-falcon|judge-qwen3-4b-2507-3605803b"),
             ("final-prefopt-vs-mobilellm|judge-opus", "final-prefopt-vs-mobilellm|judge-qwen3-4b-2507-3605803b"),
             ("final-prefopt-vs-falcon|judge-opus", "final-prefopt-vs-falcon|judge-opus-neutral"),
             ("final-prefopt-vs-mobilellm|judge-opus", "final-prefopt-vs-mobilellm|judge-opus-neutral"),
             ("final-prefopt-vs-falcon|judge-opus-neutral", "final-prefopt-vs-falcon|judge-qwen3-4b-2507-3605803b"),
             ("final-prefopt-vs-mobilellm|judge-opus-neutral", "final-prefopt-vs-mobilellm|judge-qwen3-4b-2507-3605803b")]:
    print("AGREE", a, "VS", b, agreement(a, b))
