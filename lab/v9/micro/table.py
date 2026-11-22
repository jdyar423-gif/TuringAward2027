"""Table of docs/V9.md section 8 from arms.json: relative held-out loss of every arm against
the reference at the three checkpoints (share-weighted web and chat: "mix"), the per-domain
values at the last one, and the paired sampling error (rows of the held-out set as clusters;
the formulas of bes.eval.microbench.paired_diff and compare).

    python lab/v9/micro/table.py
"""
import json
import math
from pathlib import Path

D = json.loads((Path(__file__).parent / "arms.json").read_text())
REF = D["arms"]["ref"]
SHARES = dict(D["mixture"])


def at(arm, tokens):
    return next(c for c in arm["curve"] if c["tokens"] == tokens)["eval"]


def paired(a, r):
    """(relative difference, relative standard error) of one domain."""
    d = [x - y for x, y in zip(a["ce"], r["ce"])]
    w = r["w"]
    ws = sum(w)
    delta = sum(d) / ws
    n = len(d)
    se = math.sqrt(n / (n - 1) * sum((di - delta * wi) ** 2 for di, wi in zip(d, w))) / ws
    return delta / r["loss"], se / r["loss"]


def mix(arm, tokens):
    ea, er = at(arm, tokens), at(REF, tokens)
    l_ref = l_arm = var = 0.0
    for dom, pi in SHARES.items():
        rel, se = paired(ea[dom], er[dom])
        l_ref += pi * er[dom]["loss"]
        l_arm += pi * er[dom]["loss"] * (1 + rel)
        var += (pi * se * er[dom]["loss"]) ** 2
    return (l_arm - l_ref) / l_ref, math.sqrt(var) / l_ref


def main():
    pts = [c["tokens"] for c in REF["curve"]]
    print("reference: " + "; ".join(
        f"{t:,} tokens " + " / ".join(f"{at(REF, t)[d]['loss']:.4f}" for d in ("web", "chat", "assistant"))
        for t in pts) + "  (web / chat / assistant)")
    head = ["arm"] + [f"{t // 1000}K" for t in pts] + ["SE", "web / chat / assistant at the end", "overrides"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, arm in D["arms"].items():
        if name == "ref":
            continue
        rels = [mix(arm, t) for t in pts]
        doms = " / ".join(f"{paired(at(arm, pts[-1])[d], at(REF, pts[-1])[d])[0]:+.2%}"
                          for d in ("web", "chat", "assistant"))
        print("| " + " | ".join([f"`{name}`"] + [f"{r:+.2%}" for r, _ in rels]
                                + [f"{rels[-1][1]:.2%}", doms, ", ".join(arm["overrides"]) or "-"]) + " |")


if __name__ == "__main__":
    main()
