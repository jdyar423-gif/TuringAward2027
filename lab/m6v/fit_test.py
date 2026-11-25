"""Does the reader itself know when a stored answer fits the user's question?

    BES_PROFILE=m6 JAX_PLATFORMS=cpu python lab/m6v/fit_test.py EXPORT STORED_DIR [OWN_DIR ...] [--json OUT]

STORED_DIR: the study of stored answers against the opponent (``rows.json``: one row per development
prompt and candidate entry with the index's coverages; ``pairs.json``: the texts; ``out*.jsonl``: the
judge's verdicts, A = the stored answer). OWN_DIR: judged replies of a model to the same prompts
(``judge*.jsonl``), for the comparison "the stored answer against the model's own reply".

For every pair the export's model scores the stored answer twice, as the reply to the user's question
and as the reply to the question it was stored under, with nothing else in the context:

    fit = mean over the answer's tokens of  log p(a_t | user's question, a_<t) - log p(a_t | stored question, a_<t)

An answer that suits both questions equally has fit = 0; one that answers something else is less
probable under the user's question (fit < 0). The table shows how the judge's verdict on the stored
answer depends on ``fit`` and on the index's rank (the smaller coverage, less the type penalty), and
the area under the curve of each as a predictor of "the stored answer did not lose"."""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

VAL = {"A": 1.0, "B": 0.0, "tie": 0.5}
TT = 320


def records(d):
    out = {}
    for pat in ("judge*.jsonl", "lane_*.jsonl", "out*.jsonl"):
        for f in sorted(Path(d).glob(pat)):
            for ln in open(f):
                if ln.strip():
                    r = json.loads(ln)
                    out.setdefault(r["pair_id"], r)
    return out


def auc(score, label):
    """P(score of a positive > score of a negative), ties counted half."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    exp, sdir, own_dirs = Path(argv[0]), Path(argv[1]), argv[2:]
    import jax

    from bes import export as E
    from bes.data.samples import WeightRule, conversation_sample, fit
    from bes.eval import microbench as MB
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer

    cfg = E.load_export_config(exp / "config.py")
    mcfg = E.check_model_cfg(cfg.model)
    rec = json.loads((exp / E.EXPORT_JSON).read_text())
    params = E.load_weights(exp, cfg.model)
    consts = M.make_consts(mcfg, **((rec.get("consts") or {}).get("enable") or {}))
    tok = load_tokenizer(exp / "tokenizer")
    ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])

    def nll(question, answer):
        s = conversation_sample([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], tok)
        head, _ = fit(s, TT + 1)
        if head is None:
            return None
        out = MB._empty_rows(1, TT)
        MB._put_row(out, 0, [(head.ids, head.roles)], TT, WeightRule(assistant_only=True))
        if out["w"].sum() < 2:
            return None
        m = ev(params, MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok))
        return float(m["ce_sum"]) / max(float(m["weight"]), 1e-9), float(m["weight"])

    rows = {r["pair_id"]: r for r in json.loads((sdir / "rows.json").read_text())}
    pairs = {p["pair_id"]: p for p in json.loads((sdir / "pairs.json").read_text())}
    verd = records(sdir)
    own = [(Path(d).name, records(d)) for d in own_dirs]
    rank = lambda r: min(r["cov_q"], r["cov_e"]) - (0.0 if r["type_match"] else 0.1)  # noqa: E731
    t0 = time.time()
    table = []
    for pid, r in rows.items():
        if pid not in verd or len(pairs[pid]["turns"]) != 1:
            continue
        turn = pairs[pid]["turns"][0]
        a, b = nll(turn["user"], turn["A"]), nll(r["question"], turn["A"])
        if a is None or b is None:
            continue
        row = {"pair_id": pid, "prompt_id": r["prompt_id"], "category": r["category"], "rank": rank(r),
               "cov_q": r["cov_q"], "cov_e": r["cov_e"], "type_match": r["type_match"], "meets": r["meets_constraints"],
               "nll_user": a[0], "nll_stored": b[0], "fit": b[0] - a[0], "tokens": a[1], "stored": VAL[verd[pid]["verdict"]],
               "user": turn["user"], "question": r["question"]}
        for name, rc in own:
            if r["prompt_id"] in rc:
                row["own_" + name] = VAL[rc[r["prompt_id"]]["verdict"]]
        table.append(row)
    print(f"{len(table)} pairs scored in {time.time() - t0:.0f} s (answers cut at {TT} tokens)")
    table.sort(key=lambda x: -x["fit"])
    for x in table:
        owns = " ".join(f"{x.get('own_' + n, float('nan')):.1f}" for n, _ in own)
        print(f"fit {x['fit']:+.3f}  rank {x['rank']:.2f}  stored {x['stored']:.1f}  own {owns}  {x['prompt_id']:12s} "
              f"{x['user'][:52]!r} <- {x['question'][:52]!r}")
    y = [x["stored"] >= 0.5 for x in table]
    print(f"\nstored answer did not lose: {sum(y)} of {len(y)};  AUC of fit {auc([x['fit'] for x in table], y):.3f},  "
          f"of the index's rank {auc([x['rank'] for x in table], y):.3f},  of -nll under the user's question "
          f"{auc([-x['nll_user'] for x in table], y):.3f}")
    y2 = [x["stored"] == 1.0 for x in table]
    print(f"stored answer won: {sum(y2)} of {len(y2)};  AUC of fit {auc([x['fit'] for x in table], y2):.3f},  "
          f"of rank {auc([x['rank'] for x in table], y2):.3f}")
    for lo, hi in ((-0.05, 9), (-0.1, -0.05), (-0.2, -0.1), (-0.4, -0.2), (-9, -0.4)):
        g = [x for x in table if lo <= x["fit"] < hi]
        if g:
            line = f"fit in [{lo:+.2f}, {hi:+.2f}): n {len(g):2d}  stored {np.mean([x['stored'] for x in g]):.3f}"
            for n, _ in own:
                h = [x for x in g if "own_" + n in x]
                if h:
                    line += (f"  | own {n} {np.mean([x['own_' + n] for x in h]):.3f} (stored better "
                             f"{sum(x['stored'] > x['own_' + n] for x in h)}, worse {sum(x['stored'] < x['own_' + n] for x in h)})")
            print(line)
    if out_json:
        Path(out_json).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
