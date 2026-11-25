"""Which statistic of the reader's two passes says best that a stored answer suits the user's question?

    BES_PROFILE=m6 JAX_PLATFORMS=cpu python lab/m6v/fit_stats.py SHIPPED_EXPORT POOL_DIR OUT.json

POOL_DIR: ``rows.json`` and ``pairs.json`` of ``lab/m6v/fit_pool.py``. For every candidate with a
pair the per-token losses of the stored answer are computed under the user's question and under the
stored one; ``d_t`` is their difference (positive: the user's question makes token ``t`` less
probable). OUT.json holds, per candidate, the answer's token count and ``d`` itself, so that any
statistic (mean, sum, worst token, the first tokens only) can be compared against the judge's
verdicts without another pass of the model."""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv):
    exp, pdir, out = argv[:3]
    import jax

    from bes import export as E
    from bes.data import packing as P
    from bes.data.samples import WeightRule
    from bes.decode.fitness import AnswerFit
    from bes.exact.features import exact_features
    from bes.model import model as M

    ex = E.load_export(exp, library=None)
    fit = AnswerFit(ex.params, ex.consts, ex.cfg.model, ex.tok)
    consts, cfg, tok, T = ex.consts, ex.cfg.model, ex.tok, fit.T

    def per_token(p, b):
        feats, y, w = M.split_batch(b)
        h, _ = M.forward(p, consts, feats, cfg)
        return M.token_ce(p, consts, h, feats, y, cfg), w.reshape(-1)

    f = jax.jit(per_token)

    def losses(question, answer):
        ids, roles = fit._render(question, answer)
        row = P.assemble_row([(np.asarray(ids, np.int32), np.asarray(roles, np.int8))], T, WeightRule(assistant_only=True))
        tokens = np.asarray(row.tokens, np.int32)[None]
        seg = np.asarray(row.seg, np.int32)[None]
        batch = {"tokens": tokens, "w": np.asarray(row.w, np.float32)[None], "seg": seg,
                 **exact_features(tokens[:, :T], seg, tok)}
        ce, w = f(ex.params, batch)
        ce, w = np.asarray(ce), np.asarray(w)
        return ce[w > 0]

    rows = {r["pair_id"]: r for r in json.loads((Path(pdir) / "rows.json").read_text())}
    pairs = json.loads((Path(pdir) / "pairs.json").read_text())
    res, t0 = [], time.time()
    for p in pairs:
        r = rows[p["pair_id"]]
        q, a = p["turns"][0]["user"], p["turns"][0]["A"]
        text = fit.cut((q, r["question"]), a)
        if text is None:
            continue
        lu, ls = losses(q, text), losses(r["question"], text)
        if len(lu) != len(ls):
            continue
        d = (lu - ls).astype(float)
        res.append({"pair_id": p["pair_id"], "prompt_id": r["prompt_id"], "own": r["own"], "own_rule": r["own_rule"],
                    "fit": r["fit"], "n": int(len(d)), "d": [round(float(x), 4) for x in d]})
    Path(out).write_text(json.dumps(res))
    print(f"{len(res)} candidates in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
