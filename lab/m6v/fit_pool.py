"""Every stored answer the reader's test could take for a split's single-turn prompts.

    BES_PROFILE=m6 JAX_PLATFORMS=cpu python lab/m6v/fit_pool.py SHIPPED_EXPORT SPLIT OPPONENT.jsonl OUT_DIR \\
        [--rank 0.35] [--pairs-from -0.3] [--cut]

For each single-turn prompt of SPLIT (``bes.eval.proxy.prompts``), the shipped export's library is
asked for its three best stored questions exactly as ``AnswerLibrary.verbatim`` asks; every
candidate without a contrast, whose own overlap reaches ``--rank`` and whose answer meets the
prompt's explicit constraints, is scored by the export's model (``bes.decode.fitness.AnswerFit``).

Writes ``OUT_DIR/rows.json`` (one row per prompt and candidate: the index's coverages, whether the
library's own rule takes it, the fit) and ``OUT_DIR/pairs.json`` (the stored answer as reply A
against the opponent's stored reply B, for the candidates whose fit reaches ``--pairs-from`` or
that the library's own rule takes) for ``pairjudge.py``. Rows of this file with the judge's
verdicts are what ``lab/m6v/fit_compose.py`` composes decode configurations from."""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv):
    opts = {"--rank": 0.35, "--pairs-from": -0.3}
    for k in list(opts):
        if k in argv:
            i = argv.index(k)
            opts[k] = float(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
    cut = "--cut" in argv              # decode.cut_end_markers: score and pair the answer as it would be given
    argv = [a for a in argv if a != "--cut"]
    exp, split, opp_path, out_dir = argv[:4]
    from bes import export as E
    from bes.decode.fitness import AnswerFit
    from bes.decode.respond import cut_end_markers
    from bes.eval.proxy import prompts as P
    from bes.exact import constraints

    ex = E.load_export(exp)
    lib, cfg = ex.library, ex.library.acfg
    fit = AnswerFit(ex.params, ex.consts, ex.cfg.model, ex.tok)
    opp = {}
    for line in Path(opp_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("split", split) == split:
                opp[r["prompt_id"]] = r
    looked = min(cfg.exemplar_min, cfg.second_min, cfg.verbatim_cov_q, cfg.verbatim_cov_e)
    floor = opts["--rank"] if opts["--rank"] < looked else None
    rows, pairs, t0 = [], [], time.time()
    prompts = [p for p in P.select(P.load(), split) if not p.multi_turn]
    for n, p in enumerate(prompts):
        q = p.turns[0]
        if len(q) > cfg.max_query_chars:
            continue
        try:
            spec = constraints.compile(q)
        except Exception:
            spec = None
        log = lib.new_read()
        cands = lib.candidates(q, 3, (), log) if floor is None else lib.candidates(q, 3, (), log, floor)
        for c in cands:
            if c.contrast is not None:
                continue
            own_rule = (c.type_match and c.cov_q >= cfg.verbatim_cov_q - 1e-12 and c.cov_e >= cfg.verbatim_cov_e - 1e-12
                        and max(c.miss_q, c.miss_e) <= cfg.verbatim_max_miss + 1e-12)
            own = min(c.cov_q, c.cov_e) - (0.0 if c.type_match else cfg.type_penalty)
            if not own_rule and own < opts["--rank"] - 1e-12:
                continue
            answer = c.answer or lib._answer(c.entry, log)
            if not answer:
                continue
            if spec is not None and getattr(spec, "constraints", None):
                try:
                    if constraints.verify(spec, answer):
                        continue
                except Exception:
                    continue
            given = cut_end_markers(answer.strip()) if cut else answer.strip()
            value = fit(q, c.question, given)
            row = {"pair_id": f"{p.id}#{c.entry}", "prompt_id": p.id, "category": p.category, "entry": int(c.entry),
                   "question": c.question, "cov_q": round(float(c.cov_q), 4), "cov_e": round(float(c.cov_e), 4),
                   "type_match": bool(c.type_match), "own": round(float(own), 4), "own_rule": bool(own_rule),
                   "fit": None if value is None else round(float(value), 4), "answer_chars": len(given),
                   "end_cut": given != answer.strip()}
            rows.append(row)
            if p.id in opp and (own_rule or (value is not None and value >= opts["--pairs-from"])):
                pairs.append({"pair_id": row["pair_id"],
                              "turns": [{"user": q, "A": given, "B": opp[p.id]["turns"][0].get("reply") or ""}]})
        if n % 25 == 24:
            print(f"{n + 1} of {len(prompts)} prompts, {len(rows)} candidates, {len(pairs)} pairs ({time.time() - t0:.0f} s)",
                  flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "pairs.json").write_text(json.dumps(pairs))
    asked = len({r["prompt_id"] for r in rows})
    print(f"{len(prompts)} single-turn prompts; {asked} with a candidate ({len(rows)} candidates); {len(pairs)} pairs to judge; "
          f"{fit.calls} passes of the model in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
