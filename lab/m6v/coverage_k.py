"""Development coverage of the library that fits beside the rows of K tokens (the water line):
tiers of the 200 first turns under the library's own floors and in the loose mode."""
import collections, json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import subset_answers as SA
from bes.eval.proxy import prompts as P
from bes.exact import constraints
from bes.library.answers import AnswerLibrary
N = "SCRATCH"          # the directory that holds m6v/credit_answers_v1.npy (scripts/subset_answers.py --credit)
cr = np.load(N + "/m6v/credit_answers_v1.npy")
dev = P.select(P.load(), "dev")
out = []
for K in (5120, 4096, 3072, 2048):
    weights = 74_574_216 - (5120 - K) * 7680
    room = 100_000_000 - weights - 60_000
    lib, rec = SA.subset("data_cache/library/answers_v1", room, read_cap=360_000, cr=cr, log=lambda m: None)
    row = {"K": K, "weights_and_map": weights, "library_numbers": rec["numbers"], "entries": rec["kept"]}
    for name, floor in (("default", None), ("loose0.40", 0.40), ("loose0.35", 0.35)):
        tiers = collections.Counter(); top = []
        for p in dev:
            q = p.turns[0]
            if lib.verbatim(q, constraints.compile(q)) is not None:
                tiers["verbatim"] += 1
            elif lib.exemplars(q, floor=floor):
                tiers["exemplar"] += 1
            else:
                tiers["none"] += 1
        row[name] = dict(tiers)
    ents = [c for p in dev for c in [lib.partial(p.turns[0])] ]
    row["mean_top_partial"] = float(np.mean([float(part[0]) if len(part) else 0.0 for _, part in ents]))
    out.append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open(N + "/m6v/coverage_k.json", "w"), indent=1)
