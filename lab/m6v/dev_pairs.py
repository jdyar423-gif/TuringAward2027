"""Pairs for the registered judge from replies of ours and an opponent's stored replies.

    python lab/m6v/dev_pairs.py OURS/replies.jsonl OPPONENT.jsonl SPLIT OUT_DIR

OURS: the ``replies.jsonl`` of ``scripts/gen_test_replies.py`` (``id``, ``turns``, ``replies``,
``finish``). OPPONENT: a harness output file (``prompt_id``, ``turns[].reply``). Writes
``OUT_DIR/pairs.json`` (``[{"pair_id", "turns": [{"user", "A", "B"}]}]``, A = ours; the judge
reads both orders) and ``OUT_DIR/rows.json`` (category, turns, how each reply of ours was made:
the library tier and the entries it read). Prompts of ours that are missing are listed."""
import json
import sys
from pathlib import Path


def tier_of(fin: dict) -> str:
    lib = (fin or {}).get("library")
    if not lib:
        return "none"
    if isinstance(lib, dict):
        for k in ("tier", "mode", "kind"):
            if lib.get(k):
                return str(lib[k])
        if lib.get("verbatim"):
            return "verbatim"
        return "exemplar" if (lib.get("entries") or lib.get("passages") or lib.get("exemplars")) else "none"
    return str(lib)


def main(ours_path: str, opp_path: str, split: str, out_dir: str) -> int:
    ours = {}
    for line in Path(ours_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("split", split) == split:
                ours[r["id"]] = r
    opp = {}
    for line in Path(opp_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("split", split) == split:
                opp[r["prompt_id"]] = r
    pairs, rows, missing = [], [], []
    for pid in sorted(opp):
        o = opp[pid]
        m = ours.get(pid)
        if m is None or len(m["replies"]) != len(o["turns"]):
            missing.append(pid)
            continue
        turns = [{"user": t["user"], "A": a or "", "B": t.get("reply") or ""} for t, a in zip(o["turns"], m["replies"])]
        if [t["user"] for t in o["turns"]] != list(m["turns"]):
            raise SystemExit(f"{pid}: the user turns of the two sides differ")
        pairs.append({"pair_id": pid, "turns": turns})
        fins = m.get("finish") or []
        rows.append({"pair_id": pid, "category": o.get("category") or m.get("category"), "turns": len(turns),
                     "tiers": [tier_of(f) for f in fins], "library": [(f or {}).get("library") for f in fins],
                     "tokens": [(f or {}).get("tokens") for f in fins],
                     "finish": [(f or {}).get("finish_reason") for f in fins],
                     "seconds": m.get("seconds_total")})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps(pairs))
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(pairs)} pairs of the {split} split ({len(missing)} prompts of ours missing"
          + (f": {missing[:8]}" if missing else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
