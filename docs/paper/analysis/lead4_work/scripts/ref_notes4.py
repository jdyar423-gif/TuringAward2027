"""Fourth pass (REVIEW_2 item 14): what the readers' notes on the reference pair set say where they
mention a boxed answer of MobileLLM-R1.5. Same clause attribution as lead3_work/ref_notes.py (a note split at
';' and '.', a clause attributed to the reply it names first; the key says which reply is MobileLLM-R1.5's;
in r2 the sides are exchanged). For the pairs whose MobileLLM-R1.5 clauses match \\bbox, counts how many of
those clauses, or of the pair's other MobileLLM-R1.5 clauses, also call the reply looping, repeated,
nonsense, degenerate, fragmentary or incoherent. A keyword count, not a reading. No reply of ours is in
this set. Arguments: scratchpad dir, output json.
"""
import json
import re
import sys
from pathlib import Path

N, OUT = Path(sys.argv[1]), Path(sys.argv[2])
D = N / "final_eval/ref/ref-falcon-vs-mobilellm-test"
asg = json.load(open(N / "study/proxy/keys/ref-falcon-vs-mobilellm-test.json"))["assignment"]
BOX = r"\bbox"
BAD = r"\bloop|repeat|repetit|nonsens|degenerat|fragment|garbage|gibberish|incoheren|garbled"
out = {}
examples = []
for reading in ("r1", "r2"):
    box, box_bad_same_clause, box_bad_any_clause, npairs = set(), set(), set(), 0
    for f in sorted((D / reading).glob("chunk_*.notes.json")):
        for pid, note in json.load(open(f)).items():
            npairs += 1
            x_is_a = asg[pid] == "x"
            if reading == "r2":
                x_is_a = not x_is_a
            mob = "B" if x_is_a else "A"
            mob_clauses = []
            for clause in re.split(r"[;.]", note):
                c = clause.strip()
                m = re.match(r"^(Both|A|B)\b", c)
                names = set()
                if m:
                    names = {"A", "B"} if m.group(1) == "Both" else {m.group(1)}
                else:
                    for s in ("A", "B"):
                        if re.search(rf"\b{s}\b|\b{s}'s\b", c):
                            names.add(s)
                            break
                if mob in names:
                    mob_clauses.append(c)
            bc = [c for c in mob_clauses if re.search(BOX, c, re.I)]
            if bc:
                box.add(pid)
                if any(re.search(BAD, c, re.I) for c in bc):
                    box_bad_same_clause.add(pid)
                if any(re.search(BAD, c, re.I) for c in mob_clauses):
                    box_bad_any_clause.add(pid)
                if len(examples) < 12:
                    examples.append((reading, bc[0][:120]))
    out[reading] = {"pairs_with_notes": npairs, "box": len(box), "box_and_bad_in_the_same_clause": len(box_bad_same_clause),
                    "box_and_bad_in_any_clause_on_its_reply": len(box_bad_any_clause)}
out["examples"] = examples
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
