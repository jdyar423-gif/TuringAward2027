"""Keyword count over the Opus readers' notes on the reference pair set ref-falcon-vs-mobilellm-test
(the two opponents against each other on the 200 test prompts; no reply of ours). A note is split
into clauses at ';' and '.'; a clause is attributed to the reply it names first ("A ..."/"B ...",
"A's"/"B's"); the key says which reply is MobileLLM-R1.5's (y). Counts are per pair and reading.
Heuristic: a clause can name both replies ("Both ...": counted for both)."""
import json, re, sys
from pathlib import Path
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
D = N / "final_eval/ref/ref-falcon-vs-mobilellm-test"
key = json.load(open(N / "study/proxy/keys/ref-falcon-vs-mobilellm-test.json"))
asg = key["assignment"]
KW = {"box": r"\bbox", "loop": r"\bloop|repeat|repetit", "truncated": r"truncat|cut off|unfinished|never finish|never settles|incomplete",
      "reasoning": r"reason|think|meta|deliberat|musing|monolog"}
out = {}
for reading in ("r1", "r2"):
    # in r2 the sides are exchanged: A of r2 is B of the key's assignment
    hits = {k: set() for k in KW}
    anyhit = set(); npairs = 0
    for f in sorted((D / reading).glob("chunk_*.notes.json")):
        notes = json.load(open(f))
        for pid, note in notes.items():
            npairs += 1
            x_is_a = asg[pid] == "x"
            if reading == "r2":
                x_is_a = not x_is_a
            mob = "B" if x_is_a else "A"
            for clause in re.split(r"[;.]", note):
                c = clause.strip()
                m = re.match(r"^(Both|A|B)\b", c)
                names = set()
                if m:
                    names = {"A", "B"} if m.group(1) == "Both" else {m.group(1)}
                else:
                    for s in ("A", "B"):
                        if re.search(rf"\b{s}\b|\b{s}'s\b", c):
                            names.add(s); break
                if mob in names:
                    for k, rx in KW.items():
                        if re.search(rx, c, re.I):
                            hits[k].add(pid); anyhit.add(pid)
    out[reading] = {"pairs_with_notes": npairs, **{k: len(v) for k, v in hits.items()}, "any_of_these": len(anyhit)}
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / "ref_notes.json").write_text(json.dumps(out, indent=1))
