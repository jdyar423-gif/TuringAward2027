"""Keyword count over the Opus readers' notes on the pairs they judged 'both bad' in the
development set where both replies are ours (s2-ngram3-vs-default: v9 at step 31,912, 3-gram
blocking against the default decoding; first reading). A note counts once per pattern."""
import collections, glob, json, re
from pathlib import Path
D = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/study/s2/judge/s2-ngram3-vs-default")
notes, ver = {}, {}
for f in sorted(D.glob("chunk_*.notes.json")):
    notes.update(json.load(open(f)))
for f in sorted(D.glob("chunk_*.verdicts.json")):
    ver.update(json.load(open(f)))
bb = [notes[p] for p, v in ver.items() if v == "both_bad"]
pat = {"wrong facts or arithmetic": r"wrong|incorrect|false|error", "incoherent text": r"incoheren|nonsense|garbl|gibberish",
       "loops or repetition": r"loop|repeat|repetit", "off the prompt": r"off-topic|off topic|off-prompt|off-target|ignores|irrelevant"}
out = {"verdicts": dict(collections.Counter(ver.values())), "both_bad": len(bb),
       "notes_matching": {k: sum(bool(re.search(rx, n, re.I)) for n in bb) for k, rx in pat.items()},
       "examples": bb[:3]}
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / "ours_notes.json").write_text(json.dumps(out, indent=1))
