"""Review item 42: hand labels (one annotator, the second-pass lead author, not blind to the rule,
reading only the stored first 150 characters of each assistant opening) of the 30 dropped samples
of the two sources the deflection rule hits hardest (N/deflection_check2.json, 'dropped').
Categories:
  P  pure deflection: says what the assistant is or lacks; no help in the visible opening
  H  disclaimer, then help or a pointer ('However, I can ...', 'you can visit ...')
  C  asks for input the user referred to but did not give
  S  safety-relevant referral (distress, mental health)
  A  answers directly after a persona prefix
  L  says the information is not in the given text (the behaviour the retrieval sources teach)
The kept_near samples are near misses chosen by the scan, not a random sample of what was kept:
they cannot estimate recall, and are only counted for persona-style deflections the rule keeps."""
import json, collections
labels = {
 "smoltalk_systemchats": "H H P P P S H H S H H A P P H H H H H H P H P P H H H H H H",
 "ultrachat":            "C H H C H H C H H H H H H H H H H H H H H H H H H H L H H H",
}
kept_near_persona_pure = {"smoltalk_systemchats": [3, 9, 11, 13, 18, 26, 27, 29, 39]}
d = json.load(open("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/deflection_check2.json"))
out = {}
for src, s in labels.items():
    lab = s.split()
    assert len(lab) == len(d[src]["dropped"]) == 30, (src, len(lab))
    c = collections.Counter(lab)
    out[src] = {"n": len(lab), "counts": dict(c), "labels": lab,
                "rule_share_of_conversations": d[src]["deflection"] / d[src]["conversations"]}
    print(src, dict(c))
tot = collections.Counter(l for s in labels.values() for l in s.split())
print("both", dict(tot), "of", sum(tot.values()))
out["both"] = dict(tot)
out["kept_near_persona_pure"] = kept_near_persona_pure
json.dump(out, open("deflection_labels.json", "w"), indent=1)
