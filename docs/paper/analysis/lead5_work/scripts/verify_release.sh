#!/usr/bin/env bash
# Fifth pass: the release in the repository (R/release/main16-v9) against its parts.json, against the parts fetched
# from the runtime (N/packs/official_export), and the files of its extracted copy (N/release_check/final, unpacked
# from the committed parts) against SHA256SUMS. Read-only. Prints one line per check.
set -u
R=/home/user/turingaward2027/release/main16-v9
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
cd "$R"
python3 - <<'PY'
import hashlib, json
R = "/home/user/turingaward2027/release/main16-v9"
N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
pj = json.load(open(f"{R}/parts.json"))
whole = hashlib.sha256(); ok = True; total = 0
for p in pj["parts"]:
    b = open(f"{R}/{p['name']}", "rb").read()
    h = hashlib.sha256(b).hexdigest(); whole.update(b); total += len(b)
    same = b == open(f"{N}/packs/official_export/{p['name']}", "rb").read()
    ok &= (h == p["sha256"]) and len(b) == p["bytes"] and same
    print(p["name"], len(b), h == p["sha256"], "same as the fetched part:", same)
print("parts:", len(pj["parts"]), "bytes", total, "== parts.json", total == pj["bytes"], "; whole-tar sha256 matches:", whole.hexdigest() == pj["sha256"], whole.hexdigest())
print("ALL PARTS OK:", ok)
PY
cd "$N/release_check/final" && sha256sum -c SHA256SUMS 2>&1 | awk '{print}' | sort | uniq -c | sort -rn | head -5
sha256sum -c SHA256SUMS 2>&1 | grep -vc ": OK$" | sed 's/^/files not OK: /'
