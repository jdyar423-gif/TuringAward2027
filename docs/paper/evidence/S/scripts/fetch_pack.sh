#!/usr/bin/env bash
# Pack a directory of the runtime into 16 MB parts and download them with their SHA-256 checked.
# usage: fetch_pack.sh RUNTIME_DIR NAME OUT_DIR      (OUT_DIR/parts.json, OUT_DIR/NAME.tar.NNN; then
#        `cat OUT_DIR/NAME.tar.* | tar -x -C SOMEWHERE` gives the directory)
set -u
src="$1"; name="$2"; out="$3"; ses=${SESSION:-bes24h}
export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json COLAB_CLI_ARGS=--auth=adc
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
mkdir -p "$out"
date +%s > "$S/presence_pause.txt"; timeout 60 colab --auth=adc upload -s "$ses" "$S/presence_pause.txt" /content/bes_cli/presence_pause </dev/null >/dev/null 2>&1
timeout 60 colab --auth=adc upload -s "$ses" "$S/scripts/pack_dir.py" /content/bes_cli/pack_dir.py </dev/null >/dev/null 2>&1; sleep 3
TAG=pk$name bash "$S/scripts/exec_detached.sh" "$ses" "$S/pk${name}_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/pack_dir.py $src $name 16" | tail -n 1
TAG=pk$name timeout 400 bash "$S/scripts/exec_wait.sh" "$ses" "$S/pk${name}_nonce.txt" "$out/pack.log" 360 | tail -n 1
tail -n 1 "$out/pack.log" | cut -c1-200
rm -f "$out/parts.json"; timeout 60 colab --auth=adc download -s "$ses" "/content/bes_cli/packs/$name/parts.json" "$out/parts.json" </dev/null >/dev/null 2>&1 || { echo "no parts.json"; exit 1; }
python3 -I - "$out" "$name" "$ses" <<'PY'
import hashlib, json, os, subprocess, sys
out, name, ses = sys.argv[1:4]
meta = json.load(open(os.path.join(out, "parts.json")))
bad = 0
for p in meta["parts"]:
    dst = os.path.join(out, p["name"])
    ok = False
    for attempt in range(4):
        if os.path.exists(dst) and os.path.getsize(dst) == p["bytes"] and hashlib.sha256(open(dst, "rb").read()).hexdigest() == p["sha256"]:
            ok = True
            break
        if os.path.exists(dst):
            os.remove(dst)
        subprocess.run(["timeout", "300", "colab", "--auth=adc", "download", "-s", ses, f"/content/bes_cli/packs/{name}/{p['name']}", dst],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bad += not ok
whole = hashlib.sha256()
for p in meta["parts"]:
    f = os.path.join(out, p["name"])
    if os.path.exists(f):
        whole.update(open(f, "rb").read())
print(f"FETCHED {name}: {len(meta['parts']) - bad} of {len(meta['parts'])} parts verified, {meta['bytes']} bytes, whole tar sha256 {'OK' if whole.hexdigest() == meta['sha256'] else 'MISMATCH'}")
sys.exit(1 if bad or whole.hexdigest() != meta["sha256"] else 0)
PY
