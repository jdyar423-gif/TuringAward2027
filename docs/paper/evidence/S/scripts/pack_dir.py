"""Runtime side: pack a directory into an uncompressed tar split into parts, with a SHA-256 list.

    python pack_dir.py SRC_DIR NAME [PART_MB=16] [SESSION_DIR=/content/bes_cli/packs]

Writes <SESSION_DIR>/<NAME>/<NAME>.tar.NNN and parts.json ({"name", "bytes", "sha256" of the whole
tar, "parts": [{"name", "bytes", "sha256"}], "members": n}). The tar holds SRC_DIR's content
under the directory name of SRC_DIR."""
import hashlib, json, os, sys, tarfile

src, name = sys.argv[1].rstrip("/"), sys.argv[2]
part_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 16
base = os.path.join(sys.argv[4] if len(sys.argv) > 4 else "/content/bes_cli/packs", name)
os.makedirs(base, exist_ok=True)
for f in os.listdir(base):
    os.remove(os.path.join(base, f))
tar_path = os.path.join(base, name + ".tar")
n_members = 0
with tarfile.open(tar_path, "w") as tar:
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            tar.add(p, arcname=os.path.join(os.path.basename(src), os.path.relpath(p, src)))
            n_members += 1
whole, parts, i = hashlib.sha256(), [], 0
with open(tar_path, "rb") as f:
    while True:
        chunk = f.read(part_mb * 2 ** 20)
        if not chunk:
            break
        whole.update(chunk)
        pn = f"{name}.tar.{i:03d}"
        with open(os.path.join(base, pn), "wb") as g:
            g.write(chunk)
        parts.append({"name": pn, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        i += 1
size = os.path.getsize(tar_path)
os.remove(tar_path)
json.dump({"name": name, "source": src, "bytes": size, "sha256": whole.hexdigest(), "parts": parts, "members": n_members},
          open(os.path.join(base, "parts.json"), "w"), indent=1)
print(f"PACKED {name}: {n_members} files, {size} bytes, {len(parts)} parts of at most {part_mb} MB in {base}")
