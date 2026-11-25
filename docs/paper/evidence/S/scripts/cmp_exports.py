"""Runtime side: is the export a test generation used the same model as the run's own export?
    python cmp_exports.py /content/runs/main16/final /content/bes_cli/testgen_TAG/export [...]
Prints, per directory, the files of SHA256SUMS, the decode choice and the step; then which
files differ from the first directory."""
import json, sys
from pathlib import Path

def read(d):
    d = Path(d); sums = {}
    if (d / "SHA256SUMS").exists():
        for line in (d / "SHA256SUMS").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sums[parts[-1].lstrip("*")] = parts[0]
    cfg = {}
    for name in ("config.json", "export.json"):
        if (d / name).exists():
            try:
                cfg[name] = json.loads((d / name).read_text())
            except Exception as e:  # noqa: BLE001
                cfg[name] = {"error": str(e)}
    return sums, cfg

base = None
for d in sys.argv[1:]:
    sums, cfg = read(d)
    c, e = cfg.get("config.json", {}), cfg.get("export.json", {})
    print(f"== {d}: {len(sums)} files in SHA256SUMS")
    print("   decode_choice:", json.dumps(c.get("decode_choice"))[:300])
    for k in ("step", "source", "weights", "check", "library"):
        if k in e:
            print(f"   export.json {k}:", json.dumps(e[k])[:260])
    if base is None:
        base = sums
        for k, v in sorted(sums.items()):
            print("   ", v[:16], k)
    else:
        diff = sorted(k for k in set(base) | set(sums) if base.get(k) != sums.get(k))
        print("   files that differ from the first:", diff if diff else "none (identical bundle)")
