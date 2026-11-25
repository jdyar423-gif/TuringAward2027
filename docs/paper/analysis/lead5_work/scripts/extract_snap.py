"""Fifth pass: extract the small records of the last packed snapshot N/snap_final (parts.json +
run.tar.NNN, read with lead2_work/snaptar.py; part sizes checked against parts.json) into
lead5_work/snap_final_x/: the train log, ledger, run manifest, stage ledger, run log, benchmark records,
post-training records and the two checkpoints' manifests (no weights). Also compares the extracted
files with the copies fetched separately into N/records/unpacked/ (byte for byte).

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B extract_snap.py
"""
import filecmp
import json
import sys
from pathlib import Path

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
sys.path.insert(0, str(N / "paper/lead2_work"))
import snaptar  # noqa: E402

SNAP = N / "snap_final"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "snap_final_x"
parts = json.loads((SNAP / "parts.json").read_text())
for p in parts["parts"]:
    f = SNAP / p["name"]
    assert f.stat().st_size == p["bytes"], (f, f.stat().st_size, p["bytes"])
WANT = ("run/manifest.json", "run/stages.json", "run/stage_table.json", "run/train/ledger.json",
        "run/train/run.json", "run/train/train_log.jsonl", "run/benchmark/results.jsonl",
        "run/post/prefopt_report.json", "run/post/samples.jsonl", "run.log",
        "run/train/ckpt/ckpt-000022-step-000051812/manifest.json",
        "run/post/ckpt/ckpt-000001-step-000051812/manifest.json", "run/final/export.json",
        "run/final/MODEL_CARD.md", "run/final/SHA256SUMS")
mp, buf, tf = snaptar.open_snapshot(SNAP)
try:
    names = {m.name for m in tf.getmembers()}
    for w in WANT:
        assert w in names, w
        dst = OUT / w
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(snaptar.read_member_bytes(tf, w))
finally:
    tf.close()
    mp.close()
meta = {k: v for k, v in parts.items() if k != "parts"}
meta["n_parts"] = len(parts["parts"])
(OUT / "snapshot_meta.json").write_text(json.dumps(meta, indent=1))
same = {}
REC = N / "records/unpacked"
for w in WANT:
    other = REC / w
    if other.exists():
        same[w] = filecmp.cmp(OUT / w, other, shallow=False)
OFF = N / "packs/official_export/x/final"
for w in ("export.json", "MODEL_CARD.md", "SHA256SUMS"):
    same[f"official_export/{w}"] = filecmp.cmp(OUT / "run/final" / w, OFF / w, shallow=False)
same["main16.final.log"] = filecmp.cmp(OUT / "run.log", N / "main16.final.log", shallow=False)
print(json.dumps(meta))
print("identical to the separately fetched copies:", json.dumps(same))
