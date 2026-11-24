"""Step-time estimates from compile-only accumulate programs (XLA estimated_cycles).

For every hlo/<name>.hlo (+ .json): per-piece M cycles (inside the piece loop / pieces), the
once-per-program part, the whole step of 262,144 tokens (n_pieces x per piece + once), and the
step time predicted from the chip's measured whole step of final (onetile:save+relu2):
884.5 ms at 64 x 4 x 1024 and 983.9 ms at 64 x 2 x 2048 (docs/CHANGES.md 19.6), plus the
cycle difference at 0.521 ms/M (T=1024) or 0.542 ms/M (T=2048). Mosaic kernels count 0 cycles,
so this assumes the attention kernels' time per token does not depend on the piece size or the
configuration (v9 and final have the same attention).

    python estimate.py hlo/
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlo_bytes as H  # noqa: E402

TOK = 262144
BASE = {1024: (884.5, 0.521, ("final", "take", 4)), 2048: (983.9, 0.542, ("final", "take", 2))}


def load(path):
    meta = json.load(open(path + ".json"))
    rows = H.analyse(H.parse(path), 1)
    inside = sum(r["cycles"] for r in rows if r["stack"])
    outside = sum(r["cycles"] for r in rows if not r["stack"])
    hbm_in = sum(r["rd"] + r["wr"] for r in rows if r["stack"])
    a = int(meta["pieces"])
    return dict(cfg=os.path.basename(meta["config"])[:-3], tg=meta["tg"], rows=meta["rows"], t=meta["seq_len"],
                pieces=a, per_piece=inside / a, once=outside, hbm_piece=hbm_in / a,
                peak=meta["memory"].get("peak_memory_in_bytes", 0), sets=meta.get("sets", []),
                compile_s=meta.get("compile_s"), name=os.path.basename(path)[:-4])


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "hlo"
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.hlo"))):
        if os.path.exists(p + ".json"):
            try:
                recs.append(load(p))
            except Exception as e:  # noqa: BLE001
                print("skip", p, e)
    for r in recs:
        n = TOK // (r["rows"] * r["t"])
        r["n"] = n
        r["step_mc"] = (n * r["per_piece"] + r["once"]) / 1e6
    print(f"{'program':34s} {'piece':>9s} {'n':>3s} {'Mcyc/piece':>10s} {'once':>7s} {'HBM GB/pc':>9s} "
          f"{'peak GiB':>8s} {'step Mcyc':>9s} {'d vs final@live':>15s} {'pred ms':>8s} {'tok/s':>9s}")
    for r in sorted(recs, key=lambda r: (r["t"], r["cfg"], r["tg"], r["rows"], r["name"])):
        base_ms, rate, key = BASE[r["t"]]
        b = [x for x in recs if (x["cfg"], x["tg"], x["rows"]) == key and x["t"] == r["t"] and not x["sets"]]
        if b:
            dms = (r["step_mc"] - b[0]["step_mc"]) * rate
            pred = base_ms + dms
            extra = f"{dms:+12.1f} ms {pred:8.1f} {TOK / pred * 1e3:9,.0f}"
        else:
            extra = "   (no final baseline yet)"
        print(f"{r['name']:34s} {r['rows']:3d}x{r['t']:<5d} {r['n']:3d} {r['per_piece'] / 1e6:10.3f} {r['once'] / 1e6:7.3f} "
              f"{r['hbm_piece'] / 1e9:9.3f} {r['peak'] / 2**30:8.3f} {r['step_mc']:9.1f} {extra}")


if __name__ == "__main__":
    main()
