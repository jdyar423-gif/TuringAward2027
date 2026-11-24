"""Review items 37/38 from files on disk (no new compile): for every configuration that has both a
compile-only program (wf/v9_*_work/hlo, estimate.py: XLA's estimated cycles and the HBM bytes of its
instructions, libtpu 0.0.50, v6e:1x1, jax 0.11.1) and a whole-step time on the chip (24 Nov;
jax 0.11.2), how well does each static count predict the measured step? Leave-one-out linear fits.
Arithmetic; the inputs are typed in from estimate_out.txt and the chip files named below."""
import json, math, itertools
rows = [
 # name, est Mcycles/step, HBM GB/step, analytic GFLOP/token, reads/token, measured ms, chip source
 ("final take 64x4x1024", 1944.7, 64*8.046, 0.781, 19033736, 884.5, "chip_kernels_0745.txt (ce on)"),
 ("v9 take 64x4x1024", 3141.8, 64*19.950, 0.809, 19631240, 1433.1, "v9_measA.txt"),
 ("v9 take 32x8x1024", 3237.6, 32*47.709, 0.809, 19631240, 1491.5, "v9_measA.txt"),
 ("v9 sorted 64x4x1024", 3390.7, 64*25.352, 0.809, 19631240, 1789.0, "v9_measB_sorted.txt"),
 ("v9 take+flat 64x4x1024", 3053.6, 64*19.324, 0.809, 19631240, 1406.4, "v9_measD.txt"),
 ("v9 take+flat+rows 64x4x1024", 2825.6, 64*17.656, 0.809, 19631240, 1295.2, "v9_measD.txt"),
]
def fit(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs)
    if sxx==0: return None
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx; return my-b*mx, b
def loo(col):
    errs=[]
    for i in range(len(rows)):
        tr=[r for j,r in enumerate(rows) if j!=i]
        f=fit([r[col] for r in tr],[r[5] for r in tr])
        if f is None: errs.append(None); continue
        a,b=f; pred=a+b*rows[i][col]; errs.append((pred-rows[i][5])/rows[i][5])
    return errs
def spearman(xs, ys):
    def rank(v):
        s=sorted(range(len(v)), key=lambda i:v[i]); r=[0]*len(v)
        i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[s[j+1]]==v[s[i]]: j+=1
            for k in range(i,j+1): r[s[k]]=(i+j)/2
            i=j+1
        return r
    rx,ry=rank(xs),rank(ys); n=len(xs); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else float('nan')
res={}
for col,name in ((1,"estimated cycles"),(2,"estimated HBM bytes"),(3,"analytic FLOP per token"),(4,"numbers read per token")):
    e=loo(col)
    v9=[r for r in rows if r[0].startswith("v9")]
    res[name]={"loo_rel_err":[None if x is None else round(x,3) for x in e],
               "loo_mean_abs_rel_err": round(sum(abs(x) for x in e if x is not None)/len([x for x in e if x is not None]),3) if any(x is not None for x in e) else None,
               "spearman_all": round(spearman([r[col] for r in rows],[r[5] for r in rows]),3),
               "spearman_v9_only": round(spearman([r[col] for r in v9],[r[5] for r in v9]),3)}
for k,v in res.items(): print(k, json.dumps(v))
# differences (calibration table, item 38): predicted vs measured
diffs=[("v9 - final (take, 64x4x1024)", 623.7, 1433.1-884.5),
       ("sorted - take (v9)", 1637.9-1508.2, 1789.0-1433.1),
       ("8-row - 4-row pieces (v9, take)", 1558.1-1508.2, 1491.5-1433.1),
       ("8-row - 4-row pieces (final; chip: splash kernels, 03:09)", 53.7, 1081.5-1063.3),
       ("flat (v9)", 1462.3-1508.2, 1406.4-1433.1),
       ("rows (v9, on flat)", 1343.4-1462.3, 1295.2-1406.4),
       ("tokn (v9; predicted on flat, measured on flat+rows)", 1398.7-1462.3, 1287.1-1295.2)]
print()
for n,p,m in diffs: print(f"{n}: predicted {p:+.1f} ms, measured {m:+.1f} ms, predicted/measured {p/m:.2f}, sign {'agrees' if p*m>0 else 'DISAGREES'}")
json.dump({"rows":rows,"loo":res,"diffs":[{"change":n,"predicted_ms":round(p,1),"measured_ms":round(m,1),"ratio":round(p/m,2)} for n,p,m in diffs]},open("predictor_test.json","w"),indent=1)
print()
for col,name in ((1,"estimated cycles (M/step)"),(2,"estimated HBM bytes (GB/step)")):
    a,b=fit([r[col] for r in rows],[r[5] for r in rows])
    print(f"{name}: all-6 fit ms = {a:.1f} + {b:.4f} x; residuals " + ", ".join(f"{(a+b*r[col]-r[5]):+.0f}" for r in rows))
    if col==2: print(f"  marginal {b:.3f} ms per GB = {1/b:.2f} TB/s effective; bytes per step: " + ", ".join(f"{r[2]:.0f}" for r in rows))
    # predicted differences between neighbouring configurations by this fit (in-sample)
    pairs=[(1,0,"v9 - final"),(3,1,"sorted - take"),(2,1,"8-row - 4-row"),(4,1,"flat"),(5,4,"rows")]
    print("  differences predicted by the all-6 fit vs measured: " + "; ".join(f"{n} {b*(rows[i][col]-rows[j][col]):+.0f} vs {rows[i][5]-rows[j][5]:+.0f}" for i,j,n in pairs))
