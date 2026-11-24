import math
# (1) three-point extrapolation of final's held-out loss (midrun_eval_1.json), prediction at v9's last point
x=[494.7,519.7,538.7]; y=[2.2679438350868595,2.2624,2.25984169013143]
n=3; xm=sum(x)/n; ym=sum(y)/n
sxx=sum((a-xm)**2 for a in x); sxy=sum((a-xm)*(b-ym) for a,b in zip(x,y))
b=sxy/sxx; a=ym-b*xm
res=[yy-(a+b*xx) for xx,yy in zip(x,y)]
s=math.sqrt(sum(r*r for r in res)/(n-2))
x0=592.5; pred=a+b*x0
se=s*math.sqrt(1/n+(x0-xm)**2/sxx)
print(f"slope {b:.6f}/min, pred at {x0}: {pred:.4f}, v9 2.2337 -> gap {pred-2.2337:.4f}; resid sd {s:.5f} (1 dof); SE(mean pred) {se:.4f}; 95% t(1)=12.706 -> +-{12.706*se:.3f}; normal +-{1.96*se:.4f}")
sep=s*math.sqrt(1+1/n+(x0-xm)**2/sxx)
print(f"prediction-interval SE {sep:.4f}; t(1) 95% +-{12.706*sep:.3f}")
# (2) token projection with the v9 switches (assumption: 2048-token gain equals the 1024 one)
slice_s=27420.0; p0=0.5655; late=0.67
t2048=slice_s*(1-late)/(1-p0); t1024=slice_s-t2048
live1024=196864.0; syn1024=203665.0
syn2048=262144/(1.5102*1287.1/1433.1)
live2048=syn2048*live1024/syn1024
rem=t1024*live1024+t2048*live2048
done=31912*262144
print(f"t1024 {t1024:.0f}s t2048 {t2048:.0f}s; live2048 assumed {live2048:,.0f}; remaining {rem:.3e}; total {done+rem:.4e}; final's projected total 1.588e10 -> deficit {1.588e10-(done+rem):.3e} ({(1.588e10-(done+rem))/1.588e10:.1%})")
# (3) MDE at n=200 from the reports' own power lines
for tie,n10 in ((0.25,259),(0.62,506)):
    print(f"tie {tie}: n=200 resolves about {0.10*math.sqrt(n10/200):.3f} (score points)")
