# Reviewer's recount of 'executed' FLOP/token for configs/final.py at T=1024 after the kernel switch.
# Conventions as lead_work/scripts/flops.py; differences stated per line. Arithmetic, not a measurement.
V,d,nb,R,H,hd,bias,dff,rank,T=8192,768,2,8,6,128,2,3072,512,1024
apps=nb*R
head=d*rank+rank*V
mlp=2*d*dff
# analytic (lead): Wq,Wk on 126 content dims, Wv, Wo
attn_an=d*H*(hd-bias)*2+d*H*hd+H*hd*d
# executed: rotation as two products for q and k (x@W and x@W'), reserved dims as zero columns (full 768 width)
attn_ex=2*(2*d*H*hd)+d*H*hd+H*hd*d
full_sq=2*T*hd*H            # per query token, q.k and p.v over all T keys
onetile=0.625*full_sq       # strips of 256 rows cover 62.5% of the square at T=1024
lead_exec=4*apps*(attn_an+mlp+full_sq)+3*head
for bwd_mult in (2.0,2.5):
    ex=4*apps*(attn_ex+mlp)+(1+bwd_mult)*apps*onetile+3*head   # remat: matmuls 4x; save: fwd kernel once
    for rate in (281374,):
        print(f"bwd={bwd_mult}: executed {2*ex/1e9:.4f} GFLOP/token -> {2*ex*rate/1e12:.0f} TFLOP/s ({2*ex*rate/814e12:.1%} of 814); lead's count {2*lead_exec/1e9:.4f} -> {2*lead_exec*rate/1e12:.0f} TFLOP/s")
print("q/k double product adds", 2*4*apps*(attn_ex-attn_an)/1e9, "GFLOP/token")
