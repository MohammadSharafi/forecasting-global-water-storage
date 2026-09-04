"""Evaluate saved validation predictions and blends on a layout. usage: python blend_eval.py A|B [names...]"""
import polars as pl, numpy as np, sys, glob, itertools, os
from smooth import smooth, traj_smooth
L=sys.argv[1]; va=pl.read_parquet(f"out/mats/{L}_va.parquet",columns=["lat","lon","time","t_known","horizon","tws_known","target"])
y=va["target"].to_numpy(); k=va["tws_known"].to_numpy(); h=va["horizon"].to_numpy()
r=lambda p: float(np.sqrt(np.mean((y-p)**2)))
names=sys.argv[2:] or [os.path.basename(f)[len(f"pred_{L}_"):-4] for f in sorted(glob.glob(f"out/mats/pred_{L}_*.npy"))]
P={n:np.load(f"out/mats/pred_{L}_{n}.npy") for n in names}
print(f"layout {L}: persistence={r(k):.4f}")
for n,p in P.items(): print(f"  {n:28} {r(p):.4f}  +smooth {r(smooth(va,p,0.7)):.4f}  +traj {r(traj_smooth(va,smooth(va,p,0.7),0.35)):.4f}   by h: "+" ".join(f"{np.sqrt(np.mean(((y-p)**2)[h==hh])):.3f}" for hh in range(1,8)))
if len(P)>1:
    M=np.stack(list(P.values()),1)
    # non-negative least squares weights on residuals (in-sample, indicative only)
    from scipy.optimize import nnls
    wts,_=nnls(M-k[:,None],y-k); wts=wts/wts.sum(); pb=k+(M-k[:,None])@wts
    print("  NNLS weights:",dict(zip(names,np.round(wts,3))),f"-> {r(pb):.4f}  +smooth+traj {r(traj_smooth(va,smooth(va,pb,0.7),0.35)):.4f}")
    pe=M.mean(1); print(f"  equal mean -> {r(pe):.4f}  +smooth+traj {r(traj_smooth(va,smooth(va,pe,0.7),0.35)):.4f}")
    for a,b in itertools.combinations(names,2):
        pp=0.5*P[a]+0.5*P[b]; print(f"  50/50 {a}+{b}: {r(pp):.4f}")
