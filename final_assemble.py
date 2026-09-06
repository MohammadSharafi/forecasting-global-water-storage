"""Assemble a submission from FINAL predictions. usage: python final_assemble.py OUTNAME name1:w1 name2:w2 ...
name = pred file stem after 'pred_FINAL_' (seed averages are done here by glob on _s*)."""
import polars as pl, numpy as np, sys, glob, re, os
from smooth import smooth, traj_smooth
out=sys.argv[1]; spec=[a.split(":") for a in sys.argv[2:]]
va=pl.read_parquet("out/mats/FINAL_va.parquet",columns=["ID","lat","lon","time","t_known","horizon","tws_known"])
k=va["tws_known"].to_numpy(); acc=np.zeros(len(va)); wsum=0
for name,w in spec:
    fs=sorted(f for f in glob.glob(f"out/mats/pred_FINAL_{name}_s*.npy")
              if re.fullmatch(rf"pred_FINAL_{re.escape(name)}_s\d+\.npy", os.path.basename(f)))   # exact seed files only
    assert fs, name
    p=np.mean([np.load(f) for f in fs],0); print(f"{name}: {len(fs)} seeds, mean change {np.mean(p-k):+.4f}"); acc+=float(w)*p; wsum+=float(w)
p=acc/wsum; p=smooth(va,p,w=0.7,radius=1,iters=1)   # spatial smoothing only: same-month information. Trajectory smoothing removed (it used the next horizon's prediction, i.e. information after t).
assert np.isfinite(p).all() and len(p)==280961
pl.DataFrame({"ID":va["ID"],"Target":np.round(p,6)}).write_csv(f"out/{out}.csv",float_precision=6); print("wrote",f"out/{out}.csv",len(p))
