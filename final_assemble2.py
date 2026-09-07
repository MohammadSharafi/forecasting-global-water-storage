"""Assemble a submission using the great-circle post-processing operator (session 9).

Same blending as final_assemble.py, but the post-processing step is
    p -> M_r(p) + beta * (p - M_r(p))
with r and beta chosen by postproc2.py on validation layouts A and B (pick a setting that
wins on BOTH, as for every other adopted change in this project).

usage:
  RADIUS=500 BETA=0.55 python final_assemble2.py sub_k_gc lgb_v5x_noll:0.3 xgb_v5x_noll:0.3 \
      cat_v5x_noll:0.15 mlp_v5x_noll:0.25

env:
  RADIUS   great-circle radius in km              (default 500)
  BETA     fine-scale keep fraction, 0..1         (default 0.55)
  BETA_H   optional per-horizon betas, 7 comma-separated values, overrides BETA
  GRID     "1" to apply the old grid-box residual smoothing first (default "0")
  TAG      prediction-file suffix, as in final_assemble.py
"""
import polars as pl, numpy as np, sys, glob, re, os
from postproc2 import field_mean, apply_shrink

out = sys.argv[1]; spec = [a.split(":") for a in sys.argv[2:]]
RAD = int(os.environ.get("RADIUS", "500"))
BETA = float(os.environ.get("BETA", "0.55"))
BETA_H = os.environ.get("BETA_H", "")
GRID = os.environ.get("GRID", "0") == "1"
tag = os.environ.get("TAG", "")

va = pl.read_parquet("out/mats/FINAL_va.parquet",
                     columns=["ID", "lat", "lon", "time", "t_known", "horizon", "tws_known"])
k = va["tws_known"].to_numpy(); acc = np.zeros(len(va)); wsum = 0.0
for name, w in spec:
    pat = re.compile(rf"pred_FINAL_{re.escape(name)}_s\d+{re.escape(tag)}\.npy$")
    fs = sorted(f for f in glob.glob(f"out/mats/pred_FINAL_{name}_s*{tag}.npy")
                if pat.search(os.path.basename(f)))
    assert fs, name
    p = np.mean([np.load(f) for f in fs], 0)
    print(f"{name}: {len(fs)} seeds, mean change {np.mean(p - k):+.4f}")
    acc += float(w) * p; wsum += float(w)
p = acc / wsum

if GRID:
    from smooth import smooth
    p = smooth(va, p, w=0.7, radius=1, iters=1)

s = field_mean(va, p, radii=(RAD,))[RAD]
if BETA_H:
    bs = [float(x) for x in BETA_H.split(",")]
    assert len(bs) == 7, "BETA_H needs 7 values (horizons 1..7)"
    h = va["horizon"].to_numpy()
    beta = np.array([bs[int(np.clip(hh, 1, 7)) - 1] for hh in h])
    print(f"per-horizon beta {bs} at {RAD} km" + (", after grid smooth" if GRID else ""))
else:
    beta = BETA
    print(f"beta={BETA} at {RAD} km" + (", after grid smooth" if GRID else ""))
p = apply_shrink(p, s, beta)

assert np.isfinite(p).all() and len(p) == 280961
pl.DataFrame({"ID": va["ID"], "Target": np.round(p, 6)}).write_csv(f"out/{out}.csv", float_precision=6)
print("wrote", f"out/{out}.csv", len(p), f"mean change {np.mean(p - k):+.4f}")
