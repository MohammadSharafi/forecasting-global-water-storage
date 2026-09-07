"""Assemble a submission from FINAL predictions.

usage: python final_assemble.py OUTNAME name1:w1 name2:w2 ...
  name = the prediction-file stem after 'pred_FINAL_'; seeds are averaged here by glob on _s*.

The prediction is built in the order the post-processing was fitted in (session 9m):

  1. weighted blend of the model families                       (blend_scan.py chose the weight)
  2. horizon-1 specialist spliced in, if one was adopted        (hsplice.py chose beta)
  3. spatial smoothing of the residual field                    (smooth_scan.py chose w/r/iters)
  4. per-horizon calibration of the residual magnitude          (postcal.py chose the scales)

Every stage is off unless its environment variable is set, and each scan writes that variable
only when the gain survived leave-one-layout-out validation, so a stage that could not prove
itself simply does not happen.

env
  TAG          prediction-file suffix for the blend members, e.g. TAG=_f1
  SMOOTH_W     legacy single smoothing weight (default 0.7); SMOOTH_W=0 turns smoothing off
  SMOOTH_W1 SMOOTH_W7 SMOOTH_R SMOOTH_IT SMOOTH_WRAP   the tuned smoothing
  CALIB        'a1,..,a7' per-horizon scale of the predicted change
  H1SPEC H1TAG H1BETA   the horizon-1 specialist stem, its file suffix, and its blend weight
"""
import polars as pl, numpy as np, sys, glob, re, os
from smooth import smooth_field, horizon_w

out = sys.argv[1]; spec = [a.split(":") for a in sys.argv[2:]]
va = pl.read_parquet("out/mats/FINAL_va.parquet",
                     columns=["ID", "lat", "lon", "time", "t_known", "horizon", "tws_known"])
k = va["tws_known"].to_numpy(); h = va["horizon"].to_numpy()


def avg(name, tag):
    """Mean over the seed files for one stem. Exact matches only: a bare tag must not pick up
    pred_..._s0_h1.npy, which is a different model."""
    pat = re.compile(rf"pred_FINAL_{re.escape(name)}_s\d+{re.escape(tag)}\.npy")
    fs = sorted(f for f in glob.glob(f"out/mats/pred_FINAL_{name}_s*{tag}.npy")
                if pat.fullmatch(os.path.basename(f)))
    assert fs, f"no prediction files for {name!r} (tag {tag!r})"
    return np.mean([np.load(f) for f in fs], 0), len(fs)


TAG = os.environ.get("TAG", "")
acc = np.zeros(len(va)); wsum = 0.0
for name, w in spec:
    p, n = avg(name, TAG)
    print(f"{name}: {n} seeds, mean change {np.mean(p - k):+.4f}")
    acc += float(w) * p; wsum += float(w)
p = acc / wsum

# --- horizon-1 specialist (hsplice.py)
BETA = float(os.environ.get("H1BETA", "0") or 0)
H1SPEC = os.environ.get("H1SPEC", "")
if BETA > 0 and H1SPEC:
    s, n = avg(H1SPEC, os.environ.get("H1TAG", "_h1"))
    m = h == 1
    p = np.where(m, BETA * s + (1 - BETA) * p, p)
    print(f"h1 specialist {H1SPEC}: {n} seeds, beta={BETA} over {m.sum()} rows")

# --- spatial smoothing (smooth_scan.py). Same-month information only: trajectory smoothing was
# removed because it used the next horizon's prediction, i.e. information after t.
SW = os.environ.get("SMOOTH_W", "0.7")
w1 = float(os.environ.get("SMOOTH_W1", SW)); w7 = float(os.environ.get("SMOOTH_W7", SW))
R = int(os.environ.get("SMOOTH_R", "1")); IT = int(os.environ.get("SMOOTH_IT", "1"))
WRAP = os.environ.get("SMOOTH_WRAP", "0") == "1"
if max(w1, w7) > 0:
    p = smooth_field(va, p, horizon_w(h, w1, w7), R, IT, WRAP)
print(f"smoothing w1={w1} w7={w7} radius={R} iters={IT} wrap={WRAP}")

# --- per-horizon calibration of the predicted change (postcal.py)
CAL = os.environ.get("CALIB", "").strip()
if CAL:
    a = np.array([1.0] + [float(x) for x in CAL.split(",")])
    assert len(a) == 8, "CALIB needs seven values, one per horizon"
    p = k + a[np.clip(h, 1, 7)] * (p - k)
    print("calibration " + " ".join(f"h{i}={a[i]:.3f}" for i in range(1, 8)))

N = int(os.environ.get("NROWS", "280961"))   # the real test has exactly this many rows
assert np.isfinite(p).all() and len(p) == N, f"{len(p)} rows, expected {N}, or non-finite values"
pl.DataFrame({"ID": va["ID"], "Target": np.round(p, 6)}).write_csv(f"out/{out}.csv",
                                                                   float_precision=6)
print("wrote", f"out/{out}.csv", len(p), f"mean change {np.mean(p - k):+.4f}")
