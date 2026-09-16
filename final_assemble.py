import polars as pl, numpy as np, sys, glob, re, os
from smooth import smooth_field, horizon_w

out = sys.argv[1]; spec = [a.split(":") for a in sys.argv[2:]]

FL = os.environ.get("FLAYOUT", "FINAL")
va = pl.read_parquet(f"out/mats/{FL}_va.parquet",
                     columns=["ID", "lat", "lon", "time", "t_known", "horizon", "tws_known"])
k = va["tws_known"].to_numpy(); h = va["horizon"].to_numpy()

def avg(name, tag):
    pat = re.compile(rf"pred_{re.escape(FL)}_{re.escape(name)}_s\d+{re.escape(tag)}\.npy")
    fs = sorted(f for f in glob.glob(f"out/mats/pred_{FL}_{name}_s*{tag}.npy")
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

BETA = float(os.environ.get("H1BETA", "0") or 0)
H1SPEC = os.environ.get("H1SPEC", "")
if BETA > 0 and H1SPEC:
    s, n = avg(H1SPEC, os.environ.get("H1TAG", "_h1"))
    m = h == 1
    p = np.where(m, BETA * s + (1 - BETA) * p, p)
    print(f"h1 specialist {H1SPEC}: {n} seeds, beta={BETA} over {m.sum()} rows")

SW = os.environ.get("SMOOTH_W", "0.7")
w1 = float(os.environ.get("SMOOTH_W1", SW)); w7 = float(os.environ.get("SMOOTH_W7", SW))
R = int(os.environ.get("SMOOTH_R", "1")); IT = int(os.environ.get("SMOOTH_IT", "1"))
WRAP = os.environ.get("SMOOTH_WRAP", "0") == "1"
if max(w1, w7) > 0:
    p = smooth_field(va, p, horizon_w(h, w1, w7), R, IT, WRAP)
print(f"smoothing w1={w1} w7={w7} radius={R} iters={IT} wrap={WRAP}")

SEASON = os.environ.get("SEASON", "").strip()
if SEASON:
    b = np.zeros(13)
    for part in SEASON.split(","):
        mm, bb = part.split(":"); b[int(mm)] = float(bb)
    tm = np.array([(d.month % 12) + 1 for d in va["time"].to_list()])
    p = p - b[tm]
    print("seasonal " + " ".join(f"m{m}={b[m]:+.3f}" for m in range(1, 13) if b[m]))

CAL = os.environ.get("CALIB", "").strip()
CALB = os.environ.get("CALIB_B", "").strip()
if CAL:
    a = np.array([1.0] + [float(x) for x in CAL.split(",")])
    b = np.array([0.0] + [float(x) for x in CALB.split(",")]) if CALB else np.zeros(8)
    assert len(a) == 8 and len(b) == 8, "CALIB/CALIB_B need seven values, one per horizon"

    j = np.clip(h, 1, 7).astype(int)
    p = k + a[j] * (p - k) + b[j]
    print("calibration " + " ".join(f"h{i}={a[i]:.3f}" + (f"{b[i]:+.3f}" if CALB else "")
                                    for i in range(1, 8)))

N = int(os.environ.get("NROWS", "280961"))
assert np.isfinite(p).all() and len(p) == N, f"{len(p)} rows, expected {N}, or non-finite values"
pl.DataFrame({"ID": va["ID"], "Target": np.round(p, 6)}).write_csv(f"out/{out}.csv",
                                                                   float_precision=6)
print("wrote", f"out/{out}.csv", len(p), f"mean change {np.mean(p - k):+.4f}")
