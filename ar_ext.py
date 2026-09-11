"""Two extensions of the anchor rescaling in ar_blend.py, measured before either is built out.

The principle the AR result suggests: a per-cell quantity estimated from the cell's OWN LONG
HISTORY transfers across eras, while one estimated from recent residuals does not -- which is why
AR works and REPORT §4's per-cell reliability (fitted on validation residuals) flipped sign. Two
cheap consequences, each added to the four-vector correction and scored leave-one-layout-out:

  ar_seas   the AR slope modulated by the TARGET's calendar month. One-month persistence RMSE on
            Train.csv runs from 0.507 in April to 0.658 in February, so one slope per horizon
            averages over a real seasonal difference. Per-cell-per-month slopes would have ~11
            pairs each, so the per-cell slope is kept and scaled by the pooled month ratio
            g(h,m)/g(h), which has hundreds of thousands of pairs behind it.
  ar2       a pooled AR(2) forecast b1(h)*TWS(t_known) + b2(h)*TWS(t_known - 1), i.e. whether the
            month before the anchor adds anything the anchor does not.

Nothing here reads a month at or after the validation window: every coefficient comes from the
layout's own pseudo_hist.
"""
import glob

import numpy as np
import polars as pl
from scipy.optimize import minimize

from ar_model import fit_ar, ar_predict

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])
LAYOUTS = ["Avn2", "Bvn2", "Cvn2", "D", "E"]


def _sfx(L):
    return "" if L[0] == "A" else ("_B" if L[0] == "B" else ("_C" if L[0] == "C" else f"_{L}"))


def panel(hist):
    lat = hist["lat"].to_numpy(); lon = hist["lon"].to_numpy()
    mi = hist["time"].dt.year().to_numpy() * 12 + hist["time"].dt.month().to_numpy()
    ck = np.round((lat + 90) * 1000 + (lon + 180)).astype(np.int64)
    cells, ci = np.unique(ck, return_inverse=True); months, mj = np.unique(mi, return_inverse=True)
    A = np.full((len(months), len(cells)), np.nan); A[mj, ci] = hist["TWS_t"].to_numpy().astype(float)
    return cells, months, A


def seasonal_ratio(months, A):
    """g(h, m) / g(h): the pooled slope for target calendar month m relative to the all-month slope."""
    cm = months % 12; cm[cm == 0] = 12
    R = np.ones((8, 13))
    for h in range(1, 8):
        x = A[:-h]; y = A[h:]; gap = (months[h:] - months[:-h]) == h
        m = np.isfinite(x) & np.isfinite(y) & gap[:, None]
        xs = np.where(m, x, 0.0); ys = np.where(m, y, 0.0)
        g = (xs * ys).sum() / max((xs * xs).sum(), 1e-9)
        for mo in range(1, 13):
            s = cm[h:] == mo
            if s.any():
                gm = (xs[s] * ys[s]).sum() / max((xs[s] * xs[s]).sum(), 1e-9)
                R[h, mo] = gm / g
    return R


def ar2_coefs(months, A):
    """Pooled least squares y = b1*x0 + b2*x1 per horizon, x0 = anchor, x1 = the month before it."""
    B = np.zeros((8, 2))
    for h in range(1, 8):
        x1 = A[:-h - 1]; x0 = A[1:-h]; y = A[1 + h:]
        ok_gap = ((months[1:-h] - months[:-h - 1]) == 1) & ((months[1 + h:] - months[1:-h]) == h)
        m = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y) & ok_gap[:, None]
        X = np.stack([x0[m], x1[m]], 1); Y = y[m]
        B[h] = np.linalg.lstsq(X, Y, rcond=None)[0]
    return B


def layout(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "t_known", "horizon", "tws_known", "target", "clim_next"])
    hist = pl.read_parquet(f"out/pseudo_hist{_sfx(L)}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, months, A = panel(hist)
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float); c = va["clim_next"].to_numpy().astype(float)
    lat = va["lat"].to_numpy(); lon = va["lon"].to_numpy()
    ce, po, pe = fit_ar(hist, 100)
    ar, _ = ar_predict(ce, po, pe, lat, lon, h, k)
    # seasonal modulation by the target's calendar month (t + 1)
    R = seasonal_ratio(months, A)
    tm = va["time"].dt.month().to_numpy().astype(np.int64) % 12 + 1
    # validation layouts carry some rows with horizon > 7 (GRACE gaps push t_known back); AR leaves
    # those at 0 and the test-mix score ignores them, but a direct table lookup must be clipped
    hc = np.clip(h, 0, 7)
    ar_seas = ar * R[hc, tm]
    # AR(2): TWS one month before the anchor, looked up in the history panel
    B = ar2_coefs(months, A)
    tkm = (va["t_known"].dt.year().to_numpy().astype(np.int64) * 12
           + va["t_known"].dt.month().to_numpy().astype(np.int64)) - 1
    ck = np.round((lat + 90) * 1000 + (lon + 180)).astype(np.int64)
    ci = np.clip(np.searchsorted(cells, ck), 0, len(cells) - 1)
    mj = np.searchsorted(months, tkm); mj_ok = (mj < len(months)) & (months[np.clip(mj, 0, len(months) - 1)] == tkm)
    x1 = np.where(mj_ok & (cells[ci] == ck), A[np.clip(mj, 0, len(months) - 1), ci], np.nan)
    ar2 = B[hc, 0] * k + B[hc, 1] * x1
    ar2 = np.where(np.isfinite(ar2), ar2, ar)          # no prior month in history -> fall back to AR
    p = np.mean([np.load(f) for f in sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_d0.npy"))], 0)
    ok = np.isfinite(y) & np.isfinite(k) & np.isfinite(c) & np.isfinite(ar) & np.isfinite(p) & np.isfinite(ar_seas)
    print(f"  {L}: ar2 had a prior month for {np.isfinite(x1).mean()*100:.0f}% of rows; "
          f"seasonal ratio range {R[1:,1:].min():.2f}..{R[1:,1:].max():.2f}", flush=True)
    return dict(y=y[ok], h=h[ok], p=p[ok], k=k[ok], c=c[ok], ar=ar[ok], ar_seas=ar_seas[ok], ar2=ar2[ok])


def mixmse(y, h, q):
    return sum(MIX[i - 1] * np.mean((y[h == i] - q[h == i]) ** 2) for i in range(1, 8) if (h == i).any())


def fit(train, keys, budget):
    def obj(w):
        return sum(mixmse(d["y"], d["h"], w @ np.stack([d[x] for x in keys])) for d in train) / len(train)
    w0 = np.zeros(len(keys)); w0[0] = 1.0
    return minimize(obj, w0, bounds=[(-3, 3)] * len(keys),
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                 {"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}],
                    method="SLSQP", options={"ftol": 1e-16, "maxiter": 3000}).x


def main():
    D = {L: layout(L) for L in LAYOUTS}
    base = ["p", "k", "c", "ar"]
    arms = {"+ar_seas": base + ["ar_seas"], "+ar2": base + ["ar2"], "+both": base + ["ar_seas", "ar2"]}
    for B in (2.0, 3.0):
        print(f"\nbudget {B}: marginal value over the four-vector correction, leave-one-layout-out")
        for name, keys in arms.items():
            ds = []
            for L in D:
                tr = [D[x] for x in D if x != L]
                wb, wa = fit(tr, base, B), fit(tr, keys, B); d = D[L]
                a = np.sqrt(mixmse(d["y"], d["h"], wb @ np.stack([d[x] for x in base])))
                b = np.sqrt(mixmse(d["y"], d["h"], wa @ np.stack([d[x] for x in keys])))
                ds.append(b - a)
            ds = np.array(ds)
            print(f"  {name:9} mean {ds.mean():+.4f}  worst {ds.max():+.4f}  wins {int((ds < 0).sum())}/5  "
                  f"clears 0.0003 on {int((ds < -0.0003).sum())}/5   " + " ".join(f"{v:+.4f}" for v in ds))


if __name__ == "__main__":
    main()
