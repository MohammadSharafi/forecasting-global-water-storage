"""The anchor rescaling that validation accepts: persistence and the per-cell AR, together.

What it is
----------
Fitting a blend of (model, persistence, climatology, per-cell AR) with weights summing to one, on
four layouts and scoring the fifth, wins on all five by a mean of 0.0064. The decomposition says
why, and it is not what it looks like:

    vectors allowed          mean held-out delta
    model only                     --
    + persistence + climatology   -0.0016
    + AR only                     -0.0000
    + persistence + clim + AR     -0.0064

AR on its own is worth nothing and persistence/climatology little, but together they are worth
four times their sum. The reason is that AR ~ a(h, cell) * tws_known, so the fitted pair
-0.35*persistence + 0.47*AR is tws_known * (0.47*a(h,cell) - 0.35): a per-cell, per-horizon
rescaling of the anchor, positive at h=1 where a~0.80 and negative by h=7 where a~0.56. The
*difference* between AR and persistence carries the signal; neither level does.

Why the marginal form ships
---------------------------
The submitted file is already a blend carrying negative persistence and climatology weights, so
adding the full correction would apply those twice. What is added is the INCREMENTAL piece,

    corr = w4 . (model, pers, clim, ar)  -  w3 . (model, pers, clim)

whose value measured on top of an already persistence/climatology-corrected baseline is -0.0048,
winning on all five layouts. The correction is also model-independent: fitted against one model and
applied to a different one, on a held-out layout, it is worth -0.0065 (mean, 5/5).

Leakage: the AR slopes come from `ar_model.fit_ar` on the layout's own history, and for the test
from Train.csv alone, which ends before every test month. Coordinates group a cell's own history
and are not features.

usage: python ar_blend.py            # prints the leave-one-out evidence, writes the candidate
"""
import glob

import numpy as np
import polars as pl
from scipy.optimize import minimize

from ar_model import fit_ar, ar_predict

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])
LAYOUTS = ["Avn2", "Bvn2", "Cvn2", "D", "E"]
BASE = "out/scored/sub_z_lb.csv"      # the submitted blend the correction is added to
MODEL = "out/sub_v_anwide.csv"        # the single-model file the weights were fitted against


def _hist_suffix(L):
    return "" if L[0] == "A" else ("_B" if L[0] == "B" else ("_C" if L[0] == "C" else f"_{L}"))


def layout_data(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "horizon", "tws_known", "target", "clim_next"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float); c = va["clim_next"].to_numpy().astype(float)
    hist = pl.read_parquet(f"out/pseudo_hist{_hist_suffix(L)}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, pooled, per = fit_ar(hist, 100)
    ar, _ = ar_predict(cells, pooled, per, va["lat"].to_numpy(), va["lon"].to_numpy(), h, k)
    fs = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_d0.npy"))
    p = np.mean([np.load(f) for f in fs], 0)
    ok = np.isfinite(y) & np.isfinite(k) & np.isfinite(c) & np.isfinite(ar) & np.isfinite(p)
    return dict(y=y[ok], h=h[ok], p=p[ok], k=k[ok], c=c[ok], ar=ar[ok])


def mixmse(y, h, q):
    return sum(MIX[i - 1] * np.mean((y[h == i] - q[h == i]) ** 2) for i in range(1, 8) if (h == i).any())


def fit(train, keys, budget=2.0):
    def obj(w):
        return sum(mixmse(d["y"], d["h"], w @ np.stack([d[x] for x in keys])) for d in train) / len(train)
    w0 = np.zeros(len(keys)); w0[0] = 1.0
    return minimize(obj, w0, bounds=[(-3, 3)] * len(keys),
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                 {"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}],
                    method="SLSQP", options={"ftol": 1e-16, "maxiter": 3000}).x


def main():
    D = {L: layout_data(L) for L in LAYOUTS}
    K3, K4 = ["p", "k", "c"], ["p", "k", "c", "ar"]
    print("leave-one-layout-out, marginal value of AR on a persistence/climatology-corrected base")
    ds = []
    for L in D:
        tr = [D[x] for x in D if x != L]
        w3, w4 = fit(tr, K3), fit(tr, K4)
        d = D[L]
        a = np.sqrt(mixmse(d["y"], d["h"], w3 @ np.stack([d[x] for x in K3])))
        b = np.sqrt(mixmse(d["y"], d["h"], w4 @ np.stack([d[x] for x in K4])))
        ds.append(b - a); print(f"  {L:6} {a:.4f} -> {b:.4f}   {b - a:+.4f}")
    ds = np.array(ds)
    print(f"  mean {ds.mean():+.4f}  worst {ds.max():+.4f}  wins {int((ds < 0).sum())}/5")

    w3, w4 = fit(list(D.values()), K3), fit(list(D.values()), K4)
    print(f"\n  shipping weights   w3 {np.round(w3, 4)}   w4 {np.round(w4, 4)}")

    ref = pl.read_csv("Test.csv").select("ID")
    def load(f):
        d = pl.read_csv(f); col = [x for x in d.columns if x != "ID"][0]
        v = ref.join(d.rename({col: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
        assert np.isfinite(v).all(), f
        return v
    model, pers = load(MODEL), load("out/probe_persistence.csv")
    clim, ar, base = load("out/probe_clim.csv"), load("out/probe_ar.csv"), load(BASE)
    X3 = np.stack([model, pers, clim]); X4 = np.stack([model, pers, clim, ar])
    corr = w4 @ X4 - w3 @ X3
    out = base + corr
    print(f"  correction: mean {corr.mean():+.4f} std {corr.std():.4f} range [{corr.min():.2f},{corr.max():.2f}]")
    pl.DataFrame({"ID": ref["ID"], "Target": out}).write_csv("out/sub_ac_armarg.csv")
    print(f"  wrote out/sub_ac_armarg.csv  std={out.std():.4f} "
          f"RMS distance from the current best={np.sqrt(np.mean((out - base) ** 2)):.4f}")


if __name__ == "__main__":
    main()
