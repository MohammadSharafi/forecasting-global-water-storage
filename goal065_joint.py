"""The adopted set measured JOINTLY: the AR correction on top of the GPCC-improved model.

Two changes cleared the every-layout bar in this goal -- GPCC gauge precipitation inside the model
(mean -0.0020, 5/5) and the marginal AR anchor rescaling applied after it (mean -0.0048, 5/5).
Their sum is not their joint value: the AR term rescales the anchor, and a better-informed model may
already place its prediction where that rescaling would have moved it. So the correction is re-fitted
leave-one-layout-out against the GPCC model's own predictions, and scored against that same model.

This is the number G2 is about: the best held-out gain any adopted set achieves, against -0.041.
"""
import glob

import numpy as np
import polars as pl

from ar_blend import LAYOUTS, _hist_suffix, fit, mixmse
from ar_model import fit_ar, ar_predict
from goal065_eval import summary_line

K3, K4 = ["p", "k", "c"], ["p", "k", "c", "ar"]


def build(L):
    """Load one layout with ONE shared finiteness mask over every vector, including the GPCC arm.

    ar_blend.layout_data already filters its arrays, so attaching a full-length prediction to its
    output silently misaligns every row -- which is exactly what the first version of this script
    did, and it reported +0.49 for a block measured at -0.0020 an hour earlier. Build it here.
    """
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "horizon", "tws_known", "target", "clim_next"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float); c = va["clim_next"].to_numpy().astype(float)
    hist = pl.read_parquet(f"out/pseudo_hist{_hist_suffix(L)}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, pooled, per = fit_ar(hist, 100)
    ar, _ = ar_predict(cells, pooled, per, va["lat"].to_numpy(), va["lon"].to_numpy(), h, k)
    def avg(tag):
        fs = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_{tag}.npy"))
        if not fs:
            raise SystemExit(f"{L}: no _{tag} predictions")
        return np.mean([np.load(f) for f in fs], 0)
    p, g = avg("d0"), avg("gpcc1")
    ok = (np.isfinite(y) & np.isfinite(k) & np.isfinite(c) & np.isfinite(ar)
          & np.isfinite(p) & np.isfinite(g))
    return dict(y=y[ok], h=h[ok], p=p[ok], k=k[ok], c=c[ok], ar=ar[ok], g=g[ok])


KNOWN = {"Avn2": -0.0014, "Bvn2": -0.0013, "Cvn2": -0.0040, "D": -0.0026, "E": -0.0007}


def main():
    D = {L: build(L) for L in LAYOUTS}
    print("held-out test-mix RMSE, five layouts\n")
    print(f"  {'layout':8} {'control':>9} {'+gpcc':>9} {'+gpcc+AR':>10}   {'gpcc':>9} {'joint':>9}")
    dg, dj = {}, {}
    for L in D:
        tr = [D[x] for x in D if x != L]
        d = D[L]
        # the AR correction re-fitted against the GPCC model, not against the shipped one
        trg = [dict(x, p=x["g"]) for x in tr]
        w3, w4 = fit(trg, K3), fit(trg, K4)
        base = np.sqrt(mixmse(d["y"], d["h"], d["p"]))
        gp = np.sqrt(mixmse(d["y"], d["h"], d["g"]))
        dd = dict(d, p=d["g"])
        joint = np.sqrt(mixmse(d["y"], d["h"], w4 @ np.stack([dd[x] for x in K4])))
        dg[L] = gp - base; dj[L] = joint - base
        # self-check: the gpcc arm must reproduce what goal065_eval measured from the same files
        assert abs(dg[L] - KNOWN[L]) < 3e-4, (
            f"{L}: gpcc delta {dg[L]:+.4f} does not match the recorded {KNOWN[L]:+.4f} -- rows are misaligned")
        print(f"  {L:8} {base:9.4f} {gp:9.4f} {joint:10.4f}   {gp-base:+9.4f} {joint-base:+9.4f}")
    print()
    print("  gpcc alone :", summary_line("gpcc_alone", dg))
    print("  JOINT      :", summary_line("JOINT-SET", dj))
    with open("out/goal065/joint.txt", "w") as f:
        f.write(summary_line("gpcc_alone", dg) + "\n")
        f.write(summary_line("JOINT-SET", dj) + "\n")


if __name__ == "__main__":
    main()
