import glob

import numpy as np
import polars as pl

from ar_blend import LAYOUTS, mixmse, fit, _hist_suffix
from ar_model import fit_ar, ar_predict

K3 = ["p", "k", "c", "ar"]
K4 = ["p", "k", "c", "ar", "rp"]

def layout(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "horizon", "t_known", "tws_known", "target", "clim_next"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float); c = va["clim_next"].to_numpy().astype(float)
    hist = pl.read_parquet(f"out/pseudo_hist{_hist_suffix(L)}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, pooled, per = fit_ar(hist, 100)
    ar, _ = ar_predict(cells, pooled, per, va["lat"].to_numpy(), va["lon"].to_numpy(), h, k)
    fs = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_gpcc1.npy")) or\
         sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_d0.npy"))
    p = np.mean([np.load(f) for f in fs], 0)

    key = pl.DataFrame({"lat": va["lat"], "lon": va["lon"], "tk": va["t_known"],
                        "h": h, "p": p, "i": np.arange(len(h))})
    prev = key.select(["lat", "lon", "tk", "p"]).with_columns((key["h"] + 1).alias("h"))\
              .rename({"p": "rp"})
    j = key.join(prev, on=["lat", "lon", "tk", "h"], how="left").sort("i")
    rp = j["rp"].to_numpy().astype(float)

    rp = np.where(np.isfinite(rp), rp, k)

    ok = np.isfinite(y) & np.isfinite(k) & np.isfinite(c) & np.isfinite(ar) & np.isfinite(p) & np.isfinite(rp)
    print(f"  {L}: recursive arm available on {np.isfinite(j['rp'].to_numpy().astype(float)).mean():.1%} of rows")
    return dict(y=y[ok], h=h[ok], p=p[ok], k=k[ok], c=c[ok], ar=ar[ok], rp=rp[ok])

def main():
    D = {L: layout(L) for L in LAYOUTS}
    print("\nleave-one-layout-out: does the recursive anchor add to the accepted blend?")
    ds = []
    for L in D:
        tr = [D[x] for x in D if x != L]
        wa, wb = fit(tr, K3), fit(tr, K4)
        d = D[L]
        a = np.sqrt(mixmse(d["y"], d["h"], wa @ np.stack([d[x] for x in K3])))
        b = np.sqrt(mixmse(d["y"], d["h"], wb @ np.stack([d[x] for x in K4])))
        ds.append(b - a); print(f"  {L:6} {a:.4f} -> {b:.4f}   {b - a:+.4f}   w_rp={wb[-1]:+.3f}")
    ds = np.array(ds)
    print(f"  mean {ds.mean():+.4f}  worst {ds.max():+.4f}  wins {int((ds < 0).sum())}/5")
    print(f"\n  all-five weights: {np.round(fit(list(D.values()), K4), 4)}")

if __name__ == "__main__":
    main()
