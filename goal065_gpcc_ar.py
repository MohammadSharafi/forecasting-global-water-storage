import glob

import numpy as np
import polars as pl

from ar_blend import LAYOUTS, _hist_suffix, fit, mixmse
from ar_model import ar_predict, fit_ar

K3, K4 = ["p", "k", "c"], ["p", "k", "c", "ar"]
BASE = "out/sub_gpcc_base.csv"
OUT = "out/sub_gpcc_ar.csv"

def build(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "horizon", "tws_known", "target", "clim_next"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float); c = va["clim_next"].to_numpy().astype(float)
    hist = pl.read_parquet(f"out/pseudo_hist{_hist_suffix(L)}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, pooled, per = fit_ar(hist, 100)
    ar, _ = ar_predict(cells, pooled, per, va["lat"].to_numpy(), va["lon"].to_numpy(), h, k)
    fs = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_gpcc1.npy"))
    p = np.mean([np.load(f) for f in fs], 0)
    ok = np.isfinite(y) & np.isfinite(k) & np.isfinite(c) & np.isfinite(ar) & np.isfinite(p)
    return dict(y=y[ok], h=h[ok], p=p[ok], k=k[ok], c=c[ok], ar=ar[ok])

def main():
    D = {L: build(L) for L in LAYOUTS}

    ds = []
    for L in D:
        tr = [D[x] for x in D if x != L]
        w4 = fit(tr, K4); d = D[L]
        base = np.sqrt(mixmse(d["y"], d["h"], d["p"]))
        got = np.sqrt(mixmse(d["y"], d["h"], w4 @ np.stack([d[x] for x in K4])))
        ds.append(got - base); print(f"  {L:6} gpcc {base:.4f} -> +AR {got:.4f}  {got-base:+.4f}")
    print(f"  mean {np.mean(ds):+.4f}  wins {sum(v < 0 for v in ds)}/5")

    w4 = fit(list(D.values()), K4)
    print(f"  shipping weights w4 = {np.round(w4, 4).tolist()}")
    ref = pl.read_csv("Test.csv").select("ID")
    def load(f):
        d = pl.read_csv(f); col = [x for x in d.columns if x != "ID"][0]
        v = ref.join(d.rename({col: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
        assert np.isfinite(v).all(), f
        return v
    X = np.stack([load(BASE), load("out/probe_persistence.csv"),
                  load("out/probe_clim.csv"), load("out/probe_ar.csv")])
    out = w4 @ X
    pl.DataFrame({"ID": ref["ID"], "Target": out}).write_csv(OUT)
    print(f"  wrote {OUT}  std={out.std():.4f}  range [{out.min():.2f},{out.max():.2f}]")

if __name__ == "__main__":
    main()
