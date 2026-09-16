import sys

import numpy as np
import polars as pl

def hist_path(L):
    return "out/pseudo_hist.parquet" if L[0] == "A" or L.startswith("FINAL") else (
        f"out/pseudo_hist_{L[0]}.parquet")

def cell_stats(L):

    h = (pl.read_parquet(f"out/mats/{L}_tr.parquet", columns=["lat", "lon", "time", "tws_known"])
         .rename({"tws_known": "TWS_t"}).unique(["lat", "lon", "time"]).drop_nulls())
    h = h.sort(["lat", "lon", "time"])
    rows = {}
    for (la, lo), g in h.group_by(["lat", "lon"], maintain_order=True):
        v = g["TWS_t"].to_numpy().astype(float)
        t = g["time"].to_list()
        n = len(v)
        if n < 24:
            continue
        mth = np.array([x.month for x in t])
        cm = np.array([v[mth == m].mean() if (mth == m).any() else 0.0 for m in range(1, 13)])
        clim = cm[mth - 1]
        resid = v - clim
        sd = float(v.std())
        def ac(k):
            if n <= k or v[:-k].std() < 1e-9 or v[k:].std() < 1e-9:
                return 0.0
            return float(np.corrcoef(v[:-k], v[k:])[0, 1])
        d1 = np.diff(v); d3 = v[3:] - v[:-3] if n > 3 else np.array([0.0])
        tt = np.arange(min(36, n), dtype=float)
        slope = float(np.polyfit(tt, v[-len(tt):], 1)[0]) if len(tt) > 2 else 0.0
        rows[(la, lo)] = dict(
            c_sd=sd, c_ac1=ac(1), c_ac12=ac(12), c_amp=float(cm.std()),
            c_climr2=float(1 - resid.var() / max(v.var(), 1e-9)),
            c_d1sd=float(d1.std()) if len(d1) else 0.0,
            c_d3sd=float(d3.std()) if len(d3) else 0.0,
            c_trend=slope,
            c_xfrac=float(np.mean(np.abs(v - v.mean()) > sd)) if sd > 0 else 0.0,
            c_n=float(n))
    return rows

def main():
    L = sys.argv[1]
    stats = cell_stats(L)
    cols = ["c_sd", "c_ac1", "c_ac12", "c_amp", "c_climr2", "c_d1sd", "c_d3sd", "c_trend",
            "c_xfrac", "c_n"]
    print(f"{L}: statistics for {len(stats)} cells from out/mats/{L}_tr.parquet")
    for part in ("tr", "va"):
        m = pl.read_parquet(f"out/mats/{L}_{part}.parquet", columns=["lat", "lon"])
        la = m["lat"].to_numpy(); lo = m["lon"].to_numpy()
        out = {c: np.full(len(la), np.nan) for c in cols}
        for i, (a, b) in enumerate(zip(la, lo)):
            d = stats.get((a, b))
            if d:
                for c in cols:
                    out[c][i] = d[c]
        out["cf_lat"] = la.astype(float)
        out["cf_lon"] = lo.astype(float)
        out["cf_abslat"] = np.abs(la).astype(float)
        df = pl.DataFrame({k: v.astype(np.float32) for k, v in out.items()})
        df.write_parquet(f"out/mats/{L}_{part}_cell.parquet")
        cov = float(np.isfinite(out["c_sd"]).mean())
        print(f"  {part}: {df.shape} written, cell coverage {cov:.1%}")

if __name__ == "__main__":
    main()
