import glob
import sys
import time

import numpy as np
import polars as pl

from goal065_gpcc import gpcc_table

def build(L):
    t0 = time.time()
    parts = {}
    for part in ("tr", "va"):
        m = pl.read_parquet(f"out/mats/{L}_{part}.parquet",
                            columns=["lat", "lon", "time", "t_known"])
        parts[part] = m
    allm = pl.concat(list(parts.values())).unique(["lat", "lon", "time", "t_known"])
    lats = sorted(set(allm["lat"].to_list())); lons = sorted(set(allm["lon"].to_list()))
    g = gpcc_table(lats, lons)

    g = g.with_columns(pl.col("time").dt.month().alias("m"))
    clim = g.group_by(["lat", "lon", "m"]).agg(pl.col("gpccP").mean().alias("pclim"))
    g = g.join(clim, on=["lat", "lon", "m"], how="left").with_columns(
        (pl.col("gpccP") - pl.col("pclim")).alias("panom"))
    w = g.select(["lat", "lon", "time", "panom", "gpccP", "gpccN"]).rename({"time": "tw"})
    j = (allm.join(w, on=["lat", "lon"], how="inner")
         .filter((pl.col("tw") > pl.col("t_known")) & (pl.col("tw") <= pl.col("time"))))
    acc = j.group_by(["lat", "lon", "time", "t_known"]).agg([
        pl.col("panom").sum().alias("gacc_anom"),
        pl.col("gpccP").sum().alias("gacc_raw"),
        pl.col("panom").mean().alias("gacc_rate"),
        pl.col("gpccN").min().alias("gacc_ngmin"),
        pl.len().alias("gacc_n")])
    cols = ["gacc_anom", "gacc_raw", "gacc_rate", "gacc_ngmin", "gacc_n"]
    print(f"{L}: accumulated GPCC over the anchor gap for {acc.height:,} (cell, time, anchor) keys "
          f"({time.time()-t0:.0f}s)")
    for part, m in parts.items():
        r = m.join(acc, on=["lat", "lon", "time", "t_known"], how="left")
        df = r.select(cols).with_columns([pl.col(c).cast(pl.Float32) for c in cols])
        df.write_parquet(f"out/mats/{L}_{part}_gacc.parquet")
        cov = float(np.isfinite(df["gacc_anom"].to_numpy().astype(float)).mean())
        print(f"  {part}: {df.shape} written, coverage {cov:.1%}")

if __name__ == "__main__":
    build(sys.argv[1])
