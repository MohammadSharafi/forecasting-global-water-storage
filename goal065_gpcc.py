import glob
import sys
import time

import numpy as np
import polars as pl
import xarray as xr

import features_anom as FA
from build_mats import base_of
from features_x import add_anwide

RADIUS = 4

def gpcc_table(lats, lons):
    fs = sorted(glob.glob("external/gpcc/full_data_monthly_v2022_*_10.nc"))
    if not fs:
        raise SystemExit("no external/gpcc/*.nc -- download the 1.0 degree decade files first")

    ds = xr.concat([xr.open_dataset(f)[["precip", "numgauge"]].load() for f in fs], dim="time")
    ds = ds.sortby("time")
    ds = ds.sel(lat=sorted(set(lats)), lon=sorted(set(lons)))
    d = ds.to_dataframe().reset_index()[["lat", "lon", "time", "precip", "numgauge"]]
    t = pl.from_pandas(d).rename({"precip": "gpccP", "numgauge": "gpccN"})
    return t.with_columns(pl.col("time").cast(pl.Date),
                          pl.col("gpccP").cast(pl.Float64), pl.col("gpccN").cast(pl.Float64))

def main():
    L = sys.argv[1]; t0 = time.time()
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    lats = tr["lat"].unique().to_list(); lons = tr["lon"].unique().to_list()
    tab = gpcc_table(lats, lons)
    if base_of(L) == "FINAL":
        hist = tr
    else:
        b = base_of(L); sfx = "" if b == "A" else f"_{b}"
        hist = pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    HM = hist["time"].unique().to_list()
    built = FA.build(tab, ["gpccN"], ["gpccP"], HM)
    if built is None:
        raise SystemExit("GPCC table has neither column after loading")
    atab, sz, fz = built
    print(f"{L}: GPCC {tab['time'].min()} -> {tab['time'].max()}, climatology from {len(HM)} history "
          f"months, storage_z={sz} flux_z={fz} ({time.time()-t0:.0f}s)", flush=True)
    for part in ("tr", "va"):
        m = pl.read_parquet(f"out/mats/{L}_{part}.parquet", columns=["lat", "lon", "time", "t_known"])
        r, f1 = FA.add_anom(m, atab, sz, fz)
        r, f2 = FA.add_anom_windows(r, atab, fz, sz)
        new = list(dict.fromkeys(f1 + f2))
        r, aw = add_anwide(r, radius=RADIUS, keys=[c for c in new if c.endswith(("_acc", "_d", "_w3", "_w6"))])
        cols = new + aw
        out = r.select([pl.col(c).cast(pl.Float32) for c in cols])
        assert out.height == m.height, f"{L} {part}: {out.height} rows vs {m.height}"
        out.write_parquet(f"out/mats/{L}_{part}_gpcc.parquet")
        print(f"  {part}: {out.height} rows, {len(cols)} cols ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
