import sys
import time

import numpy as np
import polars as pl
import xarray as xr

import features_anom as FA
from build_mats import base_of
from features_x import add_anwide

RADIUS = 4
SRC = "external/watergap/watergap_tws.nc"

def table(lats, lons):
    d = xr.open_dataset(SRC)["tws"].sel(time=slice("2000-01-01", "2019-12-31"))
    d = d.coarsen(lat=2, lon=2, boundary="trim").mean()
    d = d.sel(lat=sorted(set(lats)), lon=sorted(set(lons)), method="nearest")
    d = d.assign_coords(lat=sorted(set(lats)), lon=sorted(set(lons)))
    f = d.to_dataframe().reset_index()[["lat", "lon", "time", "tws"]]

    import datetime as _dt
    f["time"] = [_dt.date(int(x.year), int(x.month), 1) for x in f["time"]]
    t = pl.from_pandas(f).rename({"tws": "wgTWS"})
    return t.with_columns(pl.col("time").cast(pl.Date), pl.col("wgTWS").cast(pl.Float64))

def main():
    L = sys.argv[1]; t0 = time.time()
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    tab = table(tr["lat"].unique().to_list(), tr["lon"].unique().to_list())
    hist = tr if base_of(L) == "FINAL" else pl.read_parquet(
        f"out/pseudo_hist{'' if base_of(L)=='A' else '_'+base_of(L)}.parquet")
    HM = hist["time"].unique().to_list()
    built = FA.build(tab, ["wgTWS"], [], HM)
    if built is None:
        raise SystemExit("WaterGAP table empty after loading")
    atab, sz, fz = built
    print(f"{L}: WaterGAP {tab['time'].min()} -> {tab['time'].max()}, climatology from {len(HM)} "
          f"history months, storage_z={sz} ({time.time()-t0:.0f}s)", flush=True)
    for part in ("tr", "va"):
        m = pl.read_parquet(f"out/mats/{L}_{part}.parquet", columns=["lat", "lon", "time", "t_known"])
        r, f1 = FA.add_anom(m, atab, sz, fz)
        r, f2 = FA.add_anom_windows(r, atab, fz, sz)
        new = list(dict.fromkeys(f1 + f2))
        r, aw = add_anwide(r, radius=RADIUS, keys=[c for c in new if c.endswith(("_d", "_m3", "_m6"))])
        cols = new + aw
        out = r.select([pl.col(c).cast(pl.Float32) for c in cols])
        assert out.height == m.height, f"{L} {part}: {out.height} vs {m.height}"
        out.write_parquet(f"out/mats/{L}_{part}_wgap.parquet")
        print(f"  {part}: {out.height} rows, {len(cols)} cols ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
