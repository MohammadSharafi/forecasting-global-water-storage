"""Candidate c6: WaterGAP 2.2e modelled water storage as a side-car.

What it adds that nothing else does. The model's storage features are soil water and snow from
reanalysis (ERA5, NCEP). It has never seen a modelled GROUNDWATER or SURFACE-WATER store, and those
are exactly the compartments that make total storage lag rainfall by months. WaterGAP 2.2e is a
global hydrological model run for ISIMIP3a on observed climate forcing (GSWP3-W5E5) with historical
human water use, and its `tws` is the summed storage of all compartments.

Compliance, stated because this is a TWS product. R5 forbids GRACE and GRACE-derived products,
"including land-surface models that assimilate GRACE". WaterGAP obsclim does NOT assimilate GRACE:
it is a forward model driven by observed meteorology, and its storage is a physical simulation
independent of the satellite. The release is static, 1901-2019, against a test needing months
<= 2018-12, so every value read has a source date <= t. Licence CC BY 4.0 (Mueller Schmied et al.,
GMD 17, 8817, 2024). Coordinates define the neighbourhood for the regional means, never a feature.

Encoded as a STORAGE state, not a flux: the per-cell calendar-month z-score at t and at t_known,
their difference, and the 3/6/12-month mean state, plus regional means -- the §3.2c scale.

usage: python goal065_wgap.py <LAYOUT>   -> out/mats/{L}_{tr,va}_wgap.parquet
"""
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
    """WaterGAP 0.5-degree storage, block-averaged to the project's 1-degree grid."""
    d = xr.open_dataset(SRC)["tws"].sel(time=slice("2000-01-01", "2019-12-31"))
    d = d.coarsen(lat=2, lon=2, boundary="trim").mean()      # 0.5 -> 1 degree
    d = d.sel(lat=sorted(set(lats)), lon=sorted(set(lons)), method="nearest")
    d = d.assign_coords(lat=sorted(set(lats)), lon=sorted(set(lons)))
    f = d.to_dataframe().reset_index()[["lat", "lon", "time", "tws"]]
    # ISIMIP files use a cftime NoLeap calendar, which arrow cannot type. Every value is a month
    # start, so rebuild the stamp from year and month.
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
