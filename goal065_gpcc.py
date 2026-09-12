"""Candidate c7: gauge-based precipitation (GPCC Full Data Monthly v2022) as a side-car.

Why it is worth a measurement. Precipitation is the model's strongest forcing -- `aw_an_e5Pz_acc`
is the number two feature at 7.1% of gain -- but every precipitation the model sees is REANALYSIS
(ERA5, NCEP-R1, NCEP-R2). GPCC is the other kind of number: rain gauges, interpolated, with a gauge
count per cell. Where networks are dense it is the closer thing to a measurement; where they are
empty it is worse than reanalysis, which is why `numgauge` is carried as a feature too, so the tree
can tell the two regimes apart.

Encoding is the project's own, not a new one. Session 9's largest win was discovering that raw
millimetres are unusable to a model with no coordinates and that the per-cell standardised anomaly
is what carries the signal, so GPCC enters through exactly that path: `features_anom.build` for the
per-cell calendar-month z-score, `_acc` over (t_known, t], fixed 3/6/12-month antecedent windows,
and `add_anwide` regional means, which is the scale §3.2c showed the forcing actually lives at.

Compliance. The climatology comes from HISTORY months only, so no month at or after a target enters
it. Every value read is a month <= t. The product is rain gauges alone: no GRACE, and no model that
assimilates GRACE. Coordinates define the neighbourhood for the regional means and are not features.
Licence CC BY 4.0, Deutscher Wetterdienst; DOI 10.5676/DWD_GPCC/FD_M_V2022_100.

usage: python goal065_gpcc.py <LAYOUT>   -> out/mats/{L}_{tr,va}_gpcc.parquet
"""
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
    """Monthly gauge precipitation and gauge count on the project's own 1-degree land grid."""
    fs = sorted(glob.glob("external/gpcc/full_data_monthly_v2022_*_10.nc"))
    if not fs:
        raise SystemExit("no external/gpcc/*.nc -- download the 1.0 degree decade files first")
    # open_mfdataset would need dask; the decade files are small enough to concatenate directly
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
    built = FA.build(tab, ["gpccN"], ["gpccP"], HM)      # gauge count is a state, rainfall is a flux
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
