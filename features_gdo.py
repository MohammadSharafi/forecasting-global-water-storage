"""Copernicus GDO products as per-cell standardised anomalies (session 9v).

Why these and not more ERA5
---------------------------
The residual variogram says the remaining error is large-scale and spatially coherent (correlation
+0.967 with the neighbouring cell, +0.899 at two cells), so a finer-resolution version of the same
forcing -- ERA5-Land -- cannot reach it: the error does not live at fine scales. That objection says
nothing about NEW VARIABLES, and two GDO products are new information rather than another function
of the P, E and R this feature set already has:

  spi24, spi48   The feature set stops at a 12-month window. TWS includes groundwater, which
                 integrates over years, so nothing in the model currently sees beyond one year of
                 accumulated deficit. This is the single clearest gap in the covariate set.
  fapar          Vegetation greenness is an OBSERVATION of how much water the surface actually had,
                 integrated over weeks. It is not derived from the same reanalysis as everything
                 else here, so its errors are not the same errors.

Both have a source date at or before t, which is the condition the organisers set for external
covariates. GDO also publishes a GRACE TWS layer and seasonal forecasts: neither is used. The
first is prohibited outright; the second is deliberately avoided because a forecast issued at t
encodes t+1, which the rules prohibit elsewhere, and the question is unanswered.

Everything loaded here is fed through features_anom.build(), the SAME per-cell standardisation that
produced this project's only large leaderboard gain. That is the point: a raw level is unusable to
a model with no location features, and these products are levels.

Layout on disk
--------------
    external/gdo/<name>/*.nc      one directory per product; the directory name becomes the feature

The loader is deliberately format-tolerant rather than hardcoded to filenames: it takes the first
real data variable in each file, accepts lat/latitude and lon/longitude, folds 0..360 longitudes,
aggregates sub-monthly data (GDO ships several products by dekad) to monthly means, and
concatenates files in time. A product that is missing is skipped with a message, exactly as ERA5
already is, so nothing here can break a run that does not have the data.
"""
import glob
import os

import numpy as np
import polars as pl

# products to look for, and whether the raw value is a state/level (all of these are)
PRODUCTS = ("spi24", "spi48", "spi09", "fapar", "fapanom", "smanom")
DIR = "external/gdo"


def _name_of(ds):
    """The first variable that carries data rather than bounds or grid mapping."""
    for k, v in ds.data_vars.items():
        if k.endswith(("_bnds", "_bounds")) or "bnds" in getattr(v, "dims", ()):
            continue
        if v.ndim >= 2:
            return k
    return None


def load_gdo(lats, lons, d=DIR, start="2001-01-01", end="2019-06-01", products=PRODUCTS):
    """-> (polars table with lat, lon, time + one column per product found, [column names])."""
    import xarray as xr
    lat_c = np.array(sorted(set(lats))); lon_c = np.array(sorted(set(lons)))
    out, cols = None, []
    for name in products:
        files = sorted(glob.glob(os.path.join(d, name, "*.nc")))
        if not files:
            continue
        try:
            das = []
            for f in files:
                ds = xr.open_dataset(f)
                v = _name_of(ds)
                if v is None:
                    continue
                da = ds[v]
                ren = {}
                for a, b in (("latitude", "lat"), ("longitude", "lon"), ("Lat", "lat"),
                             ("Lon", "lon"), ("time_counter", "time")):
                    if a in da.dims or a in da.coords:
                        ren[a] = b
                if ren:
                    da = da.rename(ren)
                if "time" not in da.dims:
                    continue
                for extra in [x for x in da.dims if x not in ("time", "lat", "lon")]:
                    da = da.isel({extra: 0})
                das.append(da)
            if not das:
                print(f"gdo {name}: files present but no usable variable", flush=True); continue
            da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
            da = da.sortby("time").sel(time=slice(start, end))
            # GDO ships several products by dekad; a monthly mean is the right aggregation and is
            # a no-op for products that are already monthly.
            da = da.resample(time="1MS").mean()
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
            da = da.interp(lat=lat_c, lon=lon_c, method="linear", kwargs={"fill_value": None})
            df = da.to_dataframe(name=name).reset_index()[["time", "lat", "lon", name]]
            t = pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))
            out = t if out is None else out.join(t, on=["time", "lat", "lon"], how="full",
                                                 coalesce=True)
            cols.append(name)
            print(f"gdo {name}: {len(files)} file(s), {t['time'].n_unique()} months", flush=True)
        except Exception as e:                                              # noqa: BLE001
            print(f"gdo {name}: skipped ({type(e).__name__}: {e})", flush=True)
    if out is None:
        return None, []
    return out.select(["lat", "lon", "time"] + cols), cols
