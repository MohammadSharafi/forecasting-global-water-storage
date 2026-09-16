import glob
import os

import numpy as np
import polars as pl

PRODUCTS = ("spi24", "spi48", "spi09", "fapar", "fapanom", "smanom")
DIR = "external/gdo"

def _name_of(ds):
    for k, v in ds.data_vars.items():
        if k.endswith(("_bnds", "_bounds")) or "bnds" in getattr(v, "dims", ()):
            continue
        if v.ndim >= 2:
            return k
    return None

def load_gdo(lats, lons, d=DIR, start="2001-01-01", end="2019-06-01", products=PRODUCTS):
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

            da = da.resample(time="1MS").mean()
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
            da = da.interp(lat=lat_c, lon=lon_c, method="linear", kwargs={"fill_value": None})
            df = da.to_dataframe(name=name).reset_index()[["time", "lat", "lon", name]]
            t = pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))
            out = t if out is None else out.join(t, on=["time", "lat", "lon"], how="full",
                                                 coalesce=True)
            cols.append(name)
            print(f"gdo {name}: {len(files)} file(s), {t['time'].n_unique()} months", flush=True)
        except Exception as e:
            print(f"gdo {name}: skipped ({type(e).__name__}: {e})", flush=True)
    if out is None:
        return None, []
    return out.select(["lat", "lon", "time"] + cols), cols
