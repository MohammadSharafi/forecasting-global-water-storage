import polars as pl

def anom_table(tab, cols, hist_months, pfx):
    t = tab.with_columns(pl.col("time").dt.month().alias("cm"))
    h = t.filter(pl.col("time").is_in(hist_months))
    mu = h.group_by(["lat", "lon", "cm"]).agg([pl.col(c).mean().alias(c + "_mu") for c in cols])
    t = t.join(mu, on=["lat", "lon", "cm"], how="left")
    t = t.with_columns([(pl.col(c) - pl.col(c + "_mu")).alias(c + "_an") for c in cols])
    ha = t.filter(pl.col("time").is_in(hist_months))
    sd = ha.group_by(["lat", "lon"]).agg([pl.col(c + "_an").std().alias(c + "_sd") for c in cols])
    t = t.join(sd, on=["lat", "lon"], how="left")

    g = {c: float(ha[c + "_an"].std() or 1.0) for c in cols}
    t = t.with_columns([
        pl.col(c + "_sd").fill_null(g[c]).clip(lower_bound=0.05 * g[c]).alias(c + "_sde")
        for c in cols])
    t = t.with_columns([(pl.col(c + "_an") / pl.col(c + "_sde")).clip(-10, 10).alias(pfx + c + "z")
                        for c in cols])
    return t.select(["lat", "lon", "time"] + [pfx + c + "z" for c in cols])

def add_anom(r, atab, storage_z, flux_z):
    allz = storage_z + flux_z
    r = r.join(atab.rename({c: c + "_t" for c in allz}), on=["lat", "lon", "time"], how="left")
    if storage_z:
        r = r.join(atab.select(["lat", "lon", "time"] + storage_z)
                       .rename({"time": "t_known", **{c: c + "_k" for c in storage_z}}),
                   on=["lat", "lon", "t_known"], how="left")
        for c in storage_z:
            r = r.with_columns((pl.col(c + "_t") - pl.col(c + "_k")).alias(c + "_d"))
    feats = [c + "_t" for c in allz] + [c + "_k" for c in storage_z] + [c + "_d" for c in storage_z]
    if flux_z:

        w = atab.select(["lat", "lon", "time"] + flux_z).rename({"time": "tw"})
        j = (r.select(["lat", "lon", "time", "t_known"]).unique()
              .join(w, on=["lat", "lon"], how="inner")
              .filter((pl.col("tw") > pl.col("t_known")) & (pl.col("tw") <= pl.col("time"))))
        acc = j.group_by(["lat", "lon", "time", "t_known"]).agg(
            [pl.col(c).sum().alias(c + "_acc") for c in flux_z])
        r = r.join(acc, on=["lat", "lon", "time", "t_known"], how="left")
        feats += [c + "_acc" for c in flux_z]
    return r, feats

WINDOWS = (3, 6, 12)

def add_anom_windows(r, atab, flux_z, storage_z, windows=WINDOWS):
    if not (flux_z or storage_z):
        return r, []
    feats = []
    for W in windows:
        w = atab.select(["lat", "lon", "time"] + flux_z + storage_z).rename({"time": "tw"})
        j = (r.select(["lat", "lon", "time"]).unique()
              .join(w, on=["lat", "lon"], how="inner")
              .filter((pl.col("tw") <= pl.col("time")) &
                      (pl.col("tw") > pl.col("time").dt.offset_by(f"-{W}mo"))))
        agg = ([pl.col(c).sum().alias(f"{c}_w{W}") for c in flux_z] +
               [pl.col(c).mean().alias(f"{c}_m{W}") for c in storage_z])
        r = r.join(j.group_by(["lat", "lon", "time"]).agg(agg),
                   on=["lat", "lon", "time"], how="left")
        feats += [f"{c}_w{W}" for c in flux_z] + [f"{c}_m{W}" for c in storage_z]
    return r, feats

def add_mtws(tab, soil, snow, out, soil_scale=1.0):
    if tab is None or soil not in tab.columns or snow not in tab.columns:
        return tab
    return tab.with_columns((pl.col(soil) * soil_scale + pl.col(snow)).alias(out))

ERA5_STORAGE = ["e5SW", "e5SWE", "e5T2M", "e5MTWS", "e5SW1", "e5SW2", "e5SW3", "e5SW4"]

ERA5_FLUX = ["e5P", "e5E", "e5R", "e5PER"]
NCEP_STORAGE = ["SW", "SWE", "MTWS"]
NCEP_FLUX = ["P", "E", "R", "PER"]
COV_STORAGE = ["SOIL_MOISTURE_t"]

NCEP2_STORAGE = ["r2SW", "r2SWE", "r2MTWS"]
NCEP2_FLUX = ["r2P", "r2E", "r2R", "r2PER"]
CPC_STORAGE = ["cpcSW"]

SPEI_STORAGE = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t"]
PFX = "an_"

def build(tab, storage, flux, hist_months):
    if tab is None:
        return None
    have = [c for c in storage + flux if c in tab.columns]
    if not have:
        return None
    return (anom_table(tab, have, hist_months, PFX),
            [PFX + c + "z" for c in storage if c in have],
            [PFX + c + "z" for c in flux if c in have])
