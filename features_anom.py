"""Per-cell standardised anomalies of the external covariates (session 9c).

The gap this closes
-------------------
Every ERA5 and NCEP feature built so far is a raw physical level: `e5PER_acc` is millimetres
of accumulated precipitation minus evaporation minus runoff, `e5SW_t` is a soil-water column
in metres, `SWE_t` is snow water equivalent in millimetres.  Meanwhile `lat` and `lon` are
excluded from every final feature set, per the organisers' ruling.  So the model is shown
"accumulated P-E-R = 350 mm" with no way to learn that 350 mm is a drought in the Amazon and a
record flood in the Sahel.  It has per-cell TWS statistics (cmean, csd, clim_next, ac1, mad1)
that locate a cell in TWS space, but nothing that locates it in covariate space.

The target compounds this.  TWS here is a standardised anomaly (RMS ~1.01, mean ~0), and the
change in an anomaly is driven by the *anomaly* of the water balance, not its raw total: a
cell that receives its perfectly normal seasonal 350 mm should have zero expected change.
`add_era5` even carries the comment "anomaly of accumulated balance vs the cell's
climatological balance for those months" immediately above its `return` -- the feature was
identified and never written.

What this builds
----------------
For each covariate, the cell's own climatology for that calendar month, then

  storage-like variables (soil water, snow, temperature)
      _t, _k  z-scored level at t and at the last observed month, and their difference _d
  flux variables (P, E, R, P-E-R)
      _t      z-scored flux at t
      _acc    the ACCUMULATED ANOMALY over the months while TWS was unobserved, i.e.
              sum over (t_known, t] of (flux - that cell's climatological flux for that
              calendar month), divided by the cell's flux SD.  This is the quantity that
              should drive the change in a TWS anomaly.

Compliance: the climatology is computed from HISTORY MONTHS ONLY (the training period for
FINAL, the pseudo-history for layouts A and B), so no month at or after a prediction target
contributes to it.  Covariate values at t are permitted -- they are never masked, and the
organisers allow external covariates whose source date is <= t.  Coordinates are used only to
group a cell with its own history, never as a feature, exactly as for the TWS climatology.
"""
import polars as pl


def anom_table(tab, cols, hist_months, pfx):
    """Z-score each column against the cell's own climatology for that calendar month.

    tab: lat, lon, time + cols.  hist_months: the months allowed to form the climatology.
    Returns lat, lon, time + one '<pfx><col>z' column per input column."""
    t = tab.with_columns(pl.col("time").dt.month().alias("cm"))
    h = t.filter(pl.col("time").is_in(hist_months))
    mu = h.group_by(["lat", "lon", "cm"]).agg([pl.col(c).mean().alias(c + "_mu") for c in cols])
    t = t.join(mu, on=["lat", "lon", "cm"], how="left")
    t = t.with_columns([(pl.col(c) - pl.col(c + "_mu")).alias(c + "_an") for c in cols])
    ha = t.filter(pl.col("time").is_in(hist_months))
    sd = ha.group_by(["lat", "lon"]).agg([pl.col(c + "_an").std().alias(c + "_sd") for c in cols])
    t = t.join(sd, on=["lat", "lon"], how="left")
    # A bare per-cell SD is unsafe: snow in the subtropics and runoff in deserts are near
    # constant, so a cell SD near zero turns a rounding-level anomaly into a huge z-score.
    # Floor each cell's SD at 5% of the variable's global anomaly SD, fall back to the global
    # value where the cell has too little history, and clip the result.
    # float() matters: a numpy scalar here makes polars treat the floor as a length-1 Series,
    # which then refuses to broadcast. clip(lower_bound=...) is also clearer than max_horizontal.
    g = {c: float(ha[c + "_an"].std() or 1.0) for c in cols}
    t = t.with_columns([
        pl.col(c + "_sd").fill_null(g[c]).clip(lower_bound=0.05 * g[c]).alias(c + "_sde")
        for c in cols])
    t = t.with_columns([(pl.col(c + "_an") / pl.col(c + "_sde")).clip(-10, 10).alias(pfx + c + "z")
                        for c in cols])
    return t.select(["lat", "lon", "time"] + [pfx + c + "z" for c in cols])


def add_anom(r, atab, storage_z, flux_z):
    """r must have lat, lon, time, t_known.  storage_z / flux_z are column names in atab."""
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
        # accumulated anomaly over (t_known, t]: the months while TWS was unobserved, plus t
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
    """Fixed-length antecedent windows of the covariate anomalies, ending at t.

    The `_acc` features accumulate over (t_known, t], whose length is the horizon -- so they
    describe the gap, not the cell's condition. Drought is a memory process: what matters for
    where storage sits is how wet or dry the cell has been over the last 3, 6 and 12 months,
    regardless of when TWS was last seen. SPEI-3/6/12 supply exactly that for METEOROLOGICAL
    drought and are among the strongest features in the model; nothing supplies it for the
    actual water balance or for modelled storage.

    flux_z    summed over the window (an accumulated anomaly, like _acc but fixed length)
    storage_z averaged over the window (the mean state, not a sum)

    These depend on (cell, t) only, not on t_known, so they are cheap: one group per cell-month.
    Compliance: every month in the window is <= t, the same rule as every other covariate here."""
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
    """Modelled total water storage = soil water + snow water equivalent, in mm.

    Every reanalysis feature so far keeps these two apart, and a tree can split on each but
    cannot add them, so the quantity the literature reports as the single strongest predictor
    of GRACE TWSA -- the land-surface model's own total storage -- was never available to the
    model. Its z-scored change over the unobserved window (`an_<out>z_d`) is the closest legal
    analogue of the GRACE change being predicted.

    soil_scale converts the soil term to millimetres: ERA5's column is metres of water
    (volumetric fraction x layer thickness), NCEP's is already mm."""
    if tab is None or soil not in tab.columns or snow not in tab.columns:
        return tab
    return tab.with_columns((pl.col(soil) * soil_scale + pl.col(snow)).alias(out))


# which covariates behave like a stored quantity and which like a flux
ERA5_STORAGE = ["e5SW", "e5SWE", "e5T2M", "e5MTWS", "e5SW1", "e5SW2", "e5SW3", "e5SW4"]
# the four soil layers are present only in a profile build; build_mats filters this list to
# the columns the loaded table actually has, so the extra names are a no-op otherwise
ERA5_FLUX = ["e5P", "e5E", "e5R", "e5PER"]
NCEP_STORAGE = ["SW", "SWE", "MTWS"]
NCEP_FLUX = ["P", "E", "R", "PER"]
COV_STORAGE = ["SOIL_MOISTURE_t"]        # SPEI is already a standardised index; soil moisture is not
# NCEP-R2 and CPC were never passed to anom_build, so their 24 level features are raw millimetres
# and fractions -- the exact encoding the session-9 finding was about. Measured on A_va, the ratio
# sd(per-cell mean)/sd(overall) is 0.77-1.01 for these columns against 0.28-0.36 for the encoded
# ones, and none of them appears in any top-gain list.
NCEP2_STORAGE = ["r2SW", "r2SWE", "r2MTWS"]
NCEP2_FLUX = ["r2P", "r2E", "r2R", "r2PER"]
CPC_STORAGE = ["cpcSW"]
# the SPEI columns are standardised in the large, but their per-cell CALENDAR-MONTH means still
# run to +-0.9 with an sd of 0.24, which is what a per-cell-per-month climatology removes
SPEI_STORAGE = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t"]
PFX = "an_"                              # every feature built here starts with this, so it can be ablated


def build(tab, storage, flux, hist_months):
    """Prepare one covariate table for add_anom.  Returns (anomaly table, storage z-cols,
    flux z-cols), or None when the source table is missing or has none of the columns."""
    if tab is None:
        return None
    have = [c for c in storage + flux if c in tab.columns]
    if not have:
        return None
    return (anom_table(tab, have, hist_months, PFX),
            [PFX + c + "z" for c in storage if c in have],
            [PFX + c + "z" for c in flux if c in have])
