"""Build the feature superset once per layout (A, B, C, FINAL) and write float32 parquet matrices.
usage: python build_mats.py A|B|C|FINAL

The feature builder is exposed as prepare(L)['featfn'] so recursive.py can call it with an
extended observation table (its own predictions fed forward) without a second copy of the pipeline.
The script's output is unchanged by that refactor -- verified byte-identical."""
import re
import polars as pl, numpy as np, sys, time, gc, os, json
from features import COV, training_rows, training_rows_coherent, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
from features_ncep import load_ncep, add_ncep
import glob
from features_era5 import load_era5, add_era5, era5_feats
from features_x import load_ncep2, load_cpc, add_ext, add_wide4, WIDE4, add_covwin, COVWIN, cell_response, add_response, RESP, add_anwide, add_anwide_multi
from features_anom import build as anom_build, add_anom, add_anom_windows, add_mtws, ERA5_STORAGE, ERA5_FLUX, NCEP_STORAGE, NCEP_FLUX, COV_STORAGE, NCEP2_STORAGE, NCEP2_FLUX, CPC_STORAGE, SPEI_STORAGE
from features_scale import add_scale, SCALE
from features_gdo import load_gdo


# A layout may carry its variant in its NAME -- 'Ap3' is layout A built with PER_ROW=3, 'Ae' is
# layout A built with the ERA5 soil profile split apart, 'FINALe' likewise. The variant writes to
# out/mats/<name>_*.parquet and leaves the cached A_*/FINAL_* matrices untouched, which is what
# makes it safe to sweep a parameter that changes every matrix on a machine holding a working
# submission.
# ...and a trailing "v<tag>" namespaces the files without changing what is built, so a feature
# change that has to be measured against the cached matrices can build its own set beside them.
_LNAME = re.compile(r"(FINAL|[ABC])(?:p(\d+))?(e)?(?:v[a-z0-9]+)?$")


def _parse(L):
    m = _LNAME.fullmatch(L)
    return m.groups() if m else (L, None, None)


def base_of(L):
    """'Ap3' -> 'A', 'Ae' -> 'A', 'FINALe' -> 'FINAL'."""
    return _parse(L)[0]


def prof_of(L):
    """True when this layout wants the ERA5 soil layers kept apart instead of summed.

    The name suffix is how the A/B ablation builds its own matrices beside the cached ones
    ('Ae' against 'A'). PROF=1 is how the orchestrator turns the winner on for every layout
    once it has been gated, without renaming anything it already checkpoints."""
    return _parse(L)[2] is not None or os.environ.get("PROF", "") == "1"


def per_row_of(L, default=None):
    pr = _parse(L)[1]
    if pr is not None:
        return int(pr)
    return int(os.environ.get("PER_ROW", "3")) if default is None else default


def prepare(L, t0=None):
    """Load every source and return the state plus a feature closure `featfn(rows, obs, loyo)`.

    Splitting this out of the script body lets recursive.py reuse the exact same feature
    construction with a different observation table; the script (main) uses it identically, so
    the written matrices are unchanged."""
    if t0 is None:
        t0 = time.time()
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    lats = tr["lat"].unique().to_list(); lons = tr["lon"].unique().to_list()
    nc, cols = load_ncep(lats, lons); nc2, cols2 = load_ncep2(lats, lons); cpc, cols3 = load_cpc(lats, lons)
    # Copernicus GDO: long-window SPI and fAPAR. Both are LEVELS, so they only become usable after
    # the same per-cell standardisation that made the ERA5/NCEP block work. Absent unless
    # downloaded, and a no-op when absent, exactly like ERA5.
    gdo, gcols = load_gdo(lats, lons)
    ERA = load_era5(prof=prof_of(L)) if glob.glob("external/era5/*.nc") else None
    print("era5:", None if ERA is None else ERA.shape,
          "(soil profile split)" if prof_of(L) else "", flush=True)
    print("gdo:", gcols or "not present", flush=True)
    print("ext loaded", cols, cols2, cols3, f"({time.time()-t0:.0f}s)", flush=True)
    if base_of(L) == "FINAL":
        te = pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
        cov_all = pl.concat([tr.select(["lat", "lon", "time"]+COV), te.select(["lat", "lon", "time"]+COV)])
        obs_hist = tr.select(["lat", "lon", "time", "TWS_t"])
        obs_all = pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat", "lon", "time", "TWS_t"])])
        hist = tr
        known = (te.select(["lat", "lon", "time"]).join(obs_all.rename({"time": "t_obs"}).select(["lat", "lon", "t_obs"]), on=["lat", "lon"], how="inner")
                 .filter(pl.col("t_obs") <= pl.col("time")).group_by(["lat", "lon", "time"]).agg(pl.col("t_obs").max().alias("t_known")))
        rows_va = te.select(["ID", "lat", "lon", "time"]).join(known, on=["lat", "lon", "time"], how="left")
        meta_va = ["ID", "lat", "lon", "time", "t_known", "horizon", "tws_known"]
    else:
        sfx = {"A": "", "B": "_B", "C": "_C"}[base_of(L)]; tp = pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist = pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
        cov_all = tr.select(["lat", "lon", "time"]+COV)
        obs_hist = hist.select(["lat", "lon", "time", "TWS_t"])
        obs_all = pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat", "lon", "time", "TWS_t"])])
        rows_va = tp.select(["lat", "lon", "time", "t_known", "target"]); meta_va = ["lat", "lon", "time", "t_known", "horizon", "tws_known", "target"]
    sums = clim_sums(hist); _, cell = cell_stats(hist); resp = cell_response(hist)
    # Climatologies for the covariate anomalies come from HISTORY MONTHS ONLY, so no month at or
    # after a prediction target can enter them.
    HM = hist["time"].unique().to_list()
    # modelled total water storage: soil water + snow, the predictor the literature ranks first
    ERA = add_mtws(ERA, "e5SW", "e5SWE", "e5MTWS", 1000.0)   # e5SW is metres of water, e5SWE is mm
    nc = add_mtws(nc, "SW", "SWE", "MTWS")                   # both already mm
    nc2 = add_mtws(nc2, "r2SW", "r2SWE", "r2MTWS")           # same, for the R2 stream
    E5S = [c for c in ERA5_STORAGE if ERA is None or c in ERA.columns]
    ANOM = [x for x in (anom_build(ERA, E5S, ERA5_FLUX, HM),
                        anom_build(nc, NCEP_STORAGE, NCEP_FLUX, HM),
                        anom_build(cov_all.select(["lat", "lon", "time"]+COV_STORAGE).unique(["lat", "lon", "time"]), COV_STORAGE, [], HM),
                        # SPI and fAPAR are indices/levels, so storage-like: z-scored value at t
                        # and at t_known, their difference, and the 3/6/12-month antecedent windows
                        anom_build(gdo, gcols, [], HM),
                        # NCEP-R2 and CPC: raw levels since session 8, never standardised
                        anom_build(nc2, [c for c in NCEP2_STORAGE if c in nc2.columns], NCEP2_FLUX, HM),
                        anom_build(cpc, CPC_STORAGE, [], HM),
                        # the released SPEI columns, whose per-cell seasonal bias survives
                        anom_build(cov_all.select(["lat", "lon", "time"]+SPEI_STORAGE).unique(["lat", "lon", "time"]), SPEI_STORAGE, [], HM)) if x is not None]
    print("anomaly tables:", [(len(sz), len(fz)) for _, sz, fz in ANOM], f"({time.time()-t0:.0f}s)", flush=True)

    def featfn(rows, obs, loyo):
        r = add_recent(add_wide(add_ar(assemble2(rows, cov_all, obs, sums, cell, loyo=loyo), obs)), obs)
        r = add_scale(r, obs)   # zonal context: the widest aggregation elsewhere is only radius 4
        r, NF = add_ncep(r, nc, cols); r, NF2 = add_ext(r, nc2, cols2, acc_cols=["r2P", "r2E", "r2PER"]); r, NF3 = add_ext(r, cpc, cols3)
        r = add_wide4(r); r = add_covwin(r, cov_all); r = add_response(r, resp)
        EF = []
        if ERA is not None:
            r = add_era5(r, ERA); EF = era5_feats(ERA)
        AF = []   # per-cell standardised covariate anomalies (features_anom): the level features above
        for at, sz, fz in ANOM:   # are raw mm and unusable without lat/lon, which is not a feature
            r, f = add_anom(r, at, sz, fz); AF += f
            r, f = add_anom_windows(r, at, fz, sz); AF += f   # 3/6/12-month antecedent windows ending at t
        # regional means of the covariate-anomaly block: the scale our error actually lives at
        _rad = os.environ.get("ANWIDE_R", "4")
        if "," in _rad:
            r, AW = add_anwide_multi(r, radii=tuple(int(x) for x in _rad.split(",")))
        else:
            r, AW = add_anwide(r, radius=int(_rad))
        F = FEATS2+AR+WIDE+RECENT+NF+NF2+NF3+WIDE4+COVWIN+RESP+EF+AF+SCALE+AW
        F = list(dict.fromkeys(F)); return r, F

    return dict(tr=tr, cov_all=cov_all, obs_all=obs_all, obs_hist=obs_hist, hist=hist,
                rows_va=rows_va, meta_va=meta_va, featfn=featfn)


def main():
    L = sys.argv[1]; os.makedirs("out/mats", exist_ok=True); t0 = time.time()
    S = prepare(L, t0)
    Xva, F = S["featfn"](S["rows_va"], S["obs_all"], False)
    Xva.select(list(dict.fromkeys(S["meta_va"]+F))).with_columns([pl.col(f).cast(pl.Float32) for f in F]).write_parquet(f"out/mats/{L}_va.parquet")
    print("va", Xva.shape, f"({time.time()-t0:.0f}s)", flush=True)
    del Xva; gc.collect()
    Xtr, _ = S["featfn"](training_rows_coherent(S["hist"], np.random.default_rng(11 if base_of(L) == "FINAL" else 0), per_row=per_row_of(L)), S["obs_hist"], True)
    Xtr.select(list(dict.fromkeys(["lat", "lon", "time", "t_known", "horizon", "tws_known", "target"]+F))).with_columns([pl.col(f).cast(pl.Float32) for f in F]).write_parquet(f"out/mats/{L}_tr.parquet")
    # Per-layout, because feats.json is read by run_models at TRAIN time: a build of one layout
    # was overwriting the feature list another layout's training run depended on, which is how a
    # run of layout A died mid-experiment tonight. The shared file is still written for anything
    # that reads it, but the layout-specific one wins.
    json.dump(F, open(f"out/mats/feats_{L}.json", "w"))
    json.dump(F, open("out/mats/feats.json", "w")); print("tr", Xtr.shape, "nfeat", len(F), f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
