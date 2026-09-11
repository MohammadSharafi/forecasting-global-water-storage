"""Audit the submitted model against the competition's rules (session 9n).

The top entries go to a code review, and a rule broken there costs the placing however good the
RMSE is. Everything below is checked against the ARTEFACTS -- the matrices and the feature list
the submitted run actually used -- not against what the code is believed to do.

The rules being checked, from the organisers' rulings:
  R1  latitude and longitude may not be model features (they may be used to group a cell with
      its own history, which is what every climatology here does)
  R2  no information at or after t+1 may enter a row that predicts t+1
  R3  neighbouring cells' TWS at months <= t may be used, including spatial filtering
  R4  external covariates are allowed when their source date is <= t
  R5  no GRACE-derived product other than the competition's own data

Checks
------
  features   the exact list the run used (out/mats/used_*.json, written by run_models.py), not
             feats.json -- that is the superset and does contain lat/lon
  rows       t_known <= t for every row, and horizon = months(t_known -> t) + 1
  anchoring  tws_known equals the OBSERVED TWS at t_known, reconstructed independently from
             Train.csv and the unmasked Test.csv rows
  climatology  clim_next recomputed from Train.csv alone and compared to the matrix column. If
             they agree, no test-era month entered the climatology -- a direct test of R2 rather
             than a reading of the code
  external   an inventory of external/, so the report can state exactly what was used
  code       a grep for the patterns that have caused problems before: trajectory smoothing
             (it used the next horizon's prediction), and lat/lon in a final featset

usage: python compliance.py [FINAL]
"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import polars as pl

from build_mats import base_of as layout_base   # one parser: FINALe is a FINAL matrix
L = sys.argv[1] if len(sys.argv) > 1 else "FINAL"
LB = layout_base(L)
FAIL = []


def ok(cond, msg, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(msg)


def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def used_runs():
    return sorted(glob.glob(f"out/mats/used_{L}_*.json"))


def main():
    hdr(f"R1  FEATURES ACTUALLY USED  (layout {L})")
    runs = used_runs()
    if not runs:
        print("  no out/mats/used_*.json -- run a model first; feats.json is the SUPERSET and")
        print("  contains lat/lon by design, so auditing it would be meaningless.")
    for f in runs:
        d = json.load(open(f))
        F = d["features"]
        bad = [c for c in ("lat", "lon") if c in F]
        ok(not bad, f"{os.path.basename(f)}: {d['n_features']} features, no lat/lon",
           f"model={d['model']} featset={d['featset']} dropf={d['dropf']} "
           f"anchor={d['anchor_target']} weights={d['weights']} hmix={d['hmix'] or '-'} "
           f"hfilt={d['hfilt'] or '-'}" + (f"  FOUND {bad}" if bad else ""))
        groups = {}
        for c in F:
            g = ("anchor" if c.startswith(("sa", "dsa")) else
                 "covariate anomaly" if c.startswith("an_") else
                 "ERA5" if c.startswith("e5") else
                 "NCEP2" if c.startswith("r2") else
                 "CPC" if c.startswith("cpc") else
                 "zonal" if c in ("zm", "zn", "dz", "zd3", "zd12", "dzd3", "dzd12") else
                 "NCEP" if c.split("_")[0] in ("P", "E", "R", "SW", "SWE", "PER", "MTWS") else
                 "TWS/competition")
            groups[g] = groups.get(g, 0) + 1
        print("         " + "  ".join(f"{k} {v}" for k, v in sorted(groups.items())))

    hdr("R2  ROW GEOMETRY AND ANCHORING")
    path = f"out/mats/{L}_va.parquet"
    if not os.path.exists(path):
        print(f"  {path} missing -- build it first"); return
    cols = pl.scan_parquet(path).collect_schema().names()
    want = [c for c in ("lat", "lon", "time", "t_known", "horizon", "tws_known", "clim_next")
            if c in cols]
    va = pl.read_parquet(path, columns=want)
    t = va["time"].to_list(); tk = va["t_known"].to_list()
    dm = np.array([(a.year - b.year) * 12 + (a.month - b.month) for a, b in zip(t, tk)])
    ok(bool((dm >= 0).all()), "t_known <= t for every row",
       f"min gap {dm.min()} months, max {dm.max()}")
    ok(bool((va["horizon"].to_numpy() == dm + 1).all()),
       "horizon == months(t_known -> t) + 1 for every row")

    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    obs = tr.select(["lat", "lon", "time", "TWS_t"])
    if LB == "FINAL" and os.path.exists("Test.csv"):
        te = pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
        if "TWS_t_masked" in te.columns:
            obs = pl.concat([obs, te.filter(~pl.col("TWS_t_masked"))
                                     .select(["lat", "lon", "time", "TWS_t"])])
    j = va.join(obs.rename({"time": "t_known", "TWS_t": "truth"}),
                on=["lat", "lon", "t_known"], how="left")
    d = (j["tws_known"].cast(pl.Float64) - j["truth"].cast(pl.Float64)).abs().drop_nulls().to_numpy()
    ok(len(d) > 0 and float(np.nanmax(d)) < 1e-4,
       "tws_known equals the observed TWS at t_known, reconstructed independently",
       f"max |difference| {float(np.nanmax(d)) if len(d) else float('nan'):.2e} over {len(d)} rows")

    if "clim_next" in va.columns:
        # recompute the target-month climatology from Train.csv ALONE. If the matrix column
        # matches, no test-era observation entered it.
        hist = tr if LB == "FINAL" else pl.read_parquet(
            "out/pseudo_hist.parquet" if LB == "A" else f"out/pseudo_hist_{LB}.parquet")
        cl = (hist.with_columns(pl.col("time").dt.month().alias("m"))
                  .group_by(["lat", "lon", "m"]).agg(pl.col("TWS_t").mean().alias("ref")))
        s = (va.with_columns(((pl.col("time").dt.month() % 12) + 1).alias("m"))
               .join(cl, on=["lat", "lon", "m"], how="left"))
        dd = (s["clim_next"].cast(pl.Float64) - s["ref"]).abs().drop_nulls().to_numpy()
        ok(len(dd) > 0 and float(np.nanmax(dd)) < 1e-3,
           "clim_next reproduces a HISTORY-ONLY climatology exactly",
           f"max |difference| {float(np.nanmax(dd)) if len(dd) else float('nan'):.2e} "
           f"over {len(dd)} rows")

    hdr("R4/R5  EXTERNAL DATA INVENTORY")
    n = 0
    for root, _, files in os.walk("external"):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"    {os.path.getsize(p)/1e6:9.1f} MB  {p}")
            n += 1
    print(f"  {n} external files. Every one must be a non-GRACE covariate whose source date is <= t.")
    print("  ERA5 / NCEP / CPC are reanalysis and observation products, not GRACE derivatives.")

    hdr("CODE PATTERNS THAT HAVE CAUSED PROBLEMS BEFORE")
    for pat, why in (("traj_smooth", "trajectory smoothing uses the next horizon's prediction"),
                     ("TWS_t_masked", "reads of the masked flag -- must only ever EXCLUDE rows")):
        try:
            r = subprocess.run(["grep", "-rn", pat, "--include=*.py", "."],
                               capture_output=True, text=True)
            hits = [l for l in r.stdout.splitlines() if not l.startswith("./smooth.py")]
        except Exception:                                              # noqa: BLE001
            hits = []
        print(f"  {pat}: {len(hits)} reference(s) outside its own module  ({why})")
        for l in hits[:6]:
            print(f"      {l[:110]}")

    hdr("RESULT")
    if FAIL:
        print(f"  {len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"    - {f}")
        sys.exit(1)
    print("  every check passed")


if __name__ == "__main__":
    main()
