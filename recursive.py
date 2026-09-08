"""Recursive multi-step forecasting -- measure whether it beats the direct model (session 9t).

Explicitly permitted by the organisers (2026-09 clarification): "recursive forecasting, feeding
your own prediction forward as an input to the next month. The restriction is on future
observations, not on model output." Nothing in this project has ever used it, and it targets
exactly where the error is worst -- analyze_A shows RMSE climbing from 0.62 at h=1 to 1.14 at h=7,
and 66.5% of test rows have a TWS lead time of two months or more.

The idea
--------
The direct model predicts TWS(t+1) in ONE jump from a `tws_known` that can be up to seven months
stale. Recursion instead chains one-month steps: predict the month after the last observation,
feed that prediction in as the known value, predict the next, and so on up to the target. A row at
horizon h becomes h chained horizon-1 predictions.

Whether that helps is not obvious -- one-step error COMPOUNDS across the chain -- so this measures
it before anything is built on it, the same discipline every other session-9 avenue got. It does
NOT touch the submission; it prints direct-vs-recursive RMSE per horizon under the test mix.

Method
------
It trains a horizon-1 model on the layout's own training matrix (the h=1 slice), then walks the
forecast calendar. For each month mm it takes every cell whose chain has reached mm-1, builds a
genuine horizon-1 row (time=mm-1, t_known=mm-1, tws_known = the value carried forward), featurises
it with build_mats.prepare's own feature builder against an observation table that now contains the
predictions so far, predicts, and writes the result back as the cell's mm value. The original row's
answer is the carried value at its target month.

Compliance is inherited from the direct pipeline: every intermediate row is a legitimate
horizon-1 row that reads observations only at months <= its own t, and the only forward-fed values
are model outputs, which the ruling permits. To keep the measurement cheap and self-contained it
trains on the non-anchor featset (v5x_noll), so the chain needs only the feature builder and not a
per-step anchor rebuild; if recursion helps, the anchors are the first thing to add to it.

usage: python recursive.py A            (or B, C)
"""
import os
import sys
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from build_mats import prepare
from eval_mix import test_mix

L = sys.argv[1] if len(sys.argv) > 1 else "A"
W = test_mix()


def mdiff(a, b):
    return (a.year - b.year) * 12 + (a.month - b.month)


def add_month(d, n=1):
    import datetime as dt
    y, m = d.year, d.month - 1 + n
    return dt.date(y + m // 12, m % 12 + 1, 1)


def testmix_rmse(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def main():
    S = prepare(L)
    featfn, obs_all, hist = S["featfn"], S["obs_all"], S["hist"]

    # feature list: the non-anchor set the h1 model will use, taken from a fresh featfn build so it
    # matches exactly what the chain will produce at each step.
    ALL = json.load(open("out/mats/feats.json"))
    F = [f for f in ALL if f not in ("lat", "lon")]

    # ---- train the horizon-1 model on the training matrix's h=1 rows
    tr = pl.read_parquet(f"out/mats/{L}_tr.parquet")
    tr1 = tr.filter(pl.col("horizon") == 1)
    Xt = tr1.select(F).to_numpy(); yt = (tr1["target"] - tr1["tws_known"]).to_numpy()
    print(f"layout {L}: horizon-1 model on {len(tr1)} rows, {len(F)} features", flush=True)
    P = dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=500,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             verbose=-1, num_threads=8, max_bin=63, seed=0)
    m1 = lgb.train(P, lgb.Dataset(Xt, yt), num_boost_round=int(os.environ.get("RECUR_ROUNDS", "300")))

    # ---- the rows we must answer, and their true target for scoring
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "t_known", "horizon", "tws_known", "target"])
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy()
    kdirect = va["tws_known"].to_numpy()

    # observation table we extend with our own predictions (uniform dtypes so appends concat)
    F64 = [pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64)]
    obs = obs_all.select(["lat", "lon", "time", "TWS_t"]).with_columns(
        *F64, pl.col("TWS_t").cast(pl.Float64))
    # every (cell, target-month) we must reach: months t_known+1 .. t+1
    need = va.select(["lat", "lon", "time", "t_known"]).unique().with_columns(*F64)
    tmax = add_month(max(va["time"].to_list()), 1)
    mstart = min(add_month(t, 1) for t in need["t_known"].to_list())

    mm = mstart
    step = 0
    while mm <= tmax:
        prev = add_month(mm, -1)
        # cells whose predecessor month is available (observed or already predicted) and that still
        # need month mm on the way to their target
        have_prev = obs.filter(pl.col("time") == prev).select(["lat", "lon", "TWS_t"])
        # a cell needs month mm iff t_known < mm <= t+1 (mm is on the way to its target)
        want = need.filter((pl.col("t_known") < mm) & (
            pl.col("time").map_elements(lambda t: add_month(t, 1) >= mm, return_dtype=pl.Boolean)))
        cells = want.select(["lat", "lon"]).unique().join(have_prev, on=["lat", "lon"], how="inner")
        if len(cells):
            rows = cells.select(["lat", "lon"]).with_columns(
                pl.lit(prev).alias("time"), pl.lit(prev).alias("t_known"))
            r, Fr = featfn(rows, obs, loyo=False)
            miss = [c for c in F if c not in r.columns]
            for c in miss:
                r = r.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            Xv = r.select(F).to_numpy()
            base = cells["TWS_t"].to_numpy()
            pred = m1.predict(Xv) + base
            add = cells.select(["lat", "lon"]).with_columns(
                pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64),
                pl.lit(mm).alias("time"), pl.Series("TWS_t", pred).cast(pl.Float64))
            obs = pl.concat([obs, add.select(["lat", "lon", "time", "TWS_t"])])
            step += 1
        mm = add_month(mm, 1)
    print(f"  chained {step} monthly steps", flush=True)

    # recursive answer for each row: the carried value at its target month t+1
    tgt = va.select(["lat", "lon", "time"]).with_columns(
        pl.col("time").map_elements(lambda t: add_month(t, 1), return_dtype=pl.Date).alias("tm"))
    ans = tgt.join(obs.rename({"time": "tm", "TWS_t": "prec"}), on=["lat", "lon", "tm"], how="left")
    prec = ans["prec"].to_numpy()
    # A broken chain falls back to persistence, which is WORSE than either model being compared, so
    # a silent fallback would make recursion look worse than it is. Report it: the comparison is
    # only trustworthy while this is a negligible fraction.
    broke = ~np.isfinite(prec)
    if broke.any():
        print(f"  WARNING  {int(broke.sum())} of {len(prec)} rows ({100*broke.mean():.2f}%) never "
              f"reached their target month; those fall back to persistence", flush=True)
    else:
        print("  every row's chain reached its target month (no persistence fallback)", flush=True)
    prec = np.where(np.isfinite(prec), prec, kdirect)

    # ---- the direct model, scored the same way, for the comparison
    # If this fell through silently, "direct" would be persistence and recursion would be flattered
    # by the comparison. Say which one is being scored.
    try:
        from eval_mix import load
        direct = load(L, ["lgb_v5x_noll:_bw"])
        print("  direct = lgb_v5x_noll:_bw", flush=True)
    except SystemExit:
        direct = kdirect
        print("  WARNING  direct predictions not found -- 'direct' below IS persistence, and the "
              "comparison does not answer the question", flush=True)

    print(f"\n  {'horizon':>8} {'n':>7} {'persist':>9} {'direct':>9} {'recursive':>10} {'r-d':>8}")
    for i in range(1, 8):
        k = h == i
        if k.sum() < 20:
            continue
        rp = lambda a: float(np.sqrt(np.mean((y[k] - a[k]) ** 2)))
        print(f"  {i:>8} {int(k.sum()):>7} {rp(kdirect):>9.4f} {rp(direct):>9.4f} "
              f"{rp(prec):>10.4f} {rp(prec)-rp(direct):>+8.4f}")
    print(f"\n  test-mix RMSE   direct {testmix_rmse(y, direct, h):.4f}   "
          f"recursive {testmix_rmse(y, prec, h):.4f}   "
          f"blend .5/.5 {testmix_rmse(y, 0.5*direct+0.5*prec, h):.4f}")
    print("\n  Recursion is worth building out (adding anchors, a per-step model) only if it beats")
    print("  the direct model at the long horizons, where the direct jump is most stale.")


if __name__ == "__main__":
    main()
