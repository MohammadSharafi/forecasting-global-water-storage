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

    ALL = json.load(open("out/mats/feats.json"))
    F = [f for f in ALL if f not in ("lat", "lon")]

    tr = pl.read_parquet(f"out/mats/{L}_tr.parquet")
    tr1 = tr.filter(pl.col("horizon") == 1)
    Xt = tr1.select(F).to_numpy(); yt = (tr1["target"] - tr1["tws_known"]).to_numpy()
    print(f"layout {L}: horizon-1 model on {len(tr1)} rows, {len(F)} features", flush=True)
    P = dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=500,
             feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             verbose=-1, num_threads=8, max_bin=63, seed=0)
    m1 = lgb.train(P, lgb.Dataset(Xt, yt), num_boost_round=int(os.environ.get("RECUR_ROUNDS", "300")))

    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "t_known", "horizon", "tws_known", "target"])
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy()
    kdirect = va["tws_known"].to_numpy()

    F64 = [pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64)]
    obs = obs_all.select(["lat", "lon", "time", "TWS_t"]).with_columns(
        *F64, pl.col("TWS_t").cast(pl.Float64))

    need = va.select(["lat", "lon", "time", "t_known"]).unique().with_columns(*F64)
    tmax = add_month(max(va["time"].to_list()), 1)
    mstart = min(add_month(t, 1) for t in need["t_known"].to_list())

    mm = mstart
    step = 0
    while mm <= tmax:
        prev = add_month(mm, -1)

        have_prev = obs.filter(pl.col("time") == prev).select(["lat", "lon", "TWS_t"])

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

    tgt = va.select(["lat", "lon", "time"]).with_columns(
        pl.col("time").map_elements(lambda t: add_month(t, 1), return_dtype=pl.Date).alias("tm"))
    ans = tgt.join(obs.rename({"time": "tm", "TWS_t": "prec"}), on=["lat", "lon", "tm"], how="left")
    prec = ans["prec"].to_numpy()

    broke = ~np.isfinite(prec)
    if broke.any():
        print(f"  WARNING  {int(broke.sum())} of {len(prec)} rows ({100*broke.mean():.2f}%) never "
              f"reached their target month; those fall back to persistence", flush=True)
    else:
        print("  every row's chain reached its target month (no persistence fallback)", flush=True)
    prec = np.where(np.isfinite(prec), prec, kdirect)

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
