"""Per-cell autoregressive baseline: pred = a(h, cell) * tws_known.

Why this exists
---------------
Not to win. On validation it scores 0.6942 (layout A) and 0.6220 (B) under the test's horizon
mix, against the gradient-boosted model's 0.6423 and 0.5403 -- clearly worse. It exists because
`lb_blend.py` showed that what a blend ledger is starved of is not more capacity but structural
difference: four GBMs plus persistence and climatology beat sixty-two GBMs alone. This is the
cheapest genuinely different predictor available, and it needs no training.

The slope of TWS(t+h) on TWS(t) is the whole model. Fitted per cell and shrunk to the pooled
slope by a ridge term K, because a single cell has at most ~130 usable pairs:

    a(h,i) = (sum_t x*y + K*a_pooled) / (sum_t x*x + K)

K=100 was chosen on layouts A and B (A prefers 20 at 0.6942 vs 0.6954, B prefers 100).

Compliance: coordinates group a cell's own history and are not model features, which is the same
use `anchor3.py` and `cell_response` already make of them. Only months <= t enter the fit -- for
the test the fit uses Train.csv alone, which ends 2015-08, before every test month.

usage: python ar_model.py           # writes out/probe_ar.csv and out/probe_clim.csv
"""
import numpy as np
import polars as pl

K_DEFAULT = 100


def fit_ar(hist, K=K_DEFAULT):
    """hist: lat, lon, time, TWS_t (observed only). Returns sorted cell keys, pooled and per-cell slopes."""
    lat = hist["lat"].to_numpy(); lon = hist["lon"].to_numpy()
    mi = hist["time"].dt.year().to_numpy() * 12 + hist["time"].dt.month().to_numpy()
    v = hist["TWS_t"].to_numpy().astype(float)
    ckey = np.round((lat + 90) * 1000 + (lon + 180)).astype(np.int64)
    cells, ci = np.unique(ckey, return_inverse=True)
    months, mj = np.unique(mi, return_inverse=True)
    A = np.full((len(months), len(cells)), np.nan)
    A[mj, ci] = v
    pooled = {}; per = {}
    for h in range(1, 8):
        x = A[:-h]; y = A[h:]
        # calendar-exact pairs only: GRACE is missing months, so an index step is not a month step
        m = np.isfinite(x) & np.isfinite(y) & ((months[h:] - months[:-h]) == h)[:, None]
        xs = np.where(m, x, 0.0); ys = np.where(m, y, 0.0)
        sxy = (xs * ys).sum(0); sxx = (xs * xs).sum(0)
        g = float(sxy.sum() / max(sxx.sum(), 1e-9))
        pooled[h] = g; per[h] = (sxy + K * g) / (sxx + K)
    return cells, pooled, per


def ar_predict(cells, pooled, per, lat, lon, horizon, tws_known):
    ckey = np.round((lat + 90) * 1000 + (lon + 180)).astype(np.int64)
    pos = np.clip(np.searchsorted(cells, ckey), 0, len(cells) - 1)
    hit = cells[pos] == ckey
    p = np.zeros(len(lat))
    for h in range(1, 8):
        s = horizon == h
        if s.any():
            p[s] = np.where(hit[s], per[h][pos[s]], pooled[h]) * tws_known[s]
    return p, float(hit.mean())


def main():
    va = pl.read_parquet("out/mats/FINALvn2_va.parquet",
                         columns=["ID", "lat", "lon", "horizon", "tws_known", "clim_next"])
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()).select(["lat", "lon", "time", "TWS_t"])
    h = va["horizon"].to_numpy().astype(int); k = va["tws_known"].to_numpy().astype(float)
    cells, pooled, per = fit_ar(tr)
    ar, hit = ar_predict(cells, pooled, per, va["lat"].to_numpy(), va["lon"].to_numpy(), h, k)
    cl = va["clim_next"].to_numpy().astype(float)
    cl = np.where(np.isfinite(cl), cl, 0.0)   # 635 cells have no history climatology; anomaly 0 is the prior
    print(f"cells matched {hit*100:.1f}%  slopes " + " ".join(f"h{i}={pooled[i]:.3f}" for i in range(1, 8)))
    for nm, p in (("out/probe_ar.csv", ar), ("out/probe_clim.csv", cl)):
        assert np.isfinite(p).all()
        pl.DataFrame({"ID": va["ID"], "Target": p}).write_csv(nm)
        print(f"  wrote {nm}  std={p.std():.4f}")


if __name__ == "__main__":
    main()
