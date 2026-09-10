"""Additional validation layouts, placed to be INDEPENDENT of A, B and C.

Why this exists (session 10s)
-----------------------------
The standing rule adopts a change when it wins on all three layouts. That sounds like three
independent tests. It is not: layout B's window is 2009-10..2012-08 and layout C's is
2009-01..2012-04, they share six months, and both sample the same era. The gate is closer to two
independent tests than three, and a null change passes two coin flips 25% of the time.

Against that gate this project has tested on the order of twenty ideas, so roughly five false
adoptions are expected -- and the transfer record matches exactly. The covariate-anomaly encoding
at -0.0156 was far outside what selection can manufacture and transferred at 83% of its validation
size; everything since has been in the 0.002-0.009 range, which is what a weak gate produces by
chance, and none of it transferred.

More independent windows is the fix. Each extra one that a change must also win halves the
false-adoption rate.

Placement
---------
Same block geometry as layout C -- the test's own [1,3,4,7,1,2] at the test's own spacing -- so the
joint structure of gaps and staleness is faithful. The anchor is passed in rather than searched,
because the point here is to choose a window in an era the existing layouts do not cover, and the
binding constraint is history: a window must end before 2009-01 to miss A, B and C entirely, which
leaves at most 35 months of history before it. That is short, and a layout built on it will have a
noisier per-cell climatology and a weaker model. That is acceptable for a GATE: both arms of a
comparison carry the same handicap, so the contrast is still fair, even though the absolute score
is not comparable across layouts.

usage: python validation_extra.py <NAME> <ANCHOR yyyy-mm-dd>
       python validation_extra.py D 2005-09-01
"""
import datetime as dt
import sys
import numpy as np
import polars as pl

STARTS = (0, 4, 9, 15, 34, 38)
LENGTHS = (1, 3, 4, 7, 1, 2)
SPAN = STARTS[-1] + LENGTHS[-1]


def add_months(d, n):
    y, m = d.year, d.month - 1 + n
    return dt.date(y + m // 12, m % 12 + 1, 1)


def main():
    name = sys.argv[1]
    anchor = dt.date.fromisoformat(sys.argv[2])
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    months = sorted(tr["time"].unique().to_list())
    have = set(months)

    blocks = [[add_months(anchor, s + i) for i in range(L)] for s, L in zip(STARTS, LENGTHS)]
    missing = [m for b in blocks for m in b if m not in have]
    if missing:
        print(f"layout {name}: anchor {anchor} misses {len(missing)} months "
              f"{[str(m) for m in missing]} -- refusing to build a truncated layout")
        sys.exit(1)

    test_months = [m for b in blocks for m in b]
    first = {b[0] for b in blocks}
    cut = min(test_months)
    hist = tr.filter(pl.col("time") < cut)
    if hist["time"].n_unique() < 24:
        print(f"layout {name}: only {hist['time'].n_unique()} months of history before {cut}; "
              f"the 12-month lags and the per-cell climatology would be degenerate")
        sys.exit(1)

    te = (tr.filter(pl.col("time").is_in(test_months))
            .with_columns(pl.col("time").is_in(list(first)).not_().alias("masked")))
    obs = pl.concat([hist.select(["lat", "lon", "time", "TWS_t"]),
                     te.filter(~pl.col("masked")).select(["lat", "lon", "time", "TWS_t"])])
    known = (te.select(["lat", "lon", "time"])
               .join(obs.rename({"time": "t_obs"}).select(["lat", "lon", "t_obs"]),
                     on=["lat", "lon"], how="inner")
               .filter(pl.col("t_obs") <= pl.col("time"))
               .group_by(["lat", "lon", "time"]).agg(pl.col("t_obs").max().alias("t_known")))
    te = te.join(known, on=["lat", "lon", "time"], how="left")
    te.write_parquet(f"out/pseudo_test_{name}.parquet")
    hist.write_parquet(f"out/pseudo_hist_{name}.parquet")

    h = ((te["time"].dt.year() - te["t_known"].dt.year()) * 12
         + (te["time"].dt.month() - te["t_known"].dt.month()) + 1).to_numpy()
    print(f"layout {name}: anchored at {anchor}, exact")
    print("  blocks: " + " | ".join(f"{b[0]} x{len(b)}" for b in blocks))
    print(f"  hist months={hist['time'].n_unique()} ({hist['time'].min()}..{hist['time'].max()})"
          f"  test rows={len(te)}  masked={te['masked'].mean():.3f}")
    want = np.array([6, 4, 3, 2, 1, 1, 1], dtype=float); want /= want.sum()
    print("  horizon   " + "  ".join(f"h{i}" for i in range(1, 8)))
    print(f"  layout {name}  " + "  ".join(f"{float((h == i).mean()):.3f}" for i in range(1, 8)))
    print("  real test " + "  ".join(f"{w:.3f}" for w in want))
    bad = [i for i in range(1, 8) if not (h == i).any()]
    if bad:
        print(f"  *** horizons {bad} are EMPTY -- do NOT use this layout as a vote ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
