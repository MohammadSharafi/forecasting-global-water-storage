"""Third validation layout: the REAL test's block geometry, placed where Train can carry it.

Layouts A [3,2,3,7,3] and B [1,3,4,3,7,2] were invented before the test's own block structure was
worked out, and neither matches it. eval_mix.py repairs the horizon MARGINAL by reweighting, but
not the joint structure: in the real test a 7-month block's first month follows a 5-month gap with
no observation in it, and a 1-month block sits alone between two long ones. How stale `tws_known`
is when a block starts, and how many months of covariates have accumulated since the last
observation, are properties of the gaps, not of the horizon.

So layout C reproduces the test's exact pattern -- block lengths [1, 3, 4, 7, 1, 2] at the test's
exact spacing -- and slides it back into train.

WHY THE PLACEMENT IS SEARCHED RATHER THAN HARDCODED (session 9q)
----------------------------------------------------------------
The first version hardcoded a 40-month shift. On the real Train.csv that put block months inside
GRACE's data gaps, so blocks were silently truncated: layout C came out with 140k rows instead of
280k, horizons 3, 5, 6 and 7 EMPTY, and a horizon mix of .445/.110/0/.111/0/0/0 against the test's
.333/.222/.167/.111/.056/.056/.056. It was then used as a third vote in select_config, where a
layout that cannot score four of the seven horizons is not a vote but a corruption.

The fix is to stop guessing. This searches every possible placement of the pattern and keeps the
LATEST one whose months Train actually has, so the layout is as close to the test era as the data
allows and is structurally exact by construction rather than by luck. If no placement is perfect
it takes the best available, says so loudly, and reports exactly which months are missing --
because a layout that silently disagrees with its own specification is worse than no layout.

writes out/pseudo_test_C.parquet, out/pseudo_hist_C.parquet
"""
import datetime as dt
import sys
import numpy as np
import polars as pl

# The real test, as offsets in months from the first block month, with each block's length.
#   2015-09 | 2016-01..03 | 2016-06..09 | 2016-12..2017-06 | 2018-07 | 2018-11..12
STARTS = (0, 4, 9, 15, 34, 38)
LENGTHS = (1, 3, 4, 7, 1, 2)
SPAN = STARTS[-1] + LENGTHS[-1]          # 40 months from first block month to last


def add_months(d, n):
    y, m = d.year, d.month - 1 + n
    return dt.date(y + m // 12, m % 12 + 1, 1)


def placement(anchor):
    """The months a placement anchored at this first-block month needs, block by block."""
    return [[add_months(anchor, s + i) for i in range(L)] for s, L in zip(STARTS, LENGTHS)]


def main():
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date(),
                                               pl.col("time").str.to_date().dt.month().alias("m"))
    months = sorted(tr["time"].unique().to_list())
    have = set(months)

    # search every anchor that keeps the whole pattern inside train, newest first
    best, best_missing = None, None
    for anchor in reversed(months):
        if add_months(anchor, SPAN - 1) > months[-1]:
            continue
        need = [m for b in placement(anchor) for m in b]
        missing = [m for m in need if m not in have]
        if not missing:
            best, best_missing = anchor, []
            break
        if best_missing is None or len(missing) < len(best_missing):
            best, best_missing = anchor, missing
    if best is None:
        print("layout C: train is shorter than the test's 40-month span; cannot build it")
        sys.exit(1)

    blocks = placement(best)
    if best_missing:
        # Never silently ship a truncated layout: a vote that cannot score four of seven
        # horizons is not a weaker vote, it is a wrong one.
        print(f"layout C: WARNING -- no gap-free placement exists. Best anchor {best} still "
              f"misses {len(best_missing)} months: {[str(m) for m in best_missing]}")
        blocks = [[m for m in b if m in have] for b in blocks]
        blocks = [b for b in blocks if b]
    print(f"layout C: anchored at {best}, {'exact' if not best_missing else 'TRUNCATED'}")
    print("  blocks: " + " | ".join(f"{b[0]} x{len(b)}" for b in blocks))
    print(f"  block lengths {[len(b) for b in blocks]}   the test's are {list(LENGTHS)}")

    test_months = [m for b in blocks for m in b]
    first = {b[0] for b in blocks}
    cut = min(test_months)
    hist = tr.filter(pl.col("time") < cut)
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
    te.write_parquet("out/pseudo_test_C.parquet")
    hist.write_parquet("out/pseudo_hist_C.parquet")

    h = ((te["time"].dt.year() - te["t_known"].dt.year()) * 12
         + (te["time"].dt.month() - te["t_known"].dt.month()) + 1).to_numpy()
    print(f"  hist months={hist['time'].n_unique()} ({hist['time'].min()}..{hist['time'].max()})"
          f"  test rows={len(te)}  masked={te['masked'].mean():.3f}")
    want = np.array([6, 4, 3, 2, 1, 1, 1], dtype=float); want /= want.sum()
    print("  horizon   " + "  ".join(f"h{i}" for i in range(1, 8)))
    print("  layout C  " + "  ".join(f"{float((h == i).mean()):.3f}" for i in range(1, 8)))
    print("  real test " + "  ".join(f"{w:.3f}" for w in want))
    bad = [i for i in range(1, 8) if not (h == i).any()]
    if bad:
        print(f"  *** horizons {bad} are EMPTY -- do NOT use this layout as a vote ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
