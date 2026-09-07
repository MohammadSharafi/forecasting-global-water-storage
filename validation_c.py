"""Third validation layout: the REAL test's block structure, shifted back into train (session 9m).

Layouts A [3,2,3,7,3] and B [1,3,4,3,7,2] were invented before the test's own block structure was
worked out, and neither matches it. eval_mix.py repairs the horizon MARGINAL by reweighting, but
not the joint structure: in the real test a 7-month block's first month follows a 5-month gap
with no observation in it, and a 1-month block sits alone between two long ones. How stale
`tws_known` is when a block starts, and how many months of covariates have accumulated since the
last observation, are properties of the gaps, not of the horizon.

So layout C takes the test's exact pattern -- blocks [1, 3, 4, 7, 1, 2] with the test's exact gaps
between them -- and slides it back 40 months, so that its last month lands on the last month of
train:

    test    2015-09 | 2016-01..03 | 2016-06..09 | 2016-12..2017-06 | 2018-07 | 2018-11..12
    layout C 2012-05 | 2012-09..11 | 2013-02..05 | 2013-08..2014-02 | 2015-03 | 2015-07..08

Calendar months differ by four, so C is not a seasonal replica -- nothing here can be. What it
replicates is the geometry, which is what A and B get wrong. It is the third and most trustworthy
vote in select_config.py: a change has to survive it as well.

Note that C's history is shorter (it cuts at 2012-05, where A cuts at 2012-07 and B at 2009-10),
so its absolute RMSE is not comparable with A's -- only differences between methods are.

writes out/pseudo_test_C.parquet, out/pseudo_hist_C.parquet
"""
import datetime as dt
import polars as pl

BLOCKS = [["2012-05"], ["2012-09", "2012-10", "2012-11"],
          ["2013-02", "2013-03", "2013-04", "2013-05"],
          ["2013-08", "2013-09", "2013-10", "2013-11", "2013-12", "2014-01", "2014-02"],
          ["2015-03"], ["2015-07", "2015-08"]]

tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date(),
                                           pl.col("time").str.to_date().dt.month().alias("m"))
blocks = [[dt.date(int(x[:4]), int(x[5:7]), 1) for x in b] for b in BLOCKS]
test_months = [m for b in blocks for m in b]
have = set(tr["time"].unique().to_list())
missing = [str(m) for m in test_months if m not in have]
if missing:
    # GRACE has real gaps; a block month that train does not carry would silently shrink a block
    # and change its horizon mix, which is the one thing this layout exists to get right.
    print(f"WARNING: {len(missing)} block months are absent from Train.csv: {missing}")
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
     + (te["time"].dt.month() - te["t_known"].dt.month()) + 1)
print(f"layout C: hist months={hist['time'].n_unique()} ({hist['time'].min()}..{hist['time'].max()})"
      f"  test rows={len(te)} masked={te['masked'].mean():.3f}")
print("  horizon shares: " + "  ".join(
    f"h{i} {float((h == i).mean()):.3f}" for i in range(1, 8)))
