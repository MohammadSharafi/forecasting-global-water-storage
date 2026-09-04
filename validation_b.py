"""Second validation layout (earlier years, different block placement) -> out/pseudo_test_B.parquet"""
import polars as pl, datetime as dt, sys
BLOCKS=[["2009-10"],["2010-02","2010-03","2010-04"],["2010-08","2010-09","2010-10","2010-11"],["2011-03","2011-04","2011-05"],
        ["2011-08","2011-09","2011-10","2011-11","2011-12","2012-01","2012-02"],["2012-07","2012-08"]]
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date(), pl.col("time").str.to_date().dt.month().alias("m"))
blocks=[[dt.date(int(m[:4]),int(m[5:7]),1) for m in b] for b in BLOCKS]
test_months=[m for b in blocks for m in b]; first={b[0] for b in blocks}; cut=min(test_months)
hist=tr.filter(pl.col("time")<cut)
te=tr.filter(pl.col("time").is_in(test_months)).with_columns(pl.col("time").is_in(list(first)).not_().alias("masked"))
obs=pl.concat([hist.select(["lat","lon","time","TWS_t"]), te.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
known=(te.select(["lat","lon","time"]).join(obs.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
te=te.join(known,on=["lat","lon","time"],how="left")
te.write_parquet("out/pseudo_test_B.parquet"); hist.write_parquet("out/pseudo_hist_B.parquet")
print(f"layout B: hist months={hist['time'].n_unique()} ({hist['time'].min()}..{hist['time'].max()})  test rows={len(te)} masked={te['masked'].mean():.3f}")
