import polars as pl, numpy as np
from features import mdiff
LAGS=[1,2,3,6,12]

def add_ar(r, obs_all):
    for L in LAGS:
        r=r.with_columns(pl.col("t_known").dt.offset_by(f"-{L}mo").alias("t_lag"))
        r=r.join(obs_all.rename({"time":"t_lag","TWS_t":f"lag{L}"}),on=["lat","lon","t_lag"],how="left").drop("t_lag")
        r=r.with_columns((pl.col("tws_known")-pl.col(f"lag{L}")).alias(f"d{L}"))

    o=obs_all.rename({"time":"t_o"})
    j=r.select(["lat","lon","t_known"]).unique().join(o,on=["lat","lon"],how="inner")
    j=j.with_columns((mdiff("t_o","t_known")).alias("x")).filter((pl.col("x")<=0)&(pl.col("x")>-24))
    tr_=j.group_by(["lat","lon","t_known"]).agg(pl.cov("x","TWS_t").alias("c"),pl.col("x").var().alias("v"),pl.len().alias("n24"),pl.col("TWS_t").mean().alias("mean24"),pl.col("TWS_t").std().alias("sd24"))
    tr_=tr_.with_columns((pl.col("c")/pl.col("v")).alias("trend24")).drop("c","v")
    r=r.join(tr_,on=["lat","lon","t_known"],how="left")
    r=r.with_columns((pl.col("tws_known")+pl.col("trend24")*pl.col("horizon")).alias("trend_persist"),
                     (pl.col("tws_known")-pl.col("mean24")).alias("dev24"))
    return r
AR=[f"lag{L}" for L in LAGS]+[f"d{L}" for L in LAGS]+["trend24","n24","mean24","sd24","trend_persist","dev24"]
