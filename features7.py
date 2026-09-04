"""v7: ENSO state (NOAA ONI, monthly, published within days -> operationally available).
Features at t, at t_known, their change, and the 3-month mean ending at t. Trees can
learn regional teleconnections via lat/lon x ONI."""
import polars as pl
ONI=pl.read_parquet("external/oni.parquet")
def add_oni(r):
    o=ONI.sort("time").with_columns(pl.col("oni").rolling_mean(3).alias("oni3"))
    r=r.join(o.rename({"oni":"oni_t","oni3":"oni3_t"}),on="time",how="left")
    r=r.join(o.select(["time","oni"]).rename({"time":"t_known","oni":"oni_k"}),on="t_known",how="left")
    return r.with_columns((pl.col("oni_t")-pl.col("oni_k")).alias("oni_d"))
ONIF=["oni_t","oni3_t","oni_k","oni_d"]
