import polars as pl

BAND = 5.0
LAGS = (3, 12)
SCALE = ["zm", "zn", "dz", "zd3", "zd12", "dzd3", "dzd12"]

def zonal_table(obs, band=BAND):
    return (obs.with_columns(((pl.col("lat") + 90) // band).cast(pl.Int32).alias("zb"))
               .group_by(["time", "zb"])
               .agg(pl.col("TWS_t").mean().alias("zm"), pl.len().alias("zn")))

def add_scale(r, obs, band=BAND, lags=LAGS):
    zt = zonal_table(obs, band)
    r = r.with_columns(((pl.col("lat") + 90) // band).cast(pl.Int32).alias("zb"))
    r = r.join(zt.rename({"time": "t_known"}), on=["t_known", "zb"], how="left")
    for L in lags:
        r = r.with_columns(pl.col("t_known").dt.offset_by(f"-{L}mo").alias("tz"))
        r = r.join(zt.select(["time", "zb", "zm"]).rename({"time": "tz", "zm": f"_zm{L}"}),
                   on=["tz", "zb"], how="left").drop("tz")
        r = r.with_columns((pl.col("zm") - pl.col(f"_zm{L}")).alias(f"zd{L}"))

        r = r.with_columns((pl.col(f"d{L}") - pl.col(f"zd{L}")).alias(f"dzd{L}"))
    r = r.with_columns((pl.col("tws_known") - pl.col("zm")).alias("dz"))
    return r.drop([f"_zm{L}" for L in lags] + ["zb"])
