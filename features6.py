import polars as pl
from features import mdiff
def add_recent(r, obs_all, months=60):
    o=obs_all.rename({"time":"t_o"}).with_columns(pl.col("t_o").dt.month().alias("m_o"))
    j=r.select(["lat","lon","t_known","m_next"]).unique().join(o,on=["lat","lon"],how="inner")
    j=j.with_columns(mdiff("t_o","t_known").alias("x")).filter((pl.col("x")<=0)&(pl.col("x")>-months))
    g=j.group_by(["lat","lon","t_known","m_next"]).agg(
        pl.col("TWS_t").mean().alias("rmean60"), pl.col("TWS_t").std().alias("rsd60"), pl.len().alias("rn60"),
        pl.cov("x","TWS_t").alias("c"), pl.col("x").var().alias("v"),
        pl.col("TWS_t").filter(pl.col("m_o")==pl.col("m_next")).mean().alias("rclim_next"),
        pl.col("TWS_t").filter(pl.col("m_o")==pl.col("m_next")).count().alias("rclim_n"))
    g=g.with_columns((pl.col("c")/pl.col("v")).alias("trend60")).drop("c","v")
    r=r.join(g,on=["lat","lon","t_known","m_next"],how="left")
    return r.with_columns((pl.col("tws_known")-pl.col("rmean60")).alias("ranom_known"),
                          (pl.col("rclim_next")-pl.col("rmean60")).alias("rseas_next"),
                          (pl.col("tws_known")+pl.col("trend60")*pl.col("horizon")).alias("trend60_persist"),
                          (pl.col("rclim_next")+pl.col("tws_known")-pl.col("rmean60")).alias("ranom_persist"))
RECENT=["rmean60","rsd60","rn60","rclim_next","rclim_n","trend60","ranom_known","rseas_next","trend60_persist","ranom_persist"]
LONGTERM=["cmean","csd","clim_next","clim_known","clim_sd_next","anom_persist","anom_known","tws_ly","anom_ly"]
