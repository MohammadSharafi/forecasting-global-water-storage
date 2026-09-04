"""v3 additions: per-cell linear proxies of TWS from covariates (from history only),
applied at t and at t_known; and the implied covariate-driven change since t_known."""
import polars as pl, numpy as np
from features import COV, mdiff

def cell_proxies(hist):
    """Per cell OLS: TWS_t ~ a + b*x for x in SM, SPEI_03, SPEI_06, SPEI_12 (history only)."""
    out=hist.select(["lat","lon"]).unique()
    for x,nm in [("SOIL_MOISTURE_t","sm"),("SPEI_03_t","s3"),("SPEI_06_t","s6"),("SPEI_12_t","s12")]:
        g=hist.group_by(["lat","lon"]).agg(pl.cov("TWS_t",x).alias("cov"), pl.col(x).var().alias("vx"), pl.col(x).mean().alias("mx"), pl.col("TWS_t").mean().alias("my"))
        g=g.with_columns((pl.col("cov")/pl.col("vx")).alias(f"b_{nm}")).with_columns((pl.col("my")-pl.col(f"b_{nm}")*pl.col("mx")).alias(f"a_{nm}"))
        out=out.join(g.select(["lat","lon",f"a_{nm}",f"b_{nm}"]),on=["lat","lon"],how="left")
    return out

PROX=[f"{p}_{n}" for n in ("sm","s3","s6","s12") for p in ("hat","dhat","b")]+["pers_sm","pers_s6"]

def add_proxies(r, prox):
    r=r.join(prox,on=["lat","lon"],how="left")
    for x,nm in [("SOIL_MOISTURE_t","sm"),("SPEI_03_t","s3"),("SPEI_06_t","s6"),("SPEI_12_t","s12")]:
        r=r.with_columns((pl.col(f"a_{nm}")+pl.col(f"b_{nm}")*pl.col(x)).alias(f"hat_{nm}"),          # proxy TWS at t
                         (pl.col(f"b_{nm}")*(pl.col(x)-pl.col(x+"_k"))).alias(f"dhat_{nm}"))          # proxy change since t_known
    # proxy-adjusted persistence
    r=r.with_columns((pl.col("tws_known")+pl.col("dhat_sm")).alias("pers_sm"),(pl.col("tws_known")+pl.col("dhat_s6")).alias("pers_s6"))
    # cells with zero covariate variance give +-inf slopes; LightGBM tolerates NaN, not inf
    r=r.with_columns([pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in PROX])
    return r
