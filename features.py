"""Leak-free, horizon-aware features. Only information at or before month t is used.

A row = (cell, t, t_known) where t_known <= t is the month of the last OBSERVED TWS
for that cell. horizon = months(t_known -> t) + 1 = how far ahead of the last
observation the target (t+1) sits.
"""
import polars as pl, numpy as np
SPEI=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t"]; COV=SPEI+["SOIL_MOISTURE_t"]

def mdiff(a,b):  # months from b to a
    return (pl.col(a).dt.year()-pl.col(b).dt.year())*12+(pl.col(a).dt.month()-pl.col(b).dt.month())

def cell_stats(hist):
    """Per-cell tables from history only."""
    h=hist.with_columns(pl.col("time").dt.month().alias("m"))
    clim=h.group_by(["lat","lon","m"]).agg(pl.col("TWS_t").mean().alias("clim"), pl.col("TWS_t").std().alias("clim_sd"))
    # lag-1 dynamics per cell
    s=h.sort(["lat","lon","time"]).with_columns(pl.col("TWS_t").shift(1).over(["lat","lon"]).alias("prev"),
                                               pl.col("time").shift(1).over(["lat","lon"]).alias("tprev"))
    s=s.filter(mdiff("time","tprev")==1)
    cell=s.group_by(["lat","lon"]).agg(pl.corr("TWS_t","prev").alias("ac1"), (pl.col("TWS_t")-pl.col("prev")).abs().mean().alias("mad1"),
                                       pl.col("TWS_t").mean().alias("cmean"), pl.col("TWS_t").std().alias("csd"),
                                       pl.corr("TWS_t","SPEI_03_t").alias("r_spei3"), pl.corr("TWS_t","SOIL_MOISTURE_t").alias("r_sm"))
    return clim, cell

def assemble(rows, cov_all, obs_all, clim, cell):
    """rows: lat, lon, time(t), t_known [, target].  cov_all: covariates for every (cell, month)
    that exists (train+test).  obs_all: observed TWS (cell, month) usable at prediction time."""
    r=rows.with_columns((mdiff("time","t_known")+1).alias("horizon"), pl.col("time").dt.month().alias("m"),
                        ((pl.col("time").dt.month()%12)+1).alias("m_next"), pl.col("t_known").dt.month().alias("m_known"))
    r=r.join(obs_all.rename({"time":"t_known","TWS_t":"tws_known"}), on=["lat","lon","t_known"], how="left")
    r=r.join(cov_all, on=["lat","lon","time"], how="left")                         # covariates at t
    r=r.join(cov_all.rename({"time":"t_known",**{c:c+"_k" for c in COV}}), on=["lat","lon","t_known"], how="left")
    for c in COV: r=r.with_columns((pl.col(c)-pl.col(c+"_k")).alias(c+"_d"))
    # accumulated 1-month SPEI over (t_known, t]  and mean soil moisture over the same window
    win=cov_all.select(["lat","lon","time","SPEI_01_t","SOIL_MOISTURE_t"]).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).join(win,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg(pl.col("SPEI_01_t").sum().alias("spei1_acc"), pl.col("SOIL_MOISTURE_t").mean().alias("sm_win"), pl.len().alias("n_win"))
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")
    r=r.join(clim.rename({"m":"m_next","clim":"clim_next","clim_sd":"clim_sd_next"}),on=["lat","lon","m_next"],how="left")
    r=r.join(clim.rename({"m":"m_known","clim":"clim_known"}).select(["lat","lon","m_known","clim_known"]),on=["lat","lon","m_known"],how="left")
    r=r.join(cell,on=["lat","lon"],how="left")
    # TWS one year before the target month, if observed by then
    r=r.with_columns(pl.col("time").dt.offset_by("-11mo").alias("t_ly"))
    r=r.join(obs_all.rename({"time":"t_ly","TWS_t":"tws_ly"}),on=["lat","lon","t_ly"],how="left")
    r=r.with_columns((pl.col("tws_known")-pl.col("clim_known")).alias("anom_known"),
                     (pl.col("clim_next")+pl.col("tws_known")-pl.col("clim_known")).alias("anom_persist"),
                     (pl.col("tws_ly")-pl.col("clim_next")).alias("anom_ly"))
    return r

FEATS=["lat","lon","horizon","m","m_next","tws_known","anom_known","anom_persist","clim_next","clim_sd_next","clim_known",
       "ac1","mad1","cmean","csd","r_spei3","r_sm","tws_ly","anom_ly","spei1_acc","sm_win","n_win"]+COV+[c+"_k" for c in COV]+[c+"_d" for c in COV]

def training_rows(hist, rng, per_row=2, max_h=7):
    """From history: for each (cell,t) draw `per_row` known-months t_known in [t-6, t] that
    exist for the cell, giving horizons 1..7 with the test's mix (1/3 at h=1)."""
    cells=hist.select(["lat","lon","time","target"])
    have=hist.select(["lat","lon","time"]).with_columns(pl.lit(True).alias("ok"))
    out=[]
    for k in range(per_row):
        h=np.where(rng.random(len(cells))<0.34, 1, rng.integers(2,max_h+1,len(cells)))
        c=cells.with_columns(pl.Series("h",h)).with_columns(pl.col("time").dt.offset_by(pl.format("-{}mo",pl.col("h")-1)).alias("t_known"))
        c=c.join(have.rename({"time":"t_known"}),on=["lat","lon","t_known"],how="left").filter(pl.col("ok")).drop("ok","h")
        out.append(c)
    return pl.concat(out).unique(["lat","lon","time","t_known"])
