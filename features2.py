import polars as pl, numpy as np
from features import mdiff, COV, SPEI, cell_stats

def clim_sums(hist):
    h=hist.with_columns(pl.col("time").dt.month().alias("m"))
    return h.group_by(["lat","lon","m"]).agg(pl.col("TWS_t").sum().alias("S"), pl.len().alias("N"))

def assemble2(rows, cov_all, obs_all, sums, cell, loyo):
    r=rows.with_columns((mdiff("time","t_known")+1).alias("horizon"), pl.col("time").dt.month().alias("m"),
                        ((pl.col("time").dt.month()%12)+1).alias("m_next"), pl.col("t_known").dt.month().alias("m_known"))
    r=r.join(obs_all.rename({"time":"t_known","TWS_t":"tws_known"}), on=["lat","lon","t_known"], how="left")

    prev=obs_all.sort("time").with_columns(pl.col("TWS_t").shift(1).over(["lat","lon"]).alias("tws_prev"), pl.col("time").shift(1).over(["lat","lon"]).alias("t_prev"))
    r=r.join(prev.select(["lat","lon","time","tws_prev","t_prev"]).rename({"time":"t_known"}),on=["lat","lon","t_known"],how="left")
    r=r.with_columns(((pl.col("tws_known")-pl.col("tws_prev"))/mdiff("t_known","t_prev")).alias("slope"))
    r=r.join(cov_all, on=["lat","lon","time"], how="left")
    r=r.join(cov_all.rename({"time":"t_known",**{c:c+"_k" for c in COV}}), on=["lat","lon","t_known"], how="left")
    for c in COV: r=r.with_columns((pl.col(c)-pl.col(c+"_k")).alias(c+"_d"))
    win=cov_all.select(["lat","lon","time","SPEI_01_t","SPEI_03_t","SOIL_MOISTURE_t"]).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).join(win,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg(pl.col("SPEI_01_t").sum().alias("spei1_acc"), pl.col("SPEI_03_t").mean().alias("spei3_win"), pl.col("SOIL_MOISTURE_t").mean().alias("sm_win"), pl.len().alias("n_win"))
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")

    r=r.join(sums.rename({"m":"m_next","S":"S_next","N":"N_next"}),on=["lat","lon","m_next"],how="left")
    r=r.join(sums.rename({"m":"m_known","S":"S_known","N":"N_known"}),on=["lat","lon","m_known"],how="left")
    if loyo:
        r=r.with_columns(((pl.col("S_next")-pl.col("target"))/(pl.col("N_next")-1)).alias("clim_next"),
                         ((pl.col("S_known")-pl.col("tws_known"))/(pl.col("N_known")-1)).alias("clim_known"))
    else:
        r=r.with_columns((pl.col("S_next")/pl.col("N_next")).alias("clim_next"),(pl.col("S_known")/pl.col("N_known")).alias("clim_known"))
    r=r.join(cell,on=["lat","lon"],how="left")
    r=r.with_columns(pl.col("time").dt.offset_by("-11mo").alias("t_ly"))
    r=r.join(obs_all.rename({"time":"t_ly","TWS_t":"tws_ly"}),on=["lat","lon","t_ly"],how="left")
    r=r.with_columns((pl.col("tws_known")-pl.col("clim_known")).alias("anom_known"))
    r=r.with_columns((pl.col("clim_next")+pl.col("anom_known")).alias("anom_persist"),(pl.col("tws_ly")-pl.col("clim_next")).alias("anom_ly"))

    base=r.select(["lat","lon","time","t_known","anom_known","SPEI_03_t_d","tws_known"])
    nb=[]
    for dl in (-1,0,1):
        for dn in (-1,0,1):
            if dl==0 and dn==0: continue
            nb.append(base.with_columns((pl.col("lat")+dl).alias("lat"),(pl.col("lon")+dn).alias("lon")))
    nb=pl.concat(nb).group_by(["lat","lon","time","t_known"]).agg(pl.col("anom_known").mean().alias("nb_anom"),pl.col("SPEI_03_t_d").mean().alias("nb_spei3d"),pl.col("tws_known").mean().alias("nb_tws"),pl.len().alias("nb_n"))
    r=r.join(nb,on=["lat","lon","time","t_known"],how="left")
    return r

FEATS2=["lat","lon","horizon","m","m_next","tws_known","anom_known","anom_persist","clim_next","clim_known","slope","tws_prev",
        "ac1","mad1","cmean","csd","r_spei3","r_sm","tws_ly","anom_ly","spei1_acc","spei3_win","sm_win","n_win",
        "nb_anom","nb_spei3d","nb_tws","nb_n"]+COV+[c+"_k" for c in COV]+[c+"_d" for c in COV]
