"""Second-generation anchor features: smoothed anchor at earlier observed months (smoothed trend),
smoothed anomaly relative to the recent mean field, and multi-scale deviations.
All read TWS only at months <= t_known, which the 24 Aug ruling permits."""
import polars as pl, numpy as np, sys, time
from anchor import anchor_fields, sample
L=sys.argv[1]; t0=time.time(); RAD=(300,600)
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
if L=="FINAL":
    te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    obs_hist=tr.select(["lat","lon","time","TWS_t"]); obs_eval=pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
else:
    sfx={"A":"","B":"_B"}[L]; tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_eval=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
for part,obs in (("tr",obs_hist),("va",obs_eval)):
    m=pl.read_parquet(f"out/mats/{L}_{part}.parquet",columns=["lat","lon","t_known","tws_known","horizon"])
    cols={}
    # smoothed anchor at t_known - L months  -> smoothed trend
    allm=sorted(set(m["t_known"].to_list()))
    shifts={}
    for lag in (1,2,3,12):
        mk=m.with_columns(pl.col("t_known").dt.offset_by(f"-{lag}mo").alias("tk2")).rename({"t_known":"tk0"}).rename({"tk2":"t_known"})
        F,ti=anchor_fields(sorted(set(mk["t_known"].to_list())),obs,RAD); S=sample(F,ti,mk)
        for km,v in S.items(): shifts[(lag,km)]=v.astype(np.float32)
    F0,ti0=anchor_fields(allm,obs,RAD); S0=sample(F0,ti0,m); k=m["tws_known"].to_numpy(); h=m["horizon"].to_numpy()
    for km in RAD:
        base=np.where(np.isnan(S0[km]),k,S0[km]).astype(np.float32); cols[f"sb{km}"]=base
        for lag in (1,2,3,12):
            prev=shifts[(lag,km)]; d=np.where(np.isnan(prev),np.nan,base-prev).astype(np.float32); cols[f"sb{km}_d{lag}"]=d
        cols[f"sb{km}_trend"]=np.where(np.isnan(shifts[(3,km)]),np.nan,(base-shifts[(3,km)])/3.0).astype(np.float32)
        cols[f"sb{km}_persist"]=(base+np.nan_to_num(cols[f"sb{km}_trend"])*h).astype(np.float32)
    cols["sb_scale"]=(cols["sb300"]-cols["sb600"]).astype(np.float32)
    pl.DataFrame(cols).write_parquet(f"out/mats/{L}_{part}_anchor2.parquet")
    print(f"{L} {part}: {len(m)} rows, {len(cols)} cols ({time.time()-t0:.0f}s)",flush=True)
