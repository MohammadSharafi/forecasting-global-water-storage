"""Build the feature superset once per layout (A, B, FINAL) and write float32 parquet matrices.
usage: python build_mats.py A|B|FINAL"""
import polars as pl, numpy as np, sys, time, gc, os, json
from features import COV, training_rows, training_rows_coherent, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
from features_ncep import load_ncep, add_ncep
import glob
from features_era5 import load_era5, add_era5, ERA5F
from features_x import load_ncep2, load_cpc, add_ext, add_wide4, WIDE4, add_covwin, COVWIN, cell_response, add_response, RESP
from features_anom import build as anom_build, add_anom, ERA5_STORAGE, ERA5_FLUX, NCEP_STORAGE, NCEP_FLUX, COV_STORAGE
from features_scale import add_scale, SCALE
L=sys.argv[1]; os.makedirs("out/mats",exist_ok=True); t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
lats=tr["lat"].unique().to_list(); lons=tr["lon"].unique().to_list()
nc,cols=load_ncep(lats,lons); nc2,cols2=load_ncep2(lats,lons); cpc,cols3=load_cpc(lats,lons)
ERA=load_era5() if glob.glob("external/era5/*.nc") else None; print("era5:", None if ERA is None else ERA.shape, flush=True); print("ext loaded",cols,cols2,cols3,f"({time.time()-t0:.0f}s)",flush=True)
def feats(rows,cov_all,obs,sums,cell,resp,loyo):
    r=add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs)
    r=add_scale(r,obs)   # zonal context: the widest aggregation elsewhere is only radius 4
    r,NF=add_ncep(r,nc,cols); r,NF2=add_ext(r,nc2,cols2,acc_cols=["r2P","r2E","r2PER"]); r,NF3=add_ext(r,cpc,cols3)
    r=add_wide4(r); r=add_covwin(r,cov_all); r=add_response(r,resp)
    EF=[]
    if ERA is not None: r=add_era5(r,ERA); EF=ERA5F
    AF=[]   # per-cell standardised covariate anomalies (features_anom): the level features above
    for at,sz,fz in ANOM:   # are raw mm and unusable without lat/lon, which is not a feature
        r,f=add_anom(r,at,sz,fz); AF+=f
    F=FEATS2+AR+WIDE+RECENT+NF+NF2+NF3+WIDE4+COVWIN+RESP+EF+AF+SCALE
    F=list(dict.fromkeys(F)); return r,F
if L=="FINAL":
    te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
    obs_hist=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
    hist=tr
    known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
            .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
    rows_va=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left"); meta_va=["ID","lat","lon","time","t_known","horizon","tws_known"]
else:
    sfx={"A":"","B":"_B"}[L]; tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    cov_all=tr.select(["lat","lon","time"]+COV)
    obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
    rows_va=tp.select(["lat","lon","time","t_known","target"]); meta_va=["lat","lon","time","t_known","horizon","tws_known","target"]
sums=clim_sums(hist); _,cell=cell_stats(hist); resp=cell_response(hist)
# Climatologies for the covariate anomalies come from HISTORY MONTHS ONLY, so no month at or
# after a prediction target can enter them.
HM=hist["time"].unique().to_list()
ANOM=[x for x in (anom_build(ERA,ERA5_STORAGE,ERA5_FLUX,HM),
                  anom_build(nc,NCEP_STORAGE,NCEP_FLUX,HM),
                  anom_build(cov_all.select(["lat","lon","time"]+COV_STORAGE).unique(["lat","lon","time"]),COV_STORAGE,[],HM)) if x is not None]
print("anomaly tables:",[(len(sz),len(fz)) for _,sz,fz in ANOM],f"({time.time()-t0:.0f}s)",flush=True)
Xva,F=feats(rows_va,cov_all,obs_all,sums,cell,resp,False)
Xva.select(list(dict.fromkeys(meta_va+F))).with_columns([pl.col(f).cast(pl.Float32) for f in F]).write_parquet(f"out/mats/{L}_va.parquet"); print("va",Xva.shape,f"({time.time()-t0:.0f}s)",flush=True)
del Xva; gc.collect()
Xtr,_=feats(training_rows_coherent(hist,np.random.default_rng(11 if L=="FINAL" else 0),per_row=int(os.environ.get("PER_ROW","3"))),cov_all,obs_hist,sums,cell,resp,True)
Xtr.select(list(dict.fromkeys(["lat","lon","time","t_known","horizon","tws_known","target"]+F))).with_columns([pl.col(f).cast(pl.Float32) for f in F]).write_parquet(f"out/mats/{L}_tr.parquet")
json.dump(F,open("out/mats/feats.json","w")); print("tr",Xtr.shape,"nfeat",len(F),f"({time.time()-t0:.0f}s)",flush=True)
