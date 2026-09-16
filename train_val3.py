import polars as pl, numpy as np, lightgbm as lgb, time, sys
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features3 import cell_proxies, add_proxies, PROX
EXCLUDE_GAP = "--keepgap" not in sys.argv
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV)
obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist); prox=cell_proxies(hist)
rows=training_rows(hist,np.random.default_rng(0),per_row=3)
if EXCLUDE_GAP:
    have=tr.select(["lat","lon","time"]).with_columns(pl.col("time").dt.offset_by("-1mo").alias("time"), pl.lit(True).alias("obs_next"))
    n0=len(rows); rows=rows.join(have,on=["lat","lon","time"],how="left").filter(pl.col("obs_next")).drop("obs_next"); print(f"dropped gap-target rows: {n0-len(rows)}")
Xtr=add_proxies(assemble2(rows,cov_all,obs_hist,sums,cell,loyo=True),prox)
Xva=add_proxies(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),prox)
F=FEATS2+PROX
print(f"train={len(Xtr)} val={len(Xva)} feats={len(F)} built {time.time()-t0:.0f}s"); t0=time.time()
X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32)
y=Xtr["target"].to_numpy().astype(np.float32); yv=Xva["target"].to_numpy().astype(np.float32)
kt=Xtr["tws_known"].to_numpy().astype(np.float32); kv=Xva["tws_known"].to_numpy().astype(np.float32)
r=lambda a: float(np.sqrt(np.nanmean((yv-a)**2)))
params=dict(objective="regression",learning_rate=0.03,num_leaves=255,min_data_in_leaf=300,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=3.0,verbose=-1,num_threads=8)
ds=lgb.Dataset(X,y-kt,feature_name=F); dv=lgb.Dataset(Xv,yv-kv,reference=ds)
m=lgb.train(params,ds,num_boost_round=4000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
p=m.predict(Xv,num_iteration=m.best_iteration)+kv
print(f"\n[v3 residual, exclude_gap={EXCLUDE_GAP}] VAL RMSE={r(p):.4f}  best_iter={m.best_iteration}  persistence={r(kv):.4f}  v2 was 0.6692  ({time.time()-t0:.0f}s)")
print("   top:",[f for g,f in sorted(zip(m.feature_importance("gain"),F),reverse=True)[:14]])
h=Xva["horizon"].to_numpy()
print("  h:", "  ".join(f"{hh}={r(np.where(h==hh,p,np.nan)):.4f}" for hh in range(1,8)))
print("  proxy-only baselines: pers_sm=%.4f  pers_s6=%.4f  hat_sm=%.4f"%(r(Xva['pers_sm'].to_numpy()),r(Xva['pers_s6'].to_numpy()),r(Xva['hat_sm'].to_numpy())))
m.save_model("out/lgb_val3.txt"); np.save("out/val3_pred.npy",p)
