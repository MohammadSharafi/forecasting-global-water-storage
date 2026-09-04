"""Phase 1: confirm v5 on layout B. Phase 2: build v5 test submission (3 seeds) + smoothed.
Sequential phases, memory freed between them. Run alone."""
import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from smooth import smooth
F=FEATS2+AR+WIDE
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
def feats(rows, cov_all, obs, sums, cell, loyo): return add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs))
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
# ---- phase 1: layout B ----
t0=time.time()
tp=pl.read_parquet("out/pseudo_test_B.parquet"); hist=pl.read_parquet("out/pseudo_hist_B.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=feats(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,True)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr; gc.collect()
Xva=feats(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,False)
Xv=Xva.select(F).to_numpy().astype(np.float32); yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy()
ds=lgb.Dataset(X,y,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
m=lgb.train(P,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
p=m.predict(Xv,num_iteration=m.best_iteration)+kv; r=lambda a: float(np.sqrt(np.mean((yv-a)**2)))
print(f"[layout B] v5 RMSE={r(p):.4f} best_iter={m.best_iteration}  (v4 on B: 0.5931; persistence 0.6869)  ({time.time()-t0:.0f}s)",flush=True)
del ds,m,Xv,Xva,tp,hist,obs_hist,obs_val; gc.collect()
# ---- phase 2: final ----
t0=time.time()
te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(tr); _,cell=cell_stats(tr)
known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left"); assert rows_te["t_known"].null_count()==0
Xte=feats(rows_te,cov_all,obs_all,sums,cell,False); Xt=Xte.select(F).to_numpy().astype(np.float32); kt=Xte["tws_known"].to_numpy()
Xtr=feats(training_rows(tr,np.random.default_rng(2),per_row=3),cov_all,obs_train,sums,cell,True)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr,tr,te,cov_all; gc.collect()
ds=lgb.Dataset(X,y,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
preds=[]
for s in (0,1,2):
    m=lgb.train({**P,"seed":s},ds,num_boost_round=260); m.save_model(f"out/lgb_final5_s{s}.txt"); preds.append(m.predict(Xt)+kt); print(f"  seed {s} ({time.time()-t0:.0f}s)",flush=True)
p=np.mean(preds,axis=0); assert np.isfinite(p).all()
pl.DataFrame({"ID":Xte["ID"],"Target":np.round(p,6)}).write_csv("out/sub_v5_6dp.csv",float_precision=6)
ps=smooth(Xte.select(["lat","lon","time","tws_known"]),p,w=0.7,radius=1,iters=1)
pl.DataFrame({"ID":Xte["ID"],"Target":np.round(ps,6)}).write_csv("out/sub_v5_smooth_6dp.csv",float_precision=6)
print(f"wrote out/sub_v5_6dp.csv + smooth  rows={len(p)} mean={p.mean():.4f} std={p.std():.4f}")
