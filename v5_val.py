import polars as pl, numpy as np, lightgbm as lgb, time, gc
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=add_wide(add_ar(assemble2(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,loyo=True),obs_hist))
F=FEATS2+AR+WIDE
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr; gc.collect()
Xva=add_wide(add_ar(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),obs_val))
Xv=Xva.select(F).to_numpy().astype(np.float32); yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy(); del tr,cov_all; gc.collect()
print(f"built {time.time()-t0:.0f}s rows={len(X)} feats={len(F)}",flush=True)
ds=lgb.Dataset(X,y,feature_name=F,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
m=lgb.train(params,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
p=m.predict(Xv,num_iteration=m.best_iteration)+kv; r=lambda a: float(np.sqrt(np.mean((yv-a)**2)))
print(f"[v5 wide-nb] RMSE={r(p):.4f} best_iter={m.best_iteration}   (v4 0.6614)")
print("  top:",[f for g,f in sorted(zip(m.feature_importance("gain"),F),reverse=True)[:14]]); np.save("out/val5_pred.npy",p); m.save_model("out/lgb_val5.txt")
