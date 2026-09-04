import polars as pl, numpy as np, lightgbm as lgb, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=add_ar(assemble2(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,loyo=True),obs_hist)
Xva=add_ar(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),obs_val)
F=FEATS2+AR; print(f"train={len(Xtr)} val={len(Xva)} feats={len(F)} built {time.time()-t0:.0f}s")
X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32)
y=Xtr["target"].to_numpy(); yv=Xva["target"].to_numpy(); kt=Xtr["tws_known"].to_numpy(); kv=Xva["tws_known"].to_numpy()
r=lambda a: float(np.sqrt(np.nanmean((yv-a)**2)))
params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8)
m=lgb.train(params,lgb.Dataset(X,y-kt,feature_name=F),num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv)],callbacks=[lgb.early_stopping(200,verbose=False)])
p=m.predict(Xv,num_iteration=m.best_iteration)+kv
print(f"\n[v4 AR+trend] VAL RMSE={r(p):.4f} best_iter={m.best_iteration}  (v2 0.6692, persistence 0.7569)")
print("  top:",[f for g,f in sorted(zip(m.feature_importance("gain"),F),reverse=True)[:14]])
h=Xva["horizon"].to_numpy(); print("  h:", "  ".join(f"{hh}={r(np.where(h==hh,p,np.nan)):.4f}" for hh in range(1,8)))
print("  trend_persist baseline:", f"{r(Xva['trend_persist'].to_numpy()):.4f}")
np.save("out/val4_pred.npy",p); m.save_model("out/lgb_val4.txt")
