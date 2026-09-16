import polars as pl, numpy as np, lightgbm as lgb, time, sys
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
sfx=sys.argv[1] if len(sys.argv)>1 else ""
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=add_ar(assemble2(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,loyo=True),obs_hist)
Xva=add_ar(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),obs_val)
F=FEATS2+AR
X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32)
y=Xtr["target"].to_numpy(); yv=Xva["target"].to_numpy(); kt=Xtr["tws_known"].to_numpy(); kv=Xva["tws_known"].to_numpy()
r=lambda a: float(np.sqrt(np.nanmean((yv-a)**2)))
params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=4)
m=lgb.train(params,lgb.Dataset(X,y-kt,feature_name=F),num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv)],callbacks=[lgb.early_stopping(200,verbose=False)])
p=m.predict(Xv,num_iteration=m.best_iteration)+kv
h=Xva["horizon"].to_numpy()
print(f"[layout{sfx or ' A'}] v4 VAL RMSE={r(p):.4f} best_iter={m.best_iteration}  persistence={r(kv):.4f}  gain={(1-r(p)/r(kv))*100:.1f}%")
print("  h:", "  ".join(f"{hh}={r(np.where(h==hh,p,np.nan)):.4f}" for hh in range(1,8)))
np.save(f"out/val4{sfx}_pred.npy",p)
