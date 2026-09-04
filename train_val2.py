import polars as pl, numpy as np, lightgbm as lgb, time, sys
from features import COV, training_rows, cell_stats
from features2 import *
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV)
obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
rows=training_rows(hist,np.random.default_rng(0),per_row=2)
Xtr=assemble2(rows,cov_all,obs_hist,sums,cell,loyo=True)
Xva=assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False)
print(f"train={len(Xtr)} val={len(Xva)} built {time.time()-t0:.0f}s"); t0=time.time()
X=Xtr.select(FEATS2).to_numpy().astype(np.float32); Xv=Xva.select(FEATS2).to_numpy().astype(np.float32)
y=Xtr["target"].to_numpy().astype(np.float32); yv=Xva["target"].to_numpy().astype(np.float32)
kt=Xtr["tws_known"].to_numpy().astype(np.float32); kv=Xva["tws_known"].to_numpy().astype(np.float32)
r=lambda a: float(np.sqrt(np.nanmean((yv-a)**2)))
params=dict(objective="regression",learning_rate=0.03,num_leaves=255,min_data_in_leaf=300,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=3.0,verbose=-1,num_threads=8)
res={}
for mode in ["level","residual"]:
    yt=y if mode=="level" else y-kt; yvt=yv if mode=="level" else yv-kv
    ds=lgb.Dataset(X,yt,feature_name=FEATS2); dv=lgb.Dataset(Xv,yvt,reference=ds)
    m=lgb.train(params,ds,num_boost_round=4000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
    p=m.predict(Xv,num_iteration=m.best_iteration); p=p if mode=="level" else p+kv
    res[mode]=p; print(f"\n[{mode}] VAL RMSE={r(p):.4f}  best_iter={m.best_iteration}  persistence={r(kv):.4f}  ({time.time()-t0:.0f}s)")
    m.save_model(f"out/lgb_val2_{mode}.txt")
    imp=sorted(zip(m.feature_importance("gain"),FEATS2),reverse=True)[:12]; print("   top:",[f for g,f in imp])
p=0.5*res["level"]+0.5*res["residual"]; print(f"\n[blend] VAL RMSE={r(p):.4f}")
h=Xva["horizon"].to_numpy(); tm=Xva["time"].to_numpy()
print("\nby horizon:  h  n     persist  level  resid  blend")
for hh in range(1,8):
    k=h==hh; f=lambda a: r(np.where(k,a,np.nan)); print(f"  {hh}  {k.sum():>6}  {f(kv):.4f} {f(res['level']):.4f} {f(res['residual']):.4f} {f(p):.4f}")
print("\nby month (blend vs persist):")
for t in np.unique(tm):
    k=tm==t; print(f"  {str(t)[:7]} h={int(h[k].min())}  n={k.sum():>6}  persist={r(np.where(k,kv,np.nan)):.4f}  blend={r(np.where(k,p,np.nan)):.4f}")
