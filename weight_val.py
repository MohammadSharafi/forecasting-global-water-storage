import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
F=FEATS2+AR; t0=time.time()
TEST_MIX={1:.335,2:.223,3:.166,4:.111,5:.055,6:.055,7:.055}
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=add_ar(assemble2(training_rows(hist,np.random.default_rng(0),per_row=4),cov_all,obs_hist,sums,cell,loyo=True),obs_hist)
Xva=add_ar(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),obs_val)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); htr=Xtr["horizon"].to_numpy(); del Xtr; gc.collect()
Xv=Xva.select(F).to_numpy().astype(np.float32); yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy()
freq={h:(htr==h).mean() for h in range(1,8)}; w=np.array([TEST_MIX.get(h,0.01)/max(freq.get(h,1e-9),1e-9) for h in htr],dtype=np.float32); w/=w.mean()
print(f"built {time.time()-t0:.0f}s rows={len(X)} train horizon freq={ {k:round(v,3) for k,v in freq.items()} }")
r=lambda p: float(np.sqrt(np.mean((yv-p)**2)))
params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
for name,wt in [("per_row4 unweighted",None),("per_row4 test-mix weights",w)]:
    ds=lgb.Dataset(X,y,weight=wt,free_raw_data=False,params={"max_bin":63})
    m=lgb.train(params,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
    p=m.predict(Xv,num_iteration=m.best_iteration)+kv; print(f"[{name}] RMSE={r(p):.4f} best_iter={m.best_iteration}   (v4 per_row3: 0.6614)")
    np.save(f"out/val_{name.split()[1]}_pred.npy",p); del ds; gc.collect()
