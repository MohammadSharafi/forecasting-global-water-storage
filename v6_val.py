import polars as pl, numpy as np, lightgbm as lgb, gc, time, sys
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
def feats(rows,cov_all,obs,sums,cell,loyo): return add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs)
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); cov_all=tr.select(["lat","lon","time"]+COV)
for sfx in ["","_B"]:
    t0=time.time(); tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
    sums=clim_sums(hist); _,cell=cell_stats(hist)
    Xtr=feats(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,True)
    Xva=feats(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,False)
    yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy(); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32)
    r=lambda a: float(np.sqrt(np.mean((yv-a)**2)))
    for name,F in [("v5+recent",FEATS2+AR+WIDE+RECENT),("v6 no long-term anchors",[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM])]:
        X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32)
        ds=lgb.Dataset(X,y,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X; gc.collect()
        m=lgb.train(P,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
        p=m.predict(Xv,num_iteration=m.best_iteration)+kv
        print(f"[layout{sfx or ' A'}] {name:26} RMSE={r(p):.4f} bias={np.mean(p-yv):+.4f} best_iter={m.best_iteration}   (v5: A 0.6555)  ({time.time()-t0:.0f}s)",flush=True)
        np.save(f"out/val6{sfx}_{name.split()[0]}_pred.npy",p); del ds,m,Xv; gc.collect()
    del Xtr,Xva; gc.collect()
