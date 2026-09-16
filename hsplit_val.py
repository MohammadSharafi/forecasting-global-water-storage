import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
F=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
def feats(rows,cov_all,obs,sums,cell,loyo): return add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs)
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); cov_all=tr.select(["lat","lon","time"]+COV)
for sfx in ["","_B"]:
    t0=time.time(); tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
    sums=clim_sums(hist); _,cell=cell_stats(hist)
    Xtr=feats(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,True)
    Xva=feats(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,False)
    yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy(); hv=Xva["horizon"].to_numpy()
    y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); htr=Xtr["horizon"].to_numpy()
    yr=Xtr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
    X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32); del Xtr,Xva; gc.collect()
    r=lambda a,k=None: float(np.sqrt(np.mean(((yv-a)**2)[k if k is not None else slice(None)])))

    ds=lgb.Dataset(X,y,weight=w,free_raw_data=False,params={"max_bin":63})
    m=lgb.train(P,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
    p_sh=m.predict(Xv,num_iteration=m.best_iteration)+kv

    k1=htr==1; ds1=lgb.Dataset(X[k1],y[k1],weight=w[k1],params={"max_bin":63})
    m1=lgb.train(P,ds1,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv[hv==1],(yv-kv)[hv==1],reference=ds1)],callbacks=[lgb.early_stopping(200,verbose=False)])
    p_sp=p_sh.copy(); p_sp[hv==1]=m1.predict(Xv[hv==1],num_iteration=m1.best_iteration)+kv[hv==1]

    P2={**P,"num_leaves":255,"min_data_in_leaf":300,"learning_rate":0.01,"feature_fraction":0.5}
    m2=lgb.train(P2,ds,num_boost_round=8000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(300,verbose=False)])
    p_d=m2.predict(Xv,num_iteration=m2.best_iteration)+kv
    print(f"[layout{sfx or ' A'}] shared={r(p_sh):.4f} (h1 {r(p_sh,hv==1):.4f})  h1-specialist={r(p_sp):.4f} (h1 {r(p_sp,hv==1):.4f}, iters {m1.best_iteration})  deep={r(p_d):.4f} (iters {m2.best_iteration})  blend shared/deep={r(0.5*p_sh+0.5*p_d):.4f}  ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"out/val8{sfx}_shared.npy",p_sh); np.save(f"out/val8{sfx}_deep.npy",p_d); del X,Xv,ds,ds1,m,m1,m2; gc.collect()
