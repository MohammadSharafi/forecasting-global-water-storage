import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
from features_ncep import load_ncep, add_ncep
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
F6=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); cov_all=tr.select(["lat","lon","time"]+COV)
nc,cols=load_ncep(tr["lat"].unique().to_list(), tr["lon"].unique().to_list()); print("ncep cols:",cols, "rows",len(nc), flush=True)
def feats(rows,cov_all,obs,sums,cell,loyo):
    r=add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs); return add_ncep(r,nc,cols)
for sfx in ["","_B"]:
    t0=time.time(); tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
    sums=clim_sums(hist); _,cell=cell_stats(hist)
    Xtr,NF=feats(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,True)
    Xva,_=feats(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,False)
    yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy(); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32)
    yr=Xtr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
    r=lambda a: float(np.sqrt(np.mean((yv-a)**2)))
    for name,F in [("v6w (ref)",F6),("v6w + NCEP",F6+NF)]:
        X=Xtr.select(F).to_numpy().astype(np.float32); Xv=Xva.select(F).to_numpy().astype(np.float32)
        ds=lgb.Dataset(X,y,weight=w,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X; gc.collect()
        m=lgb.train(P,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
        p=m.predict(Xv,num_iteration=m.best_iteration)+kv
        print(f"[layout{sfx or ' A'}] {name:12} RMSE={r(p):.4f} bias={np.mean(p-yv):+.4f} best_iter={m.best_iteration}  ({time.time()-t0:.0f}s)",flush=True)
        if "NCEP" in name:
            g=dict(zip(F,m.feature_importance("gain"))); tot=sum(g.values()); print("   NCEP share of gain: %.1f%%  top:"%(100*sum(g[f] for f in NF)/tot), sorted(((int(g[f]),f) for f in NF),reverse=True)[:6])
        np.save(f"out/valN{sfx}_{name.split()[0]}_{'ncep' if 'NCEP' in name else 'ref'}.npy",p); del ds,m,Xv; gc.collect()
    del Xtr,Xva; gc.collect()
