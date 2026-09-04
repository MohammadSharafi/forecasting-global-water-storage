"""Memory-lean training/prediction from the on-disk matrices. Reuses saved seed models
where present; trains missing seeds one at a time with LightGBM holding the only copy."""
import polars as pl, numpy as np, lightgbm as lgb, gc, sys, time, os
from features2 import FEATS2
from features4 import AR
from smooth import smooth
F=FEATS2+AR; ROUNDS=240; SEEDS=[0,1,2]
Xte=pl.read_parquet("out/Xte_v4.parquet"); Xt=Xte.select(F).to_numpy().astype(np.float32); kv=Xte["tws_known"].to_numpy()
preds=[]
for s in SEEDS:
    path=f"out/lgb_final4_s{s}.txt"
    if not os.path.exists(path):
        t0=time.time(); d=pl.read_parquet(f"out/Xtr_v4_s2.parquet")   # one shared draw; seeds differ in bagging/feature sampling
        X=d.select(F).to_numpy().astype(np.float32); y=(d["target"]-d["tws_known"]).to_numpy().astype(np.float32); del d; gc.collect()
        ds=lgb.Dataset(X,y,feature_name=F,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
        params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,seed=s,max_bin=63)
        m=lgb.train(params,ds,num_boost_round=ROUNDS); m.save_model(path); del ds; gc.collect(); print(f"seed {s} trained ({time.time()-t0:.0f}s)",flush=True)
    m=lgb.Booster(model_file=path); preds.append(m.predict(Xt)+kv); print(f"seed {s} predicted",flush=True)
p=np.mean(preds,axis=0); assert np.isfinite(p).all()
pl.DataFrame({"ID":Xte["ID"],"Target":p}).write_csv("out/sub_v4_ar.csv")
ps=smooth(Xte.select(["lat","lon","time","tws_known"]),p,w=0.7,radius=1,iters=1)
pl.DataFrame({"ID":Xte["ID"],"Target":ps}).write_csv("out/sub_v4_ar_smooth.csv")
print(f"wrote out/sub_v4_ar.csv and out/sub_v4_ar_smooth.csv rows={len(p)} mean={p.mean():.4f} std={p.std():.4f} |raw-smooth| mean={np.abs(p-ps).mean():.4f}")
