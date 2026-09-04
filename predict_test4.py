"""v4 final: AR lags + trend on top of v2 features; 3 seeds; ROUNDS ~ 1.3x best_iter at lr 0.02."""
import polars as pl, numpy as np, lightgbm as lgb, time, sys
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
ROUNDS=int(sys.argv[1]) if len(sys.argv)>1 else 240; SEEDS=[0,1,2]; F=FEATS2+AR
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(tr); _,cell=cell_stats(tr)
known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left"); assert rows_te["t_known"].null_count()==0
Xte=add_ar(assemble2(rows_te,cov_all,obs_all,sums,cell,loyo=False),obs_all)
preds=[]
for seed in SEEDS:
    Xtr=add_ar(assemble2(training_rows(tr,np.random.default_rng(seed),per_row=3),cov_all,obs_train,sums,cell,loyo=True),obs_train)
    X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32)
    params=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=4,seed=seed)
    m=lgb.train(params,lgb.Dataset(X,y,feature_name=F),num_boost_round=ROUNDS)
    preds.append(m.predict(Xte.select(F).to_numpy().astype(np.float32))+Xte["tws_known"].to_numpy()); print(f"  seed {seed} ({time.time()-t0:.0f}s)",flush=True)
    m.save_model(f"out/lgb_final4_s{seed}.txt")
p=np.mean(preds,axis=0); sub=pl.DataFrame({"ID":Xte["ID"],"Target":p.astype(np.float64)})
assert len(sub)==len(te) and set(sub["ID"])==set(te["ID"]) and np.isfinite(p).all()
sub.write_csv("out/sub_v4_ar.csv"); Xte.select(["ID","lat","lon","time","t_known","horizon"]).write_parquet("out/test_rows.parquet")
print(f"wrote out/sub_v4_ar.csv rows={len(sub)} mean={p.mean():.4f} std={p.std():.4f}")
