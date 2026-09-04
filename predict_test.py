"""Final pipeline: train on ALL of Train.csv (LOYO climatology), build test features with
only information available at or before each test month, predict, write submission."""
import polars as pl, numpy as np, lightgbm as lgb, time, sys
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
ROUNDS=int(sys.argv[1]) if len(sys.argv)>1 else 130; SEEDS=[0,1,2]
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"])
obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])   # observed test TWS = block-first months
sums=clim_sums(tr); _,cell=cell_stats(tr)
# t_known for each test row = latest observed month <= t for that cell
known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left")
assert rows_te["t_known"].null_count()==0, "some test rows have no observed TWS at or before t"
Xte=assemble2(rows_te,cov_all,obs_all,sums,cell,loyo=False)
h=Xte["horizon"].to_numpy(); print("test horizons:", {int(k):int(v) for k,v in zip(*np.unique(h,return_counts=True))})
preds=[]
for seed in SEEDS:
    rows=training_rows(tr,np.random.default_rng(seed),per_row=3)
    Xtr=assemble2(rows,cov_all,obs_train,sums,cell,loyo=True)
    X=Xtr.select(FEATS2).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32)
    params=dict(objective="regression",learning_rate=0.03,num_leaves=255,min_data_in_leaf=300,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=3.0,verbose=-1,num_threads=8,seed=seed)
    m=lgb.train(params,lgb.Dataset(X,y,feature_name=FEATS2),num_boost_round=ROUNDS)
    preds.append(m.predict(Xte.select(FEATS2).to_numpy().astype(np.float32))+Xte["tws_known"].to_numpy())
    print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
p=np.mean(preds,axis=0)
sub=pl.DataFrame({"ID":Xte["ID"],"Target":p.astype(np.float64)})
assert len(sub)==len(te) and set(sub["ID"])==set(te["ID"]) and np.isfinite(p).all()
sub.write_csv("out/sub_v2_residual.csv"); print(f"wrote out/sub_v2_residual.csv rows={len(sub)} mean={p.mean():.4f} std={p.std():.4f}")
print("persistence-fallback share (h=1 rows):", float((h==1).mean()))
