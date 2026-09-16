import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
from smooth import smooth
F=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
def feats(rows,cov_all,obs,sums,cell,loyo): return add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs)
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(tr); _,cell=cell_stats(tr)
known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left")
Xte=feats(rows_te,cov_all,obs_all,sums,cell,False); Xt=Xte.select(F).to_numpy().astype(np.float32); kt=Xte["tws_known"].to_numpy(); ids=Xte["ID"]
Xte.select(["ID","lat","lon","time","t_known","horizon","tws_known"]).write_parquet("out/test_rows_v6.parquet"); del Xte; gc.collect()
Xtr=feats(training_rows(tr,np.random.default_rng(7),per_row=3),cov_all,obs_train,sums,cell,True)
yr=Xtr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr,tr,te,cov_all; gc.collect()
ds=lgb.Dataset(X,y,weight=w,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
preds=[]
for s in range(5):
    m=lgb.train({**P,"seed":s},ds,num_boost_round=280); m.save_model(f"out/lgb_final6w_s{s}.txt"); preds.append(m.predict(Xt)+kt); print(f"  seed {s} ({time.time()-t0:.0f}s)",flush=True)
p6=np.mean(preds,axis=0); np.save("out/pred_v6w.npy",p6)
v5=pl.read_csv("out/sub_v5_6dp.csv").join(pl.DataFrame({"ID":ids}).with_row_index("i"),on="ID").sort("i")["Target"].to_numpy()
pb=0.5*p6+0.5*v5
rows=pl.read_parquet("out/test_rows_v6.parquet")
ps=smooth(rows.select(["lat","lon","time","tws_known"]),pb,w=0.7,radius=1,iters=1)
pl.DataFrame({"ID":ids,"Target":np.round(ps,6)}).write_csv("out/sub_final_fallback_6dp.csv",float_precision=6)
print(f"wrote out/sub_final_fallback_6dp.csv rows={len(ps)}  RMSE(v6w vs v5)={np.sqrt(np.mean((p6-v5)**2)):.4f}")
