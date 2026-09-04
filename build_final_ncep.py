"""Final with NCEP water-balance features: v6w+NCEP (5 seeds) blended 50/50 with the
existing v5 (3 seeds), smoothed. Writes out/sub_final_ncep_6dp.csv (+ unblended)."""
import polars as pl, numpy as np, lightgbm as lgb, gc, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
from features_ncep import load_ncep, add_ncep
from smooth import smooth
F6=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
nc,cols=load_ncep(tr["lat"].unique().to_list(), tr["lon"].unique().to_list()); print("ncep cols",cols,flush=True)
def feats(rows,cov_all,obs,sums,cell,loyo):
    r=add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs); return add_ncep(r,nc,cols)
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(tr); _,cell=cell_stats(tr)
known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left")
Xte,NF=feats(rows_te,cov_all,obs_all,sums,cell,False); F=F6+NF
print("test NCEP coverage: %.3f"%Xte["P_t"].is_not_null().mean(), "acc coverage (h>1): %.3f"%Xte.filter(pl.col("horizon")>1)["PER_acc"].is_not_null().mean(),flush=True)
Xt=Xte.select(F).to_numpy().astype(np.float32); kt=Xte["tws_known"].to_numpy(); ids=Xte["ID"]; rows=Xte.select(["lat","lon","time","tws_known"]); del Xte; gc.collect()
Xtr,_=feats(training_rows(tr,np.random.default_rng(11),per_row=3),cov_all,obs_train,sums,cell,True)
yr=Xtr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr,tr,te,cov_all,nc; gc.collect()
ds=lgb.Dataset(X,y,weight=w,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
preds=[]
for s in range(5):
    m=lgb.train({**P,"seed":s},ds,num_boost_round=300); m.save_model(f"out/lgb_finalN_s{s}.txt"); preds.append(m.predict(Xt)+kt); print(f"  seed {s} ({time.time()-t0:.0f}s)",flush=True)
pn=np.mean(preds,axis=0); np.save("out/pred_v6w_ncep.npy",pn)
pl.DataFrame({"ID":ids,"Target":np.round(pn,6)}).write_csv("out/sub_v6w_ncep_6dp.csv",float_precision=6)
v5=pl.read_csv("out/sub_v5_6dp.csv").join(pl.DataFrame({"ID":ids}).with_row_index("i"),on="ID").sort("i")["Target"].to_numpy()
pb=0.5*pn+0.5*v5; ps=smooth(rows,pb,w=0.7,radius=1,iters=1)
pl.DataFrame({"ID":ids,"Target":np.round(ps,6)}).write_csv("out/sub_final_ncep_6dp.csv",float_precision=6)
print(f"wrote out/sub_final_ncep_6dp.csv rows={len(ps)}  RMSE(ncep-model vs v5)={np.sqrt(np.mean((pn-v5)**2)):.4f}")
