import polars as pl, numpy as np, lightgbm as lgb, json, os, time, gc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
from features5 import add_wide, WIDE
from features6 import add_recent, RECENT, LONGTERM
os.makedirs("out/report_v6",exist_ok=True)
F=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63)
def feats(rows,cov_all,obs,sums,cell,loyo): return add_recent(add_wide(add_ar(assemble2(rows,cov_all,obs,sums,cell,loyo=loyo),obs)),obs)
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); cov_all=tr.select(["lat","lon","time"]+COV)
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
obs_hist=hist.select(["lat","lon","time","TWS_t"]); obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=feats(training_rows(hist,np.random.default_rng(0),per_row=3),cov_all,obs_hist,sums,cell,True)
yr=Xtr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr; gc.collect()
Xva=feats(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,False)
Xv=Xva.select(F).to_numpy().astype(np.float32); yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy()
from codecarbon import EmissionsTracker
trk=EmissionsTracker(output_dir="out/report_v6",log_level="error",save_to_file=True); trk.start(); t0=time.time()
ds=lgb.Dataset(X,y,weight=w,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X,y; gc.collect()
m=lgb.train(P,ds,num_boost_round=280); em=trk.stop(); tsec=time.time()-t0
json.dump({"train_seconds":tsec,"kg_co2eq_one_seed":em,"kg_co2eq_final_ensemble_8_seeds":8*em},open("out/report_v6/carbon.json","w"),indent=1)
p=m.predict(Xv)+kv; err=p-yv
import shap
idx=np.random.default_rng(0).choice(len(Xv),40000,replace=False); sv=shap.TreeExplainer(m).shap_values(Xv[idx]); ma=np.abs(sv).mean(0); order=np.argsort(ma)[::-1]
json.dump({F[i]:float(ma[i]) for i in order},open("out/report_v6/shap_mean_abs.json","w"),indent=1)
plt.figure(figsize=(7,7)); shap.summary_plot(sv[:,order[:20]],Xv[idx][:,order[:20]],feature_names=[F[i] for i in order[:20]],show=False,max_display=20); plt.tight_layout(); plt.savefig("out/report_v6/shap_summary.png",dpi=130); plt.close()
plt.figure(figsize=(6,6)); plt.barh([F[i] for i in order[:20]][::-1],ma[order[:20]][::-1]); plt.xlabel("mean |SHAP| (TWS units)"); plt.tight_layout(); plt.savefig("out/report_v6/shap_bar.png",dpi=130); plt.close()
def grp(vals,labels=None):
    out=[]
    for i,v in enumerate(np.unique(vals) if labels is None else labels):
        k=(vals==v) if labels is None else (vals==i)
        if k.sum()<100: continue
        out.append({"group":str(v),"n":int(k.sum()),"rmse_model":float(np.sqrt(np.mean(err[k]**2))),"rmse_persist":float(np.sqrt(np.mean((kv[k]-yv[k])**2))),"bias":float(err[k].mean())})
    return out
lat=Xva["lat"].to_numpy(); band=np.digitize(lat,[-30,0,23.5,45,66.5]); bands=["<-30","-30..0","0..23.5","23.5..45","45..66.5",">66.5"]
h=Xva["horizon"].to_numpy(); an=np.abs(Xva["ranom_known"].to_numpy()); reg=np.digitize(an,[0.5,1.0,2.0]); regs=["|anom|<0.5","0.5-1","1-2",">2"]
rep={"overall":{"rmse_model":float(np.sqrt(np.mean(err**2))),"rmse_persist":float(np.sqrt(np.mean((kv-yv)**2))),"bias":float(err.mean()),"n":len(err)},
     "by_latitude_band":[dict(d,group=bands[int(d["group"])]) for d in grp(band)],"by_horizon":grp(h),"by_anomaly_regime":[dict(d,group=regs[int(d["group"])]) for d in grp(reg)]}
json.dump(rep,open("out/report_v6/error_breakdown.json","w"),indent=1)
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,(key,lab) in zip(ax,[("by_latitude_band","latitude band"),("by_horizon","forecast horizon (months)"),("by_anomaly_regime","|anomaly at last obs| (60-mo window)")]):
    d=rep[key]; x=np.arange(len(d)); a.bar(x-0.2,[r["rmse_persist"] for r in d],0.4,label="persistence"); a.bar(x+0.2,[r["rmse_model"] for r in d],0.4,label="model"); a.set_xticks(x); a.set_xticklabels([r["group"] for r in d],rotation=30); a.set_title(f"RMSE by {lab}"); a.legend()
plt.tight_layout(); plt.savefig("out/report_v6/error_breakdown.png",dpi=130); plt.close()
ce=pl.DataFrame({"lat":lat,"lon":Xva["lon"].to_numpy(),"e":err}).group_by(["lat","lon"]).agg(pl.col("e").mean()).to_pandas()
plt.figure(figsize=(11,5)); plt.scatter(ce.lon,ce.lat,c=ce.e,s=3,cmap="RdBu_r",vmin=-0.6,vmax=0.6); plt.colorbar(label="mean(pred - true)"); plt.title("Mean prediction error per cell (validation, final stack)"); plt.tight_layout(); plt.savefig("out/report_v6/error_map.png",dpi=130); plt.close()
print("done", json.dumps(rep["overall"]), "carbon", em, "top:", [F[i] for i in order[:10]])
