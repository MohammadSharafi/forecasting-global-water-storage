import polars as pl, numpy as np, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("report_final",exist_ok=True)
from smooth import smooth
L="A"; va=pl.read_parquet(f"out/mats/{L}_va.parquet",columns=["lat","lon","time","t_known","horizon","tws_known","target","ranom_known"])
ld=lambda n: np.load(f"out/mats/pred_{L}_{n}.npy")
R=0.7*np.mean([ld(f"mlp_allnoll_s{s}_e1") for s in range(4)],0)+0.3*ld("lgb_allnoll_s0")
Lg=0.5*ld("lgb_v5x_noll_s0")+0.5*ld("mlp_v5x_noll_s0_e1")
p=smooth(va,0.5*(R+Lg),0.7); y=va["target"].to_numpy(); k=va["tws_known"].to_numpy()
lat=va["lat"].to_numpy(); h=va["horizon"].to_numpy(); an=np.abs(va["ranom_known"].to_numpy())
r=lambda a,m=slice(None): float(np.sqrt(np.mean(((y-a)**2)[m])))
out={"overall":{"model":r(p),"persistence":r(k)},"by_horizon":{},"by_latitude_band":{},"by_regime":{}}
for hh in range(1,8): out["by_horizon"][f"h{hh}"]={"model":r(p,h==hh),"persistence":r(k,h==hh),"n":int((h==hh).sum())}
for lo,hi,nm in ((-60,-30,"30S-60S"),(-30,0,"0-30S"),(0,30,"0-30N"),(30,60,"30N-60N"),(60,90,"60N-90N")):
    m=(lat>=lo)&(lat<hi)
    out["by_latitude_band"][nm]={"model":r(p,m),"persistence":r(k,m),"bias":float(np.mean((p-y)[m])),"n":int(m.sum())}
for lo,hi,nm in ((0,1,"|anomaly|<1 sigma"),(1,2,"1-2 sigma"),(2,99,">2 sigma")):
    m=(an>=lo)&(an<hi)&~np.isnan(an)
    out["by_regime"][nm]={"model":r(p,m),"persistence":r(k,m),"bias":float(np.mean((p-y)[m])),"n":int(m.sum())}
json.dump(out,open("report_final/error_breakdown.json","w"),indent=1); print(json.dumps(out,indent=1)[:900])

g=pl.DataFrame({"lat":lat,"lon":va["lon"].to_numpy(),"e":(p-y)}).group_by(["lat","lon"]).agg(pl.col("e").mean().alias("bias"),(pl.col("e")**2).mean().sqrt().alias("rmse"))
M=np.full((180,360),np.nan); M[(g["lat"].to_numpy()+89.5).round().astype(int),(g["lon"].to_numpy()+179.5).round().astype(int)]=g["bias"].to_numpy()
plt.figure(figsize=(11,5)); plt.imshow(M,origin="lower",cmap="RdBu_r",vmin=-0.4,vmax=0.4,extent=[-180,180,-90,90])
plt.colorbar(label="mean error (prediction - truth)"); plt.title("Validation bias by cell (layout A, final ensemble)"); plt.tight_layout(); plt.savefig("report_final/error_map.png",dpi=110); plt.close()

import lightgbm as lgb, shap, sys
sys.argv=["x","FINAL","lgb","allnoll_sa"]; g_={}; exec(open("run_models.py").read().split("F=SETS[FS]")[0],g_)
F=g_["SETS"][g_["FS"]]+g_["SA"]
m=lgb.Booster(model_file="out/mats/model_FINAL_lgb_allnoll_s0.txt")
X=pl.read_parquet(f"out/mats/{L}_va.parquet",columns=[f for f in F if f not in g_["SA"]]).hstack(pl.read_parquet(f"out/mats/{L}_va_anchor.parquet")).select(F)
idx=np.random.default_rng(0).choice(len(X),40000,replace=False); Xs=X[idx].to_numpy().astype(np.float32)
sv=shap.TreeExplainer(m).shap_values(Xs); ma=np.abs(sv).mean(0)
order=np.argsort(-ma)[:20]; json.dump({F[i]:float(ma[i]) for i in np.argsort(-ma)[:40]},open("report_final/shap_mean_abs.json","w"),indent=1)
plt.figure(figsize=(7,7)); plt.barh([F[i] for i in order][::-1],ma[order][::-1]); plt.xlabel("mean |SHAP|"); plt.title("Final model: top drivers of the predicted change"); plt.tight_layout(); plt.savefig("report_final/shap_bar.png",dpi=110); plt.close()
print("top SHAP:",[(F[i],round(float(ma[i]),4)) for i in order[:12]])
