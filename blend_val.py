"""Second model family for blending: ridge regression on the same (standardised) features,
residual target. Evaluated on layout A alongside the saved v4 LightGBM predictions."""
import polars as pl, numpy as np, gc, time
from sklearn.linear_model import Ridge
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
F=FEATS2+AR; t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
tp=pl.read_parquet("out/pseudo_test.parquet"); hist=pl.read_parquet("out/pseudo_hist.parquet")
cov_all=tr.select(["lat","lon","time"]+COV); obs_hist=hist.select(["lat","lon","time","TWS_t"])
obs_val=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(hist); _,cell=cell_stats(hist)
Xtr=add_ar(assemble2(training_rows(hist,np.random.default_rng(0),per_row=2),cov_all,obs_hist,sums,cell,loyo=True),obs_hist)
Xva=add_ar(assemble2(tp.select(["lat","lon","time","t_known","target"]),cov_all,obs_val,sums,cell,loyo=False),obs_val)
X=Xtr.select(F).to_numpy().astype(np.float32); y=(Xtr["target"]-Xtr["tws_known"]).to_numpy().astype(np.float32); del Xtr; gc.collect()
Xv=Xva.select(F).to_numpy().astype(np.float32); yv=Xva["target"].to_numpy(); kv=Xva["tws_known"].to_numpy(); h=Xva["horizon"].to_numpy()
med=np.nanmedian(X,axis=0); X=np.where(np.isnan(X),med,X); Xv=np.where(np.isnan(Xv),med,Xv)
mu,sd=X.mean(0),X.std(0)+1e-9; X=(X-mu)/sd; Xv=(Xv-mu)/sd
print(f"built {time.time()-t0:.0f}s rows={len(X)}")
r=lambda p: float(np.sqrt(np.mean((yv-p)**2)))
p_lgb=np.load("out/val4_pred.npy")
for alpha in (1.0,30.0,300.0):
    rg=Ridge(alpha=alpha).fit(X,y); p_r=rg.predict(Xv)+kv
    print(f"ridge alpha={alpha}: {r(p_r):.4f}")
rg=Ridge(alpha=30.0).fit(X,y); p_r=rg.predict(Xv)+kv
# ridge with horizon-interaction (features x horizon one-hot) for a non-linear touch
H=np.stack([(Xv[:,F.index("horizon")]*sd[F.index("horizon")]+mu[F.index("horizon")])==k for k in range(1,8)],1).astype(np.float32)
print(f"lgb={r(p_lgb):.4f}  ridge={r(p_r):.4f}")
for w in (0.1,0.2,0.3,0.4):
    print(f"  blend w_ridge={w}: {r((1-w)*p_lgb+w*p_r):.4f}")
np.save("out/val4_ridge_pred.npy",p_r)
