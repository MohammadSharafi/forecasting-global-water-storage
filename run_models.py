"""Train one model family on cached matrices. usage: python run_models.py LAYOUT MODEL FEATSET [rounds]
LAYOUT: A|B|FINAL   MODEL: lgb|lgbd|xgb|cat|mlp   FEATSET: v6n|all|v5|allx
Writes out/mats/pred_{LAYOUT}_{MODEL}_{FEATSET}_s{SEED}{TAG}.npy (predictions in the va parquet row order).

env ANCHOR_TARGET selects what the residual target is measured against (session 9):
  tws    (default)  y = target - tws_known            -> persistence-anchored, shrinks to the last observation
  clim              y = target - clim_next            -> climatology-anchored, shrinks to the long-term normal
  decay             y = target - (lam*tws_known + (1-lam)*clim_next), lam = ANCHOR_RHO**horizon
The leaderboard has said twice that this test period rewards long-term anchors over recent
ones (recent-anchor stacks 0.7191, long-term 0.7110), and the anchor a model is trained
against sets its shrinkage far more directly than any feature can.  Use TAG to keep the
prediction files apart, e.g. ANCHOR_TARGET=decay TAG=_ct."""
import polars as pl, numpy as np, sys, time, gc, json, os
from features2 import FEATS2
from features4 import AR
from features5 import WIDE
from features6 import RECENT, LONGTERM
L,M,FS=sys.argv[1:4]; R=int(sys.argv[4]) if len(sys.argv)>4 else None; t0=time.time()
ALL=json.load(open("out/mats/feats.json"))
F6=[f for f in FEATS2+AR+WIDE+RECENT if f not in LONGTERM]
NCEP=[f for f in ALL if f.split("_")[0] in ("P","E","R","SWE","SW","PER")]
NCEP2=[f for f in ALL if f.startswith("r2")]; CPC=[f for f in ALL if f.startswith("cpc")]
ANOMF=[f for f in ALL if f.startswith("an_")]   # per-cell standardised covariate anomalies (features_anom)
from features_x import WIDE4, COVWIN, RESP
SA=["sa300","dsa300","sa500","dsa500","sa800","dsa800","sa_grad"]
SA3=["lb300","la300","ln300","lp300","lpd300","lb500","la500","ln500","lp500","lpd500"]
USE_SA3=FS.endswith("_sa3") and os.path.exists(f"out/mats/{L}_tr_anchor3.parquet")
if USE_SA3: FS=FS[:-4]
SA2=["sb300","sb300_d1","sb300_d2","sb300_d3","sb300_d12","sb300_trend","sb300_persist","sb600","sb600_d1","sb600_d2","sb600_d3","sb600_d12","sb600_trend","sb600_persist","sb_scale"]
USE_SA2=FS.endswith("_sa2") and os.path.exists(f"out/mats/{L}_tr_anchor2.parquet")
if USE_SA2: FS=FS[:-4]
USE_SA=os.path.exists(f"out/mats/{L}_tr_anchor.parquet") and (FS.endswith("_sa") or USE_SA2 or USE_SA3)
if FS.endswith("_sa"): FS=FS[:-3]
SETS={"v6n":F6+NCEP,"all":[f for f in ALL if f not in LONGTERM],"v5":FEATS2+AR+WIDE,"allx":[f for f in ALL if f not in LONGTERM and f not in NCEP2 and f not in CPC],
      "v6nw":F6+NCEP+WIDE4+COVWIN+RESP,"v6nc":F6+NCEP+NCEP2+CPC,"v5x":[f for f in ALL if f not in RECENT],"v5x_noll":[f for f in ALL if f not in RECENT and f not in ("lat","lon")],"e5only":[f for f in ALL if f not in LONGTERM and not f.startswith("r2") and not f.startswith("cpc") and f.split("_")[0] not in ("P","E","R","SWE","SW","PER")],"e5only_noll":[f for f in ALL if f not in LONGTERM and f not in ("lat","lon") and not f.startswith("r2") and not f.startswith("cpc") and f.split("_")[0] not in ("P","E","R","SWE","SW","PER")],"e5only_v5x_noll":[f for f in ALL if f not in RECENT and f not in ("lat","lon") and not f.startswith("r2") and not f.startswith("cpc") and f.split("_")[0] not in ("P","E","R","SWE","SW","PER")],"e5only_v5x":[f for f in ALL if f not in RECENT and not f.startswith("r2") and not f.startswith("cpc") and f.split("_")[0] not in ("P","E","R","SWE","SW","PER")],"noera":[f for f in ALL if f not in LONGTERM and not f.startswith("e5")],"allL_noll":[f for f in ALL if f not in ("lat","lon")],"allL":ALL,"allnoll":[f for f in ALL if f not in LONGTERM and f not in ("lat","lon")],
      # ablation pair for the covariate-anomaly features: same set with and without them
      "v5x_noll_noanom":[f for f in ALL if f not in RECENT and f not in ("lat","lon") and not f.startswith("an_")],
      "allnoll_noanom":[f for f in ALL if f not in LONGTERM and f not in ("lat","lon") and not f.startswith("an_")]}
F=SETS[FS]+(SA if USE_SA else [])+(SA2 if USE_SA2 else [])+(SA3 if USE_SA3 else [])
AT=os.environ.get("ANCHOR_TARGET","tws"); RHO=float(os.environ.get("ANCHOR_RHO","0.85"))
def base_of(df):
    """What the residual target is measured against (and what the prediction is added back to)."""
    k=df["tws_known"].to_numpy().astype(np.float64)
    if AT=="tws": return k
    c=df["clim_next"].to_numpy().astype(np.float64); c=np.where(np.isfinite(c),c,k)
    if AT=="clim": return c
    if AT=="decay":
        lam=RHO**df["horizon"].to_numpy().astype(np.float64); return lam*k+(1.0-lam)*c
    raise SystemExit(f"unknown ANCHOR_TARGET={AT}")
tr=pl.read_parquet(f"out/mats/{L}_tr.parquet",columns=list(dict.fromkeys(["time","tws_known","target","clim_next","horizon"]+[f for f in F if f not in SA and f not in SA2 and f not in SA3])))
if USE_SA: tr=tr.hstack(pl.read_parquet(f"out/mats/{L}_tr_anchor.parquet"))
if USE_SA2: tr=tr.hstack(pl.read_parquet(f"out/mats/{L}_tr_anchor2.parquet"))
if USE_SA3: tr=tr.hstack(pl.read_parquet(f"out/mats/{L}_tr_anchor3.parquet"))
seed=int(os.environ.get("SEED","0")); SUB=float(os.environ.get("SUB","1.0"))
if SUB<1.0: tr=tr.sample(fraction=SUB,seed=seed)
X=tr.select(F).to_numpy(); y=(tr["target"].to_numpy()-base_of(tr)).astype(np.float32)
yr=tr["time"].dt.year().to_numpy(); w=np.clip((yr-yr.min()+1)/(yr.max()-yr.min()+1),0.3,1.0).astype(np.float32)
if os.environ.get("WEIGHTS","ramp")=="uniform": w=np.ones_like(w)   # no recent-year emphasis (test regime differs from the last training years)
del tr; gc.collect()
va=pl.read_parquet(f"out/mats/{L}_va.parquet",columns=list(dict.fromkeys(["tws_known","clim_next","horizon"]+(["target"] if L!="FINAL" else [])+[f for f in F if f not in SA and f not in SA2 and f not in SA3])))
if USE_SA: va=va.hstack(pl.read_parquet(f"out/mats/{L}_va_anchor.parquet"))
if USE_SA2: va=va.hstack(pl.read_parquet(f"out/mats/{L}_va_anchor2.parquet"))
if USE_SA3: va=va.hstack(pl.read_parquet(f"out/mats/{L}_va_anchor3.parquet"))
Xv=va.select(F).to_numpy(); kv=base_of(va); yv=va["target"].to_numpy() if L!="FINAL" else None; del va; gc.collect()
if AT!="tws": print(f"  anchor target: {AT}" + (f" rho={RHO}" if AT=="decay" else "") + f", base mean {kv.mean():+.4f}",flush=True)
print(f"{L} {M} {FS}: X {X.shape} Xv {Xv.shape} ({time.time()-t0:.0f}s)",flush=True)
seed=int(os.environ.get("SEED","0"))
def report(p,tag=""):
    if yv is None: return
    h=None
    print(f"  {tag} RMSE={np.sqrt(np.mean((yv-p)**2)):.4f} bias={np.mean(p-yv):+.4f} ({time.time()-t0:.0f}s)",flush=True)
if M in ("lgb","lgbd"):
    import lightgbm as lgb
    P=dict(objective="regression",learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8,max_bin=63,seed=seed)
    if M=="lgbd": P.update(num_leaves=255,min_data_in_leaf=300,learning_rate=0.01,feature_fraction=0.5)
    ds=lgb.Dataset(X,y,weight=w,free_raw_data=True,params={"max_bin":63}); ds.construct(); del X; gc.collect()
    if yv is not None and R is None:
        m=lgb.train(P,ds,num_boost_round=6000,valid_sets=[lgb.Dataset(Xv,yv-kv,reference=ds)],callbacks=[lgb.early_stopping(200,verbose=False)])
        p=m.predict(Xv,num_iteration=m.best_iteration)+kv; report(p,f"best_iter={m.best_iteration}")
        g=dict(zip(F,m.feature_importance("gain"))); tot=sum(g.values()); print("   top gain:",[(f,round(100*g[f]/tot,1)) for f in sorted(F,key=lambda f:-g[f])[:15]])
    else:
        m=lgb.train(P,ds,num_boost_round=R); p=m.predict(Xv)+kv; report(p,f"rounds={R}"); m.save_model(f"out/mats/model_{L}_{M}_{FS}_s{seed}.txt")
elif M=="xgb":
    import xgboost as xgb
    P=dict(objective="reg:squarederror",eta=0.03,max_depth=9,min_child_weight=200,subsample=0.8,colsample_bytree=0.6,reg_lambda=5.0,tree_method="hist",max_bin=63,nthread=8,seed=seed)
    d=xgb.DMatrix(X,y,weight=w,nthread=8); del X; gc.collect(); dv=xgb.DMatrix(Xv,(yv-kv) if yv is not None else None,nthread=8)
    if yv is not None and R is None:
        m=xgb.train(P,d,6000,evals=[(dv,"va")],early_stopping_rounds=200,verbose_eval=False)
        p=m.predict(dv,iteration_range=(0,m.best_iteration+1))+kv; report(p,f"best_iter={m.best_iteration}")
    else:
        m=xgb.train(P,d,R); p=m.predict(dv)+kv; report(p,f"rounds={R}"); m.save_model(f"out/mats/model_{L}_{M}_{FS}_s{seed}.json")
elif M=="cat":
    from catboost import CatBoostRegressor, Pool
    m=CatBoostRegressor(iterations=R or 6000,learning_rate=float(os.environ.get("CATLR","0.03")),depth=8,l2_leaf_reg=5,loss_function="RMSE",thread_count=8,random_seed=seed,border_count=64,verbose=0,od_type="Iter" if R is None else None,od_wait=200 if R is None else None)
    pool=Pool(X,y,weight=w); del X; gc.collect()
    if yv is not None and R is None: m.fit(pool,eval_set=Pool(Xv,yv-kv),use_best_model=True); p=m.predict(Xv)+kv; report(p,f"best_iter={m.get_best_iteration()}")
    else: m.fit(pool); p=m.predict(Xv)+kv; report(p,f"rounds={R}"); m.save_model(f"out/mats/model_{L}_{M}_{FS}_s{seed}.cbm")
elif M=="mlp":
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    med=np.nanmedian(X,axis=0); X=np.where(np.isnan(X),med,X); Xv=np.where(np.isnan(Xv),med,Xv)
    mu=X.mean(0); sd=X.std(0)+1e-6; X=np.clip((X-mu)/sd,-6,6).astype(np.float32); Xv=np.clip((Xv-mu)/sd,-6,6).astype(np.float32)
    ysd=float(y.std()); dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    W=int(os.environ.get("WIDTH","256")); DO=float(os.environ.get("DROP","0.3")); WD=float(os.environ.get("WD","1e-3")); LR=float(os.environ.get("LR","1e-3"))
    net=nn.Sequential(nn.Linear(X.shape[1],W),nn.SiLU(),nn.Dropout(DO),nn.Linear(W,W),nn.SiLU(),nn.Dropout(DO),nn.Linear(W,W//2),nn.SiLU(),nn.Dropout(DO),nn.Linear(W//2,1)).to(dev)
    EP=int(os.environ.get("EPOCHS","3")); BS=4096; n=len(X); steps=EP*((n+BS-1)//BS)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.15)
    Xt=torch.from_numpy(X); yt=torch.from_numpy(y/ysd); wt=torch.from_numpy(w); Xvt=torch.from_numpy(Xv).to(dev)
    def predict():
        net.eval(); out=[]
        with torch.no_grad():
            for i in range(0,len(Xvt),65536): out.append(net(Xvt[i:i+65536]).squeeze(1).float().cpu().numpy())
        net.train(); return np.concatenate(out)*ysd+kv
    best=9; pbest=None
    for ep in range(EP):
        perm=torch.randperm(n); tl=0.0
        for i in range(0,n,BS):
            idx=perm[i:i+BS]; xb=Xt[idx].to(dev); yb=yt[idx].to(dev); wb=wt[idx].to(dev)
            loss=(wb*(net(xb).squeeze(1)-yb)**2).sum()/wb.sum(); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl+=loss.item()*len(idx)
        p=predict(); report(p,f"epoch {ep+1} train_loss={tl/n:.4f}")
        if yv is not None and (ep==0 or np.sqrt(np.mean((yv-p)**2))<best): best=float(np.sqrt(np.mean((yv-p)**2))); pbest=p.copy()
    if yv is not None: p=pbest
    torch.save({"state":net.state_dict(),"mu":mu,"sd":sd,"med":med,"ysd":ysd,"F":F},f"out/mats/model_{L}_{M}_{FS}_s{seed}.pt")
np.save(f"out/mats/pred_{L}_{M}_{FS}_s{seed}{os.environ.get('TAG','')}.npy",p); print("saved",flush=True)
