"""U-Net on global 1-degree maps: one sample = one (time, t_known) pair; channels from the cached
matrices; target = change since last observation. usage: python run_cnn.py A|B|FINAL [epochs]"""
import polars as pl, numpy as np, sys, time, os, torch, torch.nn as nn, torch.nn.functional as Fn
L=sys.argv[1]; EP=int(sys.argv[2]) if len(sys.argv)>2 else 60; seed=int(os.environ.get("SEED","0")); torch.manual_seed(seed); np.random.seed(seed); t0=time.time()
CH=["tws_known","ranom_known","dev24","trend24","slope","d2","d6","d12","rseas_next","rclim_next","rmean60","clim_next","clim_known","csd","ac1",
    "SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","SPEI_01_t_k","SPEI_06_t_k","SOIL_MOISTURE_t_k","SPEI_06_t_d","SPEI_12_t_d","SOIL_MOISTURE_t_d",
    "spei1_acc","sm_win","spei3_win","P_t","E_t","PER_acc","P_acc","SW_t","SW_d","SWE_d","cpcSW_t","cpcSW_d","tws_ly","lag1","lag3"]
def to_maps(df, has_y):
    pairs=df.select(["time","t_known"]).unique().sort(["time","t_known"]).with_row_index("pid"); d=df.join(pairs,on=["time","t_known"],how="left")
    pid=d["pid"].to_numpy(); li=(d["lat"].to_numpy()+89.5).round().astype(int); lo=(d["lon"].to_numpy()+179.5).round().astype(int); n=len(pairs)
    X=np.zeros((n,len(CH)+4,180,360),np.float32); M=np.zeros((n,180,360),np.float32); M[pid,li,lo]=1
    for i,c in enumerate(CH):
        v=d[c].to_numpy().astype(np.float32); X[i if False else slice(None),i][pid,li,lo]=np.nan_to_num(v)
    hor=d["horizon"].to_numpy().astype(np.float32); m=d["time"].dt.month().to_numpy().astype(np.float32)
    X[:,len(CH)][pid,li,lo]=hor/7; X[:,len(CH)+1][pid,li,lo]=np.sin(2*np.pi*m/12); X[:,len(CH)+2][pid,li,lo]=np.cos(2*np.pi*m/12)
    X[:,len(CH)+3]=np.linspace(-1,1,180)[None,:,None]
    Y=np.zeros((n,180,360),np.float32)
    if has_y: Y[pid,li,lo]=(d["target"]-d["tws_known"]).to_numpy().astype(np.float32)
    return X,M,Y,(pid,li,lo)
cols=list(dict.fromkeys(["lat","lon","time","t_known","horizon","tws_known"]+CH))
tr=pl.read_parquet(f"out/mats/{L}_tr.parquet",columns=cols+["target"]); Xtr,Mtr,Ytr,_=to_maps(tr,True); del tr
va=pl.read_parquet(f"out/mats/{L}_va.parquet",columns=cols+(["target"] if L!="FINAL" else [])); Xva,Mva,Yva,idx=to_maps(va,L!="FINAL"); kv=va["tws_known"].to_numpy(); yv=va["target"].to_numpy() if L!="FINAL" else None; del va
# per-channel standardisation over valid cells
mu=np.array([Xtr[:,i][Mtr>0].mean() for i in range(Xtr.shape[1])],np.float32); sd=np.array([Xtr[:,i][Mtr>0].std()+1e-6 for i in range(Xtr.shape[1])],np.float32)
Xtr=(Xtr-mu[None,:,None,None])/sd[None,:,None,None]*Mtr[:,None]; Xva=(Xva-mu[None,:,None,None])/sd[None,:,None,None]*Mva[:,None]
Xtr=np.concatenate([Xtr,Mtr[:,None]],1); Xva=np.concatenate([Xva,Mva[:,None]],1); ysd=float(Ytr[Mtr>0].std())
print(f"maps: train {Xtr.shape} val {Xva.shape} ysd={ysd:.3f} ({time.time()-t0:.0f}s)",flush=True)
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,padding_mode="circular"),nn.GroupNorm(8,o),nn.SiLU(),nn.Conv2d(o,o,3,padding=1,padding_mode="circular"),nn.GroupNorm(8,o),nn.SiLU())
class UNet(nn.Module):
    def __init__(s,ci,w=32):
        super().__init__(); s.e1=blk(ci,w); s.e2=blk(w,2*w); s.e3=blk(2*w,4*w); s.b=blk(4*w,4*w)
        s.d3=blk(8*w,2*w); s.d2=blk(4*w,w); s.d1=blk(2*w,w); s.out=nn.Conv2d(w,1,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(Fn.max_pool2d(e1,2)); e3=s.e3(Fn.max_pool2d(e2,2)); b=s.b(Fn.max_pool2d(e3,2))
        u3=s.d3(torch.cat([Fn.interpolate(b,size=e3.shape[-2:],mode="bilinear"),e3],1)); u2=s.d2(torch.cat([Fn.interpolate(u3,size=e2.shape[-2:],mode="bilinear"),e2],1))
        u1=s.d1(torch.cat([Fn.interpolate(u2,size=e1.shape[-2:],mode="bilinear"),e1],1)); return s.out(u1).squeeze(1)
net=UNet(Xtr.shape[1]).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
n=len(Xtr); BS=8; steps=EP*((n+BS-1)//BS); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=steps,pct_start=0.1)
Xt=torch.from_numpy(Xtr); Yt=torch.from_numpy(Ytr/ysd); Mt=torch.from_numpy(Mtr); Xv=torch.from_numpy(Xva).to(dev)
def predict():
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(Xv),16): out.append(net(Xv[i:i+16]).float().cpu().numpy())
    net.train(); P=np.concatenate(out)*ysd; return P[idx[0],idx[1],idx[2]]+kv
best=(9,None)
for ep in range(EP):
    perm=torch.randperm(n); tl=0
    for i in range(0,n,BS):
        j=perm[i:i+BS]; xb=Xt[j]; yb=Yt[j]; mb=Mt[j]
        sh=int(np.random.randint(0,360)); xb=torch.roll(xb,sh,-1); yb=torch.roll(yb,sh,-1); mb=torch.roll(mb,sh,-1)   # longitude roll augmentation
        xb,yb,mb=xb.to(dev),yb.to(dev),mb.to(dev)
        loss=((net(xb)-yb)**2*mb).sum()/mb.sum(); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl+=loss.item()
    p=predict()
    if yv is not None:
        rm=float(np.sqrt(np.mean((yv-p)**2))); print(f"  epoch {ep+1} train={tl/((n+BS-1)//BS):.4f} val RMSE={rm:.4f} ({time.time()-t0:.0f}s)",flush=True)
        if rm<best[0]: best=(rm,p.copy())
    else: print(f"  epoch {ep+1} train={tl/((n+BS-1)//BS):.4f} ({time.time()-t0:.0f}s)",flush=True)
p_final=p if yv is None else best[1]
np.save(f"out/mats/pred_{L}_cnn_maps_s{seed}.npy",p_final); torch.save(net.state_dict(),f"out/mats/model_{L}_cnn_s{seed}.pt"); print("saved best",best[0] if yv is not None else "",flush=True)
