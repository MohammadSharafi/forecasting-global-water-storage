import sys, time
import numpy as np, polars as pl, json, os

MIX = np.array([.3333,.2222,.1667,.1111,.0556,.0556,.0556])

def main():
    L = sys.argv[1]; t0=time.time()
    F = json.load(open(f"out/mats/feats_{L}.json" if os.path.exists(f"out/mats/feats_{L}.json") else "out/mats/feats.json"))
    F = [f for f in F if f not in ("lat","lon")]
    tr = pl.read_parquet(f"out/mats/{L}_tr.parquet", columns=list(dict.fromkeys(["target","tws_known","horizon"]+F)))

    if tr.height > 600_000:
        tr = tr.sample(600_000, seed=0)
    cols=[c for c in F if c in tr.columns]
    X = tr.select(cols).to_numpy().astype(np.float64)
    y = (tr["target"].to_numpy()-tr["tws_known"].to_numpy()).astype(np.float64)
    med = np.nanmedian(X,axis=0); X = np.where(np.isnan(X),med,X)
    mu = X.mean(0); sd = X.std(0)+1e-9
    X = np.clip((X-mu)/sd,-6,6)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]

    hz = tr["horizon"].to_numpy().astype(int)[ok]
    H = np.zeros((len(hz),7)); H[np.arange(len(hz)),np.clip(hz,1,7)-1]=1.0
    X = np.hstack([X,H])
    A = X.T@X + 300.0*np.eye(X.shape[1]); b = X.T@y
    w = np.linalg.solve(A,b)
    print(f"fitted on {X.shape} ({time.time()-t0:.0f}s)")
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=list(dict.fromkeys(["target","tws_known","horizon"]+cols)))
    Xv = va.select(cols).to_numpy().astype(np.float64)
    Xv = np.where(np.isnan(Xv),med,Xv); Xv=np.clip((Xv-mu)/sd,-6,6)
    hv = va["horizon"].to_numpy().astype(int)
    Hv = np.zeros((len(hv),7)); Hv[np.arange(len(hv)),np.clip(hv,1,7)-1]=1.0
    Xv = np.hstack([Xv,Hv])
    p = Xv@w + va["tws_known"].to_numpy().astype(np.float64)
    np.save(f"out/mats/pred_{L}_ridge_v5x_noll_s0_rg.npy", p)
    yv=va["target"].to_numpy().astype(float)
    okv=np.isfinite(yv)&np.isfinite(p)
    s=float(np.sqrt(sum(MIX[i-1]*np.mean((yv[okv&(hv==i)]-p[okv&(hv==i)])**2) for i in range(1,8) if (okv&(hv==i)).any())))
    print(f"ridge {L}: {s:.4f}")

if __name__=="__main__": main()
