import glob, os
import numpy as np, polars as pl

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])

def mixrmse(y, h, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(sum(MIX[i-1] * np.mean((y[ok & (h == i)] - p[ok & (h == i)]) ** 2)
                             for i in range(1, 8) if (ok & (h == i)).any())))

def main():
    ds = []
    print("layout  seeds  anchor=tws  anchor=fit    delta")
    for L in ("Avn2", "Bvn2", "Cvn2", "D", "E"):
        c = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_wg1.npy"))
        t = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_af1.npy"))
        seeds = sorted(set(f.split("_s")[-1].split("_")[0] for f in c) &
                       set(f.split("_s")[-1].split("_")[0] for f in t))
        if not seeds:
            continue
        va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
        y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
        a = mixrmse(y, h, np.mean([np.load(f"out/mats/pred_{L}_lgb_v5x_noll_s{s}_wg1.npy") for s in seeds], 0))
        b = mixrmse(y, h, np.mean([np.load(f"out/mats/pred_{L}_lgb_v5x_noll_s{s}_af1.npy") for s in seeds], 0))
        ds.append(b - a)
        print(f"{L:7s} {','.join(seeds):5s}  {a:.4f}      {b:.4f}    {b-a:+.4f}")
    if ds:
        ds = np.array(ds)
        print(f"\nmean {ds.mean():+.4f}   worst {ds.max():+.4f}   wins {int((ds < 0).sum())}/{len(ds)}")
        print("ADOPT" if ds.mean() < 0 and (ds < 0).sum() > len(ds) / 2 else "REJECT")

if __name__ == "__main__":
    main()
