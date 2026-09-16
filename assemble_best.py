import glob
import numpy as np
import polars as pl

from smooth_sweep import smooth

W_TREE, W_LGBD, W_MLP, W_GRU = 0.70, 0.65, 0.35, 0.30

def avg(pat):
    fs = sorted(glob.glob(pat))
    assert fs, pat
    return np.mean([np.load(f) for f in fs], 0), len(fs)

def main():
    P = "out/mats/pred_FINALvn2_"
    lgbd, n1 = avg(P + "lgbd_v5x_noll_s*_r9.npy")
    xgb, n2 = avg(P + "xgb_v5x_noll_s*_r9.npy")
    mlps = [avg(P + f"mlp_v5x_noll_s*_{t}.npy") for t in ("nn", "nnB", "nnC")]
    grus = [avg(P + f"gru_v5x_noll_s*_{t}.npy") for t in ("g1", "g36", "g06")]
    print(f"lgbd {n1} seeds, xgb {n2} seeds, "
          f"mlp {[n for _, n in mlps]} seeds, gru {[n for _, n in grus]} seeds")
    tree = W_LGBD * lgbd + (1 - W_LGBD) * xgb
    mlp = np.mean([m for m, _ in mlps], 0)
    gru = np.mean([g for g, _ in grus], 0)
    rest = (1 - W_MLP) * tree + W_MLP * mlp
    blend = (1 - W_GRU) * rest + W_GRU * gru
    print(f"blend sd {blend.std():.4f}")

    va = pl.read_parquet("out/mats/FINALvn2_va.parquet",
                         columns=["lat", "lon", "time", "tws_known"])
    k = va["tws_known"].to_numpy().astype(float)
    kk = np.where(np.isfinite(k), k, 0.0)
    lat = va["lat"].to_numpy().astype(float); lon = va["lon"].to_numpy().astype(float)
    tm = np.array([str(x)[:7] for x in va["time"].to_list()])
    sm = kk + smooth(blend - kk, lat, lon, tm, 1.0, 0.5)
    assert np.isfinite(sm).all() and len(sm) == 280961
    ref = pl.read_csv("Test.csv").select("ID")
    pl.DataFrame({"ID": ref["ID"], "Target": sm}).write_csv("out/sub_best_base2.csv")
    print(f"wrote out/sub_best_base2.csv  sd {sm.std():.4f}  "
          f"smoothing moved it {np.sqrt(np.mean((sm-blend)**2)):.4f}")

if __name__ == "__main__":
    main()
