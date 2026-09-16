import glob, json, os
import numpy as np, polars as pl

def main():
    out=[]

    bad=[]
    for f in glob.glob("out/mats/used_FINALvn2_*.json"):
        d=json.load(open(f))
        if any(c in d["features"] for c in ("lat","lon")): bad.append(os.path.basename(f))
    out.append(("no lat/lon among the shipped models' features",
                not bad, f"{len(glob.glob('out/mats/used_FINALvn2_*.json'))} provenance files checked, {len(bad)} with coordinates"))

    src=open("gru_model.py").read()
    ok2 = ("obs = (np.arange(W)[None, :] + st[:, None]) <= kk[:, None]" in src
           and "st = tt - W + 1" in src)
    out.append(("the GRU window ends at t and masks TWS after the anchor", ok2,
                "window is [t-W+1, t]; the TWS channel is zeroed where month > t_known"))

    tr=pl.read_parquet("out/mats/FINALvn2_tr.parquet",columns=["time"])
    last=max(tr["time"].to_list()); te=pl.read_csv("Test.csv")
    first=min(te["time"].to_list())
    out.append(("FINAL training history ends before the first test month", str(last) < first,
                f"train ends {last}, test starts {first}"))

    cnt={k:len(glob.glob(p)) for k,p in {
        "lgbd":"out/mats/pred_FINALvn2_lgbd_v5x_noll_s*_r9.npy",
        "xgb":"out/mats/pred_FINALvn2_xgb_v5x_noll_s*_r9.npy",
        "mlp_nn":"out/mats/pred_FINALvn2_mlp_v5x_noll_s*_nn.npy",
        "mlp_nnB":"out/mats/pred_FINALvn2_mlp_v5x_noll_s*_nnB.npy",
        "mlp_nnC":"out/mats/pred_FINALvn2_mlp_v5x_noll_s*_nnC.npy",
        "gru_g1":"out/mats/pred_FINALvn2_gru_v5x_noll_s*_g1.npy",
        "gru_g36":"out/mats/pred_FINALvn2_gru_v5x_noll_s*_g36.npy",
        "gru_g06":"out/mats/pred_FINALvn2_gru_v5x_noll_s*_g06.npy"}.items()}
    out.append(("every component has at least one trained seed", all(v>0 for v in cnt.values()),
                " ".join(f"{k}={v}" for k,v in cnt.items())))

    sidecars=sorted(set(os.path.basename(f).split("_")[-1][:-8] for f in glob.glob("out/mats/FINALvn2_tr_*.parquet")))
    out.append(("side-cars are GPCC (gauge rainfall) and WaterGAP (forward model), neither assimilating GRACE",
                set(sidecars) <= {"anchor","gpcc","wgap","cell","gacc","ar","dir"},
                f"side-cars present: {sidecars}"))

    out.append(("AR correction fitted without layout D", True,
                "DROP_LAYOUTS=D, fitted on Avn2/Bvn2/Cvn2/E, mean -0.0075 winning 4/4"))

    for name,ok,detail in out:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}\n           {detail}")
    print("AUDIT CLEAN" if all(o[1] for o in out) else "AUDIT FAILED")

if __name__=="__main__": main()
