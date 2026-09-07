"""Pick the final configuration from the pipeline13 experiment grid (session 9d).

Reads every experiment's saved validation predictions, scores each under the REAL TEST
horizon mix on both layouts, and applies this project's standing rule: a change is adopted
only if it wins on BOTH layouts.  Three decisions are made independently --

  features  e1 base | e2 +anomalies | e3 +zonal | e4 +both     (must beat e1 on both)
  capacity  e4 lgb 127 leaves | e5 lgbm 63 | e6 lgbs 31        (must beat e4 on both)
  weights   e4 recent-year ramp | e7 uniform                   (must beat e4 on both)

Choosing them independently ignores interactions; it is a heuristic, and the table printed
to stderr shows the evidence so a human can override.  Nothing is adopted by default: if a
candidate does not win on both layouts, the incumbent stays.

stdout: shell assignments for pipeline14.sh.   stderr: the table and the reasoning.
"""
import sys
import numpy as np
from eval_mix import test_mix, load
import polars as pl

EXP = {
    "e1": dict(stem="lgb_v5x_noll",  tag="_e1", dropf="anom,scale", model="lgb",  weights="ramp",    label="base (no anom, no zonal)"),
    "e2": dict(stem="lgb_v5x_noll",  tag="_e2", dropf="scale",      model="lgb",  weights="ramp",    label="+covariate anomalies"),
    "e3": dict(stem="lgb_v5x_noll",  tag="_e3", dropf="anom",       model="lgb",  weights="ramp",    label="+zonal scale"),
    "e4": dict(stem="lgb_v5x_noll",  tag="_e4", dropf="",           model="lgb",  weights="ramp",    label="+both"),
    "e5": dict(stem="lgbm_v5x_noll", tag="_e5", dropf="",           model="lgbm", weights="ramp",    label="both, 63 leaves"),
    "e6": dict(stem="lgbs_v5x_noll", tag="_e6", dropf="",           model="lgbs", weights="ramp",    label="both, 31 leaves"),
    "e7": dict(stem="lgb_v5x_noll",  tag="_e7", dropf="",           model="lgb",  weights="uniform", label="both, uniform weights"),
}
W = test_mix()


def testmix_rmse(L, e):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy()
    p = load(L, [f"{e['stem']}:{e['tag']}"])
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def main():
    S = {}
    for k, e in EXP.items():
        try:
            S[k] = {L: testmix_rmse(L, e) for L in ("A", "B")}
        except SystemExit:
            print(f"  {k} ({e['label']}): predictions missing, skipped", file=sys.stderr)
    if "e1" not in S or "e4" not in S:
        print("FINAL_DROPF=''\nFINAL_MODEL=lgb\nFINAL_WEIGHTS=ramp")
        print("  e1 or e4 missing -- falling back to the full feature set, default lgb, ramp",
              file=sys.stderr)
        return

    print("\n  experiment                      testmix A   testmix B", file=sys.stderr)
    for k in EXP:
        if k in S:
            print(f"  {k} {EXP[k]['label']:28} {S[k]['A']:.4f}      {S[k]['B']:.4f}", file=sys.stderr)

    def beats(cand, ref):
        return S[cand]["A"] < S[ref]["A"] and S[cand]["B"] < S[ref]["B"]

    # features: best of e2/e3/e4 that beats e1 on both layouts, else stay at e1
    feat = "e1"
    for k in sorted([c for c in ("e2", "e3", "e4") if c in S],
                    key=lambda c: S[c]["A"] + S[c]["B"]):
        if beats(k, "e1"):
            feat = k
            break
    # capacity: only switch away from e4's lgb if it wins on both
    model = EXP["e4"]["model"]
    for k in sorted([c for c in ("e5", "e6") if c in S], key=lambda c: S[c]["A"] + S[c]["B"]):
        if beats(k, "e4"):
            model = EXP[k]["model"]
            break
    # weights: only switch to uniform if it wins on both
    weights = "ramp"
    if "e7" in S and beats("e7", "e4"):
        weights = "uniform"

    dropf = EXP[feat]["dropf"]
    print(f"\n  features: {feat} ({EXP[feat]['label']}) -> DROPF='{dropf}'", file=sys.stderr)
    print(f"  capacity: {model}", file=sys.stderr)
    print(f"  weights : {weights}", file=sys.stderr)
    print("  (each decision requires a win on BOTH layouts; incumbent kept otherwise)",
          file=sys.stderr)
    print(f"FINAL_DROPF='{dropf}'")
    print(f"FINAL_MODEL={model}")
    print(f"FINAL_WEIGHTS={weights}")


if __name__ == "__main__":
    main()
