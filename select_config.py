"""Pick the final configuration from the pipeline13 experiment grid (session 9d).

Reads every experiment's saved validation predictions, scores each under the REAL TEST
horizon mix on both layouts, and applies this project's standing rule: a change is adopted
only if it wins on BOTH layouts.  Four decisions are made independently --

  features  e1 base | e2 +anomalies | e3 +zonal | e4 both | e8 + big anchors  (vs e1)
  capacity  e8 lgb 127 leaves | e5 lgbm 63 | e6 lgbs 31                       (vs e8)
  weights   e8 recent-year ramp | e7 uniform                                  (vs e8)
  hmix      e8 sampler default | e9 reweighted to the test horizon mix        (vs e8)

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
    "e1": dict(stem="lgb_v5x_noll",  tag="_e1", dropf="anom,scale,bigsa", model="lgb",  weights="ramp",    hmix="", label="base (pre-session-9)"),
    "e2": dict(stem="lgb_v5x_noll",  tag="_e2", dropf="scale,bigsa",      model="lgb",  weights="ramp",    hmix="", label="+covariate anomalies"),
    "e3": dict(stem="lgb_v5x_noll",  tag="_e3", dropf="anom,bigsa",       model="lgb",  weights="ramp",    hmix="", label="+zonal scale"),
    "e4": dict(stem="lgb_v5x_noll",  tag="_e4", dropf="bigsa",            model="lgb",  weights="ramp",    hmix="", label="+anomalies +zonal"),
    "e8": dict(stem="lgb_v5x_noll",  tag="_e8", dropf="",                 model="lgb",  weights="ramp",    hmix="", label="+1500/2500 km anchors"),
    "e5": dict(stem="lgbm_v5x_noll", tag="_e5", dropf="",                 model="lgbm", weights="ramp",    hmix="", label="all, 63 leaves"),
    "e6": dict(stem="lgbs_v5x_noll", tag="_e6", dropf="",                 model="lgbs", weights="ramp",    hmix="", label="all, 31 leaves"),
    "e7": dict(stem="lgb_v5x_noll",  tag="_e7", dropf="",                 model="lgb",  weights="uniform", hmix="", label="all, uniform weights"),
    "e9": dict(stem="lgb_v5x_noll",  tag="_e9", dropf="",                 model="lgb",  weights="ramp",    hmix="test", label="all, test horizon mix"),
}
FEATURE_CANDS = ("e2", "e3", "e4", "e8")   # compared against e1
REF = "e8"                                  # capacity / weights / hmix are compared against this
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
    if "e1" not in S or REF not in S:
        print("FINAL_DROPF=''\nFINAL_MODEL=lgb\nFINAL_WEIGHTS=ramp\nFINAL_HMIX=''")
        print(f"  e1 or {REF} missing -- falling back to the full feature set, default lgb, ramp",
              file=sys.stderr)
        return

    print("\n  experiment                      testmix A   testmix B", file=sys.stderr)
    for k in EXP:
        if k in S:
            print(f"  {k} {EXP[k]['label']:28} {S[k]['A']:.4f}      {S[k]['B']:.4f}", file=sys.stderr)

    def beats(cand, ref):
        return S[cand]["A"] < S[ref]["A"] and S[cand]["B"] < S[ref]["B"]

    # features: best candidate that beats the pre-session-9 baseline on both layouts
    feat = "e1"
    for k in sorted([c for c in FEATURE_CANDS if c in S], key=lambda c: S[c]["A"] + S[c]["B"]):
        if beats(k, "e1"):
            feat = k
            break
    # capacity / weights / horizon mix: switch only if the variant wins on both layouts
    model = EXP[REF]["model"]
    for k in sorted([c for c in ("e5", "e6") if c in S], key=lambda c: S[c]["A"] + S[c]["B"]):
        if beats(k, REF):
            model = EXP[k]["model"]
            break
    weights = "uniform" if ("e7" in S and beats("e7", REF)) else "ramp"
    hmix = "test" if ("e9" in S and beats("e9", REF)) else ""

    dropf = EXP[feat]["dropf"]
    print(f"\n  features: {feat} ({EXP[feat]['label']}) -> DROPF='{dropf}'", file=sys.stderr)
    print(f"  capacity: {model}", file=sys.stderr)
    print(f"  weights : {weights}", file=sys.stderr)
    print(f"  hmix    : {hmix or 'sampler default'}", file=sys.stderr)
    print("  (each decision requires a win on BOTH layouts; incumbent kept otherwise)",
          file=sys.stderr)
    print(f"FINAL_DROPF='{dropf}'")
    print(f"FINAL_MODEL={model}")
    print(f"FINAL_WEIGHTS={weights}")
    print(f"FINAL_HMIX='{hmix}'")


if __name__ == "__main__":
    main()
