"""Tune the lgb/xgb blend weight, which has been hardcoded at 50/50 forever (session 9k).

final_assemble.py has always been called with `lgb:0.5 xgb:0.5`. That weight was never fitted --
it is just the obvious default. Scanning it costs nothing (the predictions already exist) and is
worth a few 1e-4, which at the current leaderboard density of ~0.00047 RMSE per rank is one or
two places.

Scored under the REAL TEST horizon mix on both layouts, and the chosen weight is clipped to
[0.25, 0.75]: past that the blend is effectively one family and the diversity that made lgb+xgb
the best submission is thrown away for a difference validation cannot resolve.

stdout: FINAL_WLGB=<w> for pipeline14.   stderr: the scan table.

usage: python blend_scan.py lgb_v5x_noll xgb_v5x_noll [tag]
"""
import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load

LGB = sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll"
XGB = sys.argv[2] if len(sys.argv) > 2 else "xgb_v5x_noll"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
W = test_mix()
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def testmix(L, p, y, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def main():
    scores = {}
    for L in ("A", "B"):
        try:
            va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
            y = va["target"].to_numpy(); h = va["horizon"].to_numpy()
            a = load(L, [f"{LGB}:{TAG}"]); b = load(L, [f"{XGB}:{TAG}"])
        except SystemExit as e:
            print(f"  layout {L}: {e}", file=sys.stderr); continue
        scores[L] = np.array([testmix(L, w * a + (1 - w) * b, y, h) for w in GRID])
    if not scores:
        print("FINAL_WLGB=0.5")
        print("  no predictions found -- keeping the 50/50 default", file=sys.stderr); return

    print("\n  w_lgb   " + "   ".join(f"{L}" for L in scores), file=sys.stderr)
    for i, w in enumerate(GRID):
        row = "   ".join(f"{scores[L][i]:.4f}" for L in scores)
        star = "  <-" if all(scores[L][i] == scores[L].min() for L in scores) else ""
        print(f"   {w:4.2f}   {row}{star}", file=sys.stderr)

    mean = np.mean([scores[L] for L in scores], axis=0)
    w = float(GRID[int(np.argmin(mean))])
    wc = float(np.clip(w, 0.25, 0.75))
    per = {L: float(GRID[int(np.argmin(scores[L]))]) for L in scores}
    print(f"\n  best per layout: {per}", file=sys.stderr)
    print(f"  best on the mean: {w:.2f}" + (f" -> clipped to {wc:.2f}" if wc != w else ""),
          file=sys.stderr)
    if len(scores) == 2 and abs(per["A"] - per["B"]) > 0.3:
        print("  WARNING: the layouts disagree strongly; 0.5 is the safer choice", file=sys.stderr)
        wc = 0.5
    gain = float(np.mean([scores[L][int(np.argmin(mean))] - scores[L][GRID.tolist().index(0.5)]
                          for L in scores]))
    print(f"  gain over 50/50 (mean of layouts): {gain:+.5f}", file=sys.stderr)
    if gain > -0.0002:
        print("  gain is below what validation can resolve -- keeping 0.50", file=sys.stderr)
        wc = 0.5
    print(f"FINAL_WLGB={wc:.2f}")


if __name__ == "__main__":
    main()
