"""How many boosting rounds should the FINAL models get? (session 9x)

The defect
----------
`rd()` in run_night.sh takes each family's round count from what early stopping chose on layout A,
and hands that number to the FINAL training runs. But layout A's history is ~122 months and FINAL
trains on ~160 -- about 30% more data. The number of trees a gradient-boosted model wants grows
with the training set, so FINAL has plausibly been UNDER-trained in every run this project has
made, and nobody has checked.

Under-training is invisible: there is no validation set for FINAL, the model trains without
complaint, and the submission looks normal.

What this does
--------------
The layouts have different history lengths (B ~89 months, C ~120, A ~122), so their early-stopped
round counts are three points on the curve of "rounds against training size". This reads those
counts out of the orchestrator's own logs, measures each layout's history from the parquet it was
built from, fits log(rounds) against log(months), and extrapolates to FINAL.

It is deliberately conservative:
  * a power law is fitted only when there are at least three layouts and the fit explains most of
    the spread; otherwise it falls back to a straight ratio of history lengths
  * the exponent is clipped to [0, 1]. Zero means rounds do not scale with data at all, one means
    they scale linearly; outside that range the fit is noise, not a relationship
  * the recommendation is never below the incumbent count, because the failure being corrected is
    under-training
  * the increase is capped at 1.6x, since over-shooting a boosted model with a 0.02 learning rate
    costs a little accuracy and a lot of time

stdout: shell assignments (LGB_ROUNDS, XGB_ROUNDS, ...) for the orchestrator.
stderr: the table and the fit.

usage: python rounds.py [out/night]
"""
import glob
import os
import re
import sys
import numpy as np
import polars as pl

S = sys.argv[1] if len(sys.argv) > 1 else "out/night"
SFX = {"A": "", "B": "_B", "C": "_C"}
MAXUP = 1.6


def months(L):
    p = f"out/pseudo_hist{SFX[L]}.parquet"
    if not os.path.exists(p):
        return None
    return int(pl.read_parquet(p, columns=["time"])["time"].n_unique())


def final_months():
    if not os.path.exists("Train.csv"):
        return None
    return int(pl.read_csv("Train.csv", columns=["time"])["time"].n_unique())


def best_iter(path):
    if not os.path.exists(path):
        return None
    m = re.findall(r"best_iter=(\d+)", open(path, errors="replace").read())
    return int(m[-1]) if m else None


def main():
    fm = final_months()
    if fm is None:
        print("LGB_ROUNDS=\nXGB_ROUNDS=")
        print("  Train.csv not found", file=sys.stderr); return

    fams = sorted({os.path.basename(f).split("_", 2)[2][:-4]
                   for f in glob.glob(os.path.join(S, "cv_?_*.log"))})
    if not fams:
        print("  no cv_*.log files -- run phase 5 first", file=sys.stderr); return

    print(f"\n  FINAL trains on {fm} history months\n", file=sys.stderr)
    out = {}
    for fam in fams:
        pts = []
        for L in ("A", "B", "C"):
            n, r = months(L), best_iter(os.path.join(S, f"cv_{L}_{fam}.log"))
            if n and r:
                pts.append((L, n, r))
        if not pts:
            continue
        print(f"  {fam}:", file=sys.stderr)
        for L, n, r in pts:
            print(f"    layout {L}: {n:3d} months -> {r:4d} rounds", file=sys.stderr)
        inc = [r for L, n, r in pts if L == "A"] or [max(r for _, _, r in pts)]
        inc = inc[0]
        x = np.log([n for _, n, _ in pts]); y = np.log([r for _, _, r in pts])
        how, k = "ratio of history lengths", None
        if len(pts) >= 3 and x.std() > 1e-6:
            k = float(np.polyfit(x, y, 1)[0])
            pred = np.polyval(np.polyfit(x, y, 1), x)
            ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            if ss > 0.6 and 0.0 <= k <= 1.0:
                how = f"power law, exponent {k:.2f} (fit explains {ss*100:.0f}%)"
            else:
                how = (f"power law rejected (exponent {k:.2f}, fit {ss*100:.0f}%) -- "
                       f"using the ratio instead")
                k = None
        na = [n for L, n, _ in pts if L == "A"] or [max(n for _, n, _ in pts)]
        scale = (fm / na[0]) ** (k if k is not None else 1.0)
        rec = int(round(min(max(inc * scale, inc), inc * MAXUP)))
        print(f"    {how}", file=sys.stderr)
        print(f"    incumbent {inc} -> recommended {rec}  ({rec/inc:.2f}x)", file=sys.stderr)
        out[fam] = rec

    names = {"lgb": "LGB_ROUNDS", "lgbs": "LGBS_ROUNDS", "lgbm": "LGBM_ROUNDS",
             "xgb": "XGB_ROUNDS", "cat": "CAT_ROUNDS"}
    for fam, r in out.items():
        if fam in names:
            print(f"{names[fam]}={r}")
    print("\n  These are a correction for training-set size, not a tuning sweep. If a family's "
          "three\n  layouts disagree wildly, the fit is rejected and the plain ratio is used.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
