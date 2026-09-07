"""Is a horizon specialist worth splicing in? (session 9m)

Why this exists
---------------
Every model in this project is trained on all horizons at once, and h=1 and h=7 are not the same
problem. At h=1 the target is one month away from an observed value: the answer is mostly the
cell's own state plus one month of weather. At h=7 the observation is half a year stale and what
survives is regional persistence and climatology. A single tree ensemble has to spend its splits
on both regimes, and the compromise falls hardest on h=1 -- which is 33.3% of the test, the
largest single slice, and the one where there is the most signal left to lose.

`run_models.py HFILT=1` trains on the h=1 rows only. This script measures whether replacing (or
blending into) the general model's h=1 predictions with that specialist's actually helps, under
the real test horizon mix, on both layouts, with a scan over the blend weight beta:

    p'(h=1) = beta * specialist + (1 - beta) * general        p'(h>1) = general

beta is chosen cross-layout -- fitted on one, scored on the other -- and adopted only if it
clears 0.0003 on both. A specialist sees a third of the training rows, so it is noisier; the
blend is what lets it contribute without that noise being taken on whole.

stdout: FINAL_H1BETA=<beta>   (0 = do not use a specialist)
stderr: the scan.

usage: python hsplice.py lgb_v5x_noll:_bw,xgb_v5x_noll:_bw lgb_v5x_noll:_h1
"""
import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load
from xfit import layouts, loo_grid

# both may be comma-separated member lists, so the general model can be the same lgb+xgb BLEND
# the submission uses -- beta has to be measured against what it will actually be spliced into.
GEN = (sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll:_e8").split(",")
SPE = (sys.argv[2] if len(sys.argv) > 2 else "lgb_v5x_noll:_h1").split(",")
HS = int(sys.argv[3]) if len(sys.argv) > 3 else 1        # which horizon the specialist covers
W = test_mix()
GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
THRESH = -0.0003


def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def main():
    D = {}
    for L in layouts():
        try:
            va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
            g = load(L, GEN); s = load(L, SPE)
        except (SystemExit, FileNotFoundError) as e:
            print(f"  layout {L}: {e}", file=sys.stderr); continue
        D[L] = (va["target"].to_numpy(), va["horizon"].to_numpy(), g, s)
    if not D:
        print("FINAL_H1BETA=0")
        print("  no specialist predictions found", file=sys.stderr); return

    S = {}
    for L, (y, h, g, s) in D.items():
        m = h == HS
        S[L] = {float(b): mixed(y, np.where(m, b * s + (1 - b) * g, g), h) for b in GRID}
        r = lambda p: float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
        print(f"\n  layout {L}: RMSE at h={HS} -- general {r(g):.4f}, specialist {r(s):.4f}",
              file=sys.stderr)
        print("    beta " + " ".join(f"{b:6.2f}" for b in GRID), file=sys.stderr)
        print("    mix  " + " ".join(f"{S[L][float(b)] - S[L][0.0]:+.4f}" for b in GRID),
              file=sys.stderr)

    print("\n  leave-one-layout-out choice:", file=sys.stderr)
    best, _, ok, lines = loo_grid(S, 0.0, THRESH)
    for ln in lines:
        print(ln, file=sys.stderr)
    print(f"  {'adopted' if best else 'not adopted'}: beta={best:.2f}", file=sys.stderr)
    print(f"FINAL_H1BETA={best:.2f}")


if __name__ == "__main__":
    main()
