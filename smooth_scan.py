"""Tune the spatial smoothing that has been fixed at (w=0.7, radius=1, iters=1) since session 4.

Why this is worth doing
-----------------------
`final_assemble.py` blurs the predicted residual field over each cell's 8 grid neighbours with a
weight of 0.7. That weight came from one coarse table in `smooth.py.__main__`, scored on layout A
alone, under the PLAIN horizon average, on a model that no longer exists. Three things about it
have never been checked:

  * the weight is the same at every horizon. It should not be: at h=1 the residual is mostly the
    cell's own state, which the neighbours do not know; at h=7 the predictable part is almost
    entirely regional, which is exactly what the neighbours estimate.
  * the radius is fixed at one cell, and one iteration.
  * the two columns either side of the dateline smooth against half a neighbourhood, because
    lon+1 at +179.5 does not exist. `wrap` closes the grid.

Method
------
Everything is scored under the REAL TEST horizon mix, and the choice is CROSS-FITTED: the
configuration is chosen on one layout and reported on the other, so the number that decides
adoption is out of sample. The incumbent (0.7, r=1, it=1, no wrap) has to be beaten on BOTH
layouts by more than 0.0003 -- below that, validation cannot tell the two apart.

One blending pass is linear in the neighbour mean, so the whole (w1, w7) grid costs one
neighbour pass per (radius, iters, wrap); see smooth.nb_mean.

stdout: shell assignments for the orchestrator.   stderr: the tables.

usage: python smooth_scan.py lgb_v5x_noll:_bw,xgb_v5x_noll:_bw
"""
import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load
from smooth import nb_mean, smooth_field, horizon_w
from xfit import layouts, loo_grid

SPEC = (sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll,xgb_v5x_noll").split(",")
W = test_mix()
WGRID = (0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9)
CONFIGS = [(1, 1, False), (1, 1, True), (2, 1, False), (2, 1, True), (1, 2, False), (2, 2, False)]
BASE = (0.7, 0.7, 1, 1, False)          # what final_assemble.py does today
THRESH = -0.0003                        # must beat the incumbent by at least this, on both layouts


def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def layout(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "horizon", "tws_known", "target"])
    p = load(L, SPEC)
    return va, p


def scores_for(va, p, cfg_list):
    """{(w1, w7, r, it, wrap): testmix RMSE} for every weight pair, over the given configs."""
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy(); k = va["tws_known"].to_numpy()
    res0 = p - k
    out = {}
    for r, it, wrap in cfg_list:
        if it == 1:
            nb = nb_mean(va, res0, r, wrap)
            for w1 in WGRID:
                for w7 in WGRID:
                    wv = horizon_w(h, w1, w7)
                    out[(w1, w7, r, it, wrap)] = mixed(y, k + res0 + wv * (nb - res0), h)
        else:
            for w in WGRID:                      # two passes are not linear in w; scalar only
                out[(w, w, r, it, wrap)] = mixed(y, smooth_field(va, p, w, r, it, wrap), h)
    return out


def main():
    S = {}
    for L in layouts():
        try:
            va, p = layout(L)
        except (SystemExit, FileNotFoundError) as e:
            print(f"  layout {L}: {e}", file=sys.stderr); continue
        S[L] = scores_for(va, p, CONFIGS)
    if not S:
        print("FINAL_SMOOTH_W1=0.7\nFINAL_SMOOTH_W7=0.7\nFINAL_SMOOTH_R=1\n"
              "FINAL_SMOOTH_IT=1\nFINAL_SMOOTH_WRAP=0")
        print("  no predictions found -- keeping the incumbent 0.7/r1/it1", file=sys.stderr); return

    for L, sc in S.items():
        print(f"\n  layout {L}: incumbent (0.7, r1, it1, no wrap) = {sc[BASE]:.4f}", file=sys.stderr)
        print(f"  {'w1':>5} {'w7':>5} {'r':>2} {'it':>3} {'wrap':>5}   testmix     vs incumbent",
              file=sys.stderr)
        for cfg in sorted(sc, key=lambda c: sc[c])[:8]:
            print(f"  {cfg[0]:5.2f} {cfg[1]:5.2f} {cfg[2]:2d} {cfg[3]:3d} {str(cfg[4]):>5}   "
                  f"{sc[cfg]:.4f}     {sc[cfg]-sc[BASE]:+.5f}", file=sys.stderr)

    print("\n  leave-one-layout-out choice:", file=sys.stderr)
    best, _, _, lines = loo_grid(S, BASE, THRESH)
    for ln in lines:
        print(ln, file=sys.stderr)
    w1, w7, r, it, wrap = best
    print(f"  chosen: w1={w1} w7={w7} radius={r} iters={it} wrap={wrap}", file=sys.stderr)
    print(f"FINAL_SMOOTH_W1={w1}\nFINAL_SMOOTH_W7={w7}\nFINAL_SMOOTH_R={r}\n"
          f"FINAL_SMOOTH_IT={it}\nFINAL_SMOOTH_WRAP={1 if wrap else 0}")


if __name__ == "__main__":
    main()
