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
BASE = (0.7, 0.7, 1, 1, False)
THRESH = -0.0003

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
            for w in WGRID:
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
