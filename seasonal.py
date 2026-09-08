"""Per-calendar-month bias correction (session 9s).

Why this exists
---------------
analyze_A's target-month table shows the submitted blend carrying large, systematic, month-shaped
biases: June +0.2912, October -0.1262, November -0.0774, February +0.0600. Removing all of them
would be worth -0.0060 RMSE on layout A -- the same size as the remaining gap to the 0.69 target,
and four times the best case the band offset could ever give.

Two things make this worth building where the band offset was not.

1. IT IS PERIODIC BY CONSTRUCTION. The band offset failed because it is temporally white:
   corr(previous month, this month) = +0.190, so nothing at time <= t predicts it. A calendar-month
   effect is the opposite -- June recurs every year, so an estimate from other years is a legal
   prediction for this one. That is a property of the quantity, not a hope about it.

2. IT EXPLAINS WHY postcal FAILED. postcal fitted coefficients per HORIZON, and in these layouts
   horizon is confounded with calendar month: layout A's 7-month block contributes its h7 rows from
   exactly one target month, its h6 rows from one other, and so on. Fitting "per horizon" on layout
   A is really fitting "per those particular months", and the test then applies them to different
   months entirely. Its leave-one-layout-out gate could not see this because both layouts share the
   confound. Indexing by the month itself removes it.

Why the model has not already absorbed it
-----------------------------------------
`m_next` IS in the feature set, so this is not a missing input. But a gradient-boosted tree
optimises local squared error, and a small additive offset spread evenly over every cell in a month
buys almost no split gain anywhere while costing real MSE in aggregate. An additive correction
expresses in one number what the tree would need many splits to approximate, and it is fitted on
exactly the statistic that is wrong.

Honesty
-------
Estimated on the OTHER layouts and scored on the held-out one, so no month's coefficient is ever
tested on the rows that produced it. Months the fitting layouts do not cover get zero rather than a
guess. The estimate is shrunk toward zero and clipped, and nothing is adopted unless every held-out
gain clears the usual 0.0003. The table prints how many distinct month-years back each coefficient,
because a "seasonal" effect measured from one June is not a seasonal effect.

stdout: FINAL_SEASON=m:b,m:b,...    stderr: the table.

usage: python seasonal.py lgb_v5x_noll:_bw,xgb_v5x_noll:_bw
env:   SEASON_LAM (shrinkage toward zero, default 0.5), SMOOTH_* as in postcal
"""
import os
import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load
from smooth import smooth_field, horizon_w
from xfit import layouts

SPEC = (sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll,xgb_v5x_noll").split(",")
W = test_mix()
LAM = float(os.environ.get("SEASON_LAM", "0.5"))
CLIP = 0.25
THRESH = -0.0003
MINROWS = 2000
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def sm(va, p):
    w1 = float(os.environ.get("SMOOTH_W1", os.environ.get("SMOOTH_W", "0.7")))
    w7 = float(os.environ.get("SMOOTH_W7", os.environ.get("SMOOTH_W", "0.7")))
    r = int(os.environ.get("SMOOTH_R", "1")); it = int(os.environ.get("SMOOTH_IT", "1"))
    wrap = os.environ.get("SMOOTH_WRAP", "0") == "1"
    if max(w1, w7) <= 0:
        return p
    return smooth_field(va, p, horizon_w(va["horizon"].to_numpy(), w1, w7), r, it, wrap)


def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))


def main():
    D = {}
    for L in layouts():
        try:
            va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                                 columns=["lat", "lon", "time", "horizon", "tws_known", "target"])
            p = sm(va, load(L, SPEC))
        except (SystemExit, FileNotFoundError) as e:
            print(f"  layout {L}: {e}", file=sys.stderr); continue
        t = va["time"].to_list()
        # the TARGET month is t+1, which is what the bias is a property of
        mn = np.array([(d.month % 12) + 1 for d in t])
        yr = np.array([d.year + (1 if d.month == 12 else 0) for d in t])
        D[L] = (va["target"].to_numpy(), p, va["horizon"].to_numpy(), mn, yr)
    if not D:
        print("FINAL_SEASON=")
        print("  no predictions found", file=sys.stderr); return

    def fit(Ls):
        """bias per target calendar month, pooled over the given layouts, shrunk and clipped."""
        b = np.zeros(13); n = np.zeros(13, dtype=int); ny = {m: set() for m in range(1, 13)}
        for L in Ls:
            y, p, h, mn, yr = D[L]
            for m in range(1, 13):
                k = mn == m
                if k.sum() == 0:
                    continue
                b[m] += float(np.sum(p[k] - y[k])); n[m] += int(k.sum())
                ny[m] |= set(yr[k].tolist())
        out = np.zeros(13)
        for m in range(1, 13):
            if n[m] >= MINROWS:
                out[m] = float(np.clip(LAM * b[m] / n[m], -CLIP, CLIP))
        return out, n, ny

    print("\n  bias by TARGET calendar month, per layout "
          "(month-years behind each estimate in brackets):", file=sys.stderr)
    print("  layout   " + "".join(f"{m:>8}" for m in range(1, 13)), file=sys.stderr)
    for L in D:
        b, n, ny = fit([L])
        print(f"    {L}    " + "".join(
            (f"{b[m]/max(LAM,1e-9):+8.3f}" if n[m] >= MINROWS else "       .")
            for m in range(1, 13)), file=sys.stderr)
        print(f"     n/yrs " + "".join(
            (f"{len(ny[m]):>8}" if n[m] >= MINROWS else "       .") for m in range(1, 13)),
            file=sys.stderr)

    Ls = list(D)
    if len(Ls) < 2:
        print("FINAL_SEASON=")
        print("  a seasonal effect cannot be cross-validated on one layout -- not adopted",
              file=sys.stderr)
        return

    # leave one layout out: the coefficients never see the rows they are scored on
    print("\n  held out (coefficients from the OTHER layouts):", file=sys.stderr)
    best = {}
    for lam in GRID:
        gains = {}
        for L in Ls:
            others = [o for o in Ls if o != L]
            b, n, _ = fit(others)
            y, p, h, mn, _ = D[L]
            corr = lam / max(LAM, 1e-9) * b[np.clip(mn, 1, 12)]
            gains[L] = mixed(y, p - corr, h) - mixed(y, p, h)
        best[lam] = gains
        print(f"    lambda={lam:.2f}   " + "   ".join(f"{L} {gains[L]:+.5f}" for L in Ls),
              file=sys.stderr)

    ok = {lam: g for lam, g in best.items() if lam > 0 and all(v < THRESH for v in g.values())}
    if not ok:
        print(f"\n  no shrinkage clears {abs(THRESH):.4f} on every held-out layout -- not adopted",
              file=sys.stderr)
        print("FINAL_SEASON="); return
    lam = min(ok, key=lambda l: sum(ok[l].values()) / len(Ls))
    b, n, ny = fit(Ls)
    b = b * (lam / max(LAM, 1e-9))
    print(f"\n  adopted at lambda={lam:.2f}, held-out gains "
          f"{[f'{v:+.5f}' for v in best[lam].values()]}", file=sys.stderr)
    print("    " + "  ".join(f"m{m}={b[m]:+.3f}" for m in range(1, 13) if b[m] != 0),
          file=sys.stderr)
    print("FINAL_SEASON=" + ",".join(f"{m}:{b[m]:.4f}" for m in range(1, 13) if b[m] != 0))


if __name__ == "__main__":
    main()
