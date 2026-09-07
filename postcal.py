"""Per-horizon calibration of the predicted change (session 9m).

The gap this closes
-------------------
Every model here predicts a residual, target - tws_known, and that residual is added back
unscaled. Its MAGNITUDE has never been checked. Under a regime shift -- and this competition is
one, 2002-2015 training against a 2015-2018 test that begins in the GRACE/GRACE-FO gap -- a
squared-error learner trained on one era systematically mis-scales its predicted change on the
next. Too confident and every cell overshoots; too timid and it collapses onto persistence.

Fitting one or two numbers per horizon fixes that without touching the model:

    scale     p' = tws_known + a_h * (p - tws_known)
    affine    p' = tws_known + a_h * (p - tws_known) + b_h

a_h < 1 shrinks the change toward the last observation, a_h > 1 amplifies it. Each is a single
scalar per horizon estimated from ~50k validation rows, so neither can chase noise the way a
per-cell correction could, and both are exactly the quantities a regime shift breaks.

The offset b_h exists because the first full grid (session 9d, run 2026-09-07) showed every
configuration OVER-predicting: mean prediction minus truth was +0.011 on layout A and +0.055 on
layout B, on every one of the seven experiments. A bias that large is worth several thousandths of
RMSE. But it is five times larger on B than on A, so it is a property of the era rather than of
the model, and a fixed offset fitted on one era is not obviously the right one for another. That
is a question for evidence, not for judgement: both candidates are fitted, both are scored on a
held-out layout, and whichever wins there -- possibly neither -- is the one that ships.

Honesty
-------
a is fitted on ONE layout and scored on the OTHER, so the number that decides adoption is out of
sample. On top of that the estimate is pulled halfway back to 1 (LAM) and clipped, because the
test era is a third regime and neither layout is it. Nothing is adopted unless the cross-fitted
gain beats 0.0003 on BOTH layouts, the same bar every other session-9 decision has had to clear.

Calibration is fitted AFTER smoothing, because that is the order final_assemble.py applies them.

stdout: FINAL_CALIB=a1,..,a7 and FINAL_CALIB_B=b1,..,b7 (empty unless the affine form won).
stderr: the table.

usage: python postcal.py lgb_v5x_noll:_bw,xgb_v5x_noll:_bw
env:   SMOOTH_W1 SMOOTH_W7 SMOOTH_R SMOOTH_IT SMOOTH_WRAP  (the chosen smoothing, applied first)
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
LAM = float(os.environ.get("CALIB_LAM", "0.5"))     # keep this much of the estimated departure
LO, HI = 0.80, 1.20
BLO, BHI = -0.10, 0.10       # an offset larger than this is a modelling failure, not a calibration
THRESH = -0.0003
MINN = 2000
CANDS = ("scale", "affine")


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


def fit(y, k, p, h, form="scale"):
    """Per-horizon least squares -- through the origin for 'scale', with an intercept for
    'affine' -- then shrunk toward the no-op and clipped."""
    a = np.ones(8); b = np.zeros(8)
    for i in range(1, 8):
        m = h == i
        if m.sum() < MINN:
            continue
        d = p[m] - k[m]; r = y[m] - k[m]
        if form == "affine":
            dm = d.mean(); rm = r.mean()
            v = float((d - dm) @ (d - dm))
            if v <= 0:
                continue
            s = float((d - dm) @ (r - rm)) / v
            a[i] = float(np.clip(1.0 + LAM * (s - 1.0), LO, HI))
            b[i] = float(np.clip(LAM * (rm - a[i] * dm), BLO, BHI))
        else:
            dd = float(d @ d)
            if dd <= 0:
                continue
            a[i] = float(np.clip(1.0 + LAM * (float(r @ d) / dd - 1.0), LO, HI))
    return a, b


def apply(p, k, h, a, b=None):
    j = np.clip(h, 1, 7)
    return k + a[j] * (p - k) + (0.0 if b is None else b[j])


def mean_coef(fits):
    """Average a set of per-layout fits, keeping each coefficient inside its bounds."""
    a = np.clip(np.mean([f[0] for f in fits], axis=0), LO, HI)
    b = np.clip(np.mean([f[1] for f in fits], axis=0), BLO, BHI)
    a[0] = 1.0; b[0] = 0.0
    return a, b


def main():
    D = {}
    for L in layouts():
        try:
            va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                                 columns=["lat", "lon", "time", "horizon", "tws_known", "target"])
            p = sm(va, load(L, SPEC))
        except (SystemExit, FileNotFoundError) as e:
            print(f"  layout {L}: {e}", file=sys.stderr); continue
        D[L] = (va["target"].to_numpy(), va["tws_known"].to_numpy(), p, va["horizon"].to_numpy())
    if not D:
        print("FINAL_CALIB=\nFINAL_CALIB_B=")
        print("  no predictions found -- no calibration", file=sys.stderr); return

    Ls = list(D)
    F = {c: {L: fit(*D[L], form=c) for L in Ls} for c in CANDS}
    for L in Ls:
        y, k, p, h = D[L]
        print(f"\n  layout {L}: mean prediction - truth = {float(np.mean(p - y)):+.4f}",
              file=sys.stderr)
        print("    " + " ".join(f"h{i}" for i in range(1, 8)), file=sys.stderr)
        for c in CANDS:
            a, b = F[c][L]
            print(f"    scale  ({c}) " + " ".join(f"{a[i]:5.3f}" for i in range(1, 8)),
                  file=sys.stderr)
            if c == "affine":
                print("    offset (affine) " + " ".join(f"{b[i]:+.3f}" for i in range(1, 8)),
                      file=sys.stderr)

    # held-out scores: coefficients from the OTHER layouts, applied to this one
    held = {c: {} for c in CANDS}
    for c in CANDS:
        for L in Ls:
            others = [o for o in Ls if o != L] or [L]
            a, b = mean_coef([F[c][o] for o in others])
            y, k, p, h = D[L]
            held[c][L] = mixed(y, apply(p, k, h, a, b), h) - mixed(y, p, h)
        src = "the other layouts" if len(Ls) > 1 else "ITSELF (in sample)"
        print(f"\n  {c}: fitted on {src}, scored held out: "
              + "  ".join(f"{L} {held[c][L]:+.5f}" for L in Ls), file=sys.stderr)

    good = [c for c in CANDS if all(g < THRESH for g in held[c].values())]
    if not good or len(Ls) == 1 and not good:
        print(f"  no form clears {abs(THRESH):.4f} out of sample everywhere -- no calibration",
              file=sys.stderr)
        print("FINAL_CALIB=\nFINAL_CALIB_B="); return
    best = min(good, key=lambda c: sum(held[c].values()) / len(Ls))
    a, b = mean_coef([F[best][L] for L in Ls])
    ins = []
    for L in Ls:
        y, k, p, h = D[L]
        ins.append(mixed(y, apply(p, k, h, a, b), h) - mixed(y, p, h))
    if len(Ls) == 1:
        print("  only one layout available; this is IN sample, treat with suspicion",
              file=sys.stderr)
    print(f"\n  adopted the {best} form", file=sys.stderr)
    print("    scale  " + " ".join(f"{a[i]:5.3f}" for i in range(1, 8)), file=sys.stderr)
    if best == "affine":
        print("    offset " + " ".join(f"{b[i]:+.3f}" for i in range(1, 8)), file=sys.stderr)
    print(f"    on each layout: {[f'{g:+.5f}' for g in ins]}", file=sys.stderr)
    print("FINAL_CALIB=" + ",".join(f"{a[i]:.4f}" for i in range(1, 8)))
    print("FINAL_CALIB_B=" + ("" if best != "affine"
                              else ",".join(f"{b[i]:.4f}" for i in range(1, 8))))


if __name__ == "__main__":
    main()
