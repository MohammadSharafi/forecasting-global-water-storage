"""Per-horizon calibration of the predicted change (session 9m).

The gap this closes
-------------------
Every model here predicts a residual, target - tws_known, and that residual is added back
unscaled. Its MAGNITUDE has never been checked. Under a regime shift -- and this competition is
one, 2002-2015 training against a 2015-2018 test that begins in the GRACE/GRACE-FO gap -- a
squared-error learner trained on one era systematically mis-scales its predicted change on the
next. Too confident and every cell overshoots; too timid and it collapses onto persistence.

Fitting one number per horizon fixes that without touching the model:

    p' = tws_known + a_h * (p - tws_known)

a_h < 1 shrinks the change toward the last observation, a_h > 1 amplifies it. It is a single
scalar per horizon estimated from ~50k validation rows each, so it cannot chase noise the way a
per-cell correction could, and it is exactly the quantity a regime shift breaks.

Honesty
-------
a is fitted on ONE layout and scored on the OTHER, so the number that decides adoption is out of
sample. On top of that the estimate is pulled halfway back to 1 (LAM) and clipped, because the
test era is a third regime and neither layout is it. Nothing is adopted unless the cross-fitted
gain beats 0.0003 on BOTH layouts, the same bar every other session-9 decision has had to clear.

Calibration is fitted AFTER smoothing, because that is the order final_assemble.py applies them.

stdout: FINAL_CALIB=a1,..,a7    stderr: the table.

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
THRESH = -0.0003
MINN = 2000


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


def fit(y, k, p, h):
    """Least squares through the origin, per horizon, then shrunk toward 1 and clipped."""
    a = np.ones(8)
    for i in range(1, 8):
        m = h == i
        if m.sum() < MINN:
            continue
        d = p[m] - k[m]; r = y[m] - k[m]
        dd = float(d @ d)
        if dd <= 0:
            continue
        a[i] = float(np.clip(1.0 + LAM * (float(r @ d) / dd - 1.0), LO, HI))
    return a


def apply(p, k, h, a):
    return k + a[np.clip(h, 1, 7)] * (p - k)


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
        print("FINAL_CALIB=")
        print("  no predictions found -- no calibration", file=sys.stderr); return

    A = {L: fit(*D[L]) for L in D}
    print(f"\n  per-horizon scale of the predicted change (already pulled toward 1 by lam={LAM}):",
          file=sys.stderr)
    print("  layout   " + "   ".join(f"h{i}" for i in range(1, 8)), file=sys.stderr)
    for L in A:
        print(f"    {L}     " + " ".join(f"{A[L][i]:5.3f}" for i in range(1, 8)), file=sys.stderr)

    Ls = list(D)
    if len(Ls) == 1:
        L = Ls[0]; y, k, p, h = D[L]; a = A[L]
        if mixed(y, apply(p, k, h, a), h) - mixed(y, p, h) >= THRESH:
            print("  single layout and no measurable gain -- no calibration", file=sys.stderr)
            print("FINAL_CALIB="); return
        print(f"  only layout {L}; this is IN sample, treat with suspicion", file=sys.stderr)
    else:
        # fit on the other layouts, score on the held-out one: nothing chose its own coefficients
        held = {}
        for L in Ls:
            others = [o for o in Ls if o != L]
            ao = np.ones(8)
            for i in range(1, 8):
                ao[i] = float(np.clip(np.mean([A[o][i] for o in others]), LO, HI))
            y, k, p, h = D[L]
            held[L] = mixed(y, apply(p, k, h, ao), h) - mixed(y, p, h)
            print(f"  fitted on {'+'.join(others)}, scored on {L}: {held[L]:+.5f}", file=sys.stderr)
        a = np.ones(8)
        for i in range(1, 8):
            a[i] = float(np.clip(np.mean([A[L][i] for L in Ls]), LO, HI))
        if not all(g < THRESH for g in held.values()):
            print(f"  held-out gain does not clear {abs(THRESH):.4f} everywhere -- no calibration",
                  file=sys.stderr)
            print("FINAL_CALIB="); return
        ins = []
        for L in Ls:
            y, k, p, h = D[L]
            ins.append(mixed(y, apply(p, k, h, a), h) - mixed(y, p, h))
        print("  adopted: " + " ".join(f"{a[i]:5.3f}" for i in range(1, 8))
              + f"   (on each layout: {[f'{g:+.5f}' for g in ins]})", file=sys.stderr)
    print("FINAL_CALIB=" + ",".join(f"{a[i]:.4f}" for i in range(1, 8)))


if __name__ == "__main__":
    main()
