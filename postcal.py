import os
import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load
from smooth import smooth_field, horizon_w
from xfit import layouts

SPEC = (sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll,xgb_v5x_noll").split(",")
W = test_mix()
LAM = float(os.environ.get("CALIB_LAM", "0.5"))
LO, HI = 0.80, 1.20
BLO, BHI = -0.10, 0.10
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

    j = np.clip(h, 1, 7).astype(int)
    return k + a[j] * (p - k) + (0.0 if b is None else b[j])

def mean_coef(fits):
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
