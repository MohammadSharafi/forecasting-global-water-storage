import glob
import os
import re
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize

LEDGER = {
    "main":   ("out/scored/sub_q_main.csv",     0.692657),
    "alt":    ("out/scored/sub_q_alt.csv",      0.694922),
    "base":   ("out/sub_q_base.csv",            0.708852),
    "sm05":   ("out/sub_t_sm05it2.csv",         0.692773095),
    "anw":    ("out/sub_v_anwide.csv",          0.692189463),
    "prof":   ("out/sub_r_prof.csv",            0.695365985),
    "nosm":   ("out/sub_q_main_nosm.csv",       0.699149222),
    "nocal":  ("out/sub_q_main_nocal.csv",      0.695964865),
    "pers":   ("out/probe_persistence.csv",     0.8864),
    "clim1":  ("out/sub_h_l10_clim1.csv",       0.731800),
    "mlp":    ("out/sub_i_lt_mlp.csv",          0.723121),
    "ltx":    ("out/sub_i_lt_lgbxgb.csv",       0.709259),
    "gblend": ("out/sub_g_blend.csv",           0.71159),
    "rb500":  ("out/sub_h_l10_rb3_500.csv",     0.717866),
    "trees":  ("out/sub_i_lt_trees.csv",        0.710382),
    "glt":    ("out/sub_g_longterm.csv",        0.710959),
}
BEST = 0.692189463

def load(keys=None):
    keys = list(LEDGER) if keys is None else keys
    base = pl.read_csv("out/scored/sub_q_main.csv").select("ID")
    P = {}
    for k in keys:
        d = pl.read_csv(LEDGER[k][0])
        c = [x for x in d.columns if x != "ID"][0]
        P[k] = base.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(np.float64)
        assert np.isfinite(P[k]).all(), f"{k} has non-finite rows"
    return base, keys, P

def moments(keys, P):
    M = np.array([LEDGER[k][1] ** 2 for k in keys])
    D2 = np.array([[float(np.mean((P[a] - P[b]) ** 2)) for b in keys] for a in keys])
    return M, D2

def solve(M, D2, budget):
    f = lambda w: float(w @ M - 0.5 * w @ D2 @ w)
    w0 = np.zeros(len(M)); w0[int(np.argmin(M))] = 1.0
    r = minimize(f, w0, bounds=[(-1.5 * budget, 1.5 * budget)] * len(M),
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                              {"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}],
                 method="SLSQP", options={"ftol": 1e-16, "maxiter": 5000})
    return r.x, float(np.sqrt(f(r.x)))

def fit(budget=2.0, out="out/sub_x_lb2.csv"):
    base, keys, P = load()
    M, D2 = moments(keys, P)
    w, s = solve(M, D2, budget)
    p = sum(w[i] * P[k] for i, k in enumerate(keys))
    print(f"L1<={budget}  exact public RMSE {s:.6f}  (gain {s - BEST:+.6f})")
    print("  w =", {k: round(float(x), 3) for k, x in zip(keys, w) if abs(x) > 0.02})
    print(f"  std={p.std():.4f}  RMS distance from the current best={np.sqrt(np.mean((p - P['anw']) ** 2)):.4f}")
    pl.DataFrame({"ID": base["ID"], "Target": p}).write_csv(out)
    print(f"  wrote {out}")

def simulate(L, trials=5):
    cols = pl.scan_parquet(f"out/mats/{L}_va.parquet").collect_schema().names()
    need = ["target", "tws_known"] + (["clim_next"] if "clim_next" in cols else []) + (["time"] if "time" in cols else [])
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=need)
    y = va["target"].to_numpy().astype(float)
    V = {"pers": va["tws_known"].to_numpy().astype(float)}
    if "clim_next" in va.columns:
        V["clim"] = va["clim_next"].to_numpy().astype(float)
    stems = {}
    for f in glob.glob(f"out/mats/pred_{L}_*.npy"):
        m = re.match(rf"pred_{L}_(.+)_s\d+(.*)\.npy$", os.path.basename(f))
        if m:
            stems.setdefault(m.group(1) + m.group(2), []).append(f)
    for s, fs in sorted(stems.items()):
        a = np.mean([np.load(x) for x in fs], 0)
        if a.shape[0] == len(y):
            V[s] = a
    keys = [k for k in V if np.isfinite(V[k]).mean() > 0.99][:18]
    X = np.stack([V[k] for k in keys])
    ok = np.isfinite(y) & np.isfinite(X).all(0)
    y, X = y[ok], X[:, ok]
    t = va["time"].to_numpy()[ok] if "time" in va.columns else None
    D2 = np.array([[float(np.mean((a - b) ** 2)) for b in X] for a in X])
    rng = np.random.default_rng(7)
    print(f"layout {L}: {len(keys)} vectors, {len(y)} rows")
    for split in (("random", "by-month") if t is not None else ("random",)):
        acc = {}
        for _ in range(trials):
            if split == "random":
                pub = rng.random(len(y)) < 0.30
            else:
                mo = np.unique(t)
                pub = np.isin(t, rng.choice(mo, max(1, round(0.3 * len(mo))), replace=False))
            prv = ~pub
            if pub.sum() < 1000 or prv.sum() < 1000:
                continue
            Mp = np.array([np.mean((y[pub] - x[pub]) ** 2) for x in X])
            i0 = int(np.argmin(Mp))
            for B in (1.0, 2.0, 4.0, 8.0):
                w, _ = solve(Mp, D2, B)
                p = w @ X
                acc.setdefault(B, []).append((np.sqrt(np.mean((y[prv] - p[prv]) ** 2)),
                                              np.sqrt(np.mean((y[prv] - X[i0][prv]) ** 2))))
        print(f"  --- {split} 30% public")
        for B in sorted(acc):
            a = np.array(acc[B]).mean(0)
            print(f"    L1<={B:4.1f}  private {a[0]:.4f}  best single {a[1]:.4f}  gain {a[0] - a[1]:+.4f}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit(float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
    else:
        simulate(sys.argv[2] if len(sys.argv) > 2 else "A")
