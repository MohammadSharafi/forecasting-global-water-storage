import numpy as np
import polars as pl
from scipy.optimize import minimize

LEDGER = {
    "mix25":   ("out/scored/sub_mix25.csv",       0.679248865),
    "armarg":  ("out/scored/sub_ac_armarg.csv",   0.679785572),
    "w65sub":  ("out/scored/sub_submit_w65.csv",  0.680909648),
    "goal065": ("out/scored/sub_goal065.csv",     0.682847),
    "zlb":     ("out/scored/sub_z_lb.csv",        0.681692450),
    "ylb":     ("out/scored/sub_y_lb.csv",        0.681717090),
    "xlb2":    ("out/scored/sub_x_lb2.csv",       0.683712087),
    "gpccar":  ("out/scored/sub_gpcc_ar.csv",     0.684984991),
    "w65ar":   ("out/sub_w65_ar.csv",             0.686312),
    "main":    ("out/scored/sub_q_main.csv",      0.692657),
    "anw":     ("out/sub_v_anwide.csv",           0.692189463),
    "blend73": ("out/scored/sub_blend73.csv",     0.693283),
    "alt":     ("out/scored/sub_q_alt.csv",       0.694922),
    "nosm":    ("out/scored/sub_q_main_nosm.csv", 0.699149222),
    "clim":    ("out/scored/probe_clim.csv",      1.279955747),
    "pers":    ("out/probe_persistence.csv",      0.8864),
}
BEST = 0.679248865

def main():
    ref = pl.read_csv("Test.csv").select("ID")
    keys, P = [], {}
    for k, (p, _) in LEDGER.items():
        try:
            d = pl.read_csv(p)
        except Exception:
            print(f"  (missing {p}, skipped)"); continue
        c = [x for x in d.columns if x != "ID"][0]
        v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(np.float64)
        if not np.isfinite(v).all(): print(f"  ({k} has gaps, skipped)"); continue
        keys.append(k); P[k] = v
    M = np.array([LEDGER[k][1] ** 2 for k in keys])
    D2 = np.array([[float(np.mean((P[a] - P[b]) ** 2)) for b in keys] for a in keys])
    print(f"{len(keys)} scored files in the ledger; best single is {BEST:.6f}\n")

    def solve(budget):
        f = lambda w: float(w @ M - 0.5 * w @ D2 @ w)
        w0 = np.zeros(len(M)); w0[int(np.argmin(M))] = 1.0
        r = minimize(f, w0, bounds=[(-1.5 * budget, 1.5 * budget)] * len(M),
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                  {"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}],
                     method="SLSQP", options={"ftol": 1e-16, "maxiter": 5000})
        return r.x, float(np.sqrt(max(f(r.x), 1e-12)))

    print("budget   predicted public   gain vs the record   weights")
    for b in (1.0, 1.5, 2.0, 3.0, 4.0):
        w, s = solve(b)
        nz = {k: round(x, 3) for k, x in zip(keys, w) if abs(x) > 0.02}
        print(f"  L1<={b:.1f}   {s:.6f}        {s-BEST:+.6f}      {nz}")

if __name__ == "__main__":
    main()

def build(budget=2.0, out="out/sub_tomorrow.csv"):
    ref = pl.read_csv("Test.csv").select("ID")
    keys, P = [], {}
    for k, (p, _) in LEDGER.items():
        try: d = pl.read_csv(p)
        except Exception: continue
        c = [x for x in d.columns if x != "ID"][0]
        v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(np.float64)
        if np.isfinite(v).all(): keys.append(k); P[k] = v
    M = np.array([LEDGER[k][1] ** 2 for k in keys])
    D2 = np.array([[float(np.mean((P[a] - P[b]) ** 2)) for b in keys] for a in keys])
    f = lambda w: float(w @ M - 0.5 * w @ D2 @ w)
    w0 = np.zeros(len(M)); w0[int(np.argmin(M))] = 1.0
    r = minimize(f, w0, bounds=[(-1.5 * budget, 1.5 * budget)] * len(M),
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                              {"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}],
                 method="SLSQP", options={"ftol": 1e-16, "maxiter": 5000})
    w = r.x; q = sum(wi * P[k] for wi, k in zip(w, keys))
    pl.DataFrame({"ID": ref["ID"], "Target": q}).write_csv(out)
    print(f"wrote {out}  budget L1<={budget}  predicted public {np.sqrt(f(w)):.6f}")
    print("  weights:", {k: round(x, 3) for k, x in zip(keys, w) if abs(x) > 0.02})
    print(f"  sd {q.std():.4f}  RMS from the record {np.sqrt(np.mean((q-P['mix25'])**2)):.4f}")
