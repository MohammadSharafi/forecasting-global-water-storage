import numpy as np
import polars as pl
from scipy.optimize import minimize

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])
TAGS = [("lgbd", "_cap"), ("xgb", "_d9"), ("mlp", "_nn"), ("mlp", "_nnB"), ("mlp", "_nnC"),
        ("gru", "_g1")]

def load(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    P = []
    for fam, tag in TAGS:
        P.append(np.load(f"out/mats/pred_{L}_{fam}_v5x_noll_s0{tag}.npy"))
    X = np.stack(P)
    ok = np.isfinite(y) & np.isfinite(X).all(0)
    return y[ok], h[ok], X[:, ok]

def mix(y, h, p):
    return float(np.sqrt(sum(MIX[i-1] * np.mean((y[h == i] - p[h == i]) ** 2)
                             for i in range(1, 8) if (h == i).any())))

def fit(y, h, X):
    f = lambda w: mix(y, h, w @ X) ** 2
    w0 = np.ones(len(X)) / len(X)
    r = minimize(f, w0, bounds=[(0, 1)] * len(X),
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                 method="SLSQP", options={"ftol": 1e-16, "maxiter": 3000})
    return r.x

def main():
    D = {L: load(L) for L in ("Avn2", "E")}
    names = [f"{a}{b}" for a, b in TAGS]
    cur = np.array([0.80*0.65*0.65, 0.80*0.65*0.35, 0.80*0.35/3, 0.80*0.35/3, 0.80*0.35/3, 0.20])
    print("current hand-set weights:", {n: round(w, 3) for n, w in zip(names, cur)})
    for L in D:
        y, h, X = D[L]
        print(f"  {L}: hand-set {mix(y, h, cur @ X):.4f}")
    print("\nheld-out test: fit the weights on one layout, score the other")
    for a, b in (("Avn2", "E"), ("E", "Avn2")):
        w = fit(*D[a])
        yb, hb, Xb = D[b]
        s_fit = mix(yb, hb, w @ Xb); s_cur = mix(yb, hb, cur @ Xb)
        print(f"  fitted on {a:5s} -> scored on {b:5s}: {s_cur:.4f} (hand-set) vs {s_fit:.4f} (fitted)"
              f"   {s_fit-s_cur:+.4f}")
        print(f"      weights from {a}: " + " ".join(f"{n}={x:.3f}" for n, x in zip(names, w) if x > 0.01))
    wa = fit(*D["Avn2"]); we = fit(*D["E"])
    avg = (wa + we) / 2
    print("\n  mean of the two fits, the weights that would ship:")
    print("      " + " ".join(f"{n}={x:.3f}" for n, x in zip(names, avg) if x > 0.01))
    for L in D:
        y, h, X = D[L]
        print(f"      {L}: hand-set {mix(y,h,cur@X):.4f}  ->  averaged fit {mix(y,h,avg@X):.4f}"
              f"   {mix(y,h,avg@X)-mix(y,h,cur@X):+.4f}")

if __name__ == "__main__":
    main()
