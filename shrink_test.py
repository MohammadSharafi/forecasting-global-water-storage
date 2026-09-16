import numpy as np
from scipy.optimize import minimize

from ar_blend import LAYOUTS, layout_data, mixmse

K = ["p", "k", "c", "ar"]

def fit(train, keys, sum1, budget=2.0):
    def obj(w):
        return sum(mixmse(d["y"], d["h"], w @ np.stack([d[x] for x in keys])) for d in train) / len(train)
    cons = [{"type": "ineq", "fun": lambda w: budget - np.abs(w).sum()}]
    if sum1:
        cons.append({"type": "eq", "fun": lambda w: w.sum() - 1})
    w0 = np.zeros(len(keys)); w0[0] = 1.0
    return minimize(obj, w0, bounds=[(-3, 3)] * len(keys), constraints=cons,
                    method="SLSQP", options={"ftol": 1e-16, "maxiter": 3000}).x

def main():
    D = {L: layout_data(L) for L in LAYOUTS}
    print("leave-one-layout-out: sum-to-one blend  vs  free-sum blend")
    ds = []
    for L in D:
        tr = [D[x] for x in D if x != L]
        wa, wb = fit(tr, K, True), fit(tr, K, False)
        d = D[L]; X = np.stack([d[x] for x in K])
        a = np.sqrt(mixmse(d["y"], d["h"], wa @ X))
        b = np.sqrt(mixmse(d["y"], d["h"], wb @ X))
        ds.append(b - a)
        print(f"  {L:6} {a:.4f} -> {b:.4f}   {b - a:+.4f}   sum(w)={wb.sum():.3f}")
    ds = np.array(ds)
    print(f"  mean {ds.mean():+.4f}  worst {ds.max():+.4f}  wins {int((ds < 0).sum())}/5")
    w = fit(list(D.values()), K, False)
    print(f"\n  free-sum weights over all five: {np.round(w, 4)}  sum={w.sum():.4f}")

if __name__ == "__main__":
    main()
