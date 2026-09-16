import numpy as np

from ar_blend import LAYOUTS, layout_data, mixmse, fit

K = ["p", "k", "c", "ar"]

def apply_perh(w_by_h, d, pooled):
    X = np.stack([d[x] for x in K])
    out = pooled @ X
    for i in range(1, 8):
        m = d["h"] == i
        if m.any() and i in w_by_h:
            out[m] = w_by_h[i] @ X[:, m]
    return out

def main():
    D = {L: layout_data(L) for L in LAYOUTS}
    print("leave-one-layout-out: pooled blend vs per-horizon blend")
    ds = []
    for L in D:
        tr = [D[x] for x in D if x != L]
        pooled = fit(tr, K)
        w_by_h = {}
        for i in range(1, 8):
            sub = [{k: v[d["h"] == i] for k, v in d.items()} for d in tr]
            sub = [s for s in sub if len(s["y"]) > 5000]
            if sub:
                w_by_h[i] = fit(sub, K)
        d = D[L]
        a = np.sqrt(mixmse(d["y"], d["h"], pooled @ np.stack([d[x] for x in K])))
        b = np.sqrt(mixmse(d["y"], d["h"], apply_perh(w_by_h, d, pooled)))
        ds.append(b - a)
        print(f"  {L:6} {a:.4f} -> {b:.4f}   {b-a:+.4f}")
    ds = np.array(ds)
    print(f"  mean {ds.mean():+.4f}  worst {ds.max():+.4f}  wins {int((ds < 0).sum())}/5")
    print("\n  per-horizon weights fitted on all five layouts:")
    for i in range(1, 8):
        sub = [{k: v[d["h"] == i] for k, v in d.items()} for d in D.values()]
        sub = [s for s in sub if len(s["y"]) > 5000]
        if sub:
            print(f"    h{i}: {np.round(fit(sub, K), 3)}")

if __name__ == "__main__":
    main()
