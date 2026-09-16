import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load
from xfit import layouts

INC = [x.split(":") for x in (sys.argv[1] if len(sys.argv) > 1
                              else "lgb_v5x_noll:0.5,xgb_v5x_noll:0.5").split(",")]
TAG = sys.argv[2] if len(sys.argv) > 2 else "_bw"
EXTRA = [x for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else []) if x]
W = test_mix()
SHRINK = 0.5
THRESH = -0.0003

def rowweights(h):
    w = np.zeros(len(h), dtype=float)
    for i in range(1, 8):
        m = h == i
        if m.any():
            w[m] = W[i - 1] / m.sum()
    return w / w.sum()

def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))

def nnls(A, b):
    try:
        from scipy.optimize import nnls as _n
        return _n(A, b)[0]
    except Exception:
        G = A.T @ A; c = A.T @ b
        x = np.full(A.shape[1], 1.0 / A.shape[1])
        step = 1.0 / (np.linalg.eigvalsh(G).max() + 1e-9)
        for _ in range(5000):
            x = np.maximum(0.0, x - step * (G @ x - c))
        return x

def main():
    LS = layouts()
    stems = [s for s, _ in INC] + [s for s in EXTRA if s not in [t for t, _ in INC]]
    D = {}
    for L in LS:
        va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "tws_known", "target"])
        y = va["target"].to_numpy(); h = va["horizon"].to_numpy(); k = va["tws_known"].to_numpy()
        cols, names = [], []
        for s in stems:
            try:
                cols.append(load(L, [f"{s}:{TAG}"])); names.append(s)
            except (SystemExit, FileNotFoundError):
                pass
        if len(cols) >= 2:
            D[L] = (y, h, k, np.column_stack(cols), names)
        else:
            print(f"  layout {L}: fewer than two families present, skipped", file=sys.stderr)
    if not D:
        print("FINAL_STACK=")
        print("  no predictions -- keeping the incumbent blend", file=sys.stderr); return

    common = [n for n in stems if all(n in D[L][4] for L in D)]
    if len(common) < 2:
        print("FINAL_STACK=")
        print("  the layouts do not share two families -- keeping the incumbent", file=sys.stderr)
        return
    print(f"  families on every layout: {common}", file=sys.stderr)
    for L in D:
        y, h, k, P, names = D[L]
        idx = [names.index(n) for n in common]
        D[L] = (y, h, k, P[:, idx], common)

    iw = {s: float(w) for s, w in INC}
    inc = np.array([iw.get(s, 0.0) for s in common])
    inc = inc / inc.sum() if inc.sum() > 0 else np.full(len(common), 1.0 / len(common))

    def fit(Ls):
        A, b = [], []
        for L in Ls:
            y, h, k, P, _ = D[L]
            sw = np.sqrt(rowweights(h) / len(Ls))
            A.append((P - k[:, None]) * sw[:, None]); b.append((y - k) * sw)
        a = nnls(np.vstack(A), np.concatenate(b))
        if a.sum() <= 0:
            return inc.copy()
        a = a / a.sum()
        return SHRINK * a + (1 - SHRINK) * inc

    def score(L, a):
        y, h, k, P, _ = D[L]
        return mixed(y, k + (P - k[:, None]) @ a, h)

    print(f"\n  {'weights':>52}   " + "  ".join(f"{L}" for L in D), file=sys.stderr)
    fmt = lambda a: " ".join(f"{n.split('_')[0]}:{v:.2f}" for n, v in zip(common, a))
    print(f"  incumbent {fmt(inc):>42}   " + "  ".join(f"{score(L, inc):.4f}" for L in D),
          file=sys.stderr)
    a_all = fit(list(D))
    print(f"  fitted    {fmt(a_all):>42}   " + "  ".join(f"{score(L, a_all):.4f}" for L in D),
          file=sys.stderr)

    if len(D) >= 2:
        held = {}
        for L in D:
            others = [o for o in D if o != L]
            a = fit(others)
            held[L] = score(L, a) - score(L, inc)
            print(f"  fitted on {'+'.join(others)}, scored on {L}: {held[L]:+.5f}", file=sys.stderr)
        if not all(g < THRESH for g in held.values()):
            print(f"  held-out gain does not clear {abs(THRESH):.4f} everywhere -- "
                  f"keeping the incumbent blend", file=sys.stderr)
            print("FINAL_STACK=" + ",".join(f"{s}:{w:.3f}" for s, w in zip(common, inc)))
            return
    else:
        L = next(iter(D))
        if score(L, a_all) - score(L, inc) >= THRESH:
            print("  single layout and no measurable gain -- keeping the incumbent", file=sys.stderr)
            print("FINAL_STACK=" + ",".join(f"{s}:{w:.3f}" for s, w in zip(common, inc)))
            return
        print(f"  only layout {L}; this is IN sample, treat with suspicion", file=sys.stderr)

    gains = [score(L, a_all) - score(L, inc) for L in D]
    print(f"  adopted, gains {[f'{g:+.5f}' for g in gains]}", file=sys.stderr)
    print("FINAL_STACK=" + ",".join(f"{s}:{w:.3f}" for s, w in zip(common, a_all)))

if __name__ == "__main__":
    main()
