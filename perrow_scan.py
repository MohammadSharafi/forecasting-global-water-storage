import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load

SPEC = (sys.argv[1] if len(sys.argv) > 1 else "lgb_v5x_noll:_bw").split(",")
VALUES = [int(v) for v in (sys.argv[2:] or ["2", "3", "4"])]
BASE = VALUES[0]
W = test_mix()
THRESH = -0.0003

def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(W[ok] * mse[ok]) / np.sum(W[ok])))

def score(L):
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
    return mixed(va["target"].to_numpy(), load(L, SPEC), va["horizon"].to_numpy())

def main():
    rows = {}
    for b in ("A", "B", "C"):
        got = {}
        for v in VALUES:
            L = b if v == BASE else f"{b}p{v}"
            try:
                got[v] = score(L)
            except (SystemExit, FileNotFoundError):
                pass
        if len(got) >= 2:
            rows[b] = got
    if not rows:
        print(f"FINAL_PER_ROW={BASE}")
        print("  nothing to compare -- build the pN layouts first", file=sys.stderr); return

    print(f"\n  test-mix RMSE by training rows per cell-month (incumbent PER_ROW={BASE}):",
          file=sys.stderr)
    print("  layout  " + "".join(f"{'p'+str(v):>10}" for v in VALUES), file=sys.stderr)
    for b, got in rows.items():
        print(f"    {b}    " + "".join(f"{got[v]:10.4f}" if v in got else f"{'.':>10}"
                                       for v in VALUES), file=sys.stderr)
        print("         " + "".join(f"{got[v]-got[BASE]:+10.4f}" if v in got and BASE in got
                                    else f"{'':>10}" for v in VALUES), file=sys.stderr)

    best = BASE
    for v in VALUES:
        if v == BASE:
            continue
        gains = [got[v] - got[BASE] for got in rows.values() if v in got and BASE in got]
        if not gains or len(gains) < len(rows):
            print(f"  p{v}: not built on every layout -- ignored", file=sys.stderr); continue
        if all(g < THRESH for g in gains):
            if best == BASE or np.mean(gains) < bestgain:
                best, bestgain = v, float(np.mean(gains))
            print(f"  p{v}: wins on every layout, mean {np.mean(gains):+.5f}", file=sys.stderr)
        else:
            print(f"  p{v}: {[f'{g:+.5f}' for g in gains]} -- does not clear "
                  f"{abs(THRESH):.4f} everywhere", file=sys.stderr)
    print(f"  chosen PER_ROW={best}" + ("" if best != BASE else "  (incumbent kept)"),
          file=sys.stderr)
    print(f"FINAL_PER_ROW={best}")

if __name__ == "__main__":
    main()
