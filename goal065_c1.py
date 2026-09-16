import sys
import time

import numpy as np
import polars as pl

from features_x import ANWKEYS

RADII = (2, 8)

def main():
    L = sys.argv[1]; t0 = time.time()
    for part in ("tr", "va"):
        p = f"out/mats/{L}_{part}.parquet"
        cols = pl.scan_parquet(p).collect_schema().names()
        keys = [k for k in ANWKEYS if k in cols]
        m = pl.read_parquet(p, columns=["lat", "lon", "time", "t_known"] + keys)
        pairs = m.select(["time", "t_known"]).unique().sort(["time", "t_known"]).with_row_index("pid")
        rr = m.join(pairs, on=["time", "t_known"], how="left")
        pid = rr["pid"].to_numpy()
        li = (rr["lat"].to_numpy() + 89.5).round().astype(int)
        lo = (rr["lon"].to_numpy() + 179.5).round().astype(int)
        n = len(pairs); out = {}
        from scipy.ndimage import uniform_filter
        for rad in RADII:
            size = (1, 2 * rad + 1, 2 * rad + 1); k2 = size[1] * size[2]
            for k in keys:
                v = rr[k].to_numpy().astype(np.float32); ok = np.isfinite(v)
                arr = np.zeros((n, 180, 360), np.float32); arr[pid[ok], li[ok], lo[ok]] = v[ok]
                c = np.zeros((n, 180, 360), np.float32); c[pid[ok], li[ok], lo[ok]] = 1.0
                S = uniform_filter(arr, size=size, mode=("constant", "constant", "wrap")) * k2
                C = uniform_filter(c, size=size, mode=("constant", "constant", "wrap")) * k2
                mth = np.where(C > 0.5, S / np.maximum(C, 1e-6), np.nan)
                out[f"c1_{k}_r{rad}"] = mth[pid, li, lo].astype(np.float32)
        pl.DataFrame(out).write_parquet(f"out/mats/{L}_{part}_c1.parquet")
        print(f"{L} {part}: {len(m)} rows, {len(out)} cols from {len(keys)} keys x {len(RADII)} radii "
              f"({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
