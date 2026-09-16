import os
import sys
import time

import numpy as np
import polars as pl

DIRKEYS = ["anom_known", "dev24", "an_e5PERz_acc", "an_e5Pz_acc", "an_SOIL_MOISTURE_tz_d"]
RADIUS = int(os.environ.get("DIR_R", "6"))

def _halfmeans(vals, pid, li, lo, npair, radius):
    from scipy.ndimage import uniform_filter
    out = {}
    ok = np.isfinite(vals)
    arr = np.zeros((npair, 180, 360), np.float32); arr[pid[ok], li[ok], lo[ok]] = vals[ok]
    cnt = np.zeros((npair, 180, 360), np.float32); cnt[pid[ok], li[ok], lo[ok]] = 1.0
    for name, (da0, da1, db0, db1) in (
            ("w", (-radius, radius, -radius, -1)), ("e", (-radius, radius, 1, radius)),
            ("s", (-radius, -1, -radius, radius)), ("n", (1, radius, -radius, radius))):
        S = np.zeros_like(arr); C = np.zeros_like(cnt)
        for da in range(da0, da1 + 1):
            for db in range(db0, db1 + 1):
                S += np.roll(np.roll(arr, -da, axis=1), -db, axis=2)
                C += np.roll(np.roll(cnt, -da, axis=1), -db, axis=2)
        out[name] = np.where(C > 0.5, S / np.maximum(C, 1e-6), np.nan)
    return out

def main():
    L = sys.argv[1]; t0 = time.time()
    for part in ("tr", "va"):
        p = f"out/mats/{L}_{part}.parquet"
        cols = pl.scan_parquet(p).collect_schema().names()
        keys = [k for k in DIRKEYS if k in cols]
        m = pl.read_parquet(p, columns=["lat", "lon", "time", "t_known"] + keys)
        pairs = m.select(["time", "t_known"]).unique().sort(["time", "t_known"]).with_row_index("pid")
        rr = m.join(pairs, on=["time", "t_known"], how="left")
        pid = rr["pid"].to_numpy()
        li = (rr["lat"].to_numpy() + 89.5).round().astype(int)
        lo = (rr["lon"].to_numpy() + 179.5).round().astype(int)
        n = len(pairs); out = {}
        for k in keys:
            H = _halfmeans(rr[k].to_numpy().astype(np.float32), pid, li, lo, n, RADIUS)
            we = H["w"][pid, li, lo] - H["e"][pid, li, lo]
            ns = H["n"][pid, li, lo] - H["s"][pid, li, lo]
            out[f"dw_{k}"] = H["w"][pid, li, lo].astype(np.float32)
            out[f"de_{k}"] = H["e"][pid, li, lo].astype(np.float32)
            out[f"dwe_{k}"] = we.astype(np.float32)
            out[f"dns_{k}"] = ns.astype(np.float32)
        pl.DataFrame(out).write_parquet(f"out/mats/{L}_{part}_dir.parquet")
        print(f"{L} {part}: {len(m)} rows, {len(out)} cols from {len(keys)} keys "
              f"({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
