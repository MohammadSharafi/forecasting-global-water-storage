"""Direction-split neighbourhood means, appended to the cached matrices in row order.

Why this is the last untested family
------------------------------------
`used_FINALvn2_*.json` lists 340 features, of which 39 are spatial -- w_, w4_, aw_, and the
great-circle anchors sa300..sa2500. Every one of them is ISOTROPIC: a disc or a box centred on the
cell. Water does not move isotropically. The cells upslope of a basin carry the water that is about
to arrive; the cells downslope carry water that has already left, and averaging the two together
discards the asymmetry.

REPORT §4 recorded this as measured at +0.0005, but that number came from `fastval.py`'s boxmean on
the old 40x40 domain, and that code path indexes a grid-to-cell map containing -1 for positions with
no land cell -- it throws on the global grid the harness now uses. So the family was never measured
on this domain, and it is absent from the model: 0 of 340 features are direction-split.

What it builds
--------------
For each key, the mean over the cells strictly west / east / north / south within a box, plus the
west-east and north-south differences, which are the quantities that actually carry the asymmetry.
Same machinery and same (time, t_known) grouping as `add_wide4`, so it is comparable to the
isotropic block it is being tested against.

Coordinates define the neighbourhood only; they are not features, exactly as in `anchor.py`.

usage: python add_dir_feats.py <LAYOUT>     -> out/mats/{L}_{tr,va}_dir.parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl

# the keys the isotropic block already aggregates, so the contrast is direction and nothing else
DIRKEYS = ["anom_known", "dev24", "an_e5PERz_acc", "an_e5Pz_acc", "an_SOIL_MOISTURE_tz_d"]
RADIUS = int(os.environ.get("DIR_R", "6"))


def _halfmeans(vals, pid, li, lo, npair, radius):
    """West/east/north/south means over a (2r+1) box, excluding the centre column/row."""
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
