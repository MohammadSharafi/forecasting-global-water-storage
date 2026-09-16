import sys
import time

import numpy as np
import polars as pl

from ar_model import fit_ar, ar_predict
from build_mats import base_of

K = 100

def main():
    L = sys.argv[1]; t0 = time.time()
    if base_of(L) == "FINAL":
        hist = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()).select(
            ["lat", "lon", "time", "TWS_t"])
    else:
        b = base_of(L); sfx = "" if b == "A" else f"_{b}"
        hist = pl.read_parquet(f"out/pseudo_hist{sfx}.parquet").select(["lat", "lon", "time", "TWS_t"])
    cells, pooled, per = fit_ar(hist, K)
    print(f"{L}: AR fitted on {hist['time'].n_unique()} history months, "
          f"slopes " + " ".join(f"h{i}={pooled[i]:.3f}" for i in range(1, 8)), flush=True)
    for part in ("tr", "va"):
        m = pl.read_parquet(f"out/mats/{L}_{part}.parquet",
                            columns=["lat", "lon", "horizon", "tws_known"])
        lat = m["lat"].to_numpy(); lon = m["lon"].to_numpy()
        h = m["horizon"].to_numpy().astype(int); k = m["tws_known"].to_numpy().astype(float)
        ar, hit = ar_predict(cells, pooled, per, lat, lon, h, k)
        slope = np.divide(ar, k, out=np.full(len(k), np.nan), where=np.abs(k) > 1e-9)
        pl.DataFrame({"ar_slope": slope.astype(np.float32),
                      "ar_pred": ar.astype(np.float32),
                      "ar_dev": (ar - k).astype(np.float32)}).write_parquet(
            f"out/mats/{L}_{part}_ar.parquet")
        print(f"  {part}: {len(m)} rows, cells matched {hit*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
