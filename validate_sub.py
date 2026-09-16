import sys
import numpy as np, polars as pl
p = sys.argv[1] if len(sys.argv) > 1 else "out/sub_final_best.csv"
try:
    d = pl.read_csv(p); t = pl.read_csv("Test.csv")
    v = d["Target"].to_numpy().astype(float)
    checks = {
        "rows==280961": d.height == 280961,
        "columns ID,Target": d.columns == ["ID", "Target"],
        "all finite": bool(np.isfinite(v).all()),
        "IDs match Test.csv in order": bool((d["ID"].to_numpy() == t["ID"].to_numpy()).all()),
        "no duplicate IDs": d["ID"].n_unique() == d.height,
        "magnitudes sane": bool(np.abs(v).max() < 10),
    }
    for k, x in checks.items():
        print(f"  {'PASS' if x else 'FAIL'}  {k}")
    print("VALID" if all(checks.values()) else "INVALID")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {e}")
    print("INVALID")
