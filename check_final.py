import json
import os
import sys

import polars as pl

def main():
    f = "out/mats/feats.json"
    if not os.path.exists(f):
        print(f"{f} missing -- build a validation layout first"); return 1
    want = set(json.load(open(f)))
    have = set()
    for p in ("out/mats/FINAL_va.parquet", "out/mats/FINAL_va_anchor.parquet"):
        if not os.path.exists(p):
            print(f"{p} missing -- run build_mats.py FINAL and add_anchor_feats.py FINAL")
            return 1
        have |= set(pl.scan_parquet(p).collect_schema().names())
    missing = sorted(want - have)
    if missing:
        print(f"MISSING {len(missing)} of {len(want)} features, e.g. {missing[:6]}")
        return 1
    print(f"FINAL carries all {len(want)} features")
    return 0

if __name__ == "__main__":
    sys.exit(main())
