import glob

import numpy as np
import polars as pl

from ar_blend import LAYOUTS, fit, layout_data, mixmse
from goal065_eval import summary_line

K4 = ["p", "k", "c", "ar"]
EXTRA = ["sa300", "sa800"]

def main():
    D = {}
    for L in LAYOUTS:
        d = layout_data(L)
        a = pl.read_parquet(f"out/mats/{L}_va_anchor.parquet", columns=EXTRA)
        va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["target", "tws_known"])

        y = va["target"].to_numpy().astype(float)
        keep = np.isfinite(y)
        for c in EXTRA:
            v = a[c].to_numpy().astype(float)
            keep &= np.isfinite(v)
        n_expected = len(d["y"])
        for c in EXTRA:
            d[c] = a[c].to_numpy().astype(float)[keep][:n_expected] if keep.sum() >= n_expected else None
        if any(d[c] is None or len(d[c]) != n_expected for c in EXTRA):
            print(f"  {L}: anchor rows do not align ({keep.sum()} vs {n_expected}), skipping"); return
        D[L] = d
    arms = {"+sa300": K4 + ["sa300"], "+sa800": K4 + ["sa800"], "+both": K4 + EXTRA}
    out = []
    print("marginal value over the four-vector AR correction, leave-one-layout-out\n")
    for name, keys in arms.items():
        deltas = {}
        for L in D:
            tr = [D[x] for x in D if x != L]
            wb, wa = fit(tr, K4), fit(tr, keys)
            d = D[L]
            a = np.sqrt(mixmse(d["y"], d["h"], wb @ np.stack([d[x] for x in K4])))
            b = np.sqrt(mixmse(d["y"], d["h"], wa @ np.stack([d[x] for x in keys])))
            deltas[L] = b - a
        line = summary_line("hunt2" + name.replace("+", "_"), deltas)
        print(" ", line); out.append(line)
    with open("out/goal065/hunt2.txt", "w") as f:
        f.write("marginal value of a smoothed anchor on top of the AR correction, LOO, validation only\n")
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
