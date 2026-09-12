"""Which other model features still pay as a post-hoc LINEAR term? LOO, validation labels only.

The AR result is the template: `ar` is a(h,cell)*tws_known, a per-cell per-horizon rescaling of the
anchor, and it pays as a linear correction even though the tree already holds every ingredient,
because a tree cannot cheaply build `model + 0.47*ar - 0.35*k` out of splits. The candidates here
are the other quantities with that shape -- products and ratios the tree must approximate with
staircases:

  anom_persist   the model's highest-gain feature (10.1%): the cell's standardised anomaly carried
                 forward. A level, not a product, but if the tree under-uses it linearly it will show.
  trend_persist  the cell's carried-forward trend.
  csd_k          csd * tws_known: the anchor rescaled by the cell's own variability. Heteroscedastic
                 shrinkage is precisely what a staircase approximates badly.
  k_over_h       tws_known / horizon: decay with lead time, the shape AR fits per cell, but global.

Results go to out/goal065/hunt3.txt; heldout.txt's last line must stay the closed goal's ADOPTED-SET.
"""
import numpy as np
import polars as pl

from ar_blend import LAYOUTS, fit, layout_data, mixmse
from goal065_eval import summary_line

K4 = ["p", "k", "c", "ar"]
COLS = ["anom_persist", "trend_persist", "csd"]


def main():
    D = {}
    for L in LAYOUTS:
        d = layout_data(L)
        va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                             columns=["target", "tws_known", "clim_next", "horizon"] + COLS)
        y = va["target"].to_numpy().astype(float)
        k = va["tws_known"].to_numpy().astype(float)
        c = va["clim_next"].to_numpy().astype(float)
        h = va["horizon"].to_numpy().astype(float)
        extra = {x: va[x].to_numpy().astype(float) for x in COLS}
        extra["csd_k"] = extra["csd"] * k
        extra["k_over_h"] = k / np.maximum(h, 1.0)
        keep = np.isfinite(y) & np.isfinite(k) & np.isfinite(c)
        for v in extra.values():
            keep &= np.isfinite(v)
        if keep.sum() != len(d["y"]):
            print(f"  {L}: mask {keep.sum()} vs layout_data {len(d['y'])} -- realigning on the smaller set")
        n = min(keep.sum(), len(d["y"]))
        for key, v in extra.items():
            d[key] = v[keep][:n]
        for key in ("y", "h", "p", "k", "c", "ar"):
            d[key] = d[key][:n]
        D[L] = d
    arms = {"+anom_persist": ["anom_persist"], "+trend_persist": ["trend_persist"],
            "+csd_k": ["csd_k"], "+k_over_h": ["k_over_h"],
            "+anom+csd_k": ["anom_persist", "csd_k"]}
    out = []
    print("marginal value over the four-vector AR correction, leave-one-layout-out\n")
    for name, add in arms.items():
        keys = K4 + add
        deltas = {}
        for L in D:
            tr = [D[x] for x in D if x != L]
            wb, wa = fit(tr, K4), fit(tr, keys)
            d = D[L]
            a = np.sqrt(mixmse(d["y"], d["h"], wb @ np.stack([d[x] for x in K4])))
            b = np.sqrt(mixmse(d["y"], d["h"], wa @ np.stack([d[x] for x in keys])))
            deltas[L] = b - a
        line = summary_line("hunt3" + name.replace("+", "_"), deltas)
        print(" ", line); out.append(line)
    with open("out/goal065/hunt3.txt", "w") as f:
        f.write("post-hoc linear terms tried on top of the AR correction, LOO, validation only\n")
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
