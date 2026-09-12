"""Does a horizon-1 specialist beat the shared model where a third of the test lives?

h=1 is 33.3% of the test rows, it is the case where TWS is known exactly, and it is where the
shipped model beats a per-cell AR line by the least: -0.0426, against -0.136 to -0.191 at h>=5.
93 of 354 features are constant or missing on those rows, because the accumulation window
(t_known, t] is empty when t_known == t. So either the model is diluted there and a specialist
helps, or h=1 genuinely holds less information and the shared model is already at its ceiling.

Reports two things, because the h=1 delta alone would overstate the prize:
  * the h=1 RMSE of both models on identical rows, per layout
  * the whole-score consequence under the test's horizon mix, which is what a submission feels
"""
import glob

import numpy as np
import polars as pl

from ar_blend import LAYOUTS
from goal065_eval import summary_line

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])


def avg(L, tag):
    fs = sorted(glob.glob(f"out/mats/pred_{L}_lgb_v5x_noll_s*_{tag}.npy"))
    return np.mean([np.load(f) for f in fs], 0) if fs else None


def main():
    d1, dmix, rows = {}, {}, []
    for L in LAYOUTS:
        va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "target"])
        y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
        base, spec = avg(L, "d0"), avg(L, "h1x")
        if spec is None:
            d1[L] = None; dmix[L] = None; continue
        ok = np.isfinite(y) & np.isfinite(base) & np.isfinite(spec)
        s1 = ok & (h == 1)
        r_base = float(np.sqrt(np.mean((y[s1] - base[s1]) ** 2)))
        r_spec = float(np.sqrt(np.mean((y[s1] - spec[s1]) ** 2)))
        d1[L] = r_spec - r_base
        # whole-score effect: swap the specialist in on h=1 rows only, leave every other horizon alone
        def mix(p):
            return float(np.sqrt(sum(MIX[i - 1] * np.mean((y[ok & (h == i)] - p[ok & (h == i)]) ** 2)
                                     for i in range(1, 8) if (ok & (h == i)).any())))
        swapped = base.copy(); swapped[s1] = spec[s1]
        m0, m1 = mix(base), mix(swapped)
        dmix[L] = m1 - m0
        rows.append((L, int(s1.sum()), r_base, r_spec, m0, m1))
    print(f"  {'layout':7} {'h1 rows':>8} {'h1 shared':>10} {'h1 spec':>9} {'h1 delta':>9} "
          f"{'mix before':>11} {'mix after':>10} {'mix delta':>10}")
    for L, n, rb, rs, m0, m1 in rows:
        print(f"  {L:7} {n:8d} {rb:10.4f} {rs:9.4f} {rs-rb:+9.4f} {m0:11.4f} {m1:10.4f} {m1-m0:+10.4f}")
    print()
    l1 = summary_line("h1spec", d1)
    lm = summary_line("h1spec_wholescore", dmix)
    print("  h=1 only     :", l1)
    print("  whole score  :", lm)
    with open("out/goal065/h1spec.txt", "w") as f:
        f.write(lm + "\n" + l1 + "\n")


if __name__ == "__main__":
    main()
