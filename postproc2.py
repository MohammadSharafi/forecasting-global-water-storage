"""Great-circle post-processing of the *prediction field* (session 9).

Why this exists
---------------
`smooth.smooth()` averages the predicted RESIDUAL (p - tws_known) over the 8 grid
neighbours.  Two things are wrong with that operator:

1. It is a grid box, not a physical radius.  At 60 deg N a 1 deg longitude step is 55 km,
   at the equator 111 km, so the amount of smoothing depends on latitude.  Session 7
   established that the scale that matters for this field is 300-500 km (the smoothed
   anchor as a bare persistence forecast: A 0.757 -> 0.724 at 300 km, 0.717 at 500 km).
2. It smooths the residual only, so the *whole* fine-scale noise of `tws_known` is still
   carried into every prediction.  Session 7 measured that only ~20% of a cell's
   departure from its 300 km neighbourhood survives into the next month; the post-hoc
   re-base `p + a*(sa300 - tws_known)` with a fixed a=0.3 was a partial correction of
   exactly this, and a was never scanned.

This module applies one operator that subsumes both:

    p_out = M_r(p) + beta * (p - M_r(p))

where M_r is the great-circle neighbourhood mean of the predicted field for the target
month (same kernel as `anchor.anchor_fields`, longitude window scaled by 1/cos(lat)).
beta = 1 is a no-op; beta = 0 is full smoothing.  beta is the fraction of the fine-scale
content of the prediction that is worth keeping, and it has a closed form:

    beta* = sum(d * (y - s)) / sum(d * d),   s = M_r(p),  d = p - s

so no grid search is needed -- but a grid is printed anyway so the curvature is visible.

Compliance: uses only the predictions themselves plus latitude/longitude as a lookup key
for the neighbourhood, which the organisers' 24 Aug ruling permits ("neighbouring cells'
TWS at months <= t, including spatial filtering and aggregation").  Nothing from t+1 is
touched: the operator mixes predictions of the SAME target month across space only, never
across horizons (that was the non-causal trajectory smoothing removed in session 6).

usage
-----
  scan  :  python postproc2.py A lgb_v5x_noll:0.3:_sa mlp_v5x_noll:0.25:_e1sa ...
           python postproc2.py B ...
  apply :  handled by final_assemble2.py
"""
import polars as pl, numpy as np, sys, glob, re, os
from anchor import anchor_fields, sample

RADII = (150, 200, 300, 400, 500, 700, 1000)


def load_preds(L, spec):
    """spec items are 'stem:weight[:tag]' where the prediction files are
    out/mats/pred_{L}_{stem}_s<digits>{tag}.npy (seeds averaged)."""
    acc = None; wsum = 0.0
    for a in spec:
        f = a.split(":")
        stem = f[0]
        w = float(f[1]) if len(f) > 1 and f[1] != "" else 1.0
        tag = f[2] if len(f) > 2 else os.environ.get("TAG", "")
        pat = re.compile(rf"pred_{re.escape(L)}_{re.escape(stem)}_s\d+{re.escape(tag)}\.npy$")
        fs = sorted(x for x in glob.glob(f"out/mats/pred_{L}_{stem}_s*{tag}.npy")
                    if pat.search(os.path.basename(x)))
        assert fs, f"no prediction files for {stem} (tag {tag!r}) in layout {L}"
        p = np.mean([np.load(x) for x in fs], 0)
        print(f"  {stem}{tag}: {len(fs)} seeds, w={w}")
        acc = w * p if acc is None else acc + w * p
        wsum += w
    return acc / wsum


def field_mean(rows, p, radii=RADII):
    """Great-circle neighbourhood mean of the predicted field, per TARGET month.

    rows: polars frame with lat, lon, time (the target month).  Returns {km: array}
    aligned to rows, with NaN (no neighbours) filled by p itself."""
    p = np.asarray(p, dtype=np.float64)
    obs = pl.DataFrame({"lat": rows["lat"], "lon": rows["lon"],
                        "time": rows["time"], "TWS_t": p})
    months = sorted(set(rows["time"].to_list()))
    F, ti = anchor_fields(months, obs, tuple(radii))
    S = sample(F, ti, rows.select(["lat", "lon", "time"]).rename({"time": "t_known"}))
    return {km: np.where(np.isnan(v), p, v).astype(np.float64) for km, v in S.items()}


def beta_star(d, resid, fallback=1.0, min_n=1000):
    """Closed-form least-squares shrink of the fine-scale part: argmin ||resid - beta*d||.
    Groups with too little data fall back rather than returning a meaningless coefficient."""
    if len(d) < min_n: return fallback
    den = float(np.sum(d * d))
    return float(np.sum(d * resid) / den) if den > 1e-12 else fallback


def apply_shrink(p, s, beta):
    return s + beta * (p - s)


def main():
    L = sys.argv[1]; spec = sys.argv[2:]
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "t_known", "horizon", "tws_known", "target"])
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy()
    rmse = lambda a: float(np.sqrt(np.mean((y - a) ** 2)))
    print(f"layout {L}: {len(va)} rows")
    p0 = load_preds(L, spec)
    print(f"  raw blend                    RMSE={rmse(p0):.4f}")

    # reference: the operator currently in final_assemble.py (grid-box residual smoothing)
    from smooth import smooth
    p_grid = smooth(va, p0, w=0.7, radius=1, iters=1)
    print(f"  current grid smooth (w=0.7)  RMSE={rmse(p_grid):.4f}")

    # the new operator, on the raw blend and on top of the grid smooth
    for label, p in (("raw", p0), ("grid+", p_grid)):
        S = field_mean(va, p)
        print(f"\n  [{label}] great-circle high-pass shrink   p -> M_r(p) + beta*(p - M_r(p))")
        print("   radius   beta*   RMSE@beta*   " + "  ".join(f"b={b:.1f}" for b in np.arange(0.1, 1.01, 0.15)))
        best = (9.9, None)
        for km in RADII:
            s = S[km]; d = p - s
            b = beta_star(d, y - s)
            row = "  ".join(f"{rmse(apply_shrink(p, s, bb)):.4f}" for bb in np.arange(0.1, 1.01, 0.15))
            r_b = rmse(apply_shrink(p, s, b))
            print(f"   {km:5d}   {b:5.2f}   {r_b:.4f}       {row}")
            if r_b < best[0]:
                best = (r_b, (label, km, b))
        print(f"  [{label}] best: radius={best[1][1]} km  beta={best[1][2]:.2f}  RMSE={best[0]:.4f}")

        # per-horizon beta at the best radius: does the optimal shrink decay with lead time?
        km = best[1][1]; s = S[km]; d = p - s
        bg = best[1][2]
        bs = [beta_star(d[h == hh], (y - s)[h == hh], fallback=bg) for hh in range(1, 8)]
        print(f"  [{label}] per-horizon beta at {km} km: " +
              " ".join(f"h{hh}={b:.2f}" for hh, b in zip(range(1, 8), bs)))
        ph = np.array([bs[int(np.clip(hh, 1, 7)) - 1] for hh in h])
        print(f"  [{label}] per-horizon beta applied:  RMSE={rmse(apply_shrink(p, s, ph)):.4f}"
              f"   (global beta {rmse(apply_shrink(p, s, best[1][2])):.4f})")

        # global multiplicative calibration of the predicted change, after the shrink
        q = apply_shrink(p, s, best[1][2]); k = va["tws_known"].to_numpy()
        a_star = beta_star(q - k, y - k)
        print(f"  [{label}] change scaling after shrink: alpha*={a_star:.3f} -> "
              f"RMSE={rmse(k + a_star * (q - k)):.4f}")


if __name__ == "__main__":
    main()
