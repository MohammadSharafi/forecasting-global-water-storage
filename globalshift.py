"""Does the model capture the month-to-month GLOBAL shift? (session 9j)

Why this exists
---------------
Session 3 measured "month-to-month global offsets of +-0.3" in the test era and concluded the
unpredictable component is regional and spatially coherent rather than white per-cell noise.
That is the largest error component ever identified in this project -- if the global mean of the
change is off by 0.3 in a month, every cell in that month inherits it -- and nothing has ever
targeted it directly. The models predict each cell independently; whatever global shift they
produce is an accident of averaging, never a modelled quantity.

Arithmetic for why it matters: total MSE is about 0.503 (RMSE 0.709). A per-month global offset
with SD 0.3 contributes 0.09 of that, or roughly 18% of the whole error. Capturing even a third
of it would be worth about 0.014 of RMSE -- the size of the gap to the 0.69 target.

What this measures
------------------
Per target month, the TRUE mean change against the PREDICTED mean change, then:

  slope     regression of true on predicted. 1.0 means calibrated; below 1 means the model
            SHRINKS the global shift (the usual failure), above 1 means it overshoots.
  share     how much of total MSE is pure global-offset error
  oracle    RMSE if each month's global mean change were corrected perfectly. This is an upper
            bound on what any global correction can buy -- not achievable, but it says whether
            the idea is worth pursuing at all.
  zonal     the same decomposition per latitude band, since the shift may be hemispheric rather
            than global (seasonal mass exchange between hemispheres is real physics).

Nothing here is a submission step: it is a measurement that decides whether to build a global
correction, and it deliberately reports an ORACLE bound so the decision is made on the size of
the prize rather than on hope.

usage: python globalshift.py A lgb_v5x_noll [tag]
"""
import sys
import numpy as np
import polars as pl
from eval_mix import load, test_mix

L = sys.argv[1] if len(sys.argv) > 1 else "A"
STEM = sys.argv[2] if len(sys.argv) > 2 else "lgb_v5x_noll"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""


def main():
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "horizon", "tws_known", "target"])
    p = load(L, [f"{STEM}:{TAG}"])
    y = va["target"].to_numpy(); k = va["tws_known"].to_numpy(); h = va["horizon"].to_numpy()
    dy_true = y - k
    dy_pred = p - k
    months = va["time"].to_list()   # .to_numpy() gives datetime64, which will not
                                    # match the datetime.date keys polars groups on
    rmse = lambda a: float(np.sqrt(np.mean((y - a) ** 2)))
    print(f"layout {L}, predictions {STEM}{TAG}: {len(va)} rows, RMSE {rmse(p):.4f}\n")

    # ---- global: one number per target month
    d = pl.DataFrame({"m": months, "t": dy_true, "q": dy_pred})
    g = d.group_by("m").agg(pl.col("t").mean().alias("true"), pl.col("q").mean().alias("pred"),
                            pl.len().alias("n")).sort("m")
    tm = g["true"].to_numpy(); pm = g["pred"].to_numpy()
    print("  per target month, mean change over all cells:")
    print(f"  {'month':12} {'true':>8} {'predicted':>10} {'missed':>8}")
    for row in g.iter_rows(named=True):
        miss = row["true"] - row["pred"]
        print(f"  {str(row['m']):12} {row['true']:8.4f} {row['pred']:10.4f} {miss:8.4f}")
    sd_true = float(np.std(tm)); sd_pred = float(np.std(pm))
    slope = float(np.polyfit(pm, tm, 1)[0]) if sd_pred > 1e-9 else np.nan
    corr = float(np.corrcoef(pm, tm)[0, 1]) if sd_pred > 1e-9 and sd_true > 1e-9 else np.nan
    print(f"\n  sd of the true monthly global shift : {sd_true:.4f}")
    print(f"  sd of the predicted one             : {sd_pred:.4f}   (ratio {sd_pred/max(sd_true,1e-9):.2f})")
    print(f"  corr(pred, true) = {corr:+.3f}    regression slope = {slope:+.3f}"
          f"   {'-> the model SHRINKS the global shift' if slope > 1.2 else ''}")

    # ---- how much of the error is the global offset, and what would perfect correction buy
    miss = dict(zip(g["m"].to_list(), (tm - pm).tolist()))
    off = np.array([miss[m] for m in months])
    share = float(np.mean(off ** 2) / np.mean((y - p) ** 2))
    print(f"\n  share of total MSE that is pure global-offset error: {share*100:.1f}%")
    print(f"  ORACLE, perfect global correction : RMSE {rmse(p + off):.4f}   "
          f"(from {rmse(p):.4f}, i.e. {rmse(p+off)-rmse(p):+.4f})")

    # a correction is only usable if it can be PREDICTED from information at <= t. The cheapest
    # honest proxy: use the previous target month's missed offset (persistence of the bias).
    ms = sorted(miss)
    prev = {m: miss[ms[i-1]] for i, m in enumerate(ms) if i > 0}
    if prev:
        off_p = np.array([prev.get(m, 0.0) for m in months])
        for a in (0.25, 0.5, 0.75, 1.0):
            print(f"    persistence of the previous month's offset, a={a:.2f}: RMSE {rmse(p + a*off_p):.4f}")
        pv = np.array([prev.get(m, np.nan) for m in ms]); cv = np.array([miss[m] for m in ms])
        ok = np.isfinite(pv)
        if ok.sum() > 3:
            print(f"    corr(previous offset, this offset) = {np.corrcoef(pv[ok],cv[ok])[0,1]:+.3f}"
                  f"   <- if this is near zero the offset is temporally white and NOT correctable")

    # ---- zonal: is the shift hemispheric rather than global?
    #
    # This section carries the largest prize the project has measured, so it gets the same
    # reachability test as the global offset above rather than only an oracle. An oracle is not
    # an opportunity: it needs the answer to compute the correction. What decides whether the
    # band offset is worth building is whether it can be PREDICTED from information at <= t.
    band = ((va["lat"].to_numpy() + 90) // 30).astype(int)
    dz = pl.DataFrame({"m": months, "b": band, "t": dy_true, "q": dy_pred})
    gz = dz.group_by(["m", "b"]).agg(pl.col("t").mean().alias("true"), pl.col("q").mean().alias("pred"))
    key = {(m, b): t - q for m, b, t, q in
           zip(gz["m"].to_list(), gz["b"].to_list(), gz["true"].to_list(), gz["pred"].to_list())}
    offz = np.array([key[(m, b)] for m, b in zip(months, band)])
    sharez = float(np.mean(offz ** 2) / np.mean((y - p) ** 2))
    print(f"\n  same decomposition per 30-degree latitude band:")
    print(f"    share of total MSE that is band-offset error: {sharez*100:.1f}%")
    print(f"    ORACLE, perfect band correction: RMSE {rmse(p + offz):.4f}   "
          f"({rmse(p+offz)-rmse(p):+.4f})")
    # is the BAND offset predictable from the previous month's band offset?
    ms = sorted({m for m, _ in key})
    bs = sorted({b for _, b in key})
    prevb = {(m, b): key[(ms[i - 1], b)] for i, m in enumerate(ms) if i > 0
             for b in bs if (ms[i - 1], b) in key and (m, b) in key}
    if prevb:
        offz_p = np.array([prevb.get((m, b), 0.0) for m, b in zip(months, band)])
        print("\n    is any of it REACHABLE? apply the PREVIOUS month's band offset:")
        for a in (0.25, 0.5, 0.75, 1.0):
            print(f"      a={a:.2f}: RMSE {rmse(p + a*offz_p):.4f}   ({rmse(p+a*offz_p)-rmse(p):+.4f})")
        pv = np.array([prevb[k] for k in sorted(prevb)])
        cv = np.array([key[k] for k in sorted(prevb)])
        if len(pv) > 3 and pv.std() > 1e-12 and cv.std() > 1e-12:
            r = float(np.corrcoef(pv, cv)[0, 1])
            print(f"      corr(previous band offset, this band offset) = {r:+.3f}   over {len(pv)}"
                  f" band-months")
            print("      near zero means the band offset is temporally white: the oracle above is"
                  " unreachable")

    # and is it predictable from what the band ITSELF has recently been observed doing? That is
    # information at <= t_known, so a correction built on it would be legal. The zonal features
    # already expose this per cell; this asks whether it survives at band level, where the error is.
    kb = pl.DataFrame({"m": months, "b": band, "k": k}).group_by(["m", "b"]).agg(
        pl.col("k").mean().alias("kbar")).sort(["b", "m"])
    kb = kb.with_columns(pl.col("kbar").diff().over("b").alias("dk"))
    dmap = {(m, b): d for m, b, d in zip(kb["m"].to_list(), kb["b"].to_list(),
                                         kb["dk"].to_list()) if d is not None}
    both = [(dmap[x], key[x]) for x in dmap if x in key]
    if len(both) > 5:
        a1 = np.array([x for x, _ in both]); a2 = np.array([y for _, y in both])
        if a1.std() > 1e-12 and a2.std() > 1e-12:
            print(f"\n    corr(band's own recent observed change, its offset) = "
                  f"{float(np.corrcoef(a1, a2)[0, 1]):+.3f}   over {len(both)} band-months")
            print("      this one uses only observations, so a correction built on it would be"
                  " legal --")
            print("      but the zonal features already give the model this signal per cell.")

    print("\n  Read it this way: the oracle rows bound the prize. If they are small, no global or")
    print("  zonal correction can help and this avenue is closed. If they are large, the")
    print("  reachability rows decide whether any of it can actually be taken.")


if __name__ == "__main__":
    main()
