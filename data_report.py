import sys
import datetime as dt
import numpy as np
import polars as pl

def months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)

def runs(ms):
    out = []
    for m in ms:
        if out and months_between(out[-1][0], m) == out[-1][1]:
            out[-1][1] += 1
        else:
            out.append([m, 1])
    return [(a, b) for a, b in out]

def hdr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")

def main():
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    te = pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    trm = sorted(tr["time"].unique().to_list()); tem = sorted(te["time"].unique().to_list())

    hdr("COVERAGE")
    for name, d, ms in (("Train", tr, trm), ("Test", te, tem)):
        cells = d.select(["lat", "lon"]).unique().height
        print(f"  {name}: {len(d):>8} rows  {cells:>6} cells  {len(ms):>3} months  "
              f"{ms[0]} .. {ms[-1]}")
    span = months_between(trm[0], trm[-1]) + 1
    have = set(trm)
    absent = []
    for i in range(span):
        y = trm[0].year + (trm[0].month - 1 + i) // 12
        mo = (trm[0].month - 1 + i) % 12 + 1
        if dt.date(y, mo, 1) not in have:
            absent.append(dt.date(y, mo, 1))
    print(f"  Train spans {span} months and carries {len(trm)}; {len(absent)} are absent (GRACE gaps)")
    if absent:
        rs = runs(absent)
        print("    " + ", ".join(f"{a} x{n}" for a, n in rs[:12])
              + ("  ..." if len(rs) > 12 else ""))
    cpm = tr.group_by("time").len().sort("time")
    print(f"  cells per month in train: min {cpm['len'].min()}, median {int(cpm['len'].median())}, "
          f"max {cpm['len'].max()}")

    hdr("TEST GEOMETRY -- read off Test.csv, not assumed")
    mask_col = "TWS_t_masked" if "TWS_t_masked" in te.columns else None
    blocks = runs(tem)
    print(f"  {len(blocks)} blocks: " + " | ".join(f"{a} x{n}" for a, n in blocks))
    print(f"  block lengths: {[n for _, n in blocks]}")
    gaps = [months_between(blocks[i - 1][0], blocks[i][0]) - blocks[i - 1][1]
            for i in range(1, len(blocks))]
    print(f"  months skipped between blocks: {gaps}")
    if mask_col:
        obs = sorted(te.filter(~pl.col(mask_col))["time"].unique().to_list())
        print(f"  months carrying an observed TWS: {len(obs)} -- {[str(m) for m in obs]}")
        starts = [a for a, _ in blocks]
        print(f"  are those exactly the block-first months? {obs == starts}")
    try:
        from eval_mix import test_mix, TEST_BLOCKS
        got = tuple(n for _, n in blocks)
        ok = got == tuple(TEST_BLOCKS)
        print(f"\n  eval_mix.TEST_BLOCKS = {tuple(TEST_BLOCKS)}   file says {got}   "
              f"{'MATCH' if ok else '*** MISMATCH -- the horizon weighting is wrong ***'}")
        w = test_mix()
        print("  horizon mix implied by the file: " + "  ".join(f"h{i} {w[i-1]:.3f}"
                                                                for i in range(1, 8)))
    except Exception as e:
        print(f"  (could not cross-check eval_mix: {e})")
    if mask_col:

        o = set(sorted(te.filter(~pl.col(mask_col))["time"].unique().to_list()) + [trm[-1]])
        hs = []
        for m in tem:
            prev = max(x for x in o if x <= m)
            hs.append(months_between(prev, m) + 1)
        u, c = np.unique(hs, return_counts=True)
        print("  horizon per test MONTH (each covers the whole globe): "
              + "  ".join(f"h{a} x{b}" for a, b in zip(u, c)))

    hdr("REGIME -- did the distribution move between the eras?")
    print(f"  {'era':28} {'n':>9} {'mean':>8} {'sd':>7} {'p05':>7} {'p95':>7}")

    def era(name, s):
        v = s.drop_nulls().to_numpy()
        if len(v) == 0:
            return
        print(f"  {name:28} {len(v):>9} {v.mean():8.4f} {v.std():7.4f} "
              f"{np.percentile(v,5):7.3f} {np.percentile(v,95):7.3f}")

    era("Train, all months", tr["TWS_t"])
    for y0, y1 in ((2002, 2007), (2008, 2011), (2012, 2015)):
        era(f"Train {y0}-{y1}", tr.filter(pl.col("time").dt.year().is_between(y0, y1))["TWS_t"])
    if mask_col and "TWS_t" in te.columns:
        era("Test, observed months", te.filter(~pl.col(mask_col))["TWS_t"])
    print("  A shift in the mean here is inherited by every prediction: the models are anchored on")
    print("  tws_known, so a global level change is absorbed, but a change in SPREAD is not.")

    g = tr.group_by("time").agg(pl.col("TWS_t").mean().alias("mu"),
                                pl.col("TWS_t").std().alias("sd")).sort("time")
    print("\n  global mean TWS, last 18 train months (the run-up to the test era):")
    for r in g.tail(18).iter_rows(named=True):
        print(f"    {r['time']}  mean {r['mu']:+.4f}  sd {r['sd']:.4f}")
    mu = g["mu"].to_numpy()
    print(f"  sd of the global monthly mean over train: {mu.std():.4f}   "
          f"month-to-month change sd: {np.diff(mu).std():.4f}")
    print("  That second number bounds what a perfect global-offset correction could ever buy.")

    hdr("SEASONALITY AND PERSISTENCE, by latitude band")
    t = tr.with_columns(pl.col("time").dt.month().alias("m"),
                        ((pl.col("lat") + 90) // 30).cast(pl.Int32).alias("band"))
    cm = t.group_by(["lat", "lon", "m"]).agg(pl.col("TWS_t").mean().alias("cl"))
    amp = (cm.group_by(["lat", "lon"]).agg((pl.col("cl").max() - pl.col("cl").min()).alias("amp"))
             .with_columns(((pl.col("lat") + 90) // 30).cast(pl.Int32).alias("band")))
    var = (t.join(cm, on=["lat", "lon", "m"], how="left")
             .group_by(["lat", "lon"])
             .agg(pl.col("TWS_t").var().alias("v"),
                  (pl.col("TWS_t") - pl.col("cl")).var().alias("vr"))
             .with_columns(((pl.col("lat") + 90) // 30).cast(pl.Int32).alias("band")))
    s = t.sort(["lat", "lon", "time"]).with_columns(
        pl.col("TWS_t").shift(1).over(["lat", "lon"]).alias("prev"),
        pl.col("time").shift(1).over(["lat", "lon"]).alias("tp"))
    s = s.filter(((pl.col("time").dt.year() - pl.col("tp").dt.year()) * 12
                  + (pl.col("time").dt.month() - pl.col("tp").dt.month())) == 1)
    per = s.group_by("band").agg(pl.corr("TWS_t", "prev").alias("ac1"),
                                 (pl.col("TWS_t") - pl.col("prev")).abs().mean().alias("mad1"),
                                 pl.len().alias("n"))
    a = amp.group_by("band").agg(pl.col("amp").mean().alias("amp"))
    v = var.group_by("band").agg((1 - pl.col("vr").sum() / pl.col("v").sum()).alias("seas_share"))
    j = a.join(v, on="band", how="left").join(per, on="band", how="left").sort("band")
    print(f"  {'band':14} {'seas amp':>9} {'seasonal share':>15} {'lag-1 corr':>11} {'|d1|':>7}")
    for r in j.iter_rows(named=True):
        b = r["band"]; lo = b * 30 - 90
        print(f"  {f'{lo:+.0f}..{lo+30:+.0f}':14} {r['amp']:9.3f} {r['seas_share']:15.3f} "
              f"{r['ac1']:11.3f} {r['mad1']:7.3f}")
    print("  seasonal share = fraction of a cell's variance explained by its own monthly")
    print("  climatology; |d1| = mean absolute one-month change, the bar for any h=1 model.")

    hdr("COVARIATE COMPLETENESS")
    from features import COV
    for name, d in (("Train", tr), ("Test", te)):
        parts = []
        for c in COV:
            if c in d.columns:
                parts.append(f"{c} {float(d[c].is_null().mean())*100:.1f}%")
        print(f"  {name} null rate: " + "  ".join(parts))

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"data_report: {e}", file=sys.stderr); sys.exit(1)
