import sys
import numpy as np
import polars as pl

L = sys.argv[1] if len(sys.argv) > 1 else "A"

def within_cell(x, y, cell):
    d = pl.DataFrame({"c": cell, "x": x, "y": y}).drop_nulls()
    d = d.with_columns((pl.col("x") - pl.col("x").mean().over("c")).alias("xd"),
                       (pl.col("y") - pl.col("y").mean().over("c")).alias("yd"))
    xd = d["xd"].to_numpy(); yd = d["yd"].to_numpy()
    if len(xd) < 100 or xd.std() < 1e-12 or yd.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(xd, yd)[0, 1])

def main():
    schema = pl.scan_parquet(f"out/mats/{L}_va.parquet").collect_schema().names()

    PAIRS = [("e5PER_acc", "an_e5PERz_acc", "ERA5 P-E-R accumulated over the gap"),
             ("PER_acc",   "an_PERz_acc",   "NCEP P-E-R accumulated over the gap"),
             ("e5SW_d",    "an_e5SWz_d",    "ERA5 soil-water change over the gap"),
             ("e5SWE_d",   "an_e5SWEz_d",   "ERA5 snow change over the gap"),
             ("e5P_acc",   "an_e5Pz_acc",   "ERA5 precipitation accumulated")]
    PAIRS = [(a, b, t) for a, b, t in PAIRS if a in schema or b in schema]
    extra = [c for c in ("an_e5MTWSz_d", "an_MTWSz_d") if c in schema]
    cols = ["lat", "lon", "horizon", "tws_known", "target"]
    cols += [c for a, b, _ in PAIRS for c in (a, b) if c in schema] + extra
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=list(dict.fromkeys(cols)))
    dy = (va["target"] - va["tws_known"]).to_numpy()
    h = va["horizon"].to_numpy()
    cell = (va["lat"].to_numpy() * 1000 + va["lon"].to_numpy())
    print(f"layout {L}: {len(va)} rows, dy sd = {np.nanstd(dy):.4f}\n")

    def stats(name):
        x = va[name].to_numpy().astype(float)
        ok = np.isfinite(x) & np.isfinite(dy)
        if ok.sum() < 100:
            return None
        g = float(np.corrcoef(x[ok], dy[ok])[0, 1])
        w = within_cell(x[ok], dy[ok], cell[ok])
        h1 = ok & (h == 1)
        g1 = float(np.corrcoef(x[h1], dy[h1])[0, 1]) if h1.sum() > 100 else np.nan
        return g, w, g1

    print(f"  {'feature':22} {'global r':>9} {'within-cell r':>14} {'r at h=1':>9}   what it is")
    print("  " + "-" * 78)
    for a, b, title in PAIRS:
        for name, tag in ((a, "raw   "), (b, "anom  ")):
            if name not in schema:
                continue
            s = stats(name)
            if s is None:
                continue
            g, w, g1 = s
            print(f"  {tag}{name:16} {g:9.3f} {w:14.3f} {g1:9.3f}   {title if tag.strip()=='raw' else ''}")
    for name in extra:
        s = stats(name)
        if s:
            g, w, g1 = s
            print(f"  anom  {name:16} {g:9.3f} {w:14.3f} {g1:9.3f}   modelled total water storage change")

    an = [c for c in schema if c.startswith("an_")]
    if not an:
        print("\n  no an_ columns in this matrix -- rebuild first"); return
    print(f"\n  ridge on the {len(an)} an_ features (fit on {L}_tr, scored on {L}_va):")
    tr = pl.read_parquet(f"out/mats/{L}_tr.parquet",
                         columns=list(dict.fromkeys(["tws_known", "target"] + an)))
    n = min(300_000, len(tr))
    tr = tr.sample(n=n, seed=0)
    Xt = tr.select(an).to_numpy().astype(np.float64)
    yt = (tr["target"] - tr["tws_known"]).to_numpy().astype(np.float64)
    Xv = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=an).to_numpy().astype(np.float64)
    med = np.nanmedian(Xt, axis=0)
    Xt = np.where(np.isnan(Xt), med, Xt); Xv = np.where(np.isnan(Xv), med, Xv)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
    A = Xt.T @ Xt + 100.0 * np.eye(Xt.shape[1])
    w = np.linalg.solve(A, Xt.T @ (yt - yt.mean()))
    pv = Xv @ w + yt.mean()
    ok = np.isfinite(dy) & np.isfinite(pv)
    r2 = 1 - np.sum((dy[ok] - pv[ok]) ** 2) / np.sum((dy[ok] - dy[ok].mean()) ** 2)
    print(f"    R^2 on the validation change = {r2:.4f}   (corr {np.corrcoef(pv[ok],dy[ok])[0,1]:.3f})")
    print(f"    that is a LINEAR bound; the trees should exceed it")
    for hh in range(1, 8):
        m = ok & (h == hh)
        if m.sum() > 500:
            r2h = 1 - np.sum((dy[m] - pv[m]) ** 2) / np.sum((dy[m] - dy[m].mean()) ** 2)
            print(f"      h={hh}: R^2 {r2h:+.4f}  (n={m.sum()})")

if __name__ == "__main__":
    main()
