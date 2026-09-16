import sys
import numpy as np
import polars as pl
from eval_mix import test_mix, load

L = sys.argv[1] if len(sys.argv) > 1 else "A"
SPEC = (sys.argv[2] if len(sys.argv) > 2 else "lgb_v5x_noll").split(",")
W = test_mix()

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def table(name, keys, y, p, k, order=None):
    ks = order if order is not None else sorted(set(keys.tolist()))
    tot = float(np.mean((y - p) ** 2)) * len(y)
    print(f"\n  {name:>14} {'n':>8} {'RMSE':>8} {'bias':>8} {'persist':>8} {'gain':>8} {'MSE share':>10}")
    for kk in ks:
        m = keys == kk
        if m.sum() < 50:
            continue
        se = float(np.sum((y[m] - p[m]) ** 2))
        print(f"  {str(kk):>14} {m.sum():>8} {rmse(y[m], p[m]):8.4f} {float(np.mean(p[m]-y[m])):+8.4f} "
              f"{rmse(y[m], k[m]):8.4f} {rmse(y[m], p[m]) - rmse(y[m], k[m]):+8.4f} "
              f"{100*se/tot:9.1f}%")

def main():
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "horizon", "tws_known", "target"])
    p = load(L, SPEC)
    y = va["target"].to_numpy(); k = va["tws_known"].to_numpy(); h = va["horizon"].to_numpy()
    lat = va["lat"].to_numpy(); lon = va["lon"].to_numpy()
    e = p - y
    print(f"layout {L}, predictions {','.join(SPEC)}: {len(va)} rows")
    print(f"  RMSE {rmse(y, p):.4f}   persistence {rmse(y, k):.4f}   "
          f"gain {rmse(y, p) - rmse(y, k):+.4f}   bias {e.mean():+.4f}")

    print("\n  HORIZON  (share is of the test-weighted MSE: what a fix at that horizon is worth)")
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    wt = W * ok; wt = wt / wt.sum()
    print(f"  {'h':>3} {'n':>8} {'RMSE':>8} {'bias':>8} {'persist':>8} {'gain':>8} "
          f"{'test wt':>8} {'wtd share':>10}")
    for i in range(1, 8):
        m = h == i
        if not m.any():
            continue
        print(f"  {i:>3} {m.sum():>8} {rmse(y[m], p[m]):8.4f} {float(np.mean(p[m]-y[m])):+8.4f} "
              f"{rmse(y[m], k[m]):8.4f} {rmse(y[m], p[m])-rmse(y[m], k[m]):+8.4f} "
              f"{W[i-1]:8.3f} {100*wt[i-1]*mse[i-1]/np.nansum(wt*np.where(ok,mse,0)):9.1f}%")

    print("\n  VS PERSISTENCE")
    for i in range(1, 8):
        m = h == i
        if m.sum() < 50:
            continue
        worse = float(np.mean(np.abs(p[m] - y[m]) > np.abs(k[m] - y[m])))
        print(f"    h={i}: model is worse than the last observation on {worse*100:5.1f}% of rows")

    band = ((lat + 90) // 30).astype(int)
    lab = np.array([f"{b*30-90:+.0f}..{b*30-60:+.0f}" for b in band])
    table("LAT BAND", lab, y, p, k)

    cs = (pl.DataFrame({"lat": lat, "lon": lon, "y": y})
            .group_by(["lat", "lon"]).agg(pl.col("y").std().alias("sd")))
    sdv = (pl.DataFrame({"lat": lat, "lon": lon})
             .join(cs, on=["lat", "lon"], how="left")["sd"].fill_null(0.0).to_numpy())
    q = np.searchsorted(np.nanpercentile(sdv, np.arange(10, 100, 10)), sdv)
    table("CELL SD DECILE", q, y, p, k, order=list(range(10)))
    print("    (decile 0 = the steadiest cells, 9 = the most variable)")

    mth = np.array([d.month for d in va["time"].to_list()])
    table("TARGET MONTH", mth, y, p, k, order=list(range(1, 13)))

    print("\n  SPATIAL STRUCTURE OF THE RESIDUAL (correlation with the neighbour at grid lag d)")
    d = pl.DataFrame({"lat": lat, "lon": lon, "time": va["time"], "e": e})
    for lagd in (1, 2, 3, 5, 8):
        j = d.join(d.select(["lat", "lon", "time", "e"])
                    .with_columns((pl.col("lon") + lagd).alias("lon")).rename({"e": "e2"}),
                   on=["lat", "lon", "time"], how="inner")
        if len(j) < 1000:
            continue
        a = j["e"].to_numpy(); b = j["e2"].to_numpy()
        print(f"    lag {lagd} cells east: corr {np.corrcoef(a, b)[0,1]:+.3f}   (n={len(j)})")
    print("    Residual correlation still present at lag 2-3 means spatial smoothing has more to")
    print("    take; correlation gone by lag 1 means it is finished.")

    print("\n  WHERE THE MSE IS  (band x horizon, % of total)")
    tot = float(np.sum(e ** 2))
    bs = sorted(set(band.tolist()))
    print("      band      " + "".join(f"   h{i}" for i in range(1, 8)))
    for b in bs:
        row = []
        for i in range(1, 8):
            m = (band == b) & (h == i)
            row.append(f" {100*float(np.sum(e[m]**2))/tot:4.1f}" if m.any() else "    .")
        print(f"  {f'{b*30-90:+.0f}..{b*30-60:+.0f}':>10}  " + "".join(row))

    cell = (pl.DataFrame({"lat": lat, "lon": lon, "se": e ** 2})
              .group_by(["lat", "lon"]).agg(pl.col("se").sum().alias("se"), pl.len().alias("n"))
              .sort("se", descending=True))
    top = cell.head(20)
    print(f"\n  20 worst cells carry {100*float(top['se'].sum())/tot:.2f}% of total MSE:")
    print("    " + ", ".join(f"({r['lat']:.0f},{r['lon']:.0f})" for r in top.iter_rows(named=True)))
    n_cells = cell.height
    cum = np.cumsum(cell["se"].to_numpy()) / tot
    for frac in (0.25, 0.5, 0.8):
        i = int(np.searchsorted(cum, frac)) + 1
        print(f"    {frac*100:.0f}% of the error comes from {i} cells ({100*i/n_cells:.1f}% of the grid)")

if __name__ == "__main__":
    main()
