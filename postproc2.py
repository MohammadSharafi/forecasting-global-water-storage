import polars as pl, numpy as np, sys, glob, re, os
from anchor import anchor_fields, sample

RADII = (150, 200, 300, 400, 500, 700, 1000)

def load_preds(L, spec):
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
    p = np.asarray(p, dtype=np.float64)
    obs = pl.DataFrame({"lat": rows["lat"], "lon": rows["lon"],
                        "time": rows["time"], "TWS_t": p})
    months = sorted(set(rows["time"].to_list()))
    F, ti = anchor_fields(months, obs, tuple(radii))
    S = sample(F, ti, rows.select(["lat", "lon", "time"]).rename({"time": "t_known"}))
    return {km: np.where(np.isnan(v), p, v).astype(np.float64) for km, v in S.items()}

def beta_star(d, resid, fallback=1.0, min_n=1000):
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

    from smooth import smooth
    p_grid = smooth(va, p0, w=0.7, radius=1, iters=1)
    print(f"  current grid smooth (w=0.7)  RMSE={rmse(p_grid):.4f}")

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

        km = best[1][1]; s = S[km]; d = p - s
        bg = best[1][2]
        bs = [beta_star(d[h == hh], (y - s)[h == hh], fallback=bg) for hh in range(1, 8)]
        print(f"  [{label}] per-horizon beta at {km} km: " +
              " ".join(f"h{hh}={b:.2f}" for hh, b in zip(range(1, 8), bs)))
        ph = np.array([bs[int(np.clip(hh, 1, 7)) - 1] for hh in h])
        print(f"  [{label}] per-horizon beta applied:  RMSE={rmse(apply_shrink(p, s, ph)):.4f}"
              f"   (global beta {rmse(apply_shrink(p, s, best[1][2])):.4f})")

        q = apply_shrink(p, s, best[1][2]); k = va["tws_known"].to_numpy()
        a_star = beta_star(q - k, y - k)
        print(f"  [{label}] change scaling after shrink: alpha*={a_star:.3f} -> "
              f"RMSE={rmse(k + a_star * (q - k)):.4f}")

if __name__ == "__main__":
    main()
