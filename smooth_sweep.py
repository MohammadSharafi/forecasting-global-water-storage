import sys

import numpy as np
import polars as pl

MIX = np.array([.3333, .2222, .1667, .1111, .0556, .0556, .0556])

def grid(lat, lon):
    ula, ulo = np.unique(lat), np.unique(lon)
    ia = np.searchsorted(ula, lat); io = np.searchsorted(ulo, lon)
    return ula, ulo, ia, io

def smooth(vals, lat, lon, tm, sigma, lam):
    ula, ulo, ia, io = grid(lat, lon)
    R = int(np.ceil(2.5 * sigma))
    if R < 1:
        return vals.copy()
    dla = ula[1] - ula[0] if len(ula) > 1 else 1.0
    dlo = ulo[1] - ulo[0] if len(ulo) > 1 else 1.0
    out = vals.copy()
    for t in np.unique(tm):
        m = tm == t
        A = np.full((len(ula), len(ulo)), np.nan)
        A[ia[m], io[m]] = vals[m]
        ok = np.isfinite(A).astype(float)
        Af = np.where(np.isfinite(A), A, 0.0)
        num = np.zeros_like(Af); den = np.zeros_like(Af)
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):

                dy = di * dla
                dx = dj * dlo * np.cos(np.deg2rad(ula))[:, None]
                wgt = np.exp(-0.5 * ((dy ** 2 + dx ** 2) / sigma ** 2))
                num += wgt * np.roll(np.roll(Af, di, 0), dj, 1)
                den += wgt * np.roll(np.roll(ok, di, 0), dj, 1)
        nb = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        v = lam * A + (1 - lam) * nb
        v = np.where(np.isfinite(v), v, A)
        out[m] = v[ia[m], io[m]]
    return out

def main():
    L = sys.argv[1] if len(sys.argv) > 1 else "Avn2"
    va = pl.read_parquet(f"out/mats/{L}_va.parquet",
                         columns=["lat", "lon", "time", "horizon", "target", "tws_known"])
    y = va["target"].to_numpy().astype(float); h = va["horizon"].to_numpy().astype(int)
    k = va["tws_known"].to_numpy().astype(float)
    lat = va["lat"].to_numpy().astype(float); lon = va["lon"].to_numpy().astype(float)
    tm = np.array([str(x)[:7] for x in va["time"].to_list()])
    T = np.load(f"out/mats/pred_{L}_lgbd_v5x_noll_s0_cap.npy")
    X = np.load(f"out/mats/pred_{L}_xgb_v5x_noll_s0_d9.npy")
    N = np.load(f"out/mats/pred_{L}_mlp_v5x_noll_s0_nn.npy")
    p = 0.70 * (0.65 * T + 0.35 * X) + 0.30 * N
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(k)

    def mix(pred):
        m = ok & np.isfinite(pred)
        return float(np.sqrt(sum(MIX[i-1] * np.mean((y[m & (h == i)] - pred[m & (h == i)]) ** 2)
                                 for i in range(1, 8) if (m & (h == i)).any())))
    base = mix(p)
    print(f"layout {L}: unsmoothed blend {base:.4f}")
    res = p - k
    best = (base, None)
    for sigma in (0.5, 1.0, 1.5, 2.0):
        for lam in (0.5, 0.7, 0.85):
            q = k + smooth(res, lat, lon, tm, sigma, lam)
            s = mix(q)
            if s < best[0]: best = (s, (sigma, lam))
            print(f"  sigma={sigma:.1f} lam={lam:.2f}  {s:.4f}  ({s-base:+.4f})")
    print(f"  best {best[1]} -> {best[0]:.4f} ({best[0]-base:+.4f})")

if __name__ == "__main__":
    main()
