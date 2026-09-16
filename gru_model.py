import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

W = int(os.environ.get("GRU_W", "18"))
COV = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

def build_panel(L):
    tr = pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
    te = pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    keep = ["lat", "lon", "time", "TWS_t"] + COV
    a = tr.select(keep)
    b = te.select(keep).with_columns(pl.when(pl.col("TWS_t").is_null()).then(None)
                                     .otherwise(pl.col("TWS_t")).alias("TWS_t"))
    p = pl.concat([a, b]).unique(["lat", "lon", "time"]).sort(["lat", "lon", "time"])
    return p

def main():
    L = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    panel = build_panel(L)
    months = sorted(set(panel["time"].to_list()))
    mi = {m: i for i, m in enumerate(months)}
    cells = panel.select(["lat", "lon"]).unique().sort(["lat", "lon"])
    ci = {(a, b): i for i, (a, b) in enumerate(zip(cells["lat"].to_list(), cells["lon"].to_list()))}
    NC, NM = len(ci), len(months)
    F = 1 + len(COV) + 2
    G = np.full((NC, NM, F), np.nan, dtype=np.float32)
    ri = np.array([ci[(a, b)] for a, b in zip(panel["lat"].to_list(), panel["lon"].to_list())])
    mj = np.array([mi[m] for m in panel["time"].to_list()])
    G[ri, mj, 0] = panel["TWS_t"].to_numpy().astype(np.float32)
    for k, c in enumerate(COV):
        G[ri, mj, 1 + k] = panel[c].to_numpy().astype(np.float32)
    mth = np.array([m.month for m in months], dtype=np.float32)
    G[:, :, -2] = np.sin(2 * np.pi * mth / 12)[None, :]
    G[:, :, -1] = np.cos(2 * np.pi * mth / 12)[None, :]
    print(f"panel {G.shape}, {time.time()-t0:.0f}s", flush=True)

    def windows(part):

        cols = ["lat", "lon", "time", "t_known", "tws_known"] + (["target"] if part == "tr" else [])
        d = pl.read_parquet(f"out/mats/{L}_{part}.parquet", columns=cols)
        c = np.array([ci.get((a, b), -1) for a, b in zip(d["lat"].to_list(), d["lon"].to_list())])
        t = np.array([mi.get(m, -1) for m in d["time"].to_list()])
        tk = np.array([mi.get(m, -1) for m in d["t_known"].to_list()])
        ok = (c >= 0) & (t >= W) & (tk >= 0)
        return d, c, t, tk, ok

    d, c, t, tk, ok = windows("tr")
    y = (d["target"].to_numpy() - d["tws_known"].to_numpy()).astype(np.float32)
    ok &= np.isfinite(y)
    idx = np.where(ok)[0]
    print(f"train rows usable {len(idx):,} of {len(ok):,}", flush=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = nn.Sequential().to(dev)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.g = nn.GRU(F + 1, 128, num_layers=2, batch_first=True, dropout=0.2)
            s.h = nn.Sequential(nn.Linear(128 + 1, 64), nn.SiLU(), nn.Linear(64, 1))
        def forward(s, x, hz):
            o, _ = s.g(x)
            return s.h(torch.cat([o[:, -1], hz], 1)).squeeze(1)
    net = Net().to(dev)

    def batch(ii):
        cc, tt, kk = c[ii], t[ii], tk[ii]
        st = tt - W + 1
        w = np.stack([G[cc[j], st[j]:tt[j] + 1] for j in range(len(ii))])

        obs = (np.arange(W)[None, :] + st[:, None]) <= kk[:, None]
        w = np.concatenate([w, obs[:, :, None].astype(np.float32)], 2)
        w[:, :, 0] = np.where(obs, np.nan_to_num(w[:, :, 0]), 0.0)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        hz = (tt - kk).astype(np.float32)[:, None]
        return torch.from_numpy(w).to(dev), torch.from_numpy(hz).to(dev)

    ysd = float(y[idx].std())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    EP = int(os.environ.get("GRU_EP", "2")); BS = 2048
    steps = EP * (len(idx) // BS + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.2)
    for ep in range(EP):
        perm = np.random.permutation(idx); tl = 0.0; n = 0
        for i in range(0, len(perm), BS):
            ii = perm[i:i + BS]
            xb, hb = batch(ii)
            yb = torch.from_numpy(y[ii] / ysd).to(dev)
            loss = ((net(xb, hb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tl += loss.item() * len(ii); n += len(ii)
        print(f"  epoch {ep+1} loss {tl/n:.4f} ({time.time()-t0:.0f}s)", flush=True)

    dv, cv, tv, kv, okv = windows("va")
    kn = dv["tws_known"].to_numpy().astype(np.float64)
    pred = np.full(len(dv), np.nan)
    net.eval()
    with torch.no_grad():
        jj = np.where(okv)[0]
        for i in range(0, len(jj), 8192):
            ii = jj[i:i + 8192]
            cc, tt, kk = cv[ii], tv[ii], kv[ii]
            st = tt - W + 1
            w = np.stack([G[cc[j], st[j]:tt[j] + 1] for j in range(len(ii))])
            obs = (np.arange(W)[None, :] + st[:, None]) <= kk[:, None]
            w = np.concatenate([w, obs[:, :, None].astype(np.float32)], 2)
            w[:, :, 0] = np.where(obs, np.nan_to_num(w[:, :, 0]), 0.0)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            hz = (tt - kk).astype(np.float32)[:, None]
            o = net(torch.from_numpy(w).to(dev), torch.from_numpy(hz).to(dev)).float().cpu().numpy()
            pred[ii] = o * ysd + kn[ii]
    pred = np.where(np.isfinite(pred), pred, kn)
    np.save(f"out/mats/pred_{L}_gru_v5x_noll_s{seed}{os.environ.get('GTAG','_g1')}.npy", pred)
    print(f"saved, coverage {okv.mean():.1%}, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
