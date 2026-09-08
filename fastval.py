"""A validation harness that needs nothing but Train.csv, so it runs anywhere in ~1 minute.

The main pipeline's validation is honest but expensive: build_mats reads four external
reanalysis archives and writes 200-column matrices, so trying an idea costs the better part of
an hour and cannot be done on a machine that has not downloaded the data. This rebuilds the
same framing from the released columns alone -- 40x40 cells, 2002-04..2015-09 -- and trains a
single LightGBM on ~40 features. It is worse in absolute terms than the real pipeline and it is
meant to be: what it buys is a fast, reproducible A/B on a structural question.

Framing, identical to the real task
-----------------------------------
A row is (cell, t, t_known): t_known <= t is the last month whose TWS is observed, the answer is
TWS(t+1), and horizon = t - t_known + 1. Six blocks of lengths 1,3,4,7,1,2 are laid end to end
with one observed anchor month before each -- the test's own shape, which yields the test's own
horizon mix (1/3 at h=1, down to 1/18 at h=7). Scores are that mix, weighted.

Leakage discipline
------------------
The per-cell, per-calendar-month climatology of TWS and of every covariate is refitted for each
placement with the validation window excluded -- fitting it on the whole record flatters the
model by about 0.005. TWS inside the window is visible only at the anchor months, exactly as
Test.csv exposes it, so `tws_prev` at an anchor is unavailable here as it is there. Covariates
ARE readable at every month up to t, because Test.csv provides them for masked rows too.

A placement must be gap-free: GRACE is missing 12 months of this record, and this harness
counts horizons in index steps, so a block laid across a gap asks for a multi-month lead while
labelling it h=1. `check()` refuses such a placement and prints the ones the record can carry.
On this Train.csv those are starts 12..77, i.e. 2003-08..2009-01.

usage:  ANCHOR=tws,ap RECUR=0 python fastval.py [start ...]
        start = column index of the first block; ANCHOR names the residual anchors to compare
"""
import os, sys, datetime as dt
import numpy as np, polars as pl, lightgbm as lgb

SPEI = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t"]
COV = SPEI + ["SOIL_MOISTURE_t"]
MIX = np.array([6, 4, 3, 2, 1, 1, 1], float); MIX /= MIX.sum()
USE = set()
# ANCHOR names the residual anchors to compare; the first is the incumbent and the
# one every other number in this harness was produced with.
ANCHORS = os.environ.get("ANCHOR", "tws").split(",")
RECUR = os.environ.get("RECUR", "1") == "1"


def addm(d, n):
    m = d.year * 12 + d.month - 1 + n
    return dt.date(m // 12, m % 12 + 1, 1)


tr = pl.read_csv("Train.csv", try_parse_dates=True)
# TWS panel: TWS_t at every train month, plus each row's target as TWS at t+1. The union
# covers 2011-01/2011-06 (absent as rows) and 2015-09.
obs = pl.concat([
    tr.select(["lat", "lon", "time", "TWS_t"]),
    tr.select(["lat", "lon", pl.col("time").dt.offset_by("1mo").alias("time"),
               pl.col("target").alias("TWS_t")]),
]).unique(["lat", "lon", "time"]).sort(["lat", "lon", "time"])
cov = tr.select(["lat", "lon", "time"] + COV)

MON = sorted(obs["time"].unique().to_list())
IDX = {m: i for i, m in enumerate(MON)}
CELLS = obs.select(["lat", "lon"]).unique().sort(["lat", "lon"])
CID = {(r[0], r[1]): i for i, r in enumerate(CELLS.iter_rows())}
NC, NM = len(CID), len(MON)

T = np.full((NC, NM), np.nan)
for la, lo, t, v in obs.iter_rows():
    T[CID[(la, lo)], IDX[t]] = v
C = np.full((NC, NM, len(COV)), np.nan)
for row in cov.iter_rows():
    C[CID[(row[0], row[1])], IDX[row[2]]] = row[3:]
MOY = np.array([m.month for m in MON])
# GRACE has real gaps, and every index step in this harness is treated as one calendar month:
# `horizon = ti - ki + 1`, the covariate window accumulates over `ki+1 .. ti`, and the answer is
# T[:, ti+1]. Where a month is missing from the record those are not the same thing, and a block
# laid across a gap quietly asks for a two- or three-month lead while calling it h=1. Layout C in
# the main pipeline had exactly this bug and it produced a validation layout that voted for the
# wrong configuration -- so placements here are required to be gap-free, the same way
# validation_c.py searches for a placement the record can carry.
MSER = np.array([m.year * 12 + m.month - 1 for m in MON])
CONT = np.zeros(NM, bool); CONT[:-1] = np.diff(MSER) == 1
_PRE = np.concatenate([[0], np.cumsum(CONT)])


def contiguous(a, b):
    """True when MON[a..b] are consecutive calendar months."""
    return b < NM and _PRE[b] - _PRE[a] == b - a

# per-cell, per-calendar-month climatology of TWS and of each covariate
def clim(A, use):
    """Per-cell, per-calendar-month mean and sd, fitted only on the months in `use`."""
    mu = np.full((A.shape[0], 12) + A.shape[2:], np.nan)
    sd = np.full_like(mu, np.nan)
    for k in range(12):
        s = (MOY == k + 1) & use
        mu[:, k] = np.nanmean(A[:, s], axis=1)
        sd[:, k] = np.nanstd(A[:, s], axis=1)
    return mu, np.where(sd > 1e-6, sd, 1.0)

TMU = TSD = CMU = CSD = None

# 4-neighbour index on the 40x40 grid
lats = sorted({k[0] for k in CID}); lons = sorted({k[1] for k in CID})
GI = np.full((len(lats), len(lons)), -1, int)
for (la, lo), i in CID.items():
    GI[lats.index(la), lons.index(lo)] = i
NB = np.full((NC, 4), -1, int)
for a in range(len(lats)):
    for b in range(len(lons)):
        i = GI[a, b]
        for k, (da, db) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            aa, bb = a + da, b + db
            if 0 <= aa < len(lats) and 0 <= bb < len(lons):
                NB[i, k] = GI[aa, bb]


def nbmean(v):
    out = np.full((NC, 4), np.nan)
    for k in range(4):
        ok = NB[:, k] >= 0
        out[ok, k] = v[NB[ok, k]]
    return np.nanmean(out, axis=1)


NLA, NLO = len(lats), len(lons)
RC = np.full((NLA, NLO), -1, int)
for (la, lo), i in CID.items():
    RC[lats.index(la), lons.index(lo)] = i
ORD = RC.ravel()                       # grid order -> cell id


def boxmean(v, r, side=None):
    """Mean of `v` over a (2r+1) box around each cell. side='w'/'e' takes only the
    columns strictly west/east of the cell -- the Amazon drains west to east, so the
    upstream half is not the same field as the downstream half."""
    G = np.full((NLA, NLO), np.nan); G[np.unravel_index(ORD, (NLA, NLO))] = v[ORD]
    P = np.pad(G, r, constant_values=np.nan)
    acc = np.zeros_like(G); cnt = np.zeros_like(G)
    for da in range(-r, r + 1):
        for db in range(-r, r + 1):
            if side == "w" and db >= 0:
                continue
            if side == "e" and db <= 0:
                continue
            w = P[r + da:r + da + NLA, r + db:r + db + NLO]
            m = np.isfinite(w)
            acc[m] += w[m]; cnt[m] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    res = np.full(NC, np.nan); res[ORD] = out.ravel()
    return res


def feats(cells, ti, ki, Tsrc):
    """Row = cell at month MON[ti], last observed TWS at MON[ki]; answer is TWS(ti+1).
    Reads TWS only from Tsrc at columns <= ki, covariates at columns <= ti."""
    h = (ti - ki + 1).astype(float)
    mt, mn = MOY[ti], MOY[np.minimum(ti + 1, NM - 1)]
    k = Tsrc[cells, ki]
    ak = (k - TMU[cells, mt - 1]) / TSD[cells, mt - 1]     # anomaly at the known month
    kprev = np.where(ki > 0, Tsrc[cells, np.maximum(ki - 1, 0)], np.nan)
    kprev2 = np.where(ki > 1, Tsrc[cells, np.maximum(ki - 2, 0)], np.nan)
    cn = TMU[cells, mn - 1]; sn = TSD[cells, mn - 1]
    F = [h, np.sin(2 * np.pi * mt / 12), np.cos(2 * np.pi * mt / 12),
         np.sin(2 * np.pi * mn / 12), np.cos(2 * np.pi * mn / 12),
         k, ak, kprev, kprev2, k - kprev, kprev - kprev2, cn, sn,
         cn - k, TMU[cells, MOY[ki] - 1]]
    names = ["h", "sm_t", "cm_t", "sm_n", "cm_n", "tws_k", "anom_k", "tws_k1", "tws_k2",
             "d1", "d2", "clim_n", "climsd_n", "clim_gap", "clim_k"]
    # covariates at the row's own month t (observed even where TWS is masked), standardised
    for j, c in enumerate(COV):
        v = C[cells, ti, j]
        F += [v, (v - CMU[cells, mt - 1, j]) / CSD[cells, mt - 1, j],
              v - C[cells, ki, j]]
        names += [c, c + "_z", c + "_d"]
    # covariate mean over the unobserved window (t_known, t]
    for j, c in enumerate(COV):
        acc = np.full(len(cells), np.nan)
        for hh in np.unique(ti - ki):
            s = (ti - ki) == hh
            if hh == 0:
                continue
            idx = np.stack([ki[s] + o for o in range(1, hh + 1)], 1)
            z = (C[cells[s][:, None], idx, j] - CMU[cells[s][:, None], MOY[idx] - 1, j]) / \
                CSD[cells[s][:, None], MOY[idx] - 1, j]
            acc[s] = z.mean(1)
        F += [acc]; names += [c + "_acc"]
    # spatial: neighbour mean of the known-month anomaly
    nbz = np.full(len(cells), np.nan)
    for kk in np.unique(ki):
        s = ki == kk
        col = (Tsrc[:, kk] - TMU[:, MOY[kk] - 1]) / TSD[:, MOY[kk] - 1]
        nbz[s] = nbmean(col)[cells[s]]
    F += [nbz]; names += ["nb_anom_k"]

    if "mem" in USE:
        # groundwater memory: the cell's own anomaly 12 and 24 months before the anchor
        for lag in (12, 24):
            j = ki - lag
            v = np.where(j >= 0, Tsrc[cells, np.maximum(j, 0)], np.nan)
            F += [(v - TMU[cells, MOY[np.maximum(j, 0)] - 1]) /
                  TSD[cells, MOY[np.maximum(j, 0)] - 1]]
            names += [f"anom_k_lag{lag}"]
        # 24-month trend in the anomaly, ending at the anchor
        j0 = np.maximum(ki - 24, 0)
        a0 = (Tsrc[cells, j0] - TMU[cells, MOY[j0] - 1]) / TSD[cells, MOY[j0] - 1]
        F += [(ak - a0) / np.maximum(ki - j0, 1)]; names += ["anom_trend24"]

    if "box" in USE or "dir" in USE:
        cols = {}
        for kk in np.unique(ki):
            cols[kk] = (Tsrc[:, kk] - TMU[:, MOY[kk] - 1]) / TSD[:, MOY[kk] - 1]
        if "box" in USE:
            for r in (3, 6):
                v = np.full(len(cells), np.nan)
                for kk, col in cols.items():
                    s_ = ki == kk
                    v[s_] = boxmean(col, r)[cells[s_]]
                F += [v]; names += [f"box{r}_anom_k"]
        if "dir" in USE:
            for side in ("w", "e"):
                v = np.full(len(cells), np.nan)
                for kk, col in cols.items():
                    s_ = ki == kk
                    v[s_] = boxmean(col, 6, side)[cells[s_]]
                F += [v]; names += [f"box6{side}_anom_k"]

    if "upcov" in USE:
        # upstream forcing: standardised soil moisture / SPEI_03 averaged over the cells
        # to the west, at the row's own month
        for j, c in enumerate(("SOIL_MOISTURE_t", "SPEI_03_t")):
            jj = COV.index(c)
            v = np.full(len(cells), np.nan)
            for tt in np.unique(ti):
                s_ = ti == tt
                col = (C[:, tt, jj] - CMU[:, MOY[tt] - 1, jj]) / CSD[:, MOY[tt] - 1, jj]
                v[s_] = boxmean(col, 6, "w")[cells[s_]]
            F += [v]; names += [c + "_upw"]

    X = np.column_stack([np.asarray(f, float) for f in F])
    return X, names


# ---------------------------------------------------------------- the residual anchor
# Each model predicts target - anchor and the anchor is added back. `tws` is the incumbent: the
# last observed TWS. It is a poor anchor under the test's horizon mix -- two thirds of rows are
# two or more months stale, and a stale level is a weak predictor of a standardised anomaly seven
# months later -- so the model spends capacity undoing it. `ap` anchors on anomaly persistence
# instead: carry the anchor month's departure from its own climatology forward to the target
# month's climatology. Both anchors are built from columns the model already has as features
# (tws_k, clim_k, clim_n), so this changes what the model must learn, not what it may see.
def anchor_of(kind, cells, ki, mn, Tsrc):
    k = Tsrc[cells, ki]
    if kind == "tws":
        return k
    if kind == "ap":
        return TMU[cells, mn - 1] + (k - TMU[cells, MOY[ki] - 1])
    raise SystemExit(f"unknown anchor {kind!r}")


# ---------------------------------------------------------------- experiment
BLOCKS = [1, 3, 4, 7, 1, 2]          # chain lengths the test actually uses
PAR = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           verbose=-1, num_threads=4, seed=1)


def rows_for(anchors):
    """(cells, ti, ki) for every row in the chains starting at the given anchor columns."""
    cs, ts, ks = [], [], []
    for a, L in anchors:
        for o in range(L):
            cs.append(np.arange(NC)); ts.append(np.full(NC, a + o)); ks.append(np.full(NC, a))
    return np.concatenate(cs), np.concatenate(ts), np.concatenate(ks)


def place(start):
    """Lay the six blocks out end to end from column `start`, one spacer month between
    them (the anchor of the next block), and return (anchor, length) pairs."""
    out, a = [], start
    for L in BLOCKS:
        out.append((a, L)); a += L + 1
    return out, a


SPAN = sum(BLOCKS) + len(BLOCKS)          # place() reaches start+SPAN; targets need start+SPAN-1


def valid_starts():
    return [s for s in range(2, NM - SPAN - 1) if contiguous(s, s + SPAN)]


def check(start):
    """Refuse a placement the record cannot carry gap-free, and say which it can."""
    if start + SPAN >= NM:
        v = valid_starts()
        raise SystemExit(f"start={start}: the window reaches index {start + SPAN}, past the end of "
                         f"the record ({NM - 1}). Gap-free starts: {v[0]}..{v[-1]}")
    if not contiguous(start, start + SPAN):
        miss = SPAN - (_PRE[start + SPAN] - _PRE[start])
        v = valid_starts()
        raise SystemExit(
            f"start={start} ({MON[start]}..{MON[start + SPAN]}): {miss} calendar month(s) are "
            f"missing inside the window, so a block laid here would ask for multi-month leads "
            f"while calling them h=1. Gap-free starts on this record: {v[0]}..{v[-1]} "
            f"({MON[v[0]]}..{MON[v[-1]]}).")


def mixed(y, p, h):
    mse = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                    for i in range(1, 8)])
    ok = ~np.isnan(mse)
    return float(np.sqrt(np.sum(MIX[ok] * mse[ok]) / np.sum(MIX[ok]))), mse


def run(start):
    global TMU, TSD, CMU, CSD
    check(start)
    anch, end = place(start)
    va_c, va_t, va_k = rows_for(anch)
    lo, hi = anch[0][0], end                      # validation window, inclusive of targets
    use = np.ones(NM, bool); use[lo:hi + 1] = False
    TMU, TSD = clim(T, use)
    CMU, CSD = clim(C, use)
    # training rows: every (anchor, horizon) pair outside the validation window
    cs, ts, ks = [], [], []
    rng = np.random.default_rng(0)
    for a in range(2, NM - 2):
        if lo - 8 <= a <= hi:
            continue
        for hh in rng.choice(7, size=3, replace=False) + 1:
            t = a + hh - 1
            if t + 1 >= NM or lo - 8 <= t + 1 <= hi:
                continue
            cs.append(np.arange(NC)); ts.append(np.full(NC, t)); ks.append(np.full(NC, a))
    tr_c, tr_t, tr_k = np.concatenate(cs), np.concatenate(ts), np.concatenate(ks)

    # a training row is only usable where its own (k .. t+1) span is gap-free, for the same
    # reason the placement must be: otherwise its horizon label and its covariate window are
    # measured in index steps that are not months
    good = np.array([contiguous(a, b + 1) for a, b in zip(tr_k, tr_t)])
    tr_c, tr_t, tr_k = tr_c[good], tr_t[good], tr_k[good]
    Xtr, names = feats(tr_c, tr_t, tr_k, T)
    mn_tr = MOY[np.minimum(tr_t + 1, NM - 1)]
    A_tr = {a: anchor_of(a, tr_c, tr_k, mn_tr, T) for a in ANCHORS}
    # one finite mask over every anchor, so each variant trains on exactly the same rows
    ok = np.isfinite(T[tr_c, tr_t + 1]) & np.isfinite(Xtr[:, 5])
    for a in ANCHORS:
        ok &= np.isfinite(A_tr[a])
    Xtr = Xtr[ok]
    ytr = T[tr_c, tr_t + 1][ok] - A_tr[ANCHORS[0]][ok]

    # test-time TWS availability: everything before the window, plus the anchor months
    # themselves -- exactly what Test.csv exposes. Nothing else inside the window is observed.
    Tva = T.copy(); Tva[:, lo:] = np.nan
    for a, L in anch:
        Tva[:, a] = T[:, a]
    Xva, _ = feats(va_c, va_t, va_k, Tva)
    yva = T[va_c, va_t + 1]
    hva = (va_t - va_k + 1).astype(int)
    mn_va = MOY[np.minimum(va_t + 1, NM - 1)]
    A_va = {a: anchor_of(a, va_c, va_k, mn_va, Tva) for a in ANCHORS}
    pred, m = {}, None
    for a in ANCHORS:
        y = T[tr_c, tr_t + 1][ok] - A_tr[a][ok]
        ma = lgb.train(PAR, lgb.Dataset(Xtr, y, feature_name=names), num_boost_round=700)
        pred[a] = A_va[a] + ma.predict(Xva)
        if a == ANCHORS[0]:
            m = ma
    direct = pred[ANCHORS[0]]

    # --- recursive: an h=1 model chained forward, its own output fed in as tws_known.
    # Settled in session 10a/10 and skipped by default now: it costs a second training per
    # placement and answers a question that is closed.
    if not RECUR:
        finq = np.isfinite(yva)
        for a in ANCHORS:
            finq &= np.isfinite(pred[a])
        res = {a: mixed(yva[finq], pred[a][finq], hva[finq]) for a in ANCHORS}
        rp, _ = mixed(yva[finq], Tva[va_c, va_k][finq], hva[finq])
        rc, _ = mixed(yva[finq], TMU[va_c, MOY[va_t + 1] - 1][finq], hva[finq])
        return res, rp, rc, int(finq.sum())
    o1 = (tr_t - tr_k) == 0
    m1 = lgb.train(PAR, lgb.Dataset(Xtr[o1[ok]], ytr[o1[ok]], feature_name=names),
                   num_boost_round=700)
    Trec = T.copy()
    Trec[:, lo:] = np.nan                       # nothing observed inside the window
    for a, L in anch:
        Trec[:, a] = T[:, a]                    # the anchor month is given
    rec = np.full(len(va_c), np.nan)
    for a, L in anch:
        for o in range(L):                      # step from a+o to a+o+1
            c = np.arange(NC); ti = np.full(NC, a + o); ki = np.full(NC, a + o)
            Xs, _ = feats(c, ti, ki, Trec)
            Trec[:, a + o + 1] = Trec[:, a + o] + m1.predict(Xs)
            s = (va_k == a) & (va_t == a + o)
            rec[s] = Trec[va_c[s], a + o + 1]
    fin = np.isfinite(yva) & np.isfinite(direct) & np.isfinite(rec)
    rd, msed = mixed(yva[fin], direct[fin], hva[fin])
    rr, mser = mixed(yva[fin], rec[fin], hva[fin])
    rp, _ = mixed(yva[fin], Tva[va_c, va_k][fin], hva[fin])
    rc, _ = mixed(yva[fin], TMU[va_c, MOY[va_t + 1] - 1][fin], hva[fin])
    return rd, rr, rp, rc, np.sqrt(msed), np.sqrt(mser), fin.sum()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].lstrip("-").isdigit():
        USE = set(args.pop(0).split(","))
    tot = {a: [] for a in ANCHORS}
    for start in [int(x) for x in (args or ["16", "44", "72"])]:
        out = run(start)
        if not RECUR:
            res, rp, rc, n = out
            print(f"start={start} ({MON[start]}) n={n}", flush=True)
            print(f"  persistence {rp:.4f}  climatology {rc:.4f}")
            base = res[ANCHORS[0]][0]
            for a in ANCHORS:
                r, mse = res[a]
                tot[a].append(r)
                d = f"  {r - base:+.4f}" if a != ANCHORS[0] else "  (incumbent)"
                print(f"  anchor {a:4} {r:.4f}{d}")
                print("    per-h  " + " ".join(f"{v:.3f}" for v in np.sqrt(mse)), flush=True)
            continue
        rd, rr, rp, rc, ed, er, n = out
        print(f"start={start} ({MON[start]}) n={n}", flush=True)
        print(f"  persistence {rp:.4f}  climatology {rc:.4f}  direct {rd:.4f}  "
              f"recursive {rr:.4f}  delta {rr-rd:+.4f}")
        print("  per-h direct    " + " ".join(f"{v:.3f}" for v in ed))
        print("  per-h recursive " + " ".join(f"{v:.3f}" for v in er), flush=True)
    if not RECUR and len(ANCHORS) > 1:
        # the standing rule: a change is adopted only if it wins on EVERY placement
        base = np.array(tot[ANCHORS[0]])
        print(f"\n  {'anchor':6} " + " ".join(f"{s:>8}" for s in ["p1", "p2", "p3"]) + "   verdict")
        print(f"  {ANCHORS[0]:6} " + " ".join(f"{v:8.4f}" for v in base) + "   incumbent")
        for a in ANCHORS[1:]:
            v = np.array(tot[a]); d = v - base
            win = bool((d < -0.0003).all())
            print(f"  {a:6} " + " ".join(f"{x:8.4f}" for x in v) +
                  ("   ADOPT: wins by >0.0003 on every placement" if win else
                   f"   reject: deltas {' '.join(f'{x:+.4f}' for x in d)}"))
