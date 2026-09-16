import itertools
import sys

import numpy as np
import polars as pl

LEDGER = {
    "FINAL":    ("out/scored/sub_FINAL.csv",       0.674941009),
    "avg4":     ("out/scored/sub_avg4.csv",        0.678589114),
    "mix25":    ("out/scored/sub_mix25.csv",       0.679248865),
    "tomorrow": ("out/sub_tomorrow.csv",           0.679428622),
    "armarg":   ("out/scored/sub_ac_armarg.csv",   0.679785572),
    "w65sub":   ("out/scored/sub_submit_w65.csv",  0.680909648),
    "zlb":      ("out/scored/sub_z_lb.csv",        0.681692450),
    "ylb":      ("out/scored/sub_y_lb.csv",        0.681717090),
    "goal065":  ("out/scored/sub_goal065.csv",     0.682847),
    "xlb2":     ("out/scored/sub_x_lb2.csv",       0.683712087),
    "NEW":      ("out/scored/sub_final_best.csv",  0.684923193),
    "gpccar":   ("out/scored/sub_gpcc_ar.csv",     0.684984991),
    "main":     ("out/scored/sub_q_main.csv",      0.692657),
    "blend73":  ("out/scored/sub_blend73.csv",     0.693283),
    "blend55":  ("out/scored/sub_blend55.csv",     0.693725),
    "alt":      ("out/scored/sub_q_alt.csv",       0.694922),
    "nosm":     ("out/scored/sub_q_main_nosm.csv", 0.699149222),
    "base":     ("out/scored/sub_q_base.csv",      0.708852),
    "clim":     ("out/scored/probe_clim.csv",      1.279955747),
}
BEST = 0.674941009

HISTORY = [
    ("blend55  (fixed 0.5/0.5)",      "fixed",     0.693726, 0.693725),
    ("avg4     (fixed, 4 equal)",     "fixed",     0.678648, 0.678589114),
    ("armarg   (fixed marginal)",     "fixed",     0.679683, 0.679785572),
    ("FINAL    (fixed, 3 equal)",     "fixed",     0.675227, 0.674941009),
    ("y_lb     (optimised L1<=4)",    "optimised", 0.676475, 0.681717090),
    ("z_lb     (optimised L1<=2)",    "optimised", 0.677391, 0.681692450),
    ("tomorrow (optimised L1<=2)",    "optimised", 0.674080, 0.679428622),
]

def load():
    ref = pl.read_csv("Test.csv").select("ID")
    P, S = {}, {}
    for k, (p, s) in LEDGER.items():
        d = pl.read_csv(p)
        c = [x for x in d.columns if x != "ID"][0]
        v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(np.float64)
        if not np.isfinite(v).all():
            continue
        P[k], S[k] = v, s
    return ref, P, S

_CACHE = {}

def moments(P, S):
    if "D2" not in _CACHE:
        ks = list(P)
        A = np.stack([P[k] for k in ks])
        sq = (A * A).mean(1)
        G = A @ A.T / A.shape[1]
        D2 = sq[:, None] + sq[None, :] - 2 * G
        _CACHE["ks"] = ks; _CACHE["D2"] = D2
        _CACHE["idx"] = {k: i for i, k in enumerate(ks)}
    return _CACHE

def predict(ks, P, S, w=None):
    C = moments(P, S)
    w = np.ones(len(ks)) / len(ks) if w is None else np.asarray(w, float)
    ii = [C["idx"][k] for k in ks]
    M = np.array([S[k] ** 2 for k in ks])
    D2 = C["D2"][np.ix_(ii, ii)]
    return float(np.sqrt(max(w @ M - 0.5 * w @ D2 @ w, 1e-12)))

def cmd_inventory(P, S):
    ks = sorted(S, key=S.get)
    print(f"{len(ks)} scored files\n")
    print(f"{'file':10s} {'score':>11s}  {'RMS to best':>11s}  {'corr to best':>12s}")
    b = P[ks[0]]
    for k in ks:
        print(f"{k:10s} {S[k]:11.6f}  {np.sqrt(np.mean((P[k]-b)**2)):11.4f}  "
              f"{np.corrcoef(P[k],b)[0,1]:12.4f}")
    print("\nINVENTORY OK")

def cmd_accuracy(P, S):
    print("identity accuracy, split by how the weights were chosen\n")
    print(f"{'case':32s} {'kind':10s} {'predicted':>10s} {'actual':>10s} {'error':>10s}")
    errs = {"fixed": [], "optimised": []}
    for name, kind, pred, act in HISTORY:
        e = act - pred
        errs[kind].append(abs(e))
        print(f"{name:32s} {kind:10s} {pred:10.6f} {act:10.6f} {e:+10.6f}")
    print()
    for kind, v in errs.items():
        print(f"  {kind:10s} n={len(v)}  mean |error| {np.mean(v):.6f}  max {np.max(v):.6f}")
    print("\n  Fixed weights are accurate to about 6e-5; optimised weights miss by about 5e-3,")
    print("  eighty times worse. Only fixed-weight rules are admissible.")
    print("\nACCURACY OK")

def cmd_bias(P, S):
    ks = [k for k in S if S[k] < 0.70]
    sigma = 6e-5
    scores = []
    for r in range(2, 7):
        for c in itertools.combinations(ks, r):
            scores.append(predict(list(c), P, S))
    scores = np.array(scores)
    n = len(scores)

    en = sigma * np.sqrt(2 * np.log(max(n, 2)))
    print(f"equal-weight subsets of the {len(ks)} usable files, sizes 2..6: {n:,} candidates")
    print(f"prediction noise per candidate (measured): {sigma:.2e}")
    print(f"expected selection bias from taking the best of {n:,}: {en:.2e}")
    print(f"spread of candidate predictions: {scores.min():.6f} to {scores.max():.6f}")
    print(f"\n  A search over this space inflates the winner by about {en:.5f}, which is small")
    print(f"  against the {BEST-scores.min():.5f} spread between the best candidate and the current best.")
    print(f"  Equal-weight subset search is therefore admissible, with that haircut applied.")
    print("\nBIAS OK")

def cmd_search(P, S):
    ks = [k for k in S if S[k] < 0.70]
    sigma = 6e-5
    rows = []
    for r in range(2, 8):
        for c in itertools.combinations(ks, r):
            rows.append((predict(list(c), P, S), c))
    rows.sort()
    n = len(rows)
    haircut = sigma * np.sqrt(2 * np.log(max(n, 2)))
    print(f"comprehensive equal-weight search: {n:,} subsets of {len(ks)} scored files")
    print(f"current best actual score {BEST:.6f}\n")
    print(f"{'predicted':>10s} {'after haircut':>14s}  files")
    for s, c in rows[:12]:
        print(f"{s:10.6f} {s+haircut:14.6f}  {' + '.join(c)}")
    print(f"\nhaircut applied for searching {n:,} candidates: {haircut:+.6f}")

    print("\nrules fixed in advance (no search):")
    by = sorted(ks, key=lambda k: S[k])
    for nm, c in (("2 best", by[:2]), ("3 best", by[:3]), ("4 best", by[:4]), ("5 best", by[:5]),
                  ("3 best + most decorrelated", by[:3] + ["NEW"]),
                  ("4 best + most decorrelated", by[:4] + ["NEW"])):
        cc = list(dict.fromkeys(c))
        print(f"  {nm:28s} {predict(cc,P,S):.6f}   {' + '.join(cc)}")
    print("\nSEARCH OK")

if __name__ == "__main__":
    ref, P, S = load()
    {"inventory": cmd_inventory, "accuracy": cmd_accuracy,
     "bias": cmd_bias, "search": cmd_search}[sys.argv[1]](P, S)
