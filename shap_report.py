"""Feature attributions for the trustworthiness report, from the FINAL models as they were
trained -- no refit, no surrogate.

LightGBM computes exact TreeSHAP itself (`pred_contrib=True`), so this needs no extra
dependency and cannot drift from the model being explained: the feature list comes from the
booster, not from a config file that might have moved on.

Two things it reports that a single global ranking hides:

  * the split between h=1 and h>=2. At h=1 `t_known == t`, so the accumulation window is empty
    and every difference feature is identically zero -- 38 of the 201 features carry nothing on
    the largest third of the test. A global mean would average that away.
  * attribution by feature FAMILY, which is what the report actually argues about (covariate
    anomalies against raw levels, smoothed anchors against per-cell statistics).

writes out/report/shap_mean_abs.json and out/report/shap.md
"""
import json, os, re
import numpy as np
import polars as pl
import lightgbm as lgb

os.makedirs("out/report", exist_ok=True)
L = os.environ.get("SHAP_LAYOUT", "FINAL")

va = pl.read_parquet(f"out/mats/{L}_va.parquet").hstack(
    pl.read_parquet(f"out/mats/{L}_va_anchor.parquet"))
h = va["horizon"].to_numpy().astype(int)

# The boosters were trained from a numpy array, so they carry Column_0..Column_N rather than
# names. run_models writes the list it actually trained on beside them; that file is the only
# thing that knows the column ORDER, which is what makes the positional mapping sound.
used = json.load(open(f"out/mats/used_A_lgb_v5x_noll{os.environ.get('SHAP_TAG', '_bw')}.json"))
F = used["features"]
assert used["dropf"] == [], f"feature list is an ablation: dropf={used['dropf']}"
X = va.select(F).to_numpy().astype(np.float32)

models = sorted(f for f in os.listdir("out/mats")
                if re.fullmatch(rf"model_{L}_lgb_v5x_noll_s\d+\.txt", f))
assert models, f"no FINAL lgb models for layout {L}"

tot, tot1, tot2 = None, None, None
for mf in models:
    b = lgb.Booster(model_file=f"out/mats/{mf}")
    assert b.num_feature() == len(F), f"{mf} has {b.num_feature()} features, list has {len(F)}"
    c = np.abs(b.predict(X, pred_contrib=True)[:, :-1])   # last column is the bias term
    tot = c.mean(0) if tot is None else tot + c.mean(0)
    tot1 = c[h == 1].mean(0) if tot1 is None else tot1 + c[h == 1].mean(0)
    tot2 = c[h >= 2].mean(0) if tot2 is None else tot2 + c[h >= 2].mean(0)
n = len(models)
mean_abs, m1, m2 = tot / n, tot1 / n, tot2 / n
order = np.argsort(mean_abs)[::-1]

assert not any(f in ("lat", "lon") for f in F), "lat/lon reached the model"
json.dump({F[i]: {"all": float(mean_abs[i]), "h1": float(m1[i]), "h2plus": float(m2[i])}
           for i in order}, open("out/report/shap_mean_abs.json", "w"), indent=1)


def family(f):
    if f.startswith("an_"):                       return "covariate anomalies"
    if re.match(r"d?sa\d|sa_grad", f):            return "smoothed anchors"
    if f.startswith(("e5",)):                     return "ERA5"
    if f.startswith(("n", "r2", "cpc")) and f not in ("n_win",): return "NCEP / CPC"
    if f.startswith(("SPEI", "SOIL")):            return "released covariates"
    if f.startswith(("z", "band")):               return "zonal scale"
    if f in ("clim_next", "clim_known", "anom_known", "anom_persist", "anom_ly",
             "clim_sd_next", "cmean", "csd", "ac1", "mad1"):   return "climatology / cell stats"
    if f.startswith("tws") or f in ("slope", "trend24", "trend60", "trend_persist",
                                    "trend60_persist"):        return "TWS history"
    if f in ("horizon", "m", "m_next", "n_win"):  return "calendar / geometry"
    return "other"


fam = {}
for i, f in enumerate(F):
    k = family(f)
    a = fam.setdefault(k, [0.0, 0.0, 0.0, 0])
    a[0] += mean_abs[i]; a[1] += m1[i]; a[2] += m2[i]; a[3] += 1
S, S1, S2 = mean_abs.sum(), m1.sum(), m2.sum()

with open("out/report/shap.md", "w") as fh:
    fh.write(f"# Feature attribution, layout {L}, {n} seed(s), {len(va)} validation rows\n\n")
    fh.write("Exact TreeSHAP from the trained boosters. Units are the residual target.\n\n")
    fh.write("## By family (share of total attribution)\n\n")
    fh.write("| family | features | all rows | h=1 | h>=2 |\n|---|--:|--:|--:|--:|\n")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        fh.write(f"| {k} | {v[3]} | {100*v[0]/S:.1f}% | {100*v[1]/S1:.1f}% | {100*v[2]/S2:.1f}% |\n")
    fh.write("\n## Top 25 features\n\n| # | feature | mean \\|SHAP\\| | h=1 | h>=2 |\n|--:|---|--:|--:|--:|\n")
    for r, i in enumerate(order[:25], 1):
        fh.write(f"| {r} | `{F[i]}` | {mean_abs[i]:.4f} | {m1[i]:.4f} | {m2[i]:.4f} |\n")
    dead = [F[i] for i in range(len(F)) if m1[i] == 0]
    fh.write(f"\n## Features with zero attribution at h=1: {len(dead)} of {len(F)}\n\n")
    fh.write(", ".join(f"`{d}`" for d in dead[:60]) + ("" if len(dead) <= 60 else " …") + "\n")

print(open("out/report/shap.md").read())
