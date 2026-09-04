# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for 15,715
one-degree cells when the current-month TWS is hidden for two-thirds of test rows.

## 1. Problem structure

The test set is six blocks of consecutive months (2015-09; 2016-01→03; 2016-06→09;
2016-12→2017-06; 2018-07; 2018-11→12). `TWS_t` is observed only in each block's first
month and masked afterwards, so the horizon from the last observed TWS to the target runs
from **1 to 7 months** (test mix 33/22/17/11/5.5/5.5/5.5%). Persistence — repeat the last
observed value — is the baseline that matters; the seasonal climatology is *not* competitive
(1.12 vs 0.76 on validation) because this TWS record is dominated by interannual anomalies.
On the public leaderboard persistence scores **0.886** (measured with a dedicated
submission); the organisers' starter, which median-imputes masked `TWS_t`, scores 0.90.

## 2. Validation that mirrors the test

Two independent pseudo-test layouts carved from the training years with the same
block/mask structure (A: 2012–2015, 18 months, 72% masked; B: 2009–2012, 23 months, 70%
masked). Every statistic used for a validation row comes strictly from months before the
first pseudo-test month; training rows use a leave-one-year-out climatology so no feature
can see its own target. All numbers below are held-out.

| | layout A | layout B | public LB |
|---|---|---|---|
| persistence | 0.757 | 0.687 | 0.886 |
| previous final stack (4 Sep) | 0.652 | 0.578 | 0.714 (v5-smooth) |
| **final ensemble (5 Sep)** | **0.636** | **0.550** | pending |
| gain vs persistence | −16.0% | −20.0% | — |

The gain transferred to the test period in full — it is larger there because 2015–2018
(the 2015–16 El Niño) has bigger anomalies, where the model helps most (§4.1).

## 3. Model

An ensemble of four model families, all predicting the **residual** `TWS(t+1) − TWS(last observed)`
from the same 142 features, blended 0.7 (neural) / 0.3 (trees), then two light post-processing
steps: spatial smoothing of the predicted change over the 8 neighbours (w = 0.7) and **trajectory
smoothing** along the horizon within a block (each change is blended 0.35 with the mean of the
changes predicted for the neighbouring horizons of the same cell and last-observed month; −0.004/−0.002).

- **MLP** (PyTorch): 3 hidden layers (256-256-128, SiLU, dropout 0.3, weight decay 1e-3), a single
  one-cycle epoch (~25 s on an Apple M1 GPU). It overfits cell identity from the second epoch on,
  so it is deliberately under-trained; 7 seeds (2 without latitude/longitude) are averaged. Best single
  model on both layouts (A 0.648, B 0.551).
- **LightGBM / XGBoost / CatBoost** (3/3/2 seeds; 127 leaves or depth 8–9, 63-bin histograms,
  learning rate 0.02–0.03, ~400–1000 rounds). A 0.646–0.650, B 0.566–0.576. Trees get 0.1 each: on
  layout A the trees and the MLP are equally good and blend well; on layout B the MLP dominates.

Inputs, all available at or before month t:

- last observed TWS, its month, horizon, previous observation and slope; TWS lags 1/2/3/6/12
  months before the last observation; 24-month trend and deviation; the same *recent-window*
  statistics over the 60 months before the last observation (mean, monthly climatology, trend).
  The 2002–2015 long-term climatology anchors were removed: they bias every model upward
  because the record trends drier.
- covariates at t (never masked): SPEI-1/3/6/12, soil moisture; the same at the last observed
  month; their differences; SPEI-1 accumulated and soil moisture averaged / min / max over the months
  while TWS was unobserved; SPEI-1 and soil moisture one to three months before t.
- spatial context at three scales: 8-neighbour means, **5×5 and 9×9** neighbourhood means of the
  dynamic features (anomaly, SPEI-6/12 change, accumulated SPEI-1, soil-moisture change, trend),
  computed by grid convolution on the 1° grid (wrapping in longitude). The 9×9 SPEI-6/12 changes
  are the most important features after the cell's own anomaly: TWS responds to basin-scale forcing.
- per-cell response coefficients from history only: slope of the one-month TWS change on SPEI-1 and
  on the soil-moisture change, their correlations, and the standard deviation of the change.
- **water balance from two public, operational reanalyses/analyses** (no login; NOAA PSL):
  NCEP/NCAR R1 and NCEP-DOE R2 precipitation, evaporation (from latent heat), runoff, snow water
  equivalent and 0–200 cm soil water at t and at the last observed month, their changes, and
  P − E − R accumulated over the unobserved months; and the CPC leaky-bucket soil-moisture analysis
  (0.5°, mm). All contain no GRACE/TWS information; monthly means are published within days.
- latitude, longitude, calendar month.

Training rows: for each training month and each of three draws, one horizon (1–7, one third at
horizon 1) is drawn **for the whole globe**, mirroring the test blocks where all cells share the
last-observed month; ~4.6 M rows for the final models. Training weights ramp toward recent years.
Features are built once to disk (`build_mats.py`, ~11 min, float32 parquet); every model trains from
that cache (`run_models.py`); the blend is assembled by `final_assemble.py`.

Tested and **not** adopted (both layouts, `NOTES.md`): ENSO index (overfits a few cycles); ridge
blend; horizon-specialist model; per-horizon bias correction; long-term anchors in the MLP (+0.006 /
+0.034 worse); wider or longer-trained MLPs; a U-Net on global maps (see NOTES).

## 4. Trustworthiness

### 4.1 Data & model bias (≤100 words)
Errors are not uniform in space or regime. Validation mean error is +0.11 in the southern
tropics (30°S–0°), −0.11 in the northern tropics and −0.14 south of 30°S, near zero in
mid-latitudes (`error_breakdown.png`, `error_map.png`): tropical cells have larger, faster
swings and fewer analogues. Gains are largest where drought monitoring needs them — RMSE
falls 1.17→0.85 when the last-observed anomaly exceeds 2σ — but the model under-corrects
extremes (+0.27 bias there). Temporal: 22 GRACE gap months and the 2017–18 GRACE/GRACE-FO
transition are unrepresented in training; the training record trends drier, so 2002–2015
anchors bias predictions upward — motivating Model B. No demographic attributes; AI
Fairness 360 not applicable.

### 4.2 Model transparency (≤100 words)
SHAP (TreeExplainer, 40k validation rows; `shap_summary.png`, `shap_bar.png`). Top drivers
of the predicted change: last observed TWS (mean |SHAP| 0.059), current SPEI-6 (0.057),
deviation from the 24-month mean (0.051), and the 5×5-neighbourhood changes in SPEI-12 and
SPEI-6 since the last observation (0.049, 0.047), then neighbourhood anomaly, accumulated
SPEI-1 and the 60-month trend. Unexpected: SPEI-6 outranks soil moisture and SPEI-1 — a
six-month water balance tracks storage better than the surface layer — and regional (5×5)
drought signals outrank the cell's own, i.e. TWS responds to basin-scale, not cell-scale,
forcing.

### 4.3 Approach reusability (≤100 words)
The pipeline is variable-agnostic: any gridded monthly target with gaps plus any covariate
set; the horizon is an explicit feature, so one model serves 1–7-month leads and longer
leads need one constant changed. Per-cell statistics are computed from history at run time
(recent-window variants adapt to trends), so new regions need no code changes. Limitation:
the "what happened while unobserved" signal relies on Copernicus SPEI/soil moisture; a
region without them degrades toward persistence. External inputs are limited to a
freely available, operational reanalysis (NCEP R1); ENSO indices were tested and rejected.

### 4.4 Sustainability & efficiency (≤100 words)
CodeCarbon: one training run takes 46 s on a laptop CPU and emits **0.0002 kg CO₂e**; the
full 8-seed ensemble 0.0017 kg (`carbon.json`). Features are built once and written to
disk as float32; each seed trains from disk with LightGBM holding the only copy (63-bin
histograms), keeping peak memory under 8 GB — after an out-of-memory failure when three
jobs ran concurrently, the pipeline was restructured to one process at a time.
Gradient-boosted trees on tabular features were chosen over a deep sequence model; the
residual target lets a small model do the work.

## 5. Reproducibility
`build_features.py` → `train_final.py`; final blend `build_final_ncep.py` (fallback without reanalysis: `build_final_fallback.py`) (Python
3.10; polars, LightGBM 4.7, shap, codecarbon — pinned in `requirements.txt`). Validation:
`validation.py`, `validation_b.py`, `v6_val.py`. Seeds fixed; no randomness beyond seeded
bagging. Scored predictions use the challenge data plus NCEP R1 monthly means (six files,
`external/ncep/`, retrieved 4 Sep 2026 from downloads.psl.noaa.gov; retrieval script in
`features_ncep.py` docstring).
