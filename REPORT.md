# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for 15,715
one-degree cells, when the current-month TWS is hidden for two-thirds of test rows.

## 1. Problem structure

The test set is six blocks of consecutive months (2015-09; 2016-01→03; 2016-06→09;
2016-12→2017-06; 2018-07; 2018-11→12). `TWS_t` is observed only in each block's first
month and masked afterwards, so the effective horizon from the last observed TWS to the
target runs from **1 to 7 months** (test mix: 33% / 22% / 17% / 11% / 5.5% / 5.5% / 5.5%).
Persistence — repeat the last observed value — is the baseline everything must beat; the
seasonal climatology is *not* competitive here (RMSE 1.12 vs 0.76 on our validation),
because this TWS series is dominated by interannual anomalies rather than the seasonal
cycle. The organisers' starter, which median-imputes the masked `TWS_t`, scores 0.90.

## 2. Validation that mirrors the test

Two independent pseudo-test layouts were carved out of the training years with the same
block/mask structure (layout A: 2012–2015, 18 months, 72% masked; layout B: 2009–2012, 23
months, 70% masked). All climatological and per-cell statistics used for a validation row
come strictly from months before the first pseudo-test month; for *training* rows the
climatology is leave-one-year-out so the model cannot see its own target through a
feature. Every result below is on these held-out layouts.

| | layout A | layout B |
|---|---|---|
| persistence | 0.757 | 0.687 |
| **model (v4)** | **0.661** | **0.593** |
| gain | −12.6% | −13.7% |

## 3. Model

LightGBM (open source) predicting the **residual** `TWS(t+1) − TWS(last observed)`.
Inputs, all available at or before month t:

- last observed TWS, its month, horizon, previous observation and slope; TWS lags 1/2/3/6/12
  months before the last observation; 24-month trend and deviation from the 24-month mean
- per-cell climatology of the target month and of the last-observed month (→ anomaly
  persistence), per-cell mean, std, lag-1 autocorrelation
- covariates at t (never masked): SPEI-1/3/6/12, soil moisture; the same at the last
  observed month; their differences; SPEI-1 accumulated and soil moisture averaged over the
  months between the last observation and t (the wetting/drying that happened while TWS
  was unobserved)
- 8-neighbour means of the anomaly, TWS and SPEI-3 change (spatial coherence)
- latitude, longitude, calendar month

Three seeds are averaged, then a light spatial smoothing of the predicted residual over the
8 neighbours (w = 0.7) is applied; it improves both layouts by ~0.003. Training: 4.9 M rows
(each training month paired with randomly drawn horizons 1–7), 240 rounds, 127 leaves.

## 4. Trustworthiness

### 4.1 Data & model bias (≤100 words)
Error is not uniform in space. On validation the model's mean error is +0.10 in the southern
tropics (30°S–0°) and −0.14 in the northern tropics (0–23.5°N), near zero in mid-latitudes
(figure `error_breakdown.png`, `error_map.png`): tropical cells have larger, faster TWS
swings and fewer well-fitting analogues. Gains are largest where it matters for drought —
RMSE falls from 1.17 to 0.80 when the last-observed anomaly exceeds 2σ — but the model
under-corrects extremes (positive bias +0.10 there). Temporal bias: GRACE gaps (22 missing
training months) and the 2017–18 GRACE/GRACE-FO transition are unrepresented in training.
No demographic attributes are involved; AI Fairness 360 does not apply.

### 4.2 Model transparency (≤100 words)
SHAP (TreeExplainer, 40k validation rows; `shap_summary.png`, `shap_bar.png`). Top drivers
of the predicted change: the anomaly at the last observation (mean |SHAP| 0.088), current
SPEI-6 (0.067), the cell's long-term mean (0.063), anomaly persistence (0.056), deviation
from the 24-month mean (0.051), and the change in SPEI-12 and SPEI-6 since the last
observation. Unexpected: SPEI-6 outranks soil moisture and SPEI-1 — a six-month water
balance tracks storage better than the surface layer does — and the sign of the anomaly
feature is mean-reverting: large positive anomalies are predicted to decay.

### 4.3 Approach reusability (≤100 words)
The pipeline is variable-agnostic: it takes any gridded monthly target with optional gaps
plus any set of covariates, and the horizon is an explicit feature rather than a fixed lag,
so the same model serves 1- to 7-month leads and can be retrained for longer ones by
changing one constant. Per-cell statistics are computed from history at run time, so new
regions need no code changes. Limitation: the covariate proxies (SPEI, soil moisture) are
Copernicus products; a region without them loses the "what happened while TWS was
unobserved" signal and falls back toward persistence.

### 4.4 Sustainability & efficiency (≤100 words)
Measured with CodeCarbon: one training run of the final model takes 45 s on a laptop CPU
and emits **0.0002 kg CO₂e**; the three-seed ensemble 0.0006 kg (`carbon.json`). Feature
construction runs once and is written to disk, so retraining does not repeat it.
Efficiency choices: gradient boosting on tabular features instead of a deep sequence model,
63-bin histograms, 240 rounds selected by early stopping on held-out blocks, float32
storage. The residual target lets a small model do the work — persistence carries the
level, the model learns only the change.

## 5. Reproducibility
`build_features.py` → `train_final.py` (Python 3.10; polars, LightGBM 4.7, shap,
codecarbon; pinned in `requirements.txt`). Validation: `validation.py`, `validation_b.py`,
`eval_layout.py`. Only the challenge data is used; no external datasets.
