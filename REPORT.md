# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for 15,715
one-degree cells when the current-month TWS is hidden for two-thirds of test rows.

## 1. Problem structure

The test set is six blocks of consecutive months (2015-09; 2016-01→03; 2016-06→09;
2016-12→2017-06; 2018-07; 2018-11→12). `TWS_t` is observed only in each block's first month and
masked afterwards, so the horizon from the last observed TWS to the target runs from **1 to 7
months** (test mix 33/22/17/11/5.5/5.5/5.5%). Persistence — repeating the last observed value — is
the baseline that matters; the seasonal climatology is not competitive (1.12 vs 0.76 on validation),
because this record is dominated by interannual anomalies. Persistence scores **0.886** on the
public leaderboard.

## 2. Validation that mirrors the test

Two independent pseudo-test layouts carved from the training years with the same block and mask
structure (A: 2012–2015, 18 months, 72% masked; B: 2009–2012, 23 months, 70% masked). Every
statistic used for a validation row comes strictly from months before the first pseudo-test month;
training rows use leave-one-year-out climatologies so no feature can see its own target. Two layouts
are used because they disagree in an informative way: 2012–2015 is a drying period that favours
recent-window anchors, while the 2015–2018 test period does not. Only changes that improve **both**
layouts were adopted.

| | layout A | layout B | public LB |
|---|---|---|---|
| persistence | 0.757 | 0.687 | 0.886 |
| **final ensemble** | **0.638** | **0.552** | **0.712** |
| gain vs persistence | −15.7% | −19.7% | −19.7% |

## 3. Model

Two stacks of four model families, blended 50/50. Each model predicts the **residual**
`TWS(t+1) − TWS(last observed)`; the ensemble mean is then smoothed over the eight neighbouring
cells of the same month (w = 0.7).

- **Recent-anchor stack**: per-cell statistics from the 60 months before the last observation.
  MLP 0.7, LightGBM / XGBoost / CatBoost 0.1 each.
- **Long-term-anchor stack**: the same features with the 2002–2015 climatology anchors instead.
  LightGBM 0.3, XGBoost 0.3, MLP 0.25, CatBoost 0.15.
- The MLP is 3 hidden layers (256-256-128, SiLU, dropout 0.3, weight decay 1e-3) trained for a
  **single** one-cycle epoch (~25 s on an Apple M1 GPU); it overfits cell identity from the second
  epoch on, so it is deliberately under-trained. Trees use 127 leaves or depth 8–9 with 63-bin
  histograms. Seeds: 4 MLP, 3 LightGBM, 3 XGBoost, 2 CatBoost per stack.

Inputs, all available at or before month t:

- last observed TWS, its month, horizon, previous observation and slope; TWS lags 1/2/3/6/12 months
  before the last observation; 24- and 60-month trends and deviations; per-cell mean, variance and
  lag-1 autocorrelation.
- **Smoothed anchor (the largest single gain of the final round).** The mean of *observed*
  neighbouring-cell TWS at the last observed month, over great-circle radii of 300, 500 and 800 km,
  with the longitude window scaled by 1/cos(latitude) so the neighbourhood is a physical radius
  rather than a grid box; plus the cell's deviation from each and the 300–800 km scale gradient.
  GRACE's effective resolution (~300 km) is coarser than the 1° grid, so a cell's departure from its
  neighbourhood is largely observation noise: only ~20% of it survives into the next month. Using
  the raw anchor alone put that noise into every prediction. As a bare baseline the 300 km mean
  improves persistence from 0.757 to 0.724 (A) and 0.687 to 0.654 (B); as features it improves every
  model family on both layouts (LightGBM A 0.6483→0.6452, B 0.5629→0.5555; MLP A 0.6453→0.6437,
  B 0.5538→0.5452).
- covariates at t (never masked): SPEI-1/3/6/12 and soil moisture; the same at the last observed
  month; their differences; SPEI-1 accumulated and soil moisture averaged, min and max over the
  months while TWS was unobserved; covariate lags 1–3 months before t.
- spatial context at three scales: 8-neighbour, 5×5 and 9×9 neighbourhood means of the dynamic
  features, by grid convolution wrapping in longitude. The 9×9 SPEI-6/12 changes are among the
  strongest features: TWS responds to basin-scale, not cell-scale, forcing.
- per-cell response coefficients from history: slope of the monthly TWS change on SPEI-1 and on the
  soil-moisture change.
- **water balance from public operational reanalyses**: ERA5 monthly means (precipitation,
  evaporation, runoff, snow depth, four soil-water layers, 2 m temperature), NCEP/NCAR R1 and
  NCEP-DOE R2 surface fields, and the NOAA CPC leaky-bucket soil moisture — at t and at the last
  observed month, their changes, and P − E − R accumulated over the unobserved months.
- calendar month. **Latitude and longitude are not features** (§4).

Training rows pair each training month with a horizon (1–7) drawn once per month for the whole
globe, mirroring the test blocks where all cells share a last-observed month; ~3.5 M rows. Sample
weights ramp toward recent years. Features are built once to disk as float32
(`build_mats.py`, ~11 min); every model trains from that cache.

**Tested and rejected** (both layouts required; full log in `NOTES.md`): ENSO/ONI indices; ridge
blending; horizon-specialist models; deeper trees; per-horizon bias correction; a U-Net on global
maps (0.677 alone, no blend gain); MLP bagging, wider and longer-trained MLPs; global low-rank (EOF)
denoising of the TWS field (worse at every rank); smoothing the *anomaly* field instead of the raw
field; per-cell local-deviation persistence coefficients (helped trees, hurt the MLP on both
layouts); smoothed-anchor trend features; disagreement-weighted shrinkage; cross-layout stacking.

## 4. Compliance

Every rule clarification issued by the organisers was applied, including two that removed score:

- **No coordinates as predictive features.** `lat`/`lon` are used only to join a cell to its own
  history, to define neighbourhoods, and to sample external grids. Verified programmatically: the
  final feature lists (168 and 166 features) contain no coordinate, cell identifier or
  coordinate-derived encoding.
- **No information from t+1 or later, from any source.** An earlier version of this pipeline smoothed
  each prediction along the horizon within a block, which indirectly used covariates at t+1. It was
  removed even though the causal replacement is worse (A 0.6395→0.6446), because the gain came
  entirely from the non-causal direction.
- **No GRACE or GRACE-derived product** of any kind is used; all external inputs are non-TWS
  reanalysis or land-surface fields, documented in §3 with retrieval scripts.
- Neighbouring cells' TWS is read only at months ≤ t, which the organisers confirmed on 24 August is
  permitted, including spatial filtering and aggregation.
- ERA5 final is used as the documented retrospective proxy for ERA5T, which is not archived for the
  challenge period; ERA5T is published ~5 days after month end, so the operational analogue exists.

## 5. Trustworthiness

### 5.1 Data & model bias (≤100 words)
Errors are not uniform in space or regime. Validation mean error is +0.13 in the southern tropics
(0–30°S) and −0.11 south of 30°S, against +0.00 in northern mid-latitudes
(`report_final/error_breakdown.json`, `error_map.png`): tropical cells swing faster and have fewer
analogues. Gains are largest where drought monitoring needs them — RMSE falls 1.17→0.84 when the
last-observed anomaly exceeds 2σ — but the model under-corrects extremes (+0.27 bias there). GRACE
gap months and the 2017–18 GRACE/GRACE-FO transition are unrepresented in training, and the training
record trends drier, which biases long-term anchors upward; this motivates the two-stack blend. No
demographic attributes are involved.

### 5.2 Model transparency (≤100 words)
SHAP (TreeExplainer, 40k validation rows; `shap_bar.png`, `shap_mean_abs.json`). Top drivers of the
predicted change: the last observed TWS (mean |SHAP| 0.054), its deviation from the 24-month mean
(0.052), current SPEI-6 (0.051), and the 9×9-neighbourhood changes in SPEI-6 and SPEI-12 (0.045,
0.038), then accumulated SPEI-1, the recent-anomaly persistence term and the 800 km anchor deviation
(0.029). Two findings are worth stating plainly: regional drought signals outrank the cell's own,
and the cell's departure from its neighbourhood is mostly observation noise.

### 5.3 Approach reusability (≤100 words)
The pipeline is variable-agnostic: any gridded monthly target with gaps, plus any covariate set. The
horizon is an explicit feature, so one model serves 1–7-month leads and longer leads need one
constant changed. Per-cell and per-neighbourhood statistics are computed from history at run time,
so new regions need no code changes, and the smoothed anchor is defined by a physical radius rather
than a grid stencil, so it transfers to other resolutions. Limitation: the "what happened while
unobserved" signal relies on Copernicus SPEI and soil moisture; a region without them degrades
toward persistence.

### 5.4 Sustainability & efficiency (≤100 words)
CodeCarbon measured every family of the final configuration (`report_final/carbon.json`): one seed of
each of the eight models emits **0.0077 kg CO₂e** in total, of which the MLP — the single strongest
model — accounts for 0.00006 kg and trains in 25 seconds. The full 24-model ensemble is under
0.03 kg CO₂e. Features are built once and cached as float32; each model trains from disk with
63-bin histograms, keeping peak memory under 8 GB on a 16 GB laptop after an out-of-memory failure
forced the pipeline to run one process at a time.

## 6. Reproducibility

Python 3.10; polars, LightGBM 4.7, XGBoost, CatBoost, PyTorch, xarray, shap, codecarbon — pinned in
`requirements.txt`. All seeds fixed; no randomness beyond seeded bagging and the MLP initialisation.

1. `cds_download.py` (ERA5, needs a free CDS token) and the NOAA retrieval commands in
   `features_ncep.py` / `features_x.py` docstrings.
2. `build_mats.py A|B|FINAL` → cached feature matrices.
3. `add_anchor_feats.py A|B|FINAL` → smoothed-anchor features.
4. `run_models.py FINAL {lgb|xgb|cat|mlp} {allnoll_sa|v5x_noll_sa} [rounds]` per seed
   (`pipeline9.sh` runs all 24).
5. `final_assemble.py sub_g_blend ...` → the submitted CSV.

Validation: `validation.py`, `validation_b.py`, then the same `run_models.py` on layouts A and B.
Evidence: `report_final.py`. Experiment log with every accepted and rejected change: `NOTES.md`.
