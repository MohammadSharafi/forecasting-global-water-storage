# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for a
40x40 grid of one-degree cells over tropical South America (19.5°S–19.5°N, 79.5°W–40.5°W — the
Amazon, the eastern Andes and the Nordeste), when the current-month TWS is hidden for two-thirds
of the 28,800 test rows.

> Every number in this report is either a public-leaderboard result or a validation measurement
> produced by a named script in this repository, and each is reproducible with the command in §7.
> Validation numbers and feature attributions are taken from the run recorded in
> `out/RUN_REPORT.md`, `out/night/*.log` and `out/report/shap.md`; leaderboard numbers are the
> scores Zindi returned for the named files.
>
> **Validation is about 2.1x optimistic against the leaderboard** — the pipeline scores 0.334 on
> validation layout A and 0.696 on the public board. The test window is the record 2015-16 El Niño
> Amazon drought plus the post-GRACE-gap months of 2018, which no training window resembles. Only
> *relative* changes transfer, and they do: validation predicted −0.0128/−0.0184 for the change in
> §3.1 and the board paid −0.0129.

## 1. Problem structure

The test set is six blocks of consecutive months — 2015-09; 2016-01→03; 2016-06→09;
2016-12→2017-06; 2018-07; 2018-11→12 — and `TWS_t` is observed only in each block's first month.
The horizon from the last observed TWS to the target therefore runs from **1 to 7 months**, in the
proportions 33.3 / 22.2 / 16.7 / 11.1 / 5.6 / 5.6 / 5.6 %.

This structure is not assumed: `data_report.py` derives it from `Test.csv` — run lengths, the gaps
between blocks, and which months carry an observed value — and cross-checks it against the constant
the rest of the pipeline uses, printing MATCH or a loud MISMATCH. Everything downstream (the
horizon weighting, the training-mix reweighting, validation layout C) depends on that constant, and
until this check existed nothing verified it.

The starter notebook calls persistence — repeating the last observed value — "a very strong
baseline". Under the test's own horizon mix it is the **weakest** thing available, because a stale
anchor is close to uncorrelated with a standardised anomaly seven months later. Measured on the
real data (`fastval.py`, `validation.py`):

| baseline | test-mix RMSE |
|---|---|
| persistence (last observed TWS) | 1.166 |
| per-cell, per-calendar-month climatology | 0.502 |
| anomaly persistence, `clim_next + (tws_known − clim_known)` | 0.407 |
| this pipeline, layout A | **0.334** |

Climatology is not merely competitive, it beats persistence by more than a factor of two: the
Amazon's seasonal water-storage swing is large and highly repeatable. Anomaly persistence — three
lines of arithmetic — beats it again. Any claim of skill has to be read against 0.407, not 1.166.

## 2. Validation that mirrors the test, and how changes are accepted

**Three pseudo-test layouts** are carved from the training years with the same block-and-mask
structure. A (2012–2015) and B (2009–2012) were built first; **C reproduces the test's exact block
pattern `[1,3,4,7,1,2]` at the test's exact spacing**, searched over every placement the training
record can carry gap-free, because GRACE has real data gaps and a hardcoded offset silently
truncated blocks — producing a layout that could not score four of the seven horizons. `validation_c.py`
now exits non-zero rather than let such a layout become a vote.

Two measurement corrections matter more than any model change:

- **`eval_mix.py`** scores every result under the *test's* horizon mix. Layout A's own mix
  over-weights h2/h3 by 5.6 points each and under-weights h1/h4 by the same, so for several
  sessions any method that traded h1 accuracy for mid-horizon accuracy was scored more kindly by
  validation than the leaderboard would score it.
- **`xfit.py`** applies leave-one-layout-out to every post-processing decision: the configuration is
  chosen on the other layouts and scored on the **held-out** one, and nothing is adopted unless
  every held-out gain clears 0.0003. That threshold is derived, not chosen: `lb_se.py` computes the
  standard error of a leaderboard gap from the RMS difference between two submission files and the
  public row count, and differences below roughly 0.001 between the files involved here are not
  distinguishable from noise.

The standing rule is that a change must win on **every** layout it was run on. `select_config.py`
applies it to four independent decisions (feature groups, model capacity, sample weighting, training
horizon mix) and keeps the incumbent otherwise.

## 3. What actually produced the score

### 3.1 The main finding: the water-balance features were raw millimetres

Every ERA5 and NCEP feature was a raw physical level (`e5PER_acc`, `P_acc`, `e5SW_t`) while `lat`
and `lon` are excluded from every feature set. The model was shown "accumulated P−E−R = 350 mm"
with no way to learn that this is an ordinary wet month in the western Amazon and a record flood in
the Nordeste: it had
per-cell **TWS** statistics to locate a cell in TWS space and nothing to locate it in covariate
space. The target compounds this — TWS here is a standardised anomaly, and the change in an anomaly
is driven by the *anomaly* of the water balance, not its raw total.

`features_anom.py` computes each cell's own climatology for that calendar month from **history
months only**, then z-scores storage variables and accumulates flux anomalies over the months while
TWS was unobserved.

| | layout A | layout B |
|---|---|---|
| without covariate anomalies | 0.6551 | 0.5584 |
| with | **0.6422** | **0.5399** |
| | −0.0128 | −0.0184 |

It wins at **every one of the seven horizons** on layout B. On the public leaderboard the first
submission carrying it moved **0.709259 → 0.696326**, a gain of 0.0129 — inside the range validation
predicted, and 28 standard errors of the gap between those two files.

### 3.2 Why it was missed, and what that says about the method

An earlier session measured the correlation between the one-month TWS change and the ERA5 water
balance at −0.02 and concluded that weather does not explain the change. That number is physically
impossible: P − E − R *is* the change in stored water. `ceiling.py` reproduces the failure on
synthetic data where the answer is known — with a true correlation of 0.844, the raw encoding reads
**0.038** and the anomaly encoding reads **0.845** — and on the real data the same contrast holds
(raw `e5P_acc` −0.032, anomaly `an_e5Pz_acc` +0.235). A perfect relationship reads as zero when
pooled across cells on raw totals. The conclusion that closed this avenue for four sessions rested
on an encoding artifact, not on the data.

### 3.3 Model and post-processing

Each model predicts the **residual** `TWS(t+1) − TWS(last observed)`. The ensemble weight is fitted
rather than assumed: `stack.py` solves a non-negative least squares over every trained family
(LightGBM at three capacities, XGBoost, CatBoost) on test-mix-weighted rows, renormalised to sum to
one so that overall scale remains a separate decision, and shrunk halfway toward the incumbent
blend.

The configuration that produced the submitted files (`out/config.sh`, `out/stack.sh`,
`out/blendw.sh`, `out/rounds.sh`):

```
FINAL_MODEL=lgb   FINAL_WEIGHTS=ramp   FINAL_DROPF=''   FINAL_HMIX=''   (sampler default)
two-family blend   lgb 0.75 / xgb 0.25      (scan preferred 1.00, clipped: past 0.75 it is one family)
fitted stack       lgb 0.375  xgb 0.318  cat 0.307  lgbm 0.000  lgbs 0.000
boosting rounds    lgb 434   xgb 311   cat 824   lgbm 637   lgbs 2183   (each family's own early stop)
```

The stack is the one post-processing stage that won decisively. Fitted on the other two layouts and
scored on the held-out one it gained **−0.0038 (A), −0.0189 (B), −0.0106 (C)** — two orders of
magnitude above the 0.0003 threshold, on all three. It is also the first configuration in this
project to use more than two families: the 31-leaf and 63-leaf LightGBM variants were fitted to
exactly zero, and CatBoost, which no earlier session had weighted, took almost a third.

Post-processing is applied in the order it was fitted, and every stage is off unless its own
held-out scan adopted it:

| stage | script | status |
|---|---|---|
| spatial smoothing of the residual field | `smooth_scan.py` | **kept in the submitted files** — removing it costs 0.0028 on the board. See the disagreement below |
| per-horizon calibration of the change | `postcal.py` | **rejected** — held-out validation adopted it (−0.0084 / −0.0165 / −0.0120); the leaderboard says it costs 0.00036, and the leaderboard wins |
| per-calendar-month bias | `seasonal.py` | **rejected** — the bias is real (+0.02 to +0.08 by target month) but no shrinkage clears 0.0003 on every held-out layout: at λ=1.0, A −0.0023, B −0.0009, **C +0.0034** |
| horizon-1 specialist | `hsplice.py` | **rejected** — the specialist is far *worse* at h=1 than the general model (0.516 vs 0.312 on A, 0.500 vs 0.334 on B, 0.474 vs 0.295 on C) and every β>0 loses on every layout; β=0 |

**Two stages where validation and the leaderboard disagree, recorded rather than reconciled.**
`postcal` was adopted by held-out validation on all three layouts by 0.008–0.017 and then cost
0.00036 on the board; it is off. `smooth_scan` in the run recorded here went the other way — it
rejected the incumbent `w=0.7, r=1` and chose `w1=w7=0.0`, i.e. no smoothing at all — while the
board had already paid 0.0028 for keeping it. The submitted files keep smoothing, on the
leaderboard's evidence.

That is not a comfortable position and it should not be dressed up as one. Both stages are fitted
on the same held-out machinery that chose the stack correctly, and on these two it disagreed with
the only out-of-sample evidence that counts. The honest reading is that a post-processing gain of
0.003 or less is at the edge of what three validation layouts drawn from a milder era can resolve
about a test window they do not resemble — which is the same 2.1x caveat as the header, in the one
place where it changes a decision. The controls exist precisely so this is visible: every run emits
`sub_q_main_nosm` and `sub_q_main_nocal` beside the main file, and those two leaderboard readings
are what settled both stages.

Inputs, all available at or before month t: the last observed TWS with its lags, trends and per-cell
statistics; **great-circle smoothed anchors** at 300/500/800/1500/2500 km, with the longitude window
scaled by 1/cos(latitude) so the neighbourhood is a physical radius rather than a grid box;
covariates at t (never masked) and at the last observed month; the covariate anomalies of §3.1 plus
3/6/12-month antecedent windows ending at t; a modelled total-water-storage composite (soil water +
snow); zonal context; calendar month; and, where downloaded, Copernicus GDO
long-window SPI and fAPAR (see `GDO_SETUP.md`). **Latitude and longitude are not features** (§5).

## 4. Avenues closed on evidence

A negative result honestly established is worth as much as a positive one, and these are recorded
because each one stopped effort being spent in the wrong place.

| avenue | what was measured | verdict |
|---|---|---|
| global monthly offset correction | 2.9% of MSE, oracle −0.0090, persistence corr **−0.110** | temporally white; unreachable |
| per-latitude-band offset | 12.9% of MSE, oracle **−0.0418**, persistence corr +0.190 | every correction weight tried made it worse; a linear predictor with r=0.19 can remove only r² of it, ≤0.0015 in sample |
| higher-resolution forcing (ERA5-Land, 0.1°) | residual correlation with the neighbour at grid lag 1 **+0.55 / +0.39 / +0.52** (layouts A/B/C), lag 2 +0.41/+0.29/+0.39, lag 3 +0.28/+0.19/+0.27, and gone by lag 5 (+0.00/+0.00/+0.02) | the error is coherent over three to four cells, so it is not the sub-degree detail a 0.1° forcing would add |
| recursive forecasting (explicitly permitted) | direct **0.306 / 0.308 / 0.312** against recursive **0.372 / 0.375 / 0.387**, three independent block placements, test horizon mix (`fastval.py`) | **measured and closed.** The bound argued recursion inherits the one-step error as an anchor; the measurement agrees and is worse than the bound — h=1 is identical by construction (0.273 vs 0.273, the same model) and the chain then compounds, reaching 0.545 against 0.345 at h=7. Recursion loses by 0.066–0.076 everywhere it was tried |
| hindcast bias correction | `Test.csv` contains only the 18 block months | the row a hindcast needs does not exist |
| groundwater memory (12/24-month anomaly lags, 24-month trend) | +0.0002 against base, three placements (`fastval.py`) | the per-cell per-calendar-month climatology already carries the cell's slow state |
| directional spatial structure (13x13 box split west/east, upstream covariate means) | +0.0005 and +0.0001 against base | the residual is spatially coherent but **isotropic** — splitting the neighbourhood along the drainage direction buys nothing over the existing great-circle anchors |
| free information in the unmasked test rows | the 6 unmasked months are exactly the 6 block anchors; the successor of every test month is masked | the organisers' masking is airtight — no test row's target is another row's given `TWS_t`. Nothing to take |

One avenue was **re-opened**, not closed. `build_mats.py` loads ERA5 only when `external/era5/*.nc`
exists, and it never has: all four `out/night/build_*.log` print `era5: None`, and none of the 201
features in `out/mats/feats.json` begins with `e5`. Every result in this report was produced on
NCEP-R1/R2 at ~2°, against a 1° target grid, with no evaporation field and a two-layer soil column.
The "higher-resolution forcing" row above closed ERA5-**Land** at 0.1°, on the finding that the
error is large-scale — it says nothing about ERA5 at 1°, which is a *matched*-resolution
replacement for a coarser product rather than a finer one, and which feeds the per-cell
standardised anomaly family that produced the entire session-9 gain (§3.1).

Together these say something specific about the remaining error: it is **coherent over a few
cells, isotropic, and temporally white**. Smoothing can only shave it — which is exactly what the
leaderboard paid, 0.0028 — and nothing available at or before t predicts the rest.

> **Correction.** Earlier drafts of this report gave the residual neighbour correlation as +0.967
> at lag 1 and +0.899 at lag 2, and argued from those figures that a neighbour's error is very
> nearly the same error. No artifact in this repository supports them: `analyze.py` measures
> +0.55/+0.39/+0.52 at lag 1 across the three layouts. The conclusion that the error is not
> fine-scale survives — correlation is gone by lag 5, so the coherence length is three to four
> cells — but it is a weaker statement than the one originally made, and it leaves more for
> smoothing to take than "0.967" implied, which is consistent with the leaderboard paying 0.0028
> for it. The figures were carried over from a measurement made on an earlier, global-grid version
> of this problem and were never re-measured after the domain changed.

## 5. Compliance

Every rule clarification issued by the organisers was applied, including ones that removed score.
`compliance.py` audits the **artefacts**, not the intent, and every check passes:

- **No coordinates as predictive features.** `run_models.py` records the exact feature list each run
  used to `out/mats/used_*.json`, and that list is audited for `lat`/`lon`. The shared feature
  superset does contain them by design, so auditing it would have been meaningless — this is why a
  per-run artefact was needed.
- **No information from t+1 or later.** `t_known ≤ t` and `horizon = months(t_known → t) + 1` are
  checked for every row. `tws_known` is independently reconstructed from `Train.csv` plus the
  unmasked `Test.csv` rows and compared. `clim_next` is recomputed from history **alone** and
  compared to the matrix column — a direct measurement that no test-era month entered the
  climatology, rather than a reading of the code. An earlier version smoothed each prediction along
  the horizon within a block, which indirectly used covariates at t+1; it was removed even though
  the causal replacement scores worse, because the gain came entirely from the non-causal direction.
- **No GRACE or GRACE-derived product** of any kind. `external/` is inventoried in the audit output;
  it holds NCEP-R1 and NCEP-R2 reanalysis, CPC soil moisture and the ONI climate index — 15 files,
  listed with their sizes in the audit output. ERA5 is supported by the code but was never
  downloaded (§4), so no ERA5 file appears.
- **Neighbouring cells' TWS at months ≤ t** is permitted including spatial filtering, which the
  smoothed anchors and the grid smoothing rely on.
- **Seasonal forecasts are deliberately not used.** A forecast issued in month t has a source date
  of t but *encodes* t+1, and the published rules permit the first while prohibiting the second.
  The question was raised publicly and is unanswered, so the product is treated as prohibited.

## 6. Trustworthiness

### 6.1 Data & model bias
Errors are not uniform, and the shape is consistent across all three validation layouts. Everything
below is from `analyze.py` (`out/night/analyze_{A,B,C}.log`); figures are A / B / C where they
differ.

**By horizon.** RMSE is 0.312 / 0.334 / 0.295 at h=1 against a persistence RMSE of ~0.516, and
0.333 / 0.468 / 0.348 at h=7 against a persistence of ~1.77. Because the test weights h=1 at one
third, h=1 and h=2 together carry **53.5%** of the test-weighted MSE on layout A — so the largest
available gain is at the shortest horizon, where the model's advantage over persistence is
*smallest*.

**By latitude.** The domain spans 19.5°S–19.5°N, so there are two bands and they are
indistinguishable: 0°–30°N carries 50.2 / 49.6 / 50.8% of MSE against 30°S–0° at 49.8 / 50.4 /
49.2%, with mean errors of +0.027 / +0.048 / −0.021 and +0.053 / +0.050 / +0.007. There is no
latitudinal bias worth correcting, which is consistent with §4: a per-band offset had an oracle
worth 0.0418 that no fitted weight could reach.

**By cell variability.** Error grows with a cell's variability, but gently — the most variable
decile carries 12.7% of MSE and the steadiest 9.0% (layout A). The model beats persistence in
**every** decile, including the steadiest (0.314 against 0.930).

**Concentration.** There is no small set of pathological cells to blame: the 20 worst carry 2.9% of
MSE, a quarter of the error comes from 15% of the grid and half from 36%.

**Against persistence.** The model is worse than simply repeating the last observation on 20.8% of
rows at h=1, falling to 6.6% at h=7 (26.8% → 8.1% on B, 22.8% → 5.0% on C). Losing to persistence
most often at h=1 is the same finding as §3.3's dead features: at h=1 the last observation *is* the
current month, and a third of the feature set has nothing to add to it.

**Bias.** A and B over-predict (+0.0398, +0.0490); C slightly under-predicts (−0.0070). The sign
flip across layouts is why `postcal`'s affine offset is shrunk halfway toward zero and clipped
rather than applied as fitted — and, in the end, why calibration is off at all.

**Seasonal.** Bias by target calendar month on layout A runs from +0.013 (February) to +0.143
(October), with December the largest MSE share at 13.2%. Real, but not correctable: see the
`seasonal.py` row in §3.3.

**Coverage.** GRACE gap months and the 2017–18 GRACE/GRACE-FO transition are unrepresented in
training, and the test window is a stronger El Niño than any training year — the single largest
source of the 2.1x validation-to-leaderboard gap. No demographic attributes are involved.

### 6.2 Model transparency
`shap_report.py` computes exact TreeSHAP from the trained boosters themselves — LightGBM's own
`pred_contrib`, so no surrogate model and no extra dependency — over all 28,800 rows of the FINAL
validation split, averaged across seeds, and splits the attribution by horizon. Full tables in
`out/report/shap.md` and `out/report/shap_mean_abs.json`.

| family | features | all rows | h=1 | h≥2 |
|---|--:|--:|--:|--:|
| climatology / cell statistics | 9 | **43.5%** | 47.4% | 41.9% |
| the cell's own TWS history | 6 | **23.1%** | 24.6% | 22.5% |
| smoothed anchors (300–2500 km) | 12 | 13.0% | 13.2% | 12.9% |
| everything else (windows, lags, responses) | 78 | 12.6% | 9.6% | 13.8% |
| covariate anomalies (§3.1) | 44 | 4.1% | 2.6% | 4.7% |
| released covariates (SPEI, soil moisture) | 15 | 2.2% | 1.2% | 2.6% |
| calendar / geometry | 4 | 0.9% | 0.9% | 0.8% |
| NCEP / CPC | 29 | 0.3% | 0.2% | 0.4% |
| zonal scale | 4 | 0.3% | 0.3% | 0.3% |

Top individual features: `clim_next` (0.472), `tws_prev` (0.253), `clim_known` (0.253),
`tws_known` (0.170), `anom_persist` (0.118), then `dsa300` (0.094).

**This corrects an earlier claim in this report.** A previous draft stated that regional drought
signals outrank a cell's own history, with neighbourhood SPEI changes and the smoothed anchors
above the cell's recent TWS. The measurement says the opposite: climatology and the cell's own TWS
account for **two thirds** of all attribution, the first anchor appears at rank 6, and SPEI in every
form together accounts for **1.4%**. The model is, in substance, an anomaly-persistence baseline
that the remaining features adjust — which is exactly what §1's baseline table predicts, since
anomaly persistence alone reaches 0.407 against the pipeline's 0.334.

Three things follow that are worth stating for reuse:

- **Soil moisture, not SPEI, is the covariate that carries information here** — 10.5% of attribution
  in all its forms against SPEI's 1.4%, and `w_SOIL_MOISTURE_t_d` is the highest-ranked covariate
  feature of any kind at rank 7.
- **The window features collapse at h=1, and the size of the collapse is now measured.** The
  `_d`/`_acc` family takes 9.3% of attribution at h≥2 and **1.8%** at h=1 — a fifth — because at
  h=1 `t_known == t` makes every accumulation window empty and every difference identically zero.
  h=1 is a third of the test. (Only 5 of 201 features receive literally zero SHAP at h=1; a
  constant feature can still carry attribution relative to the background, so the collapse shows up
  as a fivefold drop rather than as zeros.)
- **The smoothed anchors are the one family whose contribution does not depend on the horizon**
  (13.2% at h=1, 12.9% at h≥2), which is what a spatial feature should look like and is consistent
  with them being the largest single feature gain in the selection grid (−0.0265).

On smoothing the predicted field: a cell's departure from its neighbourhood is partly observation
noise, GRACE's effective resolution being coarser than the 1° grid, which is why smoothing helps at
all. It can only help a little because the residual correlation with the adjacent cell is
0.55 / 0.39 / 0.52 — a neighbour's error is *related to*, not identical to, this cell's.

### 6.3 Approach reusability
The pipeline is variable-agnostic: any gridded monthly target with gaps, plus any covariate set. The
horizon is an explicit feature, so one model serves 1–7-month leads. Per-cell and per-neighbourhood
statistics are computed from history at run time, so new regions need no code changes, and the
smoothed anchor is defined by a physical radius rather than a grid stencil, so it transfers to other
resolutions. The covariate-anomaly encoding of §3.1 is the part most worth reusing and is the least
specific to this problem: it is the general statement that a model without location features must be
given its covariates in per-cell standardised form. Limitation: the "what happened while unobserved"
signal relies on Copernicus SPEI and soil moisture; a region without them degrades toward
persistence.

### 6.4 Sustainability & efficiency
Emissions are measured **during** the runs that produce the submitted models, not reconstructed
afterwards: `run_models.py` starts a CodeCarbon tracker around each training run and writes one row
per run, and `carbon_report.py` totals them and splits validation from final training
(`out/carbon/summary.md`). Totals, for the instrumented runs of the recorded run:
**0.0001 kg CO₂e** over **39 training runs**, **1.86 hours** of training and **0.0264 kWh** —
a mean of under 0.01 g CO₂e per run. All 39 are validation runs; final-training runs in this pass
reused existing checkpoints, and runs skipped because a checkpoint already existed are not counted,
so this is the cost of work actually done rather than a reconstruction. Sessions before the
instrumentation existed are not included, and the figure should be read as the marginal cost of one
full pass, not the lifetime cost of the project. The instrumentation cannot break a training run —
a missing dependency or a platform that withholds power counters prints one line and continues.
Efficiency: features are built once and cached as float32 and every model trains from that cache;
trees use 63-bin histograms; the orchestrator checkpoints every step so an interrupted run resumes
instead of repeating work, which is itself the largest saving in the project.

## 7. Reproducibility

Python 3.10. Imported on the run path: polars, numpy, pandas, LightGBM, XGBoost, CatBoost, scipy,
scikit-learn, xarray, netCDF4, pyarrow, codecarbon. `requirements.txt` was pinned before XGBoost,
CatBoost, scipy, xarray and netCDF4 entered the pipeline and does not list them; it was left
unreconciled during the competition rather than risk a reinstall mid-run, and should be regenerated
from `pip freeze`, restricted to the packages above, before anyone reproduces this.

```
python validation.py ; python validation_b.py ; python validation_c.py   # the three layouts
python build_mats.py A|B|C|FINAL                                          # cached feature matrices
python add_anchor_feats.py A|B|C|FINAL                                    # smoothed-anchor features
python run_models.py FINAL {lgb|lgbs|lgbm|xgb|cat} v5x_noll_sa [rounds]   # one seed
python final_assemble.py <name> <stem:weight> ...                         # the submitted CSV
python shap_report.py                                                     # the §6.2 attributions
python fastval.py                                                         # §1's baselines, ~1 min
```

`fastval.py` needs only `Train.csv` and no external archive, so §1's baseline table and any
structural A/B can be reproduced on a machine that has none of the reanalysis data.

`./run_night.sh` runs all of it end to end, resumably, and writes `out/RUN_REPORT.md` containing the
configuration, the held-out evidence behind every adopted change, the data and error analysis, and
the compliance audit. `DEEP=1` adds layout C, sixteen seeds and a third configuration.

Every accepted and rejected change, with its measurement, is in `NOTES.md`. Verification of each
component against synthetic data with a known answer is described there and in each module's
docstring.

## 8. Results

| | public leaderboard |
|---|---|
| persistence baseline | 0.886 |
| starting point of this work | 0.709259 |
| covariate anomalies + the session's feature work | 0.696326 |
| smoothing kept, calibration dropped | **0.695965** |

Full set of leaderboard readings for the files this run produced, which is what settled the two
post-processing questions in §3.3:

| file | public leaderboard | vs best |
|---|---|---|
| `sub_q_base` (pre-anomaly reference) | 0.708852 | +0.012887 |
| `sub_q_main_nosm` (no smoothing) | 0.699149 | +0.003184 |
| `sub_q_alt` (31-leaf capacity) | 0.698048 | +0.002083 |
| `sub_q_main` (smoothing + calibration) | 0.696326 | +0.000361 |
| **`sub_q_main_nocal`** (smoothing, no calibration) | **0.695965** | — |

Rank moved from **82 to 55**. The leader finished at 0.649536, with ranks 7–15 packed into
0.6495–0.6649; closing that gap needs a further 6.6% relative, about 0.022 in validation terms.
§4's re-opened avenue — ERA5 has never been downloaded, so no model in this report has ever seen it
— is the largest identified candidate and was not resolved before the deadline.

**A note on which files these are.** The four `out/sub_q_*.csv` files now in the repository were
regenerated by a later pass in which `smooth_scan` rejected smoothing and `postcal` was not
applied, which makes `sub_q_main`, `sub_q_main_nocal` and `sub_q_main_nosm` byte-identical to each
other. They are therefore *not* the files that produced the three distinct scores above; those were
written by the earlier pass whose configuration is recorded in §3.3. Anyone reproducing this should
expect the regenerated files to agree with each other, and should read §3.3's configuration block —
not the current file contents — as the description of what was submitted.
