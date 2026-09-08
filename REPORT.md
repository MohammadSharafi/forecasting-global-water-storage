# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for 15,715
one-degree land cells when the current-month TWS is hidden for two-thirds of test rows.

> Numbers marked **[pending]** are filled from the final run before submission. Every other number
> in this report is either a public-leaderboard result or a validation measurement produced by a
> named script in this repository, and each is reproducible with the command given in §7.

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

Persistence — repeating the last observed value — is the baseline that matters; per-cell
climatology is not competitive, because this record is dominated by interannual anomalies rather
than by the seasonal cycle.

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
with no way to learn that this is a drought in the Amazon and a record flood in the Sahel: it had
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
blend. Configuration and weights: **[pending]**.

Post-processing is applied in the order it was fitted, and every stage is off unless its own
held-out scan adopted it:

| stage | script | status |
|---|---|---|
| spatial smoothing of the residual field | `smooth_scan.py` | **kept** — the leaderboard says removing it costs 0.0028 |
| per-horizon calibration of the change | `postcal.py` | **rejected** — held-out validation adopted it; the leaderboard says it costs 0.00036 |
| per-calendar-month bias | `seasonal.py` | **[pending]** |
| horizon-1 specialist | `hsplice.py` | **[pending]** |

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
| higher-resolution forcing (ERA5-Land) | residual spatial correlation +0.967 at lag 1, +0.899 at lag 2 | the error is large-scale and coherent; it does not live at fine scales |
| recursive forecasting (explicitly permitted) | chained one-step forecasts against the direct model, all three layouts | **measured, and it loses**: worse at 21 of 21 (layout, horizon) cells, and by more as the horizon grows. Test-mix RMSE direct/recursive: A 0.6383/0.6657, B 0.5356/0.5909, C 0.5315/0.5692; a free 50/50 blend also loses everywhere. The cause is the one the bound predicted — the h=1 model is no better than the direct model at h=1 (0.6232 vs 0.6238 on A), so recursion pays that error as an anchor and then compounds it. `recursive.py A|B|C` |
| hindcast bias correction | `Test.csv` contains only the 18 block months | the row a hindcast needs does not exist |

Together these say something specific about the remaining error: it is **large-scale, spatially
coherent, and temporally white**. Smoothing can only shave it — which is exactly what the
leaderboard paid, 0.0028 — and nothing available at or before t predicts the rest.

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
  it holds NCEP reanalysis and a climate index only.
- **Neighbouring cells' TWS at months ≤ t** is permitted including spatial filtering, which the
  smoothed anchors and the grid smoothing rely on.
- **Seasonal forecasts are deliberately not used.** A forecast issued in month t has a source date
  of t but *encodes* t+1, and the published rules permit the first while prohibiting the second.
  The question was raised publicly and is unanswered, so the product is treated as prohibited.

## 6. Trustworthiness

### 6.1 Data & model bias
Errors are not uniform. By latitude band the mean error runs from **+0.11** in the southern tropics
to **−0.13** south of 30°S against +0.01 in northern mid-latitudes; the tropics and northern
mid-latitudes carry 24% and 31% of total MSE. Error concentrates with variability — the most
variable decile of cells carries 19.2% of MSE and the steadiest 1.4% — and on the steadiest decile
the model does not beat persistence at all (0.2305 against 0.2301). There is no small set of
pathological cells to blame: the 20 worst carry 0.67% of MSE and half the error comes from 23% of
the grid. The model is worse than plain persistence on 38–45% of rows at every horizon. Errors also
have a strong seasonal signature, with a +0.29 mean bias in June against −0.13 in October. GRACE gap
months and the 2017–18 GRACE/GRACE-FO transition are unrepresented in training. No demographic
attributes are involved. Produced by `analyze.py`.

### 6.2 Model transparency
Two findings from the error anatomy and the feature-gain rankings are worth stating plainly.
Regional drought signals outrank a cell's own history: the neighbourhood-scale SPEI changes and the
smoothed anchors rank above the cell's own recent TWS. And a cell's departure from its neighbourhood
is largely observation noise — GRACE's effective resolution is coarser than the 1° grid — which is
why smoothing the predicted residual field helps at all, and why it can only help a little: the
residual correlation with the neighbouring cell is 0.967, so a neighbour's error is very nearly the
same error. Detail: **[pending]** SHAP rankings from the final configuration.

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
(`out/carbon/summary.md`). Totals: **[pending]**. The instrumentation cannot break a training run —
a missing dependency or a platform that withholds power counters prints one line and continues.
Efficiency: features are built once and cached as float32 and every model trains from that cache;
trees use 63-bin histograms; the orchestrator checkpoints every step so an interrupted run resumes
instead of repeating work, which is itself the largest saving in the project.

## 7. Reproducibility

Python 3.10; polars, LightGBM, XGBoost, CatBoost, xarray, scipy, codecarbon.

```
python validation.py ; python validation_b.py ; python validation_c.py   # the three layouts
python build_mats.py A|B|C|FINAL                                          # cached feature matrices
python add_anchor_feats.py A|B|C|FINAL                                    # smoothed-anchor features
python run_models.py FINAL {lgb|lgbs|lgbm|xgb|cat} v5x_noll_sa [rounds]   # one seed
python final_assemble.py <name> <stem:weight> ...                         # the submitted CSV
```

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
| final submission | **[pending]** |
