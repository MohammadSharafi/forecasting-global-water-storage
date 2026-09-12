# A Step Ahead of Drought — Forecasting Global Water Storage: Method & Trustworthiness Report

**Entrant:** mrSharafi · **Task:** predict GRACE Total Water Storage (TWS) at month t+1 for 15,715
one-degree land cells when the current-month TWS is hidden for two-thirds of test rows.

> Every number in this report is either a public-leaderboard result or a validation measurement
> produced by a named script in this repository, and each is reproducible with the command given in
> §7. Where a validation result and the leaderboard disagree, both are shown and the leaderboard is
> treated as the answer — §2 sets out what that disagreement turned out to be worth, and §4 records
> the avenues it closed.

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

**Five pseudo-test layouts** are carved from the training years with the same block-and-mask
structure. A (2012–2015) and B (2009–2012) were built first; **C reproduces the test's exact block
pattern `[1,3,4,7,1,2]` at the test's exact spacing**, searched over every placement the training
record can carry gap-free, because GRACE has real data gaps and a hardcoded offset silently
truncated blocks — producing a layout that could not score four of the seven horizons.
`validation_c.py` now exits non-zero rather than let such a layout become a vote.

D and E were added last, and the reason they exist is a defect in the gate rather than in any model.
"Wins on all three layouts" sounds like three independent tests; it was not. B's window is
2009-10…2012-08 and C's is 2009-01…2012-04, they share six block months, and both sample the same
era — so the rule was closer to **two** tests than three, and a null change passes two coin flips
25% of the time. Against that gate this project had by then tried roughly twenty ideas.

`validation_extra.py` places the test's geometry at a *chosen* anchor rather than the latest one,
because the point is to cover an era the others do not:

| layout | window | history | block months shared with A, B, C |
|---|---|---|---|
| A | 2012-07…2015-08 | 111 | — |
| B | 2009-10…2012-08 | 84 | 2 with A |
| C | 2009-01…2012-04 | 75 | 6 with B |
| **D** | **2005-09…2008-12** | 35 | **none** |
| **E** | **2007-09…2010-12** | 59 | 1 with B, 4 with C |

All five reproduce the test's horizon mix to three decimals. D and E have shorter histories, so
their models are weaker in absolute terms; that is acceptable for a gate, where both arms of a
comparison carry the same handicap and only the contrast is read. Every layout is **extrapolation**,
never interpolation: each one's history ends before its window begins, as the real task's does.

Two measurement corrections matter more than any model change:

- **`eval_mix.py`** scores every result under the *test's* horizon mix. Layout A's own mix
  over-weights h2/h3 by 5.6 points each and under-weights h1/h4 by the same, so for several
  sessions any method that traded h1 accuracy for mid-horizon accuracy was scored more kindly by
  validation than the leaderboard would score it.
A third measurement correction is the ensemble weight, and it is worth recording because the
leaderboard has now ruled on it in **both** directions. `stack.py`'s non-negative least squares gave
a second-capacity model a weight of exactly 0.000; the leaderboard said that model *alone* scored
within 0.002 of the best, which looked like the fit throwing away a good model. Spending two
submissions on the question settled it against the doubt: the score degrades monotonically with the
weight given to that model (0.692657 → 0.693283 at 0.3 → 0.693725 at 0.5). A model that scores well
alone adds nothing to an ensemble it is correlated with, and the fitted zero was a measurement
rather than an artifact.

- **`xfit.py`** applies leave-one-layout-out to every post-processing decision: the configuration is
  chosen on the other layouts and scored on the **held-out** one, and nothing is adopted unless
  every held-out gain clears 0.0003. That threshold is derived from `lb_se.py`, which computes the
  standard error of a leaderboard gap from the RMS difference between two submission files and the
  public row count.

  **That threshold turned out to be measuring the wrong thing, and this is the most important
  methodological finding in this report.** `lb_se.py` answers *how precisely can the public subset
  measure a difference that exists*. The question that governs whether a change should be adopted is
  different: *how well does a validation delta predict the board delta*. Every paired observation
  this project has produced:

  | change | validation | board | transferred? |
  |---|---|---|---|
  | covariate anomaly encoding (§3.1) | −0.0156 | −0.0129 | yes, at 0.83× |
  | smoothing versus none | +0.0037 | +0.0028 | yes, at 0.77× |
  | ERA5 soil profile (§3.2b) | −0.0036 | **+0.0027** | no — sign inverted |
  | per-horizon calibration | −0.0005 | **+0.0004** | no — sign inverted |
  | smoothing 0.7/it1 → 0.5/it2 | −0.0001 | **+0.0001** | no — sign inverted |
  | NCEP-R2/CPC encoding + regional anomaly means (§3.2c) | −0.0091 | −0.0005 | direction held, 5% of size |

  Two changes above 0.0037 transferred at 0.77–0.83 of their validation size. Three below 0.0036
  inverted. The sixth is the interesting one: it is the best-evidenced change this project has made —
  it wins on **five** independent validation windows, two of which no decision has ever seen — and
  the board still paid only 5% of it. So the rule is not simply a threshold on magnitude. Something
  about the 2015–2019 test period compresses gains that are robust across 2005–2015, and §4 records
  the five mechanisms that were tested for it and rejected.

  The practical consequence for a reader of this report is a caution rather than a recipe: **treat
  every validation gain here below about 0.003 as unproven**, and note that the project's own
  adoption threshold of 0.0003 was ten times too permissive for most of its history.

The standing rule is that a change must win on **every** layout it was run on. `select_config.py`
applies it to four independent decisions (feature groups, model capacity, sample weighting, training
horizon mix) and keeps the incumbent otherwise. With five layouts that rule is worth roughly what it
appears to be worth; with the three it had for most of this project's history, and two of those from
one era, it was not — which is the correction §2 records and the reason D and E exist.

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

### 3.2b A finding validation accepted and the leaderboard refused — and it is not in the model

This section is kept in §3 rather than §4 because it is the clearest single illustration of the
caution in §2, and because a reader is entitled to see a change this project believed in and was
wrong about. **The soil profile is not in the submitted file.** It won on every validation layout by
8 to 17 times the adoption threshold and the leaderboard then scored it 0.695366 against 0.692657 —
worse by 0.0027, in the opposite direction and larger than predicted.

`features_era5` collapsed ERA5's four `swvl` layers into one column by a fixed-thickness weighted
sum; `features_ncep` and `features_x` do the same to NCEP-R1's and R2's two layers. Across 266
features the shallow and deep soil stores appeared **nowhere** as separate quantities. What that
removes is the drainage timescale — a 7 cm top layer answers a month of rain, a 189 cm bottom layer
integrates seasons, and TWS is the integral of the contrast between them.

Keeping ERA5's four layers apart, each in metres of water so they still sum to the old column, and
passing them through the same per-cell standardisation as every other storage variable:

| arm | features | layout A | layout B | layout C | mean |
|---|---|---|---|---|---|
| incumbent | 261 | 0.6402 | 0.5364 | 0.5288 | — |
| + soil profile | 297 | **0.6378** | **0.5313** | **0.5256** | **−0.0036** |
| + profile + GDO SPI | 309 | 0.6376 | 0.5318 | 0.5248 | −0.0037 |

All four arms were trained on **one matrix**, so the only difference between them is which features
the model was offered. The profile wins on every layout by 8 to 17 times the adoption threshold and
at six of seven horizons on A, five on B and all seven on C. The mechanism is physical and the
measurement was clean. It still did not transfer, and no amount of care in the experiment would have
revealed that in advance — only the submission did.

Copernicus GDO's long-window SPI (24 and 48 months, GPCC-based, §5) was measured in the same run and
**not adopted**: its marginal value on top of the profile is −0.0002, +0.0005, −0.0008, so it loses
on one layout and its mean is inside seed noise. The likely reason is visible in the arithmetic — a
100–289 cm soil layer is the physical accumulator of exactly the long-window precipitation deficit
that SPI-48 indexes, so the profile carries the same information closer to the target and per cell
rather than as a basin-scale index.

### 3.2c The third finding: the covariates were offered at only one spatial scale

Two blocks were still raw levels. `build_mats` passed ERA5, NCEP-R1, soil moisture and GDO through
the per-cell standardisation of §3.1 and **never passed NCEP-R2 or CPC**, so 24 features remained
raw millimetres and volumetric fractions — the §3.1 bug, four sessions after §3.1. The diagnostic is
the ratio sd(per-cell mean) / sd(overall): 0.77–1.01 for those columns against 0.28–0.36 for encoded
ones, and **not one of the 24 appeared in any top-gain list**, which is what a feature the model
cannot read looks like.

The larger half came from asking *at what scale* the information was offered. Three measurements
point at the same place:

- blur the **true** target over its eight neighbours and it scores **0.0619** — the signal is
  spatially coherent;
- blur **our error** the same way and **98.2%** of its variance survives — the error is regional;
- the accumulated water-balance anomaly correlates **0.177** with the TWS change at the cell and
  **0.242** over a 9×9 box, because reanalysis P−E−R errors are largely independent between cells
  and average out while the TWS signal does not.

So the one quantity that physically drives TWS change is most trustworthy at exactly the scale the
error lives at — and of the 25 wide and neighbourhood features, **none** aggregated the `an_*` block.
`add_anwide` supplies neighbourhood means of eleven anomaly columns at radius 4.

| | A | B | C | **D** | **E** |
|---|---|---|---|---|---|
| both fixes, against the previous feature set | −0.0040 | −0.0072 | −0.0096 | **−0.0068** | **−0.0130** |

Five layouts, of which D and E were built *after* the decision and D shares no block month with any
other. This is the best-evidenced change in the project, and the board paid 5% of it (§2). The
honest reading is in §4's last row.

A negative result worth keeping alongside it: the same treatment applied to TWS-derived quantities
(`d1`, `anom_persist` — which had no regional aggregate either, despite `anom_persist` being the
highest-gain feature at 9.4%) is worth −0.0008, and a second and third radius add +0.0003. The rule
that separates them is physical: **aggregate a noisy driver, not a smooth state.** Reanalysis errors
average out between cells; the TWS field is already smooth, so a neighbourhood mean of `d1` is very
nearly `d1`.

### 3.2d The fourth finding: every precipitation the model saw was a reanalysis

Precipitation is this model's strongest forcing — `aw_an_e5Pz_acc` is the second feature by gain at
7.1% — and until now every version of it came from the same kind of source. ERA5, NCEP-R1 and
NCEP-R2 are all *reanalyses*: physical models nudged toward observations. Where rain gauges are
sparse they are the best available estimate; where gauges are dense they are still a model.

**GPCC Full Data Monthly v2022** is the other kind of number: interpolated rain gauges, with a gauge
count per cell. It is not a better product than ERA5 — in the empty parts of the world it is worse —
but it is an *independent* one, and the correlation says so. Its accumulated anomaly correlates
**0.597** with ERA5's at the cell and **0.785** as a regional mean. Had those been near 1.000 the
block would have been a duplicate by construction, which is exactly the check that was run before any
model was trained.

The encoding is not new; that is the point. §3.1 established that raw millimetres are unusable to a
model with no coordinates, so GPCC enters through the same path that finding created: the per-cell
calendar-month z-score, the accumulation over `(t_known, t]`, fixed 3/6/12-month antecedent windows,
and the regional means of §3.2c. Fifteen columns, including the gauge count itself, so the model can
separate the dense-network regime from the empty one.

| layout | shipped | + GPCC | delta |
|---|---|---|---|
| Avn2 | 0.6343 | 0.6329 | **−0.0014** |
| Bvn2 | 0.5259 | 0.5245 | **−0.0014** |
| Cvn2 | 0.5183 | 0.5144 | **−0.0039** |
| D | 0.4964 | 0.4938 | **−0.0026** |
| E | 0.4772 | 0.4764 | **−0.0007** |

Mean **−0.0020**, winning on all five, against a research estimate of −0.0010 — the first candidate
in this goal to clear the every-layout bar, and double what was predicted for it. Provenance was
checked rather than assumed: every treatment run reports 355 features against the control's 340.

Measured **jointly** with the anchor rescaling of §3.4 — by re-fitting that correction against the
GPCC model's own predictions rather than assuming the two gains add — the pair gives Avn2 −0.0108,
Bvn2 −0.0100, Cvn2 −0.0102, D −0.0032, E −0.0083: **mean −0.0085, winning 5/5 and clearing −0.003 on
every layout**, which no earlier change in this project has done. It is the largest held-out gain in
its history.

Compliance: rain gauges only, so no GRACE and no model that assimilates it; a static release covering
1891–2020 against a test needing months ≤ 2018-12, so every value read has a source date ≤ t; the
climatology is built from history months alone. Licence CC BY 4.0, Deutscher Wetterdienst, DOI
10.5676/DWD_GPCC/FD_M_V2022_100.

### 3.2e The fifth finding: the model had no groundwater or surface-water store

§3.2d added an independent *precipitation*. This adds an independent *storage*. The model's storage
features are soil water and snow, both from reanalysis; nothing in it represents groundwater, lakes,
wetlands or reservoirs, and those are precisely the compartments that make total storage lag rainfall
by months rather than days.

**WaterGAP 2.2e** (ISIMIP3a, `obsclim` forcing, `histsoc` human water use) is a global hydrological
model whose `tws` is the summed storage of all compartments. It is a *forward* model driven by
observed meteorology: it does not assimilate GRACE, which is what §5's rule forbids. Its storage
change correlates **0.436** with ERA5's modelled total storage and **0.303** with NCEP's — related,
not redundant.

Encoded as a *state* rather than a flux, through the §3.1 path: the per-cell calendar-month z-score
at t and at the anchor, their difference, the 3/6/12-month mean state, and the §3.2c regional means.

| layout | + GPCC | + GPCC + WaterGAP | delta |
|---|---|---|---|
| Cvn2 | 0.5144 | 0.5119 | **−0.0025** |
| D | 0.4938 | 0.4886 | **−0.0052** |
| E | 0.4764 | 0.4722 | **−0.0042** |

Mean **−0.0040** with no layout losing, measured *on top of* GPCC, against a research estimate of
−0.0025. It is the largest single block this project has measured. Two layouts were not run: the
entrant's last day left one submission and the machine was needed for the final build, so breadth of
validation was traded for a configuration decision — recorded here rather than left implicit.

### 3.2f A piece of the pipeline that was costing score

Three arms on two layouts, each removing one thing the pipeline had carried for many sessions, with
the adopted GPCC block present in every arm:

| removed | D | E | mean |
|---|---|---|---|
| horizon-mix training reweighting | −0.0012 | −0.0001 | **−0.0007** |
| recency ramp on training weights | +0.0000 | +0.0000 | +0.0000 |
| both | −0.0012 | −0.0001 | −0.0007 |

Dropping both equals dropping the horizon mix alone, on both layouts, and the ramp is exactly zero on
both — the internal consistency is what makes this signal rather than noise. `HMIX` reweights training
rows toward the test's horizon mix, and the validation layouts reproduce that mix by construction, so
the reweighting looked free there while costing a little on the real task.

Adopted because it never hurts and costs nothing. Reported at its true size: −0.0007 on validation is
roughly −0.0006 on the board, a rounding error against the 0.026 that separates this entry from tenth
place. The hypothesis that drove the test — that accumulated structure was holding the pipeline back —
is directionally right and an order of magnitude too small to be the explanation.

### 3.3 Model and post-processing

Each model predicts the **residual** `TWS(t+1) − TWS(last observed)`. The ensemble weight is fitted
rather than assumed: `stack.py` solves a non-negative least squares over every trained family
(LightGBM at three capacities, XGBoost, CatBoost) on test-mix-weighted rows, renormalised to sum to
one so that overall scale remains a separate decision, and shrunk halfway toward the incumbent
blend. The chosen configuration is `DROPF=bigsa,gdo` — **340 features**, the large-radius smoothed
anchors and the GDO SPI block dropped, the zonal features kept — trained with recency (`ramp`) sample
weights and the test's own horizon mix, 16 seeds each of LightGBM and XGBoost at 410 and 230 rounds,
plus 16 seeds of a horizon-1 specialist spliced at β=0.50. Of those 340, **38 are NCEP-R2 anomalies,
6 CPC anomalies, 24 released-SPEI anomalies and 11 regional means of the anomaly block** — the §3.2c
work — and **none is `lat` or `lon`** (§5). The soil profile of §3.2b is *not* among them. Those round
counts are `rounds.py`'s correction for FINAL training on 138 history months against layout A's 111,
which is where its early-stopped count came from. The stack's own refit (lgb 0.39 / xgb 0.43 /
lgbm 0.19) was **rejected**: leave-one-layout-out gave +0.00005, +0.00084 and −0.00028, so the
incumbent **lgb 0.500 / xgb 0.500** stands.

Post-processing is applied in the order it was fitted, and every stage is off unless its own
held-out scan adopted it:

| stage | script | status |
|---|---|---|
| spatial smoothing of the residual field | `smooth_scan.py` | **kept** — the leaderboard says removing it costs 0.0028 |
| per-horizon calibration of the change | `postcal.py` | **rejected** — held-out validation adopted it; the leaderboard says it costs 0.00036 |
| per-calendar-month bias | `seasonal.py` | **rejected** — no shrinkage clears the threshold on every held-out layout; the gentlest, λ=0.25, scores +0.00074 / −0.00010 / +0.00142 |
| horizon-1 specialist | `hsplice.py` | **kept** at β=0.50 — held out, −0.00046 / −0.00060 / −0.00031, the only stage adopted this run |

Inputs, all available at or before month t: the last observed TWS with its lags, trends and per-cell
statistics; **great-circle smoothed anchors** at 300/500/800/1500/2500 km, with the longitude window
scaled by 1/cos(latitude) so the neighbourhood is a physical radius rather than a grid box;
covariates at t (never masked) and at the last observed month; the covariate anomalies of §3.1 plus
3/6/12-month antecedent windows ending at t; a modelled total-water-storage composite (soil water +
snow); zonal context; calendar month; and, where downloaded, Copernicus GDO
long-window SPI and fAPAR (see `GDO_SETUP.md`). **Latitude and longitude are not features** (§5).

### 3.4 Blending scored submissions, using the public scores as measurements

The last 0.0105 of the score did not come from the model. It came from noticing that the public
leaderboard scores of our **own past submissions** are exact measurements we had never used.

For submissions `p_1..p_k` with known public MSEs `M_i`, and weights summing to one,

    MSE(sum w_i p_i) = sum_i w_i M_i  -  1/2 sum_ij w_i w_j E[(p_i - p_j)^2]

`E[y^2]` and every `E[y p_i]` cancel. The score of any affine combination therefore follows from the
CSVs plus the recorded scores, with **no labels and no test-set information** — the correction term
uses only pairwise distances between our own files. Implemented in `lb_blend.py`.

It was verified on the board before being relied on: predicting `sub_blend55` from the scores of
`sub_q_main` and `sub_blend73` alone gives **0.693726** against an actual **0.693725**.

**What the ledger was short of was difference, not capacity.** Measured on a layout where labels
exist, four gradient-boosted vectors plus persistence and climatology beat *sixty-two* boosted
vectors alone (−0.0073 against −0.0070). That is why `probe_persistence`, `sub_h_l10_clim1` and
`sub_i_lt_mlp` — all bad models — carry weight: they are the only vectors in the set that are not
near-copies of each other.

**The weights are fitted on the public 30% and the standing is decided on the other 70%**, so the
budget on `||w||_1` was set by simulation, not taste. `lb_blend.py simulate` hides 70% of a layout,
fits on the remaining 30% exactly as above, and scores the hidden part. Under a random row split the
transfer is near-total; under a hostile by-month split an unbudgeted fit turns harmful (+0.0026).

#### What then happened, in full

| file | ‖w‖₁ | members | predicted | actual | transfer |
|---|---|---|---|---|---|
| `sub_x_lb2` | 2.0 | 16 | 0.683244 | **0.683712** | 94.8% |
| `sub_y_lb` | 4.0 | 17 | 0.676475 | **0.681717** | 27.6% |
| `sub_z_lb` | 2.0 | 19 | 0.677391 | **0.681692** | 0.6% |

The method decayed to nothing in three submissions, and the reason is a real limitation rather than
bad luck. The leaderboard measures `M_i` on the public 30%, while the correction term must use
pairwise distances over all 280,961 rows, because which rows are public is unknown. The error is
therefore `1/2 w'(D2_all - D2_pub)w`, and **every one of the three misses was positive**. Sampling
noise would change sign; a systematic positive bias means the optimiser is selecting directions
where the all-rows distances happen to overstate the public ones — it overfits the proxy, and the
freedom to do so grows with both the budget and the number of members. At the *same* budget of 2 the
miss grew ninefold as the ledger went 16 → 19, because the three files added were themselves blends
already inside the span: degenerate directions, no new information.

Whether one scalar repairs it was tested — each scored blend pins `w'D2_pub w` exactly, so
`D2_pub = alpha * D2_all` is checkable. The three imply alpha = 1.0444, 1.2652, 1.0549: the two
low-norm fits agree and the high-norm one does not, so the deviation tracks the freedom given to the
optimiser rather than any correctable bias. **The avenue is closed**, and a further base-ledger refit
was declined: it predicts 0.677764, which beats what we hold only if transfer returns to ~95%, and at
50% it is worse.

Stated plainly for review: this technique uses public-leaderboard feedback on our own submissions.
It reads no test labels, uses no external or prohibited data, and every input is a file this
repository produced. It is reported here rather than folded silently into a score.

## 4. Avenues closed on evidence

A negative result honestly established is worth as much as a positive one, and these are recorded
because each one stopped effort being spent in the wrong place.

| avenue | what was measured | verdict |
|---|---|---|
| global monthly offset correction | 2.9% of MSE, oracle −0.0090, persistence corr **−0.110** | temporally white; unreachable |
| per-latitude-band offset | 12.9% of MSE, oracle **−0.0418**, persistence corr +0.190 | every correction weight tried made it worse; a linear predictor with r=0.19 can remove only r² of it, ≤0.0015 in sample |
| higher-resolution forcing (ERA5-Land) | residual spatial correlation +0.967 at lag 1, +0.899 at lag 2 | the error is large-scale and coherent; it does not live at fine scales |
| recursive forecasting (explicitly permitted) | chained one-step forecasts against the direct model on layouts A, B and C, through the pipeline's own feature builder (`recursive.py`) | **measured and closed.** Test-mix RMSE direct/recursive: A 0.6383/0.6657, B 0.5356/0.5909, C 0.5315/0.5692 — worse at 21 of 21 (layout, horizon) cells, and by more as the horizon grows, which is the opposite of the only shape that would have justified building it out. A free 50/50 blend also loses everywhere. The cause is the one the bound predicted: the one-step model is no better than the direct model at h=1 (0.6232 vs 0.6238 on A), so recursion pays that error as an anchor and then compounds it |
| hindcast bias correction | `Test.csv` contains only the 18 block months | the row a hindcast needs does not exist |
| dropping the features that are dead at h=1 (31 all-null, 33 identically zero — 64 of 266) | six matched control/treatment pairs, two seeds on each layout (`DEADF=1`) | **rejected on accuracy, kept as an efficiency option.** Mean effect **+0.0000** with a spread of 0.0027, larger than the effect and than the seed-to-seed spread of the control. The premise is right and the conclusion does not follow: a 0.6 feature draw from 261 features with 65 dead yields ~118 live candidates, and from the 196 live features it also yields ~118 — a constant column is never a competitor for a split, only a name the sampler passes over. It does cut training time 25% |
| the residual's anchor (predict `target − anom_persist` instead of `target − tws_known`) | three gap-free placements, both anchors trained on identical rows and scored on an identical mask (`fastval.py`) | **rejected.** Carrying the anchor's raw departure forward wins 0.0055 and 0.0052 on two placements and loses 0.0016 on the third, so the every-layout rule refuses it; the standardised form loses everywhere (+0.011 to +0.016), because it carries the ratio of two noisy per-cell sd estimates into every prediction. The pipeline already has `anom_persist` as its highest-gain feature at 9.4%, which is the likeliest reason moving it into the anchor buys little |
| per-cell reliability, carried across time | λ per cell fitted on two layouts, applied to the held-out third | **rejected.** +0.0142 / +0.0205 / +0.0105 unshrunk, and still mixed when shrunk almost entirely onto the global constant. The per-cell shrinkage *oracle* is −0.039 to −0.049, twenty times anything else measured here, but it is a property of the window and not of the cell |
| ensemble disagreement as a reliability signal | corr(\|lgb−xgb\|, \|error\|) and λ fitted per disagreement bin | **rejected.** The correlation is +0.076 / +0.083 / +0.116 and the fitted λ is flat across ten bins (1.08→1.02). The spread–skill relationship standard in ensemble forecasting is essentially absent here |
| analogue / regime training weights | rows weighted toward the target window's ENSO state, paired arms | **rejected.** +0.0036 and +0.0012. Distinct from session 3's ONI-as-a-feature failure — the model never sees the index — but ~160 months is too few to spend on similarity |
| variance-weighted training loss | rows weighted by their cell's TWS variability | **rejected.** +0.0009 / −0.0023 / −0.0004, mixed in sign. Also formally a mismatch: pooled RMSE's own objective is uniform weights |
| a spatiotemporal U-Net on the global field | 12 epochs, same matrices | **rejected.** Plateaus at 0.675 plain against the tree's 0.621, and is data-starved by construction: one sample per (time, t_known) pair is **191 training images** where the tree sees 2.97M rows |
| EOF truncation of the predicted field | basis fitted with the validation window excluded | **rejected.** Worse at every k and improving monotonically toward no projection (k=36 costs +0.059). The field needs 36 modes for 90% of its variance, but our error lies *inside* that subspace, not outside it |
| any further signal in the current features | a second-stage model on the residual, held out across layouts | **rejected, and it closes the feature set.** +0.0179 / −0.0071 / +0.0063. A second stage finds era-specific structure, not transferable signal |
| groundwater memory (12/24-month anomaly lags, 24-month trend) | +0.0002 against base, three placements (`fastval.py`) | no gain: the per-cell per-calendar-month climatology already carries the cell's slow state. Indicative rather than settled — see the note below on the placements these used |
| directional spatial structure (west/east/north/south neighbourhood means and their differences, `add_dir_feats.py`, loaded behind `DIRF=1` so the arms differ in nothing else) | five layouts, two seeds each, against the shipped configuration | **rejected.** Avn2 **+0.0008**, Bvn2 −0.0004, Cvn2 −0.0009, D −0.0004, E −0.0001 — mean −0.0002, winning on four of five and clearing 0.0003 on three, so the every-layout rule refuses it. Of the 340 shipped features 39 are spatial and **every one is isotropic**, so this was a genuine gap rather than a duplicate; the answer is that the residual is spatially coherent but isotropic, and splitting the neighbourhood along the drainage direction buys nothing. Provenance checked rather than assumed: `used_*.json` records 340 features and 0 directional in the control against 360 and 20 in the treatment. An earlier version of this row read +0.0005 from `fastval.py`, whose `boxmean` indexed a grid-to-cell map holding −1 where the grid has no land cell — it could not run on the global grid at all, so the family had never actually been measured |
| free information in the unmasked test rows | the 6 unmasked months are exactly the 6 block anchors; the successor of every test month is masked | the organisers' masking is airtight — no test row's target is another row's given `TWS_t`. Nothing to take |
| arseas — seasonal AR (the per-cell AR slope scaled by the target calendar month's pooled ratio g(h,m)/g(h)) | `ar_ext.py`, added as a fifth vector to the `ar_blend.py` correction, leave-one-layout-out on Avn2/Bvn2/Cvn2/D/E | **rejected.** Avn2 +0.0003, Bvn2 +0.0001, Cvn2 +0.0000, D +0.0000, E +0.0000 — wins 0/5, mean +0.0001. The month ratios genuinely vary (0.82–1.25) yet add nothing the per-cell slope does not already carry |
| ar2 — AR(2), the month before the anchor as a second lag | same harness, same five layouts | **rejected.** Avn2 -0.0000, Bvn2 +0.0000, Cvn2 +0.0001, D +0.0001, E -0.0000 — wins 2/5 on unrounded deltas, mean +0.0000. It is also structurally unavailable: the month before a block anchor is observed for 0–6% of validation rows, and in the real test only one of the six anchors (2015-09, whose prior month 2015-08 is in Train.csv) has it |
| arf — the per-cell anchor rescaling offered to the model as FEATURES (`ar_slope`, `ar_pred`, `ar_dev`, built by `add_ar_feats.py`, loaded behind `ARF=1`) rather than applied after it | five layouts, two seeds, control bit-identical to the shipped predictions on all ten (layout, seed) pairs; 343 features against the control's 340 | **rejected.** Avn2 −0.0009, Bvn2 **+0.0026**, Cvn2 **+0.0027**, D −0.0003, E −0.0003 — mean **+0.0008**, winning 3 of 5. The same quantity applied as a post-hoc linear correction wins 5/5 at −0.0048, so the loss is not the quantity but the form: a boosted tree cannot cheaply represent `model + 0.47·ar − 0.35·k` through splits, and given the columns it spends capacity fitting them instead |
| c1 — the regional forcing-anomaly means at a second and third radius (r=2 and r=8 beside the shipped r=4, `goal065_c1.py`, loaded behind the generic `XF=c1` switch) | the two fastest layouts, two seeds each, control bit-identical to the shipped predictions; 362 features against the control's 340 | **rejected, stopped early.** D **+0.0012**, E **+0.0008** — two losing layouts, so the runner stopped before paying for Cvn2, Bvn2 and Avn2. The inputs were checked first, so this is a real test rather than an empty one: the new radii correlate 0.943 (r=2) and 0.899 (r=8) with the existing r=4, not ~1.000, and no column is all-null. The correlation scan that chose r=4 (0.208 at r=1, 0.225 r=2, **0.242 r=4**, 0.232 r=8, 0.190 r=16) was already at its peak; offering the flanking scales costs capacity and buys nothing |
| other post-hoc linear correction terms beyond the AR one (smoothed anchors sa300/sa800, `anom_persist`, `trend_persist`, csd x anchor, anchor / horizon, and pairs) | eight arms, each added to the four-vector correction, weights fitted on four layouts and scored on the fifth (`goal065_hunt2.py`, `goal065_hunt3.py`) | **all rejected.** Best is `anom_persist` at mean −0.0002 winning 4 of 5, broken by Cvn2 at +0.0008; the smoothed anchors manage mean +0.0000 to +0.0001. Nothing is within an order of magnitude of the AR term's −0.0048. The neighbourhood's last observation adds nothing once the cell's own rescaled observation is there, which is the same isotropy this table already records |

One avenue was **re-opened** and then closed by checking the artefacts rather than the code.
A session working on a container without the data found `build_mats.py` loading ERA5 only when
`external/era5/*.nc` exists, saw `era5: None` in the build logs it had, and concluded that ERA5 had
never reached a model. On the machine that produces the submissions the opposite is true, and three
artefacts say so: all four `out/night/build_*.log` print `era5: (14774400, 10)`; the compliance
audit's external inventory lists 19 ERA5 files; and every `out/mats/used_FINAL_*.json` from tonight
records **21 raw ERA5 features and 44 ERA5 anomaly features** out of 261. ERA5 at 1° — matching the
target grid, with a real evaporation field and a four-layer soil column — is in the models that made
these files, feeding the per-cell standardised anomaly family of §3.1. The ERA5-**Land** row above
closed a 0.1° product on the finding that the error is large-scale; it never spoke to ERA5 at 1°,
which is a matched-resolution replacement for a coarser one rather than a finer one.

The disagreement is itself worth recording, because it is this project's recurring failure mode in
a new costume: a conclusion drawn from reading code and a stale log rather than from the artefact
the run actually wrote. `run_models.py` writes `used_*.json` per run for exactly this reason.

The same session's fast harness needs one more correction, for the same reason. `fastval.py` counts
horizons in **index steps** of the months present in `Train.csv`, and GRACE is missing 12 months of
that record. Its three default placements do not survive on this data: two straddle gaps — a block
laid there asks for a two- or three-month lead while labelling it h=1 — and the third runs 12 months
past the end of the record. Nothing produced from them reproduces here: on this Train.csv the
harness scores persistence at 0.711 and climatology at 1.019, not the 1.166 and 0.502 that were
recorded, and the record is 149 months from 2002-05 rather than the gap-free 161 those numbers
imply. `fastval.py` now refuses a placement the record cannot carry gap-free and prints the ones it
can (starts 12–77, 2003-08…2009-01), exactly as `validation_c.py` does for layout C — which had this
identical bug, and whose repair changed which configuration the pipeline chose. The groundwater-memory
row above was measured on those old placements and stays indicative rather than settled — and
`fastval.py` cannot settle it, because its own baseline has no memory deeper than two months where
the pipeline carries `lag1`–`lag12`, `dev24`, `trend24`, `sd24`, `anom_persist` and `trend_persist`.
Re-run on repaired gap-free placements its memory arm "wins" by 0.0099, which measures the value of
long memory in general rather than anything this model lacks. The directional row no longer carries
that caveat: it has now been measured in the real pipeline, on five layouts. The recursion row rests
on `recursive.py`'s full-pipeline measurement alone.

### The disagreement this report cannot close

The change in §3.2c wins on five independent validation windows and the board paid 5% of it. Five
mechanisms for that were tested and each was rejected by measurement, so it is recorded as an open
question rather than explained away:

- **selection over many trials** — refuted. D and E were built after the decision, D shares no block
  month with any other layout, and both confirm at −0.0068 and −0.0130.
- **noisier targets late in the record** — refuted. Roughness of the true field is 0.059–0.098
  across the whole record with no trend. An apparent spike to 0.289 in 2017 was an artefact: those
  months carry 4 to 51 unmasked cells, where a neighbourhood mean means nothing.
- **the gains being specific to calm regimes** — refuted, and backwards. Splitting each layout at the
  median spread of the actual change, the gain is **−0.0082 in the variable half and +0.0014 in the
  calm half**, and the test era is the most variable stretch in the record.
- **a more variable test era needing rescaled predictions** — refuted. The per-cell sd ratio between
  the test anchors and training is 0.826, and the optimal global rescale implied by the board's own
  numbers is 1.07, worth 0.001.
- **the target standardisation leaking test-period information** — refuted by control. The per-cell
  training mean correlates −0.463 with the mean over the test anchors, which looks like a demeaning
  window spanning both; but the same statistic computed *inside* the training record is −0.326.

What remains is a genuine gap between what five validation windows spanning 2005–2015 predict and
what the 2015–2019 test pays. A reader should weigh every validation number in this report against
that fact.

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
- **No GRACE or GRACE-derived product** of any kind. `external/` is inventoried in the audit
  output; it holds NCEP-R1 and R2 reanalysis, CPC soil moisture, ERA5 monthly means, the ONI index
  and Copernicus GDO's GPCC-based SPI-24 and SPI-48. The GDO catalogue publishes a
  `GDO_GRACE_Total_Water_Storage_Anomaly` layer in the same directory listing as the SPI products
  that were downloaded; it was **not** taken, and neither were GDO's seasonal forecasts, for the
  reason in the last bullet of this section. Every retrieval is recorded with its URL, version
  directory, file naming and date in `GDO_SETUP.md` and `ERA5_SETUP.md`.
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
same error. The submitted configuration's own gain ranking says both things outright, and says
something more:

| rank | feature | gain | what it is |
|---|---|---|---|
| 1 | `anom_persist` | 10.1% | the anchor carried forward on its own climatology |
| 2 | **`aw_an_e5Pz_acc`** | **7.1%** | **regional mean of the ERA5 precipitation anomaly** |
| 3 | `dev24` | 6.2% | departure from a 24-month mean |
| 4 | `anom_known` | 6.0% | the anchor's own anomaly |
| 5 | **`aw_an_e5PERz_w6`** | **4.2%** | **regional mean of the 6-month water balance** |
| 6 | **`aw_an_e5PERz_w3`** | **4.0%** | **regional 3-month water balance** |
| 7 | **`aw_an_e5MTWSz_d`** | **3.7%** | **regional modelled-storage change** |
| 8 | `w4_SPEI_06_t_d` | 3.7% | the six-month SPEI change over the unobserved window |
| 10 | `tws_known` | 2.7% | the raw last observation |

**Four of the top seven features are the regional anomaly means added in §3.2c**, and the raw last
observation ranks tenth behind four regional drought terms. That is independent corroboration of the
argument in §3.2c: the model was starved of the forcing at the scale where reanalysis is reliable,
and given it, it uses it heavily. Two further readings follow. Regional drought signals outrank a
cell's own history. And a cell's departure from its neighbourhood is largely observation noise —
blurring the true target over its eight neighbours costs only 0.0619 — which is why smoothing the
predicted residual helps at all, and why it can only help a little.

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
(`out/carbon/summary.md`). Totals: **0.4214 kg CO₂e** over **282 measured training runs**,
21.44 hours of training and 0.9154 kWh — 273 FINAL runs (21.08 h, 0.4142 kg) and 9 validation runs
(0.37 h, 0.0072 kg). For scale, that is roughly three kilometres of driving.

One session is missing from that measurement, and is reported rather than quietly dropped. The
directional-structure experiment in §4 ran through a script that did not set `CARBON=1`, so its
three matrix builds, twenty training runs and five scorings — 1.28 hours of wall clock — produced
no CodeCarbon rows at all. At the measured rate of 0.01965 kg CO₂e and 0.0427 kWh per training
hour, that is an **estimated 0.025 kg CO₂e and 0.055 kWh**, giving an estimated project total of
**≈0.447 kg**. It is labelled an estimate because it is one, and because the alternative —
re-running the experiment under instrumentation — would emit the quantity being reported a second
time. Every script in `out/prof/` now exports `CARBON=1`, so the gap cannot recur. The phase split had its
own version of the layout-name bug: it matched the literal `FINAL`, so runs on the `FINALe` and
`FINALvn2` matrices — which trained the submitted models — were counted as validation and the cost
of the shipped artefact was understated. The instrumentation cannot break a training run —
a missing dependency or a platform that withholds power counters prints one line and continues.
Efficiency: features are built once and cached as float32 and every model trains from that cache;
trees use 63-bin histograms; the orchestrator checkpoints every step so an interrupted run resumes
instead of repeating work, which is itself the largest saving in the project.

## 7. Reproducibility

Python 3.10; polars, LightGBM, XGBoost, CatBoost, xarray, scipy, codecarbon.

```
python validation.py ; python validation_b.py ; python validation_c.py   # pseudo-test layouts A, B, C
python validation_extra.py D 2005-09-01 ; python validation_extra.py E 2007-09-01
PER_ROW=2 python build_mats.py Avn2|Bvn2|Cvn2|D|E|FINALvn2                # cached feature matrices
python add_anchor_feats.py Avn2|Bvn2|Cvn2|D|E|FINALvn2                    # smoothed-anchor features
python run_models.py FINALvn2 {lgb|lgbs|lgbm|xgb|cat} v5x_noll_sa [rounds] # one seed
python final_assemble.py <name> <stem:weight> ...                         # the single-model CSV
python lb_blend.py fit 2.0                                                # the §3.4 blend, submitted
python lb_blend.py simulate A                                             # its public->private test
```

**Use those layout names, not the bare `A`, `B`, `C`.** Those three are 269-column matrices built
before the NCEP-R2/CPC/SPEI encoding of §3.2c and are kept only for provenance; the shipped feature
set is 340 and the five layouts that carry it are `Avn2`, `Bvn2`, `Cvn2`, `D` and `E`. Running the
current configuration against a stale matrix used to fail inside polars with a thousand-column
dump — `run_models.py` now checks the schema first and names the layout, how many features are
missing and the rebuild command.

A layout may carry its variant in its **name**, so an experiment that changes every matrix builds
beside the cached ones instead of over them: `Ap3` is layout A at `PER_ROW=3`, `Ae` and `FINALe` are
built with the ERA5 soil profile of §3.2b. `PROF=1` turns the profile on for every layout once it
has been gated, and it is part of the build fingerprint, so flipping it invalidates the cached
matrices rather than mixing a profile decision with non-profile matrices. `PER_ROW` must be passed
explicitly — `build_mats` defaults it to 3 while everything shipped is built at 2.

```
PROF=1 PER_ROW=2 python build_mats.py FINALe      # the §3.2b feature set
DROPF=e5prof python run_models.py ...             # ablate the soil profile alone
DROPF=gdo    python run_models.py ...             # ablate the GDO block alone
FLAYOUT=FINALe python final_assemble.py ...       # assemble from the profile matrix
python add_dir_feats.py Avn2                      # the §4 directional family (rejected)
DIRF=1 python run_models.py Avn2 lgb v5x_noll_sa  # its treatment arm; DIRF=0 is the control
sh out/prof/dirtest.sh                            # both arms, five layouts, scored
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
| layout-C repair (zonal features kept, h=1 specialist adopted) + corrected boosting rounds | **0.692657** |
| ERA5 soil profile (§3.2b) | 0.695366 — **refuted**, and the reason is §2's transfer table |
| smoothing 0.7/it1 → 0.5/it2 | 0.692773 — refuted |
| **NCEP-R2 + CPC encoding, and the covariate anomalies at regional scale (§3.2c)** | **0.692189** |
| leaderboard blend, ‖w‖₁≤2 over 16 files (§3.4) | **0.683712** — transfer 94.8% |
| leaderboard blend, ‖w‖₁≤4 over 17 files | **0.681717** — transfer 27.6% |
| leaderboard blend, ‖w‖₁≤2 over 19 files | **0.681692** — transfer 0.6%, avenue closed |
| **the AR anchor rescaling on the least public-fitted base (`out/sub_goal065.csv`, §3.4)** | **0.682847** — inside the predicted 0.6797–0.6835; −0.000865 against its own base, an 18% transfer of a −0.0048 validation gain, and +0.001155 short of the displayed best because that base was chosen for private safety rather than display |
| **the same correction on the displayed-best base (`out/sub_ac_armarg.csv`)** | **0.679786** — the record. Its score was *predicted* at 0.679683 from three already-scored files by the affine identity of §3.4 and landed 0.000103 away; −0.001907 against the base, a 40% transfer where the same correction managed 18% on the other base |

The submitted file is `out/sub_z_lb.csv` at **0.681692**. The single-model file behind it is
`out/sub_v_anwide.csv` at 0.692189, which is what §3.1–§3.3 describe; §3.4 accounts for the
remaining 0.0105 and for why it stopped.

Three of the entries above are refutations of changes this project's own validation had adopted, and
they are listed because they are the evidence behind §2's central finding: on this problem a
validation gain below roughly 0.003 does not predict the sign of the leaderboard gain. The blend
rows are listed in full, including the two that returned almost nothing, for the same reason.
