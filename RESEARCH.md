# RESEARCH — can a legal change take the public board below 0.65?

Leaf 1.1 of `PLAN.md`, written 2026-09-11. **Nothing here trained a model and nothing was
downloaded.** Every new number below was computed read-only from files already on disk
(`out/mats/<L>_va.parquet`, `<L>_tr.parquet`, the side-cars `<L>_va_{ar,dir}.parquet`, and the shipped
control predictions `pred_<L>_lgb_v5x_noll_s{0,1}_d0.npy`) on the five layouts Avn2 Bvn2 Cvn2 D E,
with throwaway scripts kept in the session scratchpad, not in the repository. Facts about external
products are metadata looked up on 2026-09-11.

## Bottom line

- The goal needs a board gain of **0.0317** (0.681692 → below 0.65). A single candidate reaches it
  only with a mean held-out validation gain of **−0.041** or better.
- The best legal candidate found is worth about **−0.0012** on validation — thirty times too small.
  **None** of the candidates reaches the 0.003 size below which this project's changes have flipped
  sign on the board, so by the adoption arithmetic every expected board gain is ~0.
- The budget is below 0.0317 on every reading, including the most generous one (below). **On this
  evidence the public board cannot be taken below 0.65 legally from here.** The measurements are
  still worth running in the order given, because each is cheap and each closes a question.

## The numbers that frame the gap

| quantity | value | source |
|---|---|---|
| our public best / single-model file / persistence | 0.681692 / 0.692189 / 0.8864 | board |
| our skill ratio vs persistence on the board (best file / single model) | 0.769 / 0.781 | board |
| skill ratio an entrant needs at 0.65 / 0.62 / 0.56 | 0.733 / 0.699 / 0.632 | arithmetic |
| our skill ratio on validation layouts A / B / C | 0.83 / 0.77 / 0.72 | NOTES 10q |
| h=1 skill ratio (worst horizon, 33.3% of test weight) | 0.862–0.876 on A/B/C | NOTES 10r |
| largest validation gain ever measured | −0.0156 (covariate-anomaly encoding) | REPORT §2 |
| share of our error variance shared with the 8 neighbours | 98.2% | NOTES 10n |

## How the new estimates were made, and how far to trust them

The quick estimator used throughout is a **held-out linear correction of the control's residual**:
residual = target − mean(s0, s1 control prediction). A candidate's quantities (standardised, NaN → 0)
are regressed on that residual on four layouts, each layout given equal weight and its rows reweighted
to the test horizon mix, and applied to the fifth. The delta is test-mix RMSE against an
intercept-only held-out correction, so a global offset is never credited to a candidate. It costs
~25 s and no training.

Before trusting it on anything new it was run on families whose answer is already known:

| family given to the proxy | proxy mean (wins/5) | known result |
|---|---|---|
| two features the model already has (`aw_an_e5Pz_acc`, `anom_persist`) | −0.0002 (3/5), per layout ±0.0018 | 0 by construction |
| directional side-car, 20 columns | −0.0005 (3/5) | trained: −0.0002 (REPORT §4) |
| AR vector alone | −0.0002 (4/5) | ar_blend "AR only": −0.0000 (NOTES 11i) |
| persistence + climatology | −0.0012 (3/5) | ar_blend: −0.0016 |
| AR + persistence + climatology | −0.0051 (4/5) | ar_blend: −0.0064 (5/5) |

Reading: it reproduces known linear corrections at about 0.8 of their size; its noise floor is about
±0.0005 on the five-layout mean and ±0.002 on a single layout. It is **not** a lower bound for a
trained model: the directional family came in at 0.4 of its proxy, and the t+1-covariate test came
in at a third of its linear fit (−0.0055 linear vs −0.0018 GBM, NOTES 10p/10q). Proxy numbers below
are therefore treated as optimistic.

Three further facts measured in the same pass, used by several candidates:

- **Test-era extremeness.** Share of rows outside the training 0.5–99.5% range on at least one of the
  nine top-gain features: validation 8.7–14.8% (carrying 10.8–19.5% of MSE); the FINAL test rows
  **11.9%**. The test is not more extreme than validation on the features the model leans on.
- **History-gap mismatch.** The previous observation behind `tws_prev`/`slope` is 1 month before the
  anchor for ≥90% of training rows (p10/p50/p90 = 1/1/1 on every layout) but 3–11 months for
  validation rows (p50 = 5 on Bvn2, Cvn2, D, E and 10 on Avn2). `lag1`/`d1` are null on 3.4% of
  Avn2 training rows and on 100% of rows at its main validation anchors.
- **Horizon profile.** Ratio of the control's h=1 RMSE to its test-mix RMSE: 0.975 / 0.936 / 0.893 /
  0.946 / 0.932 (Avn2 Bvn2 Cvn2 D E), mean **0.936**.

## Hypotheses for the gap

H1. **Prohibited GRACE information explains the 0.56–0.60 entries, and nothing legal can match it.** The target is a standardised GRACE anomaly, so any GRACE or GRACE-FO solution, a GRACE-assimilating land-surface model (GLDAS-2.2 CLSM DA1), a GRACE-trained reconstruction, or GDO's TWS layer, used for the masked months or for t+1, hands over the answer. Measured here: filling the masked `TWS_t` alone, which turns every row into h=1, would give our model 0.681692 × 0.936 ≈ **0.638** — inside the 0.62–0.65 band but nowhere near 0.56, so 0.56 needs the target month itself. Predicts: those entries' error is flat across h1–h7 where legal models degrade with horizon, they fall at the code and trustworthiness review, and no candidate below closes the distance.

H2. **Selection on the public 30% inflates part of the 0.62–0.65 band.** Blending scored files against their public scores took our own single-model 0.692189 to 0.681692, and 99% of that file's edge was specific to the public rows (NOTES 11j); the L1-budget-8 oracle reached 0.6726 (NOTES 11). An entrant with more submissions, or less restraint, gets more. Predicts: those entries lose roughly 0.005–0.02 on the private 70%, and the true legal gap is smaller than the board shows. This is also why the private-safer base (`sub_x_lb2`, 5% public-specific) matters more than any candidate here.

H3. **Legal storage compartments were never offered.** Every covariate on disk (ERA5, NCEP-R1/R2, CPC, the released soil moisture) carries soil water and snow only; GRACE also sees groundwater, lakes, wetlands, reservoirs, river storage and human abstraction. Predicts: gains concentrated where those stores dominate (floodplains, irrigated aquifers) and at longer horizons; tested by C5 and C6. The nearest measured analogue — the ERA5 four-layer soil profile, −0.0036 on validation and +0.0027 on the board (NOTES 10f, 10i) — says the size is small and the transfer unreliable.

H4. **The test era compresses whatever validation measures.** The regional anomaly means won on five independent windows at −0.0091 and the board paid −0.0005; five mechanisms for that were tested and refuted (REPORT §4). Entrants whose methods are less fitted to 2005–2015 structure may simply transfer better. Predicts: every candidate below pays a small fraction of its validation size, and entrants reorder between public and private.

H5. **Coordinates or cell identity (prohibited, R1) let an entrant learn per-cell dynamics.** Measured: adding lat/lon back is worth −0.0011 (NOTES 10n). Predicts nothing near the gap; rejected as an explanation.

H6. **Information at or after t+1 inside the provided files (prohibited, R2).** Covariates at t+1 in a GBM: −0.0018, mixed sign; interpolating between observed blocks: 0.634 against our legal 0.636 on layout A; a model given the next block's TWS: 0.006 (NOTES 10q, 11e). Predicts at most ~0.006 from within-file leakage, which could explain a 0.675 entry but not 0.65, let alone 0.56.

H7. **Some entrants have a genuinely better legal model, most plausibly at h=1.** h=1 is a third of the weight and our worst horizon (skill 0.86–0.88 against 0.59–0.69 at h7 on A/B/C). A legal 0.62 needs a board skill ratio of 0.70, better than any of our five layouts. Predicts: their advantage sits at h=1–2 and survives the private split. The legal h=1 levers left here are the anchor rescaling already in flight and the within-month forcing timing of C8.

## Candidates

Every candidate below is legal under R1–R5, is not one of the two AR items already in flight, and has
not been measured in this project before. Candidates C1–C4 and C9 were estimated with the read-only
proxy above; C5–C8 need data that is not on disk and are estimated by analogy to measured results.

### C1: Regional forcing-anomaly means at more than one radius
Type: representation
Rules: R1 ok (box means over grid neighbours; coordinates only index the grid, exactly as the shipped `aw_*` columns do) · R2 ok (inputs are shipped `an_*` columns built from covariates at months ≤ t) · R3 n/a (no TWS used) · R4 ok (ERA5 already on disk, source months ≤ t) · R5 ok (ERA5 assimilates no GRACE; no new product)
Expected validation gain: -0.0012
Estimated by: read-only proxy on five layouts. NaN-aware box means (longitude wrapped) of `an_e5PERz_acc`, `an_e5Pz_acc` and `an_e5MTWSz_d` at r = 2, 8 and 16, entered as differences from r = 4 (my r = 4 reproduces the shipped `aw_*` column at correlation 1.000): Avn2 +0.0022, Bvn2 −0.0019, Cvn2 −0.0029, D −0.0010, E −0.0040, mean **−0.0015**, 4/5. Ablations: r = 16 alone −0.0011, r = 8 alone −0.0005, P−E−R alone −0.0001 — the signal sits at ~1600 km in precipitation and modelled storage, consistent with NOTES 10o's scan, where P−E−R alone peaked at r = 4. Credited at 0.8 of the proxy, the calibration ratio above.
Cost (hours): 2
Download: none
Closed-list check: not on the list. Distinct from NOTES 10r's multi-radius TWS-momentum aggregates (+0.0003, dead), which aggregated a smooth state; this aggregates the noisy driver, the case 10r itself says pays. Isotropic, so not the closed directional family.
Risk: the most recent window, Avn2, is the one that loses (+0.0022) — the same era-ordering that preceded §3.2c's 5% transfer.

### C2: Per-cell water-balance response coefficient as a feature
Type: representation
Rules: R1 ok (a per-cell slope computed by grouping a cell with its own history, like the shipped `b_spei1`/`resp_sm`; no coordinate enters the model) · R2 ok (fitted on the layout's training rows, all before the window, and applied to covariates ≤ t) · R3 ok (the cell's own TWS history only) · R4 ok (ERA5 on disk) · R5 ok (no new product)
Expected validation gain: +0.0000
Estimated by: read-only proxy. Per-cell OLS slope β_c of (target − tws_known) on `aw_an_e5PERz_acc`, fitted on each layout's own training rows at h ≥ 2 and shrunk toward the global slope (λ = 50; global slope 0.230–0.264, sd of β_c 0.084–0.112), offered as β_c·x and (β_c − β̄)·x: Avn2 −0.0004, Bvn2 −0.0001, Cvn2 +0.0009, D −0.0001, E +0.0003, mean **+0.0001**, 3/5. Nothing to take.
Cost (hours): 2
Download: none
New reason: per-cell reliability (closed) rescaled the model's OUTPUT by a λ fitted on other validation windows, which did not transfer across time. This is a physical response slope fitted on each layout's own 35–111 months of history and offered as an INPUT the tree may ignore, so that measurement does not directly apply. The proxy nonetheless says the practical verdict carries over: the tree already gets what this adds from `b_spei1`, `resp_sm` and `csd`.

### C3: Piecewise-linear leaves for out-of-range forcing (LightGBM linear_tree)
Type: model
Rules: R1 ok (same 340 features, no coordinates) · R2 ok (no new inputs; only the leaf form changes) · R3 n/a (no change to TWS use) · R4 n/a (no new covariate) · R5 ok (no product involved)
Expected validation gain: +0.0000
Estimated by: two read-only measurements of the premise that constant leaves saturate in an extreme era. (a) Rows beyond the training 0.5–99.5% range on any of the nine top-gain features: 8.7–14.8% of validation rows against 11.9% of FINAL test rows, so the test is no more extreme than validation. (b) A held-out linear correction on the nine exceedances x − clip(x) makes the residual worse: Avn2 +0.0013, Bvn2 +0.0051, Cvn2 −0.0002, D −0.0011, E +0.0015, mean **+0.0013**, 2/5. The control does not saturate in any way a slope could fix.
Cost (hours): 3
Download: none
Closed-list check: a model structure, not post-processing — nothing is fitted after prediction. It needs its own training script, because `run_models.py` may only gain the XF switch.

### C4: Train on rows whose history is masked the way the test's is
Type: model
Rules: R1 ok (no coordinates) · R2 ok (removes information from training rows and adds none) · R3 ok (own-cell TWS at ≤ t_known only) · R4 n/a (no covariate change) · R5 ok (no product involved)
Expected validation gain: -0.0003
Estimated by: the mismatch is measured and real: the gap behind `tws_prev`/`slope` is 1 month in training against 3–11 months in validation, and `lag1`/`d1` are null on 3.4% of training rows against all rows at the main validation anchors. But a held-out linear correction on [`slope`, `tws_known − tws_prev`, log gap] finds nothing to correct: Avn2 +0.0004, Bvn2 +0.0008, Cvn2 +0.0001, D +0.0001, E −0.0009, mean **+0.0001**, 1/5. None of the gap-dependent features is in the top-10 gain ranking (REPORT §6.2), which bounds what a misreading can cost. The −0.0003 credits the chance that the tree's null routing, learned from GRACE-gap rows, costs something a linear probe cannot see; it is inside the proxy's noise floor.
Cost (hours): 3
Download: none
Closed-list check: not on the list, and not DEADF (REPORT §4): DEADF drops columns that are constant on the training rows, and `lag1` is not constant there. Implementation: rebuild the lag/`slope`/`tws_prev` columns for training rows with a test-like gap drawn from {4, 5, 6} months, via side-car plus DROPF of the originals.

### C5: GLDAS-2.1 Catchment open-loop groundwater and total storage
Type: external
Source: NASA GES DISC, GLDAS_CLSM10_M v2.1 — Catchment-F2.5 in LIS 7, 1°, monthly, 34 fields including groundwater storage (GWS_tavg), terrestrial water storage (TWS_tavg), the soil moisture profile, SWE and canopy water
URL: https://disc.gsfc.nasa.gov/datacollection/GLDAS_CLSM10_M_2.1.html (DOI 10.5067/FOUXNLXFAZNY)
Licence: US Government work, no restrictions (https://www.usa.gov/government-works); access needs a NASA Earthdata login, which the user must supply
Available through: 2026-08-31 (NASA Open Data Portal listing, checked 2026-09-11; coverage starts 2000-01, main-stream latency about 1–2 months, with an Early Product stream GLDAS_CLSM10_M_EP also listed). The test needs months ≤ 2018-12, all long published, so every value read has source date ≤ t.
Rules: R1 ok (per-cell anomaly and r = 4 regional means; no coordinates) · R2 ok (month-t values only; the product contains no forecast fields) · R3 n/a (no TWS used) · R4 ok (archive runs to 2026-08; each row reads months ≤ t only) · R5 ok — GLDAS-2.1 is open loop with NO data assimilation (GES DISC: GLDAS-2.0 and 2.1 are "open-loop", only GLDAS-2.2 assimilates GRACE); GLDAS-2.2 CLSM GRACE-DA1 must NOT be taken, and TWS_tavg here is modelled, not GRACE-derived
Expected validation gain: -0.0020
Estimated by: analogy. The only storage-family addition ever gated on validation is the ERA5 four-layer soil profile: −0.0036 on three layouts (NOTES 10f), which the board inverted at +0.0027 (NOTES 10i). The genuinely new compartment here is groundwater — CLSM's soil and snow duplicate ERA5, NCEP-R1/R2 and CPC, all already encoded — and the shipped model already carries slow storage from the TWS record itself (`lag12`, `dev24`, `trend24`, `sd24`, `anom_persist`, `trend_persist`; NOTES 11f found a memory arm adds nothing). Credited at ~0.55 of the soil-profile analogue, which also matches the other analogue, a further reanalysis in the right encoding (NCEP-R2/CPC: −0.0022, NOTES 10m).
Cost (hours): 4
Download: yes, needs user approval (GWS_tavg and TWS_tavg for 2001–2019 by OPeNDAP subset, about 100 MB — size not verified)
Closed-list check: not on the list, and not "groundwater memory via fastval.py", which added TWS lags; this adds a modelled groundwater store driven by forcing.

### C6: WaterGAP 2.2e surface-water, reservoir and groundwater storage
Type: external
Source: WaterGAP v2.2e global hydrological model, ISIMIP3a obsclim run (gswp3-w5e5 forcing, histsoc human water use), total water storage and its compartments (groundwater, lakes, wetlands, reservoirs, rivers, soil, snow, canopy), 0.5°, monthly
URL: https://data.isimip.org (ISIMIP3a / water_global / watergap2-2e / gswp3-w5e5 obsclim histsoc default, variable tws, monthly); model paper https://gmd.copernicus.org/articles/17/8817/2024/ ; daily-storage release https://gude.uni-frankfurt.de/handle/gude/346
Licence: CC BY 4.0 according to the model paper (Müller Schmied et al., GMD 17, 8817, 2024); both repository pages returned bot protection or HTTP 502 when checked, so confirm on the dataset page before downloading
Available through: 2019-12 (static ISIMIP3a obsclim release covering 1901–2019, with no operational updates). The test needs months ≤ 2018-12, so every value read has source date ≤ t.
Rules: R1 ok (regridded to 1°, per-cell anomaly and regional means) · R2 ok (month-t storage only) · R3 n/a (no TWS used) · R4 ok (release covers 1901–2019; each row reads months ≤ t only) · R5 ok — standard WaterGAP 2.2e is calibrated against river discharge only and does not assimilate GRACE (GMD 2024; GRACE is used only to evaluate it); GLWS2.0, which assimilates GRACE into WaterGAP, and any GRACE-calibrated WGHM variant must NOT be taken
Expected validation gain: -0.0025
Estimated by: analogy to the same soil-profile result as C5 (−0.0036 validation, +0.0027 board), credited at 0.7. Its new compartments — lakes, wetlands, reservoirs, river storage, and groundwater with human abstraction — appear in no current covariate. Against that, the shipped model already sees their slow trend through `dev24`/`trend24`, and its forcing overlaps ERA5 (W5E5 precipitation is bias-adjusted ERA5).
Cost (hours): 5
Download: yes, needs user approval (tws plus two or three compartments at 0.5° monthly, subset to 2001–2019, estimated 0.3–1.5 GB — not verified)
Closed-list check: not on the list. Overlaps C5, since both add groundwater, so measure it on top of C5 if C5 is adopted.

### C7: Gauge-based precipitation (GPCC Full Data Monthly v2022) in the water balance
Type: external
Source: Deutscher Wetterdienst, Global Precipitation Climatology Centre — GPCC Full Data Monthly Product Version 2022, 1.0°, monthly precipitation with gauge counts
URL: https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-monthly_v2022_doi_download.html (DOI 10.5676/DWD_GPCC/FD_M_V2022_100)
Licence: CC BY 4.0 (DWD open-data terms; attribution to Deutscher Wetterdienst); no registration
Available through: 2020-12 (static v2022 release covering 1891-01 to 2020-12; the GPCC Monitoring Product continues with about 2 months' latency). The test needs months ≤ 2018-12, so every value read has source date ≤ t.
Rules: R1 ok (per-cell anomaly and regional means) · R2 ok (monthly totals for months ≤ t; no forecast) · R3 n/a (no TWS used) · R4 ok (static release through 2020-12; each row reads months ≤ t only) · R5 ok (rain gauges only; no GRACE anywhere in the product)
Expected validation gain: -0.0010
Estimated by: analogy. Precipitation is the model's strongest forcing (`aw_an_e5Pz_acc`, the #2 feature at 7.1% of gain), but GPCC has been in a gated model once already: the GDO SPI-24/SPI-48 block is GPCC-based and measured −0.0002 / +0.0005 / −0.0008 marginal (NOTES 10f), and was not adopted. What remains untested is GPCC at short accumulations (`_acc`, w3, w6) and inside P−E−R. The analogue for a further forcing source in the right encoding is NCEP-R2/CPC's −0.0022 (NOTES 10m); half of that is credited, because NCEP-R2 precipitation is already encoded and ERA5 already carries most of the gauge signal where networks are dense.
Cost (hours): 3
Download: yes, needs user approval (the 1° file is about 20 MB gzip per the DWD page — the smallest download here)
Closed-list check: not on the list.

### C8: Within-month timing of month t's water balance (ERA5 daily)
Type: external
Source: Copernicus Climate Change Service — ERA5 post-processed daily statistics on single levels (daily means of total precipitation, evaporation and runoff)
URL: https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics
Licence: CC-BY (CDS dataset page); needs the CDS account already used for ERA5 monthly (`~/.cdsapirc`)
Available through: present (1940 to present, updated daily with a 6-day delay). The test needs days ≤ 2018-12-31, so every value read has source date ≤ t.
Rules: R1 ok (per-cell anomaly of a weighted accumulation; no coordinates) · R2 ok ONLY if the weighting is confined to the days of month t — no day of t+1 may enter, and a unit test must assert that the last day used is the last day of month t · R3 n/a (no TWS used) · R4 ok (source dates are days within month t) · R5 ok (ERA5 assimilates no GRACE; the same family is already in the compliance inventory)
Expected validation gain: -0.0009
Estimated by: physical decomposition plus a measured bound. GRACE's target is a monthly mean, so S̄(t+1) − S̄(t) = [S_end(t) − S̄(t)] + [S̄(t+1) − S_end(t)]. The first bracket depends only on how month t's net flux is spread within the month — rain in the last week lifts end-of-month storage above the month's mean — which is knowable at t and invisible in monthly means. The second bracket is month t+1's weather. The WHOLE t+1 covariate family, which bears on the second bracket, measured −0.0018 in a GBM on three layouts, mixed in sign (NOTES 10q); the first bracket is credited at half of that.
Cost (hours): 6
Download: yes, needs user approval (three variables at 1°, daily, 2001–2019: roughly 2–5 GB — the largest download here; not verified)
Closed-list check: not on the list. It is neither a t+1 covariate nor smoothing along the horizon.

## Budget

Expected board gain uses the adoption arithmetic: validation gain × 0.77 when its magnitude is at
least 0.003; below that, three of three measured changes flipped sign on the board (REPORT §2), so the
board gain is treated as ~0. **No candidate reaches 0.003, so every board entry is ~0.**

| candidate | type | expected validation gain | expected board gain | cost (hours) | needs download |
|---|---|---|---|---|---|
| C1 multi-radius forcing means | representation | −0.0012 | ~0 (magnitude < 0.003) | 2 | no |
| C2 per-cell response slope | representation | +0.0000 | ~0 (magnitude < 0.003) | 2 | no |
| C3 linear-tree leaves | model | +0.0000 | ~0 (magnitude < 0.003) | 3 | no |
| C4 test-like history masking | model | −0.0003 | ~0 (magnitude < 0.003) | 3 | no |
| C5 GLDAS-2.1 CLSM groundwater | external | −0.0020 | ~0 (magnitude < 0.003) | 4 | yes (~100 MB) |
| C6 WaterGAP 2.2e storages | external | −0.0025 | ~0 (magnitude < 0.003) | 5 | yes (0.3–1.5 GB) |
| C7 GPCC gauge precipitation | external | −0.0010 | ~0 (magnitude < 0.003) | 3 | yes (~20 MB) |
| C8 ERA5 within-month timing | external | −0.0009 | ~0 (magnitude < 0.003) | 6 | yes (2–5 GB) |
| **total** | | **−0.0079** | **~0** | **28** | 4 downloads |

For context only, not counted: the two AR items already in flight (ARF, and the seasonal-AR/AR(2)
corrections) start from ar_blend's −0.0064 full and −0.0048 marginal on validation, which is also
below the −0.041 a single change needs.

BUDGET TOTAL: 0.0000
That is below 0.0317 — far below it. Even the most generous reading, crediting every validation
estimate at 0.77 regardless of size and assuming the gains simply add, gives 0.77 × 0.0079 = 0.0061,
under a fifth of 0.0317. Adding the in-flight AR marginal (−0.0048 × 0.77 = 0.0037) as well still
gives only 0.0098. The gains also overlap — C5 and C6 both add groundwater, and C1, C7 and C8 all
refine the same forcing — so the naive sum is an upper bound, not an expectation.

## Recommended measurement order

Ranked by expected gain per hour, since every expected board gain is ~0; runs that need no download
come first, downloads last because they need the user's approval.

1. **C1** multi-radius forcing means — −0.0012 in 2 h, no download, the only proxy signal clearly
   above the noise floor. Early-stop if Avn2 and one more layout go positive.
2. **C4** test-like history masking — −0.0003 in 3 h, no download; worth running because the
   mismatch is real even though the proxy sees nothing.
3. **C2** and **C3** — expected +0.0000. Recommend **not training them**: the read-only proxies
   already answer the question, and that verdict should be recorded in NOTES rather than paid for
   in 5 h of training.
4. **C5** GLDAS-2.1 CLSM (download, ~100 MB, Earthdata login) — −0.0020 in 4 h.
5. **C6** WaterGAP 2.2e (download, 0.3–1.5 GB) — −0.0025 in 5 h; measure it on top of C5.
6. **C7** GPCC Full Data (download, ~20 MB) — −0.0010 in 3 h. The smallest download: if the
   connection allows only one, take this one first.
7. **C8** ERA5 daily within-month timing (download, 2–5 GB) — −0.0009 in 6 h, last.

## Considered and not ranked

- **Prohibited, never to be taken:** SEAS5 on disk and GDO seasonal forecasts (R2: they encode t+1);
  GLDAS-2.2 CLSM GRACE-DA1, GLWS2.0, GRACE-trained TWS reconstructions, and GDO's TWS layer (R5).
- **ESA CCI Soil Moisture v09.1** (combined, 1978–2023, free and open, CDS operational stream every
  10 days): surface layer only, duplicating the released soil moisture, CPC and ERA5's four layers;
  the GPCC-based SPI result (−0.0002) is the nearest analogue.
- **GLEAM4 evaporation and root-zone moisture** (0.1°, 1980–2024, SFTP after registration, which the
  user would have to do): same rationale as C7, applied to E; to be ranked only if C7 pays.
- **GloFAS v4 river discharge** (0.05° daily, 1980-01 to 2022-07): LISFLOOD forced by ERA5, so it is
  ERA5 routed; tens of GB for a signal the isotropic regional means already approximate.
- **Climate indices and SST fields:** ONI as a feature failed (NOTES session 3) and analogue/ENSO
  weights are closed; nothing new to say.
- **Per-calendar-month deseasonalised covariates:** already the encoding — `features_anom.anom_table`
  z-scores against each cell's own calendar-month climatology.

## What could not be verified

- The WaterGAP 2.2e licence and file sizes: data.isimip.org and gude.uni-frankfurt.de were
  unreachable (bot protection, HTTP 502); the CC BY 4.0 statement comes from the GMD paper.
- All download sizes except GPCC's (~20 MB, from the DWD page) are estimates.
- GLDAS-2.1 CLSM's "through 2026-08-31" comes from the NASA Open Data Portal listing; the GES DISC
  landing page did not render for the fetch tool.
- Every proxy estimate is a linear, read-only stand-in for a trained measurement. The calibration
  table shows it tracks linear corrections at ~0.8 and overstated the one trained family on record
  (directional) by 2.5×.
