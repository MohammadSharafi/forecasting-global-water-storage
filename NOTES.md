# Experiment log (validation layout A unless stated; persistence = 0.757)

| id | change | RMSE | verdict |
|---|---|---|---|
| v1 | LGB, level target, in-sample climatology | 0.712 | train/val mismatch (early stop @37) |
| v2 | LOYO climatology, neighbours, slope; residual target | 0.669 | keep residual target |
| v3 | + per-cell covariate proxies, drop gap-target rows | 0.672 | no gain, dropped |
| B  | regularised params / huber | 0.669–0.671 | plateau -> features, not params |
| v4 | + AR lags 1/2/3/6/12, 24-mo trend/dev | **0.661** (layout B 0.593, −13.7%) | keep |
| v5 | + 5x5 neighbourhood means of dynamic features | **0.6555** (layout B **0.5829**, −1.7% vs v4 on both) | keep; wide SPEI-6/12 deltas rank 3rd/4th |
| smooth | residual smoothing r=1, w=0.7 | 0.658 / B 0.591 | keep as post-process |
| bias | per-horizon bias correction across layouts | worse both ways | reject (block noise) |
| ridge | ridge blend w=0.1–0.4 | 0.663–0.675 | reject |

Memory rule: one heavy process at a time (16 GB machine); features built once to disk.
| weights | per_row=4, test-horizon-mix sample weights | 0.6607 (unweighted 0.6615) | within noise; feature set plateaued |

# Leaderboard calibration (public, 30% of test)
| file | val A | public LB |
|---|---|---|
| sub_v2_residual | 0.669 | **0.735** |
| probe_persistence | 0.757 | pending — defines the test baseline |
| sub_v4_ar / smooth | 0.661 / 0.658 | pending |
| sub_v4_ar_6dp | 0.661 | **0.7188** |
| sub_v4_ar_smooth_6dp | 0.658 | **0.7168** |
| sub_v5_6dp / sub_v5_smooth_6dp | 0.6555 / ~0.652 | pending |
| probe_persistence | 0.757 | pending |
| v5+recent | + 60-mo recent anchors | A 0.6585 / B 0.5809 | mixed |
| v6 | recent anchors, long-term anchors removed | A 0.6571 / B **0.5760** | bias unchanged (+0.014/+0.057); blend with v5 |
| probe_persistence | 0.757 | **0.8864** — test baseline; v5-smooth is −19.4% vs it (val gain was −13.4%) |
| sub_v5_smooth_6dp | ~0.652 | **0.7142** |
| sub_v6_6dp / sub_v56_blend(_smooth) | v6 A 0.6571 B 0.5760; blend A 0.6552 | pending — v6 vs v5 test preds differ by 0.1125 RMSE |
| v7 ONI | + NOAA ONI (t, known, delta, 3-mo) | A 0.6598 / B 0.5774 (v6 ref 0.6569 / 0.5765) | **reject**: heavily used yet worse -> overfits few ENSO cycles |
| v6+w | recent-year sample weights (ramp 0.3->1) | A 0.6565 / B 0.5749 | small consistent gain; adopt |
| sub_final_fallback_6dp | v6w (5 seeds) 50/50 v5 (3 seeds), smoothed | pending (tomorrow's first upload) |
| h1 specialist | separate model for h=1 rows | A 0.6568 (shared 0.6560) / B 0.5727 (0.5747) | mixed; not adopted |
| deep variant | 255 leaves, lr 0.01, ff 0.5 | A 0.6559 / B 0.5760; blend no gain | not adopted |
Plateau of the feature-only stack: A ≈ 0.656, B ≈ 0.575. Remaining lever: ERA5 water-balance covariates (needs CDS key).
| v6w + NCEP | NCEP R1 monthly P, E(latent heat), runoff, SWE, soil water; at t, at t_known, accumulated P-E-R over hidden months | A **0.6533** (ref 0.6555) / B **0.5711** (ref 0.5750) | adopt; public, no-login, operational (days latency) |
Final for tomorrow: sub_final_ncep_6dp = 0.5*(v6w+NCEP, 5 seeds) + 0.5*(v5, 3 seeds), smoothed. Backup: sub_final_fallback_6dp.
| sub_final_ncep_6dp | built 4 Sep; NCEP test coverage 99.9%; differs from fallback by 0.033 RMSE | pending (tomorrow #1) |

# Session 2 (5 Sep) — cached matrices (out/mats), new features, model families, post-processing
Reference (previous final recipe, smoothed v6w+NCEP/v5 blend): A 0.6520 / B 0.5779.
| id | change | A | B | verdict |
|---|---|---|---|---|
| traj | within-block trajectory smoothing of the predicted change along the horizon (w=0.35) | −0.004 | −0.002 | adopt (post-process) |
| coherent rows | one horizon per (month, draw) for the whole globe (test-like neighbourhoods) | 0.6554 (v6n) | 0.5757 (v6n) | slightly worse than random draws alone, but enables grid features; 25% subsample loses nothing |
| all feats | + radius-4 grid-convolution neighbourhood means, NCEP R2, CPC leaky-bucket soil moisture, covariate window stats/lags, per-cell response slopes | **0.6499** | **0.5666** | adopt; w4_SPEI_06/12_d and w4_spei1_acc are top features |
| xgb | XGBoost hist, depth 9, eta 0.03, same features | 0.6494 | 0.5657 | adopt |
| cat | CatBoost depth 8, lr 0.03 | 0.6463 | 0.5763 (early stop 134, bias +0.12) | use with low weight |
| mlp | 3-layer MLP (256, SiLU, dropout 0.3, wd 1e-3), 1 epoch one-cycle, MPS, ~25 s | **0.6479** | **0.5513** | adopt; overfits after epoch 1 (cell memorisation) |
| blend | cat+mlp 50/50 | 0.6407 (+smooth+traj **0.6353**) | mlp-dominated: NNLS 0.92 mlp (0.5469) | final: mlp-heavy blend |
Memory rule confirmed the hard way: the FINAL matrix build died silently (OOM) while tree models ran beside it. Builder alone: ~11 min, 4.6 M rows.
| final v1 | 0.5 mlp(5 seeds) + 0.2 mlp-no-latlon(2) + 0.1 lgb(3) + 0.1 xgb(3) + 0.1 cat(2), smooth + traj | A **0.6363** (bias −0.010) | B **0.5497** (bias +0.041) | sub_s2_v1.csv — upload #1 |
| mlp sweep | lr 5e-4 / 2e-3, dropout 0.2 / 0.45, wd 1e-2, width 512, 2 epochs, no lat/lon | mixed, all within ±0.003 | | keep default; no-lat/lon used as diversity |
| mlp + long-term anchors | featset with clim_next etc. | 0.6552 | 0.5849 | reject (as for trees) |
| U-Net maps | 3-level U-Net on global 1° maps (45 channels from the cache), one sample per (t, t_known), longitude-roll augmentation | 0.6769 alone (best epoch 3, then overfits); blend w=0.1–0.3 no gain after post-processing | — | reject |
| mlp bagging | 4 MLPs on different 50% subsamples vs 3 full-data seeds | bag 0.6357 vs full 0.6367 (pp blend) | bag 0.5525 vs full 0.5498 | mixed; keep full-data seeds |
| final v1 alt | tree-heavier: 0.5 mlp / 0.5 trees | A 0.6361 | B 0.5523 | sub_s2_v1_treeheavy.csv — upload #2 to let the LB arbitrate the MLP weight |
Remaining lever: ERA5 / ERA5-Land water balance (needs the CDS token, see ERA5_SETUP.md); the loader `features_era5.py` and the cache builder are ready for it.

# Session 3 (5 Sep) — after LB feedback: sub_s2_v1 = 0.7191 public (v5-smooth 0.7142)
Diagnosis (see scripts in the session transcript; numbers on train/validation):
- The val→LB ratio is stable: persistence 0.757→0.886 (×1.17), our model 0.636→0.719 (×1.13). Gains transfer; the test period is simply noisier (late GRACE mission). Train months in 2015 already show h=1 persistence RMSE 1.0–1.2 vs 0.54 for 2002–2012, with month-to-month global offsets of ±0.3.
- Temporal variogram of TWS: 2σ² intercept 0.30 (σ≈0.39) + 0.068/month; the field is spatially very smooth (cell vs 8-neighbour mean RMSE 0.055), so the unpredictable part is regional, not white per-cell noise. A Kalman/EWMA level as prediction base is worse than the last observation on both layouts (the models already handle it).
- Leaderboard top: 0.560, 0.589, 0.623, then a cluster at 0.63. Prohibited information tested for scale only: interpolation between observed test blocks scores 0.634 on layout A (worse than our legit 0.636); next-month covariates give ~0.025; a model given the next observed block's TWS gains 0.006. None of these explains 0.56 → the top entries most likely use an external gap-free TWS product (prohibited by the rules; code review applies to the top 10).
- Gap-month targets (7 of 18 test months, 39% of rows) are not interpolations of neighbouring months in train (checked on 2014-12 and 2015-06); they sit a bit closer to the previous real month. sub_s2_v2_fillshrink.csv shrinks the predicted change on those rows by (h−0.5)/h as a one-submission LB experiment.
| sub_s2_v2_fillshrink | change shrunk on gap-month targets | public LB **0.7217** (v1 0.7191) | reject: gap-month targets are not smoother on the test |
Public LB so far: v5-smooth 0.7142 (best), final_ncep ?, s2_v1 0.7191, s2_v2_fillshrink 0.7217. All within ±0.004: the stack is at its information limit; next input = ERA5.

# Session 4 (5 Sep) — ERA5 monthly means (CDS token in place)
Fields: tp, e, ro, sd, swvl1-4, t2m at t and t_known, differences, and P−E−R accumulated over the hidden months (21 features, prefix e5).
Control on identical rows (no-ERA5) vs with ERA5:
| model | A | B |
|---|---|---|
| lgb | 0.6498 → 0.6483 | 0.5669 → 0.5634 |
| mlp (seed 0) | 0.6479 → 0.6453 | 0.5509 → 0.5472 |
| xgb | 0.6494 → 0.6458 | 0.5657 → 0.5660 |
| cat | 0.6463 → 0.6434 | 0.5763 → 0.5762 |
| blend + post-processing | 0.6363 → **0.6347** | 0.5497 → **0.5459** |
FINAL matrix rebuilt with PER_ROW=2 (3 draws exceeded memory with 181 columns). Loader note: CDS delivers a zip with two netcdf streams; joined on cell-month (features_era5.py).
| sub_s3_era5 | ERA5 ensemble | public LB **0.7194** (v1 0.7191) | no LB gain despite val gain |
Hypothesis after four LB results: every "recent-anchor" (v6-style) model scores ~0.719; the v5 stack with long-term climatology anchors scores 0.714. Validation years 2012–15 were drying (recent anchors win); the 2015–18 test likely reversed (long-term anchors win). LB experiments: sub_s4_blend_v5s_era5 (50/50 v5-smooth + ERA5 ensemble), and the v5x stack (all new features, LONGTERM in, RECENT out). v5x validation: lgb A 0.6481 / B 0.5708, mlp A 0.6518 / B 0.5846 (wet bias +0.02..+0.07, as expected on dry layouts).
| sub_s4_v5x | new stack, long-term anchors (lgb .3 xgb .3 cat .15 mlp .25), smooth+traj | LB pending | differs from ERA5 ensemble by 0.121, from v5-smooth by 0.135 |
| sub_s4_v5x_era5_blend | 0.5 v5x + 0.5 ERA5 ensemble | LB pending | |
| sub_s4_v5x | | public LB **0.7141** | long-term anchors confirmed (recent-anchor versions 0.719) |
| sub_s4_blend_v5s_era5 | | public LB **0.7107** (best) | blending anchor philosophies works on the test period |
Next uploads: sub_s5_blend3 (equal v5s + v5x + era5), sub_s4_v5x_era5_blend, sub_s5_blend_x4e3v3 (0.4/0.3/0.3). In parallel: models with BOTH anchor sets (featset allL) as a third family.
| allL stack (both anchor sets) | lgb/xgb A 0.6498/0.6499, B 0.5684/0.5711 | FINAL preds differ from v5x by only 0.030 (trees pick the long-term anchors) | little diversity; sub_s5_allL, sub_s5_blend4 (v5s+v5x+era5+allL) built |

# Session 5 — first-principles re-audit (all rule-compliant diagnostics)
| test | result | conclusion |
|---|---|---|
| E1 EOF/low-rank denoising of the last observed field (k=10..200) as base | persistence 0.757→0.81–0.93 (A), 0.687→0.81–0.93 (B); worse at every rank | the fast component is real and persists; not removable by spatial filtering |
| E2 upper bound: one-month TWS change vs ERA5 water balance of the TARGET month (perfect next-month weather) | R²≈0.006, corr −0.02; LGB holdout 0.6355→0.6346 | next month's weather does not explain the change → seasonal forecasts (C3S SEAS5) cannot help; not pursued |
| E3 per-cell mean over 2004–09 | 0.20±0.60 (not a 2004–09 baseline); full-train mean 0.12±0.39 | standardisation period unknown; inferring the test-period mean from it would be future information → not used |
| D3 disagreement shrinkage | corr(disagreement,|err|)=0.19–0.22 but shrinkage hurts | rejected |
| D3 cross-layout NNLS stacking | A→B 0.5553 (equal 0.5581), B→A 0.6443 (equal 0.6387) | unstable; keep near-equal weights |
| regime shrinkage toward slow levels (mean24/rmean60/cmean/clim/lag123), tested on noisy 2015 rows | best 0.8077→0.8069; others worse | models already shrink optimally; rejected |
Conclusion: the unpredictable component is weather-independent, spatially coherent and temporally white; no legal input or representation found that predicts it. Remaining lever = blend weights across anchor philosophies (LB-determined, ~0.005).

# Session 6 — forensic audit
- COMPLIANCE FIX: trajectory smoothing used the prediction of the NEXT horizon (same cell, same t_known), which embeds covariates at t+1 → information after t. Causal version (previous horizon only) hurts (A 0.6395→0.6446, B 0.5471→0.5525). Removed from final_assemble.py. All sub_s2…sub_s5 files contained it; compliant rebuilds: sub_c_era5, sub_c_v5x, sub_c_blend3, sub_c_blend_v5s_era5, sub_c_blend_x4e3v3 (sub_v5_smooth_6dp was already compliant). Expect ≈+0.004 on the LB vs the non-compliant versions.
- Starter notebook: an earlier gridded release had "future-looking _tp1 columns" and unmasked test TWS_t; masking was added "to prevent TWS_t at one test row from revealing the hidden target of another test row". Top LB scores (0.56 two months ago, 0.589 one month ago) plausibly come from that regime.
- Organizers' starter says "Lat/Lon should not be used as features!" → review-safe final should avoid lat/lon (MLP no-lat/lon variant exists; trees to retrain).
- Test observed months: mean −0.07..+0.09, std 0.87–0.99 → no scale/normalisation mismatch with train.
- External data: NCEP R1/R2 and CPC are NOAA products, not on the permitted list (Copernicus / satellite / pretrained / open tools). ERA5-only feature sets under validation; if within noise, drop the NOAA inputs for the final.
| review-safe sets (ERA5-only, no lat/lon) | lgb e5only_noll A/B, mlp, lgb v5x-noll: see logs | FINAL 24 models trained | sub_rs_recent, sub_rs_longterm, sub_rs_blend (50/50) — the Code-Review-safe configuration |

# Organizer rulings (discussion "Addressing Some Open Chats", 13 Aug; "Neighbouring cells" 19 Aug)
- NOT permitted as predictive features: raw lat/lon, cell IDs, coordinate-derived encodings, spatial embeddings. → v5-smooth, sub_c_* and all lat/lon-using stacks are OUT of the final. Only *_noll models qualify.
- Permitted: lat/lon as lookup key; per-cell statistics from the cell's own past TWS; neighbouring cells' TWS at ≤ t incl. spatial filtering/aggregation; external non-TWS covariates with every source date ≤ t (NOAA NCEP/CPC therefore permitted if documented); ERA5 final as retrospective proxy for ERA5T (document in report); recursive forecasting.
- "Any feature encoding month t+1 or later, from any source, remains grounds for disqualification" → trajectory smoothing removal was mandatory.
- Scoring targets match the current CSVs; benchmark 0.8999 = starter notebook. A compliant participant (uzbtrust) reports "just above 0.70" with provided data + ERA5, no coordinates.
- CodeCarbon instrumentation required for top-10 (already in place from session 1: report/carbon.json; re-run for the final models).
- 9 Jul (AJoel, "Competition Update"): NetCDF → CSV relaunch; test TWS masked; SPEI_tp1/SM_tp1 removed; "Latitude and longitude must not be used as features"; external "publicly available satellite or reanalysis data (e.g. from Copernicus)" welcome; any t+1 feature from any source = disqualification.
- 13 Jul ("Data Update"): served files had been mismatched with the scoring targets (v2 vs pre-fix); "submissions against the mismatched version will be re-evaluated" — participants report the LB was never updated. 10 Aug probe analysis (uzbtrust): all-zeros 1.0138 → E[y²]=1.028; persistence corr 0.61; starter recipe scored 0.659 on v2 vs 0.900 now. → The 0.56–0.63 scores are legacy/mismatch-era; the compliant frontier on current files is ~0.70.
- Zindi platform rule quoted by participants: external data should be declared on the discussion board (“sent to Zindi for confirmation”). Recommend a declaration post listing ERA5, NCEP R1/R2, CPC soil moisture.
| final compliant candidates | sub_f_recent / sub_f_longterm (no lat/lon, NOAA+ERA5), sub_f_blend (50/50), sub_f_blend4 (+ ERA5-only stacks), sub_rs_blend (ERA5-only) | validation: no-lat/lon costs nothing (lgb A 0.6483/0.6483, B 0.5629 vs 0.5634) | upload order: sub_f_blend, sub_rs_blend, then adjust |
