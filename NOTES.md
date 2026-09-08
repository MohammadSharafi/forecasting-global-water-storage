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

# Session 7 — smoothed anchor (the missed signal)
Organiser ruling 24 Aug explicitly permits neighbouring cells' TWS at months <= t incl. spatial filtering/aggregation.
Great-circle neighbourhood mean of the last-observed TWS field (longitude window scaled by 1/cos(lat)), radii 300/500/800 km:
| | layout A | layout B |
|---|---|---|
| raw anchor as persistence | 0.7569 | 0.6869 |
| 300 km smoothed anchor as persistence | 0.7239 | 0.6535 |
| 500 km | 0.7172 | 0.6520 |
| post-hoc re-base of the blend, 300 km, a=0.3 | 0.6385 -> 0.6346 | 0.5560 -> 0.5523 |
As features (sa300/500/800, deviations, sa_grad), retrained:
| lgb recent-anchor | 0.6483 -> **0.6452** | 0.5629 -> **0.5555** |
| lgb long-term | 0.6483 -> **0.6463** | 0.5692 -> **0.5630** |
| mlp recent-anchor | 0.6453 -> **0.6437** | 0.5538 -> **0.5452** |
Why it was missed: earlier neighbourhood features were boxes in grid cells (radius 1-4) applied mostly to dynamic covariates, and the global EOF denoising test (session 5) rejected low-rank filtering of the field. A physical-radius mean of the anchor itself is a different operator and is the one that works.
CodeCarbon (final config, one seed per family, both anchor sets): 0.0076 kg CO2e total; report_final/carbon.json.
| sub_f_blend | compliant (no lat/lon, no traj), NOAA+ERA5, 50/50 anchors | public LB **0.7157** | best COMPLIANT score; earlier 0.7107/0.7141 files used coordinates |
| sub_g_blend | + smoothed-anchor features | val A 0.6382 / B 0.5515 (single seed; ref 0.6385/0.5560) | next upload |
Bug found and fixed in final_assemble.py: the seed glob `pred_FINAL_{name}_s*.npy` also matched the CodeCarbon probe files `..._s0_carbon.npy`, mixing pre-anchor models into two blends. Now matched with an exact `_s<digits>.npy` regex; probe files deleted.
Note: run_models.py strips the `_sa` suffix before naming outputs, so anchor models overwrite same-featset predictions. All FINAL preds for allnoll / v5x_noll are now the smoothed-anchor versions (intended); sub_f_blend.csv is frozen on disk as the 0.7157 reference.
| sub_g_blend | smoothed-anchor blend | public LB **0.71159** | best compliant; beats the old non-compliant 0.7107-class files |
| sa2 (smoothed trends, 2 radii) | A lgb 0.6460 / mlp 0.6419; B lgb 0.5551 / mlp 0.5523 | mixed vs sa (A 0.6452/0.6437; B 0.5555/0.5452) | reject |
| sa3 (per-cell local-deviation persistence, LOYO betas; mean beta 0.20) | standalone predictor h=1: A 0.7078->0.6799, B 0.5693->0.5500. In models: lgb v5x better both layouts (0.6463->0.6454, 0.5630->0.5604); lgb recent mixed; **MLP much worse both layouts** (0.6437->0.6573, 0.5452->0.5640) | reject (net effect on the blend ~0.001, adds review surface) |
| anomaly-field smoothing (de-mean neighbours before averaging) | A 0.7400 vs raw-field 0.7239; B 0.6649 vs 0.6535 | reject — the raw field is the more coherent one |
| sub_g_recent | recent-anchor stack alone (compliant, anchors) | public LB **0.7191** | unchanged vs pre-anchor recent (0.7194): the test gain of sub_g_blend (0.7116) came from the long-term stack |
| re-base after anchor features | p + a*(sa300 - tws_known), a=0.3 | A −0.0026, B −0.0017 at every w_long | adopt (post-process; uses only t_known information) |
Candidates: sub_h_l5_rb (50/50 + rebase), sub_h_l7 (0.3/0.7), sub_h_l7_rb, sub_h_l85_rb, sub_h_l10_rb. Upload order: sub_g_longterm (end point), sub_h_l7_rb, then interpolate.
| sub_g_longterm | long-term stack alone (anchors, compliant) | public LB **0.7110** (best) | recent 0.7191, 50/50 0.7116, 0.3/0.7+rebase.3 0.7113 → weight curve flat from 0.5 to 1.0; recent stack adds nothing on the test |
Noise regime check (information at <= t only): RMS of the cell's deviation from its 300 km neighbourhood mean, train 2002-14 vs the six fully observed test months — see session-8 output. Next probes: sub_h_l10_rb (long-term + rebase 0.3), sub_h_l10_rb5 (0.5), sub_h_l10_rb3_500 (500 km, 0.3).
| allL_noll_sa (both anchor sets, no lat/lon, smoothed anchors) | A lgb 0.6461 / mlp 0.6536; B lgb 0.5612 / mlp 0.5859 | FINAL 12 models: sub_i_both; sub_i_lt_both = 50/50 with the long-term stack | LB probe |
| family split of the long-term stack | sub_i_lt_trees (lgb .4 xgb .4 cat .2), sub_i_lt_mlp, sub_i_lt_lgbxgb | | LB probes: is the MLP what fails in the test regime? |
| uniform training weights (no recent-year ramp) for the long-term stack | pipeline11 running | | LB probe (test regime punishes trend extrapolation) |
| uniform weights, long-term lgb | A 0.6470 (ramp 0.6463), B 0.5664 (ramp 0.5630) | validation prefers the ramp, as expected for the drying layouts | sub_j_lt_uniform built (12 models) — LB decides |

# Session 9 — leaderboard statistics, post-processing shape, anchor target (no data in this environment; nothing here is measured)
- LB significance has a closed form: SE(gap) ~= RMS(pB-pA)/sqrt(N_public), N_public ~= 84,288 (`lb_se.py`).
  Recent-vs-long-term (0.0081 gap, RMS diff ~0.11) = ~21 SE: certain. sub_h_l7_rb vs sub_g_longterm
  (0.00037, RMS diff ~0.04) = ~2.7 SE: marginal. Check every future sub-0.001 difference with it.
- The anchor-philosophy axis is the ONLY effect above ±0.001 found on this test set. Push along it
  rather than hunting features.
- `final_assemble.py`'s final filter is a grid box on the residual: latitude-dependent width, and it
  leaves all of tws_known's fine-scale noise in the prediction. `postproc2.py` scans the physically
  correct operator instead -- p -> M_r(p) + beta*(p - M_r(p)) on the target-month prediction field,
  great-circle radius, closed-form beta -- on both layouts. Subsumes the grid smooth and the re-base
  (whose a=0.3 at 300 km was never scanned, though session 7's own numbers favour 500 km and a ~0.8).
  Smoke-tested on a synthetic smooth-field+white-noise case: closed form matched the grid minimum at
  every radius, 0.4252 -> 0.3198 against a 0.3002 oracle. Applied by `final_assemble2.py`.
- `run_models.py` gained ANCHOR_TARGET=tws|clim|decay: what the residual target is measured against.
  `decay` (lam = ANCHOR_RHO**horizon) keeps h=1 near persistence and pulls long horizons toward the
  long-term normal -- the direction the LB has twice rewarded, expressed in the target rather than in
  the feature set. clim_next is LOYO for training rows, so no leakage; no matrix rebuild needed.
- Free and risk-free: more seeds on the long-term stack (MLP is 1 epoch, highest variance, weight 0.25).
- Known bug, left alone mid-competition: the MLP branch picks its best epoch using yv, so any layout
  A/B MLP number from an EPOCHS>1 run is optimistically biased (pipelines mostly use EPOCHS=1, where
  no selection happens). Do not quote such a figure in the report.
- Competition structure confirmed by search (zindi.world itself is egress-blocked here): LB RMSE is
  50% of the final score, trustworthiness 30%, innovation 20%; 5 submissions/day, 200 overall.
  Close date NOT verified.
- Full write-up and run order: SESSION9.md.
- Deadline confirmed (organiser email, 7 Sep): closes 13 Sep 21:59; code review now covers the top
  TWENTY (was top 10); results by 4 Oct. Trustworthiness rubric (Trustworthiness_Evaluation.pdf) has
  four <=100-word sections -- bias, transparency (LIME/SHAP + figures), reusability, sustainability
  (CodeCarbon) -- all four already covered by REPORT.md section 5. Day-by-day schedule in SESSION9.md.
- pipeline*.sh had the author's absolute macOS path hardcoded, so none of them would run for a
  reviewer. Now `cd "$(cd "$(dirname "$0")" && pwd)"` with `PY=${PY:-./.venv/bin/python}`.

# Session 9b (7 Sep) — LB results: family ladder, and both post-processing directions refuted
| file | public LB | note |
|---|---|---|
| sub_i_lt_lgbxgb (lgb .5 xgb .5) | **0.709259** | new best compliant |
| sub_i_lt_trees (lgb .4 xgb .4 cat .2) | 0.710382 | CatBoost costs ~0.0011 |
| sub_g_longterm (lgb .3 xgb .3 cat .15 mlp .25) | 0.710959 | previous best |
| sub_h_l10_rb3_500 (500 km re-base a=0.3) | 0.717866 | +0.0069: MORE smoothing is worse |
| sub_i_lt_mlp (MLP alone) | 0.723121 | +0.0121 |
| sub_h_l10_clim1 (climatology pull) | 0.731800 | +0.0208: climatology pull is badly wrong |
- Family ladder is monotone and runs OPPOSITE to validation: removing capacity always helps on the
  test. lgb+xgb 0.70926 < +cat 0.71038 < +cat+mlp 0.71096 << mlp alone 0.72312. On validation the
  MLP was the best single family on BOTH layouts (A 0.6437 / B 0.5452 vs lgb 0.6452 / 0.5555) and cat
  was best on A (0.6463). Adopt lgb+xgb only; the MLP memorises cell identity, which pays on layouts
  carved from the training years and fails across the 2015-18 regime shift.
- Session 9's two recommendations are both REFUTED by these probes: more spatial shrinkage (§3.1) and
  a climatology-anchored target (§3.2). The reasoning "recent anchors lose to long-term, so push
  further that way" conflated long-term ANCHORS with more SHRINKAGE. Consistent with session 3's
  measurement that the test error is regional and spatially coherent (+-0.3 global month-to-month
  offsets in 2015), not white per-cell noise: smoothing cannot touch large-scale error but does
  destroy real cell-scale signal.
- Direction now: less capacity, less post-processing, more regularisation.
- final_assemble.py gained SMOOTH_W (default 0.7, unchanged behaviour). The w=0.7 grid smooth is in
  EVERY submission ever made and has never been LB-ablated -- top probe now that two smoothing-like
  post-processes have just cost 0.007 and 0.021.
- sub_i_both vs sub_i_lt_both differ by RMS 0.0166 (SE 6e-5): indistinguishable, do not spend two
  slots on them. sub_j_lt_uniform differs from sub_i_lt_both by RMS 0.0390.

# Session 9c (7 Sep) — the external covariates were never converted to anomalies
Prompted by: compliant entries reported at 0.69-0.70, i.e. ~0.019 below our 0.7093. That is far
more than blend tuning can explain, so the gap is structural.
- EVERY ERA5 and NCEP feature is a RAW PHYSICAL LEVEL in mm (e5PER_acc, P_acc, e5SW_t, SWE_t...),
  while lat/lon are excluded from every final feature set. The model is shown "accumulated
  P-E-R = 350 mm" with no way to know that is a drought in the Amazon and a record flood in the
  Sahel. It has per-cell TWS statistics (cmean, csd, clim_next, ac1, mad1) to locate a cell in TWS
  space, and NOTHING to locate it in covariate space.
- The target is a standardised anomaly (RMS 1.014), and the change in an anomaly is driven by the
  ANOMALY of the water balance, not its raw total: a cell receiving its normal seasonal 350 mm
  should have zero expected change. add_era5 carries the comment "anomaly of accumulated balance vs
  the cell's climatological balance for those months" directly above its `return` -- identified,
  never implemented. This is the most likely explanation for ERA5 buying only 0.0016 on validation
  and nothing on the LB despite P-E-R being definitionally the change in stored water.
- features_anom.py: per-cell, per-calendar-month climatology from HISTORY MONTHS ONLY, then
  storage vars -> z-scored _t/_k/_d; flux vars -> z-scored _t plus _acc = accumulated anomaly over
  (t_known, t]. ~34 new features, all prefixed an_ so they can be ablated.
  Robustness: cell SD floored at 5% of the variable's global anomaly SD and z clipped to +-10 --
  without that, near-constant cells (subtropical snow, desert runoff) produced z-scores of 1e8.
  Verified on synthetic data: per-cell mean 0 / sd 1 over the history period, degenerate cells give
  exact zeros, accumulated anomaly matches hand computation to 1e-6.
- Featsets v5x_noll_noanom / allnoll_noanom added for the ablation.
- REQUIRES a matrix rebuild (build_mats.py A|B|FINAL) and then add_anchor_feats.py A|B|FINAL, since
  the anchor parquets are row-aligned to the matrices. Memory: ~34 extra float32 columns is ~630 MB
  on the 4.6M-row FINAL matrix -- keep PER_ROW=2 and run the build with nothing else alongside.
- pipeline12.sh: the whole session-9c chain unattended -- preflight, backup of the pre-anomaly
  predictions and CSVs, rebuild A/B/FINAL + anchor feats, lgb ablation with/without the anomaly
  features on BOTH layouts with a printed ADOPT/DO-NOT-ADOPT verdict, then FINAL lgb+xgb at 5 seeds
  each (cat and mlp dropped per the family ladder) and two submission files (with and without the
  grid smooth). lgb rounds are taken from layout A's early stopping rather than the stale 560.

# Session 9d — full R&D audit: two structural findings, one measurement fix
Audited every feature module, the training-row sampler, both validation layouts and the block
structure against the real test.

1. VALIDATION MEASUREMENT BIAS (free to fix, affects every past decision).
   Layout A blocks are [3,2,3,7,3] -> horizon mix h1 .278 h2 .278 h3 .222 h4 .056 h5-7 .056.
   The real test blocks are [1,3,4,7,1,2] -> h1 .333 h2 .222 h3 .167 h4 .111 h5-7 .056.
   Layout A therefore OVER-weights h2 and h3 by 5.6 points each and UNDER-weights h1 and h4 by the
   same, while layout B ([1,3,4,3,7,2]) is much closer to the test. Since RMSE rises steeply with
   horizon, every method that trades h1 accuracy for mid-horizon accuracy has been scored more
   kindly on validation than the LB scores it -- a plausible contributor to the MLP / re-base /
   climatology-pull transfer failures, all of which should hurt most at h1 (a third of the test).
   The aggregate level barely moves (0.6312 vs 0.6294 for the same model); it is the differential
   weighting that distorts the RANKING. eval_mix.py reports plain RMSE, testmix-reweighted RMSE and
   a per-horizon breakdown for any saved prediction set, so past experiments can be re-scored free.
2. NO SPATIAL SCALE ABOVE ~4 DEGREES ANYWHERE IN THE FEATURE SET.
   The widest aggregation is add_wide4's radius-4 grid convolution (~440 km at the equator, less
   toward the poles) plus the 800 km smoothed anchor. Session 3 measured that this period's
   unpredictable component is regional and spatially coherent (+-0.3 global month-to-month offsets
   in 2015) -- the largest identified error term -- and the model has no way to see it.
   features_scale.py adds a zonal decomposition of the anchor field: band mean zm, the cell's
   departure dz, band momentum zd3/zd12, and the cell's change with the band's removed dzd3/dzd12.
   Deliberately NO global index: session 3's ONI result showed a single per-month global series is
   heavily used and makes both layouts worse (few ENSO cycles, and it doubles as a month id).
   A band-month value has 36 bands x ~160 months behind it and the deviations are cell-relative.
   Verified on synthetic data: zm matches direct computation, zd12 recovers the true per-band drift.
3. CAPACITY LADDER, untested downward. The LB ladder (lgb+xgb 0.70926 < +cat 0.71038 < +cat+mlp
   0.71096 << mlp alone 0.72312) says capacity is what fails across the regime shift, yet nobody has
   tried a SMALLER lgb. Added lgbm (63 leaves, min_data 1000, l2 10) and lgbs (31, 2000, l2 20).
4. run_models.py gained DROPF=anom,scale to ablate a feature group without a new featset name.
   NOT called DROP -- that env var is already the MLP dropout rate, and the collision would have
   crashed every MLP run.
- Training-row horizon mix is h1 34% then uniform 11% on h2..h7, against the test's 22/17/11/5.6/
  5.6/5.6. weight_val.py tested test-mix training weights in the v4 era and found it within noise;
  worth revisiting only if the horizon breakdown shows a systematic long-horizon deficit.
- pipeline13.sh runs the 7-experiment grid on both layouts and reports under the test mix;
  pipeline14.sh builds the submission from the chosen configuration.

# Session 9c RESULT (7 Sep, measured on the real data) — covariate anomalies ADOPTED
Matrices rebuilt with features_anom: 205 features, 34 of them anomalies (215 parquet columns).
Row counts PER_ROW=2: A tr 2,971,466 / va 281,003; B tr 2,193,850 / va 312,102; FINAL tr 3,470,979.
| layout | lgb without anomalies | with | delta |
|---|---|---|---|
| A | 0.6429 (best_iter 223, bias +0.0117) | **0.6307** (best_iter 255, bias +0.0118) | **-0.0122** |
| B | 0.5615 (best_iter 178, bias +0.0636) | **0.5442** (best_iter 210, bias +0.0563) | **-0.0173** |
VERDICT: ADOPT — wins on both layouts. This is the largest validation gain in the project's
history: ERA5 as raw levels was worth 0.0016 and the smoothed anchor 0.0031, so this is 4-5x
either, from the same lgb at similar depth (223 -> 255 rounds). Information, not capacity.
Note the noanom reference (A 0.6429) differs from session 7's 0.6463 because the matrices were
rebuilt with PER_ROW=2 for A and B; both ablation arms come from the same rebuild, so the
comparison is paired and clean.
- FINAL: lgb+xgb, 5 seeds each, 255/400 rounds -> out/sub_n_anom_lgbxgb.csv and
  sub_n_anom_lgbxgb_nosm.csv (SMOOTH_W=0). Mean predicted change -0.0037 (lgb) / -0.0035 (xgb).
- lb_se vs the 0.709259 file: RMS(difference) 0.1100, max |diff| 0.6517, SE 0.00038, so gaps
  below 0.00095 are noise. RMS 0.11 is the same magnitude as the recent-vs-long-term anchor
  switch (which was worth 0.008 on the LB): the anomalies moved the predictions substantially,
  so the LB verdict will be unambiguous in either direction.
- Prediction recorded BEFORE upload, for calibration: sub_n_anom_lgbxgb 0.704 (80% interval
  0.698-0.710); sub_n_anom_lgbxgb_nosm ~0.001 worse, near a coin flip. Reasoning: the val->LB
  transfer record here is erratic (smoothed anchor 1.3x, ERA5-raw ~0x, session-2 expansion
  negative), so ~0.35x applied to -0.014. The ERA5-raw row is the same experiment done wrong,
  which is the reason to expect better transfer this time.
- If it lands >= 0.709, that is the important negative result: a -0.014 gain on BOTH layouts
  buying nothing would mean the validation->LB link is broken, and the remaining days are better
  spent on the report (50% of the final score) than on more validation-driven modelling.
- run_all.sh + select_config.py: one unattended command for the overnight run. Order is
  (1) eval_mix on the 9c ablation BEFORE anything is rebuilt, since pipeline13 overwrites the
  matrices those predictions are aligned to, (2) the pipeline13 grid, (3) select_config.py picks
  features / capacity / weights independently, each requiring a win on BOTH layouts under the
  test horizon mix and keeping the incumbent otherwise, (4) pipeline14 builds the submission.
  Independent selection ignores interactions -- it is a heuristic and the printed table is the
  evidence for overriding it. Selector verified on two synthetic fixtures: it recovers planted
  winners, and it correctly refuses a candidate that wins on layout A but loses on B.
- Literature pass (session 9d). The Bayesian-network feature-selection study for global TWSA
  reconstruction (ESSD) ranks CLIMATE/LSM-SIMULATED TWSA as the top predictor, selected for 71%
  of grid cells, ahead of ONI at 56%. Our pipeline had every component of that quantity (ERA5
  and NCEP soil water, snow water equivalent) but NEVER THEIR SUM -- and a tree can split on each
  but cannot add them. features_anom.add_mtws now forms it: e5MTWS = e5SW*1000 + e5SWE and
  MTWS = SW + SWE (units checked: ERA5's soil column is metres of water, NCEP's is mm), z-scored
  per cell like any other storage variable. an_e5MTWSz_d -- the standardised change in modelled
  storage over the unobserved window -- is the closest legal analogue of the GRACE change itself.
  Bundled into the an_ prefix, so DROPF=anom ablates it with the rest.
- Also from the literature, NOT adopted: SST fields / teleconnections help at longer leads but
  fail for the same reason ONI already failed here (few independent cycles, doubles as a month
  id). "Annual memory in the terrestrial water cycle" (HESS 2025) finds memory absent in
  precipitation but strong in water stores, and warns that short-lag autocorrelation is a poor
  predictability indicator under strong interannual variability -- already covered by lag12,
  tws_ly, mean24 and rmean60.

# Session 9e — second R&D batch (built while the 9d run was in flight; NOT in that run)
1. MODELLED TOTAL WATER STORAGE (features_anom.add_mtws). See the 9d literature note: LSM-simulated
   TWSA is the top-ranked predictor in the reconstruction literature, and every component was
   present while the SUM never was. e5MTWS = e5SW*1000 + e5SWE, MTWS = SW + SWE.
2. CONTINENTAL ANCHORS at 1500 and 2500 km (add_anchor_feats, ANCHOR_RADII env; plus sa_grad2 =
   sa800 - sa2500). Nothing aggregated above 800 km, and session 3 says the dominant error is
   regional and coherent. A great-circle disc is also more hydrologically coherent than the zonal
   band added in 9d: the 45N ring mixes Oregon, Iowa, France, Kazakhstan and Mongolia. Both are in
   the grid so the data decides.
   BUG FIXED IN anchor.py: cos(lat) is floored at 0.1, so at 2500 km the polar longitude window
   asked for 451 of 360 cells. dlon is now capped at 179. Verified: a latitude-only field is
   recovered at every radius and white-noise sd falls monotonically 0.062 -> 0.012 with radius.
3. TRAINING HORIZON MIX (HMIX=test). The sampler draws h1 at 34% then h2..h7 uniformly at ~11%,
   against the test's 22/17/11/5.6/5.6/5.6 -- h5-h7 oversampled 2x, h2-h3 undersampled 2x.
   HMIX=test reweights to the test mix exactly (verified: effective shares match to 0.003).
   weight_val.py tested this in the v4 era and called it noise, but that was 30 features, layout A
   only, and scored with the PLAIN RMSE that itself over-weights h2/h3.
4. ROBUSTNESS: run_models now reads the anchor feature names from the parquet schema instead of a
   hardcoded list, so adding a radius needs no second edit. DROPF gained "bigsa". pipeline14 checks
   the FINAL matrix carries every feature in feats.json before training, so a stale matrix fails in
   one second instead of deep into a seed.
5. The grid is now 9 experiments (e1 base, e2 anomalies, e3 zonal, e4 both, e8 +big anchors,
   e5/e6 capacity, e7 uniform weights, e9 test horizon mix) and select_config makes four
   independent decisions, each requiring a win on BOTH layouts. Verified on a synthetic fixture
   with all four planted: it recovers each one and refuses the deliberately-worse uniform variant.

# Session 9f — STRATEGY CORRECTION: the leaderboard is a GATE, not a weighted component
Entrant states that after the private scores are published, only the TOP 10 go to code review.
(The 7 Sep organiser email said "top twenty"; plan for 10, which is the stricter requirement.)
This changes the objective and invalidates earlier advice in this file:
- Previously framed as "LB RMSE is 50% of the final score, so a 0.002 gain is worth less than a
  strong report". That is WRONG under a gate. Outside the top 10 the report is worth nothing at
  all, because it is never evaluated. RMSE is both the qualifying filter AND 50% of the score
  awarded afterwards, so it dominates completely until qualification is secure.
- The objective is therefore no longer "maximise expected RMSE" but "maximise P(top 10)". Those
  differ: when behind, the correct choice is the higher-variance candidate even at a worse mean;
  when comfortably inside, the correct choice is the low-variance one.
- Public-to-private sampling noise for the SAME file is about +-0.004 (SD ~0.002). If the top ten
  are packed inside that, rank is substantially luck and the only defence is a clear margin.
- REQUIRED INPUT that only the entrant can supply (zindi.world is egress-blocked from this
  session): current public rank, and the scores at ranks 1-15. Without those the risk posture
  cannot be chosen rationally.
- Open question with large leverage: session 3/6 concluded the 0.56-0.63 public scores are
  legacy artifacts of the mismatched data release. If they collapse when re-scored against the
  current private targets, several ranks come free; if they are real, the top is unreachable.

# Session 9g — LEADERBOARD SNAPSHOT (7 Sep, from the entrant): rank 82 of the public board
Top: 1 MOHAR 0.5596 | 2 Shankar 0.5893 | 3 lode4 0.6231 | 4 GIrum 0.6234 | 5 Jisoo 0.6317 |
6 OverfitStorage 0.6319 | 7 Emo HedgeHog 0.6474 | 8 Ahsan_496 0.6518 | 9 awxlong 0.6534 |
10 Ramjas 0.6559 | 11 MosCraciunXXX 0.6592 | 12 H2-Oh 0.6611
Around us: 73 fishnchips 0.70505 | 74 AbolfazlB 0.70514 | 75 Verlon 0.70605 | 76 medamin 0.70634 |
77 Milan 0.70644 | 78 IvanTeselskyi 0.70711 | 79 Aqua Vectors 0.70774 | 80 kdylkykdylky 0.70782 |
81 Kingstone 0.70896 | 82 mrSharafi 0.70926
- Reaching the DISPLAYED top 10 needs -0.0533, from 0.7093 to 0.6559. The compliant frontier is
  ~0.70 (confirmed by the dense cluster at ranks 73-82), so that is not achievable legitimately.
- LOCAL RANK DENSITY is the real lever: ranks 73-82 span 0.0042 over 9 ranks = 0.00047 RMSE per
  rank. A 0.005 gain is worth roughly 10 ranks, a 0.015 gain roughly 30 (thinning higher up).
  This is the tightest part of the board and we sit at the bottom of it.
- STALE-SCORE HYPOTHESIS (entrant's, and supported by our own session-6 forensics): the sub-0.66
  scores are pre-fix/v2-era. Anchor: the starter recipe scored 0.659 on v2 and 0.900 on the
  current data, a shift of about +0.24 for the same method. Applying it, rank 1 becomes ~0.80,
  rank 5 ~0.87, rank 10 ~0.90 -- all well behind us. If Zindi computes the private board fresh
  against current targets (which is the normal mechanism, since a stored public score is not
  recomputed when the target file changes), those entries collapse and the ~10-person cluster at
  0.705-0.709 is the real field.
- UNVERIFIED and outside our control: we cannot see submission dates, so we cannot confirm which
  entries are stale. Strategy does not depend on it -- maximising the score maximises rank under
  both scenarios -- but the EXPECTATION does: "top 10 on the current public board" is out of
  reach, while "top of the current-data cluster" is live and needs only 0.005-0.015.

# Session 9h — target set to 0.69 (-0.0193 from 0.709259). Extra independent gain source added.
- ANTECEDENT WINDOWS (features_anom.add_anom_windows). The _acc features accumulate the covariate
  anomaly over (t_known, t], whose LENGTH IS THE HORIZON -- so they describe the gap, not the
  cell's condition. Drought is a memory process. SPEI-3/6/12 give exactly the fixed-window view
  for METEOROLOGICAL drought and rank among the strongest features we have; nothing gave it for
  the actual water balance or for modelled storage. Now: flux anomalies summed and storage
  anomalies averaged over the 3, 6 and 12 months ending at t, independent of t_known.
  ~48 new features. Verified on a synthetic cell with a planted 6-month dry spell: w3 recovers
  exactly 3 dry months, w6 and w12 exactly 6, while _acc sees only the 2-month gap.
- BUG FIXED in anom_table: pl.max_horizontal with a numpy scalar floor made polars treat the floor
  as a length-1 Series that then refused to broadcast (it happened to survive on the real data
  because many cells give the sd column full length, but it failed immediately on a small case).
  Now float() + clip(lower_bound=...). Verified byte-identical output on a multi-cell fixture, so
  nothing already computed changes.
- MEMORY WARNING: the matrix is heading for ~260 features. On FINAL (3.47M rows) that is roughly
  3.6 GB for the training frame before LightGBM's binned copy. Keep PER_ROW=2; drop to 1 on OOM.

# Session 9i — E2 WAS AN ARTIFACT. The covariate avenue was closed on a broken measurement.
Session 5's E2 reported corr = -0.02 and R^2 ~ 0.006 between the one-month TWS change and the
ERA5 water balance, and concluded "next month's weather does not explain the change -> seasonal
forecasts cannot help". That closed covariate work for four sessions.
The number is physically impossible: P - E - R IS the change in stored water, so the correlation
must be strongly positive. ceiling.py reproduces the failure on synthetic data where the answer
is known -- dy driven perfectly by the water-balance ANOMALY (true r = 0.844), with a realistic
Amazon-vs-Sahara spread of raw scale between cells:
    raw  e5PER_acc      global r = 0.038   <- reproduces E2's ~zero
    anom an_e5PERz_acc  global r = 0.845   <- recovers the truth
A perfect relationship reads as zero when measured on RAW totals pooled across cells, because the
seasonal cycle and the between-cell scale spread dominate. This is the same encoding bug as 9c,
and it explains why the anomaly features then produced the largest validation gain in the
project's history (-0.0122 / -0.0173).
CONSEQUENCE: session 5's conclusion that "the unpredictable component is weather-independent and
no legal input predicts it" rests on a broken measurement and should not be treated as settled.
More covariate information (ERA5-Land, more variables, longer windows) is worth more investment
than sessions 5-8 assumed.
ceiling.py measures the real ceiling on the actual data three ways -- pooled, within-cell
(demeaned per cell, which removes scale entirely) and per horizon -- plus a ridge on the whole
an_ block fitted on tr and scored on va as an honest LINEAR bound the trees should exceed.
Run it after the current rebuild: python ceiling.py A ; python ceiling.py B

# Session 9j — the largest measured error component has never been targeted: the GLOBAL SHIFT
Session 3 measured "month-to-month global offsets of +-0.3" and concluded the unpredictable part
is regional and spatially coherent. That is the biggest error component ever identified here, and
NOTHING has ever targeted it. Every model predicts cells independently; whatever global shift the
predictions contain is an accident of averaging, never a modelled quantity.
Arithmetic: total MSE ~0.503 (RMSE 0.709). A per-month global offset with sd 0.3 is 0.09 of that,
about 18% of the whole error. Capturing a third of it is worth ~0.014 RMSE -- the whole gap to
the 0.69 target.
globalshift.py measures it: per target month the true vs predicted mean change, the regression
slope (does the model SHRINK the shift?), the share of MSE that is pure global offset, an ORACLE
bound for perfect correction, the same per 30-degree band, and -- crucially -- whether the offset
is temporally correlated, since a correction is only usable if it can be predicted from
information at <= t.
Verified on synthetic data with a planted shift the model cannot see: it reports the captured
ratio as 0.02, the MSE share as 68%, the oracle as 0.3188 -> 0.1796, AND correctly refuses the
persistence correction because the planted offset was temporally white. That last behaviour is
the point -- the script is built to say "closed" as readily as "open".
Two follow-ups, to build ONLY if the oracle is large AND the persistence correlation is non-zero:
 1. hindcast bias correction. At each block's t_known T the truth is observed, and so is some
    earlier month T'. Predict T from T', compare to truth, smooth the error field spatially, and
    carry it forward. Uses only observations <= t_known, so it is compliant. This is the legal
    way to estimate the CURRENT regime's regional bias.
 2. a dedicated small model for the global/zonal index itself, predicted from its own history and
    global covariate anomalies. Higher overfit risk (one series, ~160 months) -- the same trap
    that sank ONI -- so only worth it if the oracle is large.
Run after the rebuild: python globalshift.py A lgb_v5x_noll ; python globalshift.py B lgb_v5x_noll

# Session 9k — run_night.sh: the resumable overnight orchestrator
- FULLY RESUMABLE. Every step writes a marker in out/night/ on success; a rerun skips what is
  done, down to individual grid runs and individual training seeds. Interrupt it at any point and
  rerun the same command. FORCE=1 redoes everything.
- A failing step never aborts the run. Verified on a fixture: a failing subprocess is retried on
  the next run and is NOT marked done, and later phases still execute.
- eval runs in a SUBSHELL. Without that, any step whose command called exit would kill the
  orchestrator, and a cd would move it. Verified: a step calling `exit 9` no longer stops the run.
- Phases: (1) diagnostics on the CURRENT matrices, deliberately before the rebuild -- eval_mix,
  ceiling, globalshift; (2) rebuild A/B with every new feature; (3) the 9-experiment grid, each
  run its own step; (3b) scored under the test horizon mix; (4) select_config + post-choice
  diagnostics; (5) xgb on validation and blend_scan -- the lgb/xgb weight has been hardcoded at
  50/50 forever and never fitted; (6) FINAL matrix + the feature-completeness guard; (7) FINAL
  config 1 = the grid winner; (8) config 2 = a deliberately different capacity, so the two private
  slots hold genuinely different files; (8b, DEEP) config 3 = the pre-session-9 feature set as an
  insurance file; (9) assemble + a single report.
- blend_scan.py: scans the lgb/xgb weight under the test mix on both layouts, clips to
  [0.25,0.75] so the blend never collapses to one family, refuses to move off 0.50 when the gain
  is below what validation can resolve, and warns when the two layouts disagree by more than 0.3.
- Honest runtime: ~4 h default (SEEDS=8), ~8 h with DEEP=1 (16 seeds, 3 configs). There is not
  50 h of USEFUL work here -- seeds are 1/sqrt(n) so going past ~16 buys ~0.0003. The value of
  the resumability is being able to run it in chunks, not to fill wall-clock time.

# Session 9m — four new post-processing ideas, a third validation layout, and the orchestrator
## that runs them

Everything below is fitted OUTSIDE the model, on predictions that already exist, so nothing here
costs a training run. Each idea has its own gate, and each gate is leave-one-layout-out
(`xfit.py`): the configuration is chosen on the other layouts and scored on the held-out one, and
nothing is adopted unless every held-out gain clears 0.0003. Under 0.0003 a validation difference
is inside seed noise, and at ~84k public rows a leaderboard difference below ~0.00014 is
unreadable anyway (lb_se.py).

## 1. The smoothing weight was never tuned  (smooth_scan.py, smooth.nb_mean, smooth.smooth_field)
`final_assemble.py` has blurred the residual field over the 8 grid neighbours at w=0.7 since
session 4. That number came from one coarse table, on layout A alone, under the PLAIN horizon
average, on a model that no longer exists. Three things about it had never been checked:
 - the weight is the same at every horizon, and should not be. At h=1 the residual is mostly the
   cell's own state, which the neighbours do not know; at h=7 the predictable part is almost
   entirely regional, which is exactly what the neighbours estimate. `horizon_w(h, w1, w7)` ramps
   it linearly and the scan fits both ends.
 - radius and iteration count were fixed at 1.
 - the two columns either side of the dateline smoothed against half a neighbourhood, because
   lon+1 at +179.5 does not exist. `wrap=True` closes the grid.
One blending pass is LINEAR in the neighbour mean (res' = res + w*(nb - res)), so the whole
(w1, w7) grid costs one neighbour pass per (radius, iters, wrap) rather than one pass each. That
is what makes the scan cheap. Verified: `smooth_field` reproduces `smooth` bit for bit at scalar
w for every (radius, iters) tested, so adopting it changes nothing until the scan says to.

## 2. The magnitude of the predicted change was never calibrated  (postcal.py)
Every model here predicts target - tws_known and the residual is added back unscaled. Under a
regime shift -- and this is one, 2002-2015 training against a 2015-2018 test that starts in the
GRACE/GRACE-FO gap -- a squared-error learner systematically mis-scales its predicted change on
the next era. One scalar per horizon fixes that without touching the model:
    p' = tws_known + a_h * (p - tws_known)
a_h < 1 shrinks toward the last observation, a_h > 1 amplifies. Fitted by least squares through
the origin on ~50k rows per horizon, then pulled halfway back to 1 and clipped to [0.80, 1.20],
because the test era is a third regime and no layout is it. Fitted AFTER smoothing, since that is
the order final_assemble applies them. Sanity-checked on a synthetic fixture whose predictions
were deliberately over-scaled by 1/0.85: the fit recovers ~0.84 and the held-out gain is large;
on the same data after smoothing removes the noise it correctly returns ~1.03 instead.

## 3. h=1 is a different problem and had no specialist  (run_models.py HFILT, hsplice.py)
h=1 is 33.3% of the test -- the largest single slice -- and is not the same problem as h=7. At
h=1 the target is one month from an observed value and the answer is mostly the cell's own state
plus a month of weather; at h=7 the observation is half a year stale and what survives is
regional persistence and climatology. One ensemble has to spend its splits on both, and the
compromise falls hardest on h=1. `HFILT=1` trains on the h=1 rows only (and early-stops on the
h=1 slice of validation, not the pooled one -- otherwise the specialist is stopped by rows it
will never see). `hsplice.py` scans the splice weight beta:
    p'(h=1) = beta*specialist + (1-beta)*general,   p'(h>1) = general
The specialist sees a third of the rows so it is noisier; beta is what lets it contribute without
taking that noise on whole. Measured against the same lgb+xgb BLEND it will be spliced into, not
against lgb alone, or beta would be fitted for a model that is never submitted.

## 4. A third validation layout with the TEST's own geometry  (validation_c.py)
A [3,2,3,7,3] and B [1,3,4,3,7,2] were both invented before the test's block structure was worked
out. eval_mix repairs the horizon MARGINAL by reweighting, but not the joint structure: how stale
tws_known is when a block starts, and how many months of covariates have accumulated since the
last observation, are properties of the gaps, not of the horizon. Layout C takes the test's exact
pattern [1,3,4,7,1,2] WITH the test's exact gaps and slides it back 40 months, so its last month
is the last month of train:
    test     2015-09 | 2016-01..03 | 2016-06..09 | 2016-12..2017-06 | 2018-07 | 2018-11..12
    layout C 2012-05 | 2012-09..11 | 2013-02..05 | 2013-08..2014-02 | 2015-03 | 2015-07..08
Calendar months differ by four, so C is not a seasonal replica -- nothing can be. What it
replicates is the geometry, which is what A and B get wrong. C's history is shorter, so its
absolute RMSE is not comparable with A's; only differences between methods are. It runs only the
finalists (e1, e4, e8) plus the xgb and specialist runs: C exists to CONFIRM a choice made on A
and B, not to make it, and a full grid there would cost as much as the grid itself.

## Why the hindcast bias correction is NOT here
NOTES session 9j listed "hindcast bias correction" as follow-up #1: at each block's t_known the
truth is observed, so predict t_known from an earlier month, measure the error field, smooth it,
carry it forward. Worked through, it does not survive contact with the test file. A row in this
project's format at month t predicts TWS(t+1), so hindcasting a block's first month f needs a row
at month f-1 -- and Test.csv contains ONLY the 18 block months. f-1 is never one of them, so the
competition's own covariates (SPEI, soil moisture) do not exist for that row, and the hindcast
would be made by a model looking at a feature vector unlike anything it was trained on. The
measured "bias" would be that mismatch, not the regime. The gaps also make the hindcast horizons
1, 4, 5, 6, 19 and 4 months, so the error field would be estimated at horizons far longer than
the h=1..7 it would be applied to. Dropped, with the reasoning recorded so it is not re-proposed.

## Orchestrator (run_night.sh)
- Priority order changed so a usable submission exists as EARLY as possible: the moment config 1
  finishes training, a file is written; every later phase only adds to it. `astep` never caches
  an assemble, so each call picks up whatever seeds have landed since.
- `BUDGET_H` (default 12, 24 with DEEP=1) with a `have_time` guard before every optional phase,
  so a long run stops adding work rather than overrunning.
- New phases: layout C (DEEP), the h=1 specialist on every layout, and 5b -- blend weight,
  smoothing, splice weight and calibration, fitted in the order final_assemble applies them.
- Three control files are now produced beside the main one: `sub_q_main_nosm` (no smoothing at
  all) and `sub_q_main_nocal` (tuned smoothing, no calibration). If the leaderboard disagrees
  with validation about the post-processing, those two say which stage caused it.

# Session 9n — the run now explains itself: data, error anatomy, stacking, and an audit

The complaint this answers: a run produced a submission and a number, and nothing else. No
description of the data, no account of where the error actually is, no record of why each choice
was made, and nothing a code reviewer could read. Five additions.

## 1. data_report.py — what is actually in the files
Every design decision in this project rests on beliefs about the test file's structure that were
inferred once, by hand, and never re-derived. This derives them:
 - the BLOCK STRUCTURE read off Test.csv -- run lengths, the gaps between blocks, which months
   carry an observed TWS -- and cross-checks it against eval_mix.TEST_BLOCKS. If those disagree,
   the horizon weighting, HMIX=test and layout C are all wrong, and until now nothing would have
   noticed. It prints MATCH or a loud MISMATCH.
 - GRACE gaps in train, cells per month, coverage
 - per-era TWS statistics, so the size of the 2002-2015 -> 2015-2018 regime shift is a measured
   number rather than an assumption, plus the sd of the month-to-month global mean change, which
   is the hard bound on what any global-offset correction could ever buy
 - seasonal amplitude and the seasonal share of variance per latitude band (what climatology alone
   can do), and lag-1 autocorrelation and mean |one-month change| (the bar any h=1 model must beat)
 - covariate null rates, so a 40%-missing feature is not mistaken for a weak one

## 2. analyze.py — where the error is
Every number ever quoted here was a single RMSE, which says whether a change helped and never
where the error lives. This decomposes the residuals by horizon (with each horizon's share of the
TEST-weighted MSE, the only ranking that says where effort pays), by latitude band, by the cell's
own variability, by calendar month, and as a band x horizon matrix. Two rows matter most:
 - "model is worse than the last observation on X% of rows", per horizon. A model losing to
   persistence on a third of rows has a routing problem, not an accuracy problem.
 - the residual VARIOGRAM: correlation of the residual with its neighbour at grid lags 1, 2, 3, 5,
   8. If residual correlation survives at lag 2-3 there is more for spatial smoothing to take; if
   it is gone by lag 1, smoothing is finished and smooth_scan will find nothing. This is the first
   direct measurement of that, after four sessions of tuning a smoother blind.

## 3. stack.py — the ensemble weight is a vector, not a scalar
blend_scan tunes one number, lgb against xgb. That is the right question only while there are two
members. With a 31-leaf LightGBM, a 63-leaf one and CatBoost also trained -- each failing
differently on the regime shift, which is exactly why the leaderboard's family ladder ran opposite
to validation -- the answer is a weight vector. Non-negative least squares on the test-mix-weighted
rows, renormalised to sum to 1 so the overall SCALE stays postcal's job, pulled halfway back to
the incumbent blend, and adopted only if the leave-one-layout-out gain clears 0.0003 everywhere.
The incumbent is evaluated over the SAME member list with zero weight on the newcomers, so
"adopted" means better than what would otherwise be submitted, never better than an equal-weight
blend nobody proposed.

## 4. compliance.py — the audit the code review will do
Checked against the ARTEFACTS, not against what the code is believed to do:
 - the exact feature list each run used (run_models.py now writes out/mats/used_*.json), audited
   for lat/lon. feats.json is the SUPERSET and does contain lat/lon by design, so auditing it
   would have been meaningless -- which is why this needed a new artefact.
 - t_known <= t and horizon == months(t_known -> t) + 1 for every row
 - tws_known equals the observed TWS at t_known, reconstructed independently from Train.csv and
   the unmasked Test.csv rows
 - clim_next recomputed from history ALONE and compared to the matrix column. Agreement is a
   direct test that no test-era month entered the climatology -- a measurement of R2, not a
   reading of the code. Verified on a fixture: sabotaging the climatology with test months is
   detected (max difference 2e-1), and so is adding lat/lon to the feature list.
 - an inventory of external/, and a grep for the patterns that have caused problems before

## 5. report_night.py — one file at the end
out/RUN_REPORT.md: the configuration and the held-out evidence behind every adopted change first,
then the grid, then the data and error analysis, then the audit. This is most of what a code
review needs, written as the run happened rather than reconstructed afterwards.

## A real bug this batch fixed
blend_scan was being called as `blend_scan.py ${FM}_v5x_noll xgb_v5x_noll _bw`, but only xgb was
ever trained under the tag `_bw` -- the lgb side was tagged `_e8`. The lgb predictions did not
exist under that tag, so the scan would have found nothing and silently kept 50/50, and every
later post-processing scan would have been fitted on the wrong pair. Phase 5 now trains EVERY
family, including the chosen one, on the chosen configuration under the single tag `_bw`. This
also matters beyond the bug: the chosen configuration can differ from any single grid experiment
(features from e4, capacity from e5, uniform weights), so no grid tag ever names it.

## Orchestrator
New phases: 1 data_report, 5 all families on the chosen configuration under one tag, 5b stacking
between the blend scan and the smoothing scan, 5c the error anatomy per layout, 9 compliance +
report. WIDE=1 (implied by DEEP=1) adds CatBoost and the other capacities so the stack has
something to fit. `rd()` takes each family's boosting rounds from what early stopping chose for
THAT family on the chosen configuration, instead of reusing lgb's count for everything.

## Still open before submitting
requirements.txt lists lightgbm, polars and friends but not xgboost, catboost or scipy, all of
which the run path imports. Deliberately not touched mid-competition -- a reinstall now is a risk
not worth taking -- but run `pip freeze` and reconcile it before the code review.

# Session 9d results — the first full grid, run 2026-09-07 12:05-13:22

Run on the entrant's machine at commit 05b940b, so WITHOUT the 1500/2500 km anchors, the
antecedent windows, the modelled-TWS composite or any of the session-9m post-processing. Scored
under the real test horizon mix on both layouts.

  experiment                 testmix A   testmix B    vs e4 (A / B)
  e1 base, no anom no zonal     0.6551      0.5584    +0.0132  +0.0201
  e2 +covariate anomalies       0.6422      0.5399    +0.0003  +0.0016
  e3 +zonal scale only          0.6551      0.5574    +0.0132  +0.0191
  e4 +both                      0.6419      0.5383         --       --
  e5 both, 63 leaves            0.6432      0.5375    +0.0013  -0.0008
  e6 both, 31 leaves            0.6451      0.5387    +0.0032  +0.0004
  e7 both, uniform weights      0.6432      0.5424    +0.0013  +0.0041

Chosen: DROPF='' (e4), lgb, ramp weights. Five seeds each of lgb and xgb -> out/sub_p_final.csv.

## What it settles
1. THE COVARIATE ANOMALIES ARE REAL AND LARGE. -0.0128 on A, -0.0184 on B, and they win at every
   horizon on B and at six of seven on A. This is now confirmed twice on independent runs, and it
   is an order of magnitude above the 0.00096 leaderboard-noise threshold for this pair of files,
   so if it does not show up on the public board the failure is in TRANSFER, not in the feature.
2. THE ZONAL FEATURES ARE NEARLY WORTHLESS ALONE: +0.0001 on A, -0.0010 on B. On top of the
   anomalies they are worth -0.0003 (A) and -0.0016 (B) -- inside noise on A. Kept because they
   do not hurt, but they are not where the next gain is. The ONI trap warning in features_scale.py
   was the right instinct; the band-relative encoding avoided the disaster but did not find much.
3. CAPACITY IS GENUINELY AMBIGUOUS. 63 leaves is the best model on layout B (-0.0008 vs e4) and
   the third best on A (+0.0013). The both-layouts rule keeps 127 leaves, correctly, but this is
   the closest call in the grid and it is exactly why layout C was built.
4. UNIFORM WEIGHTS ARE THE BEST THING AT h=1 ON LAYOUT A (-0.0224 against the base, better than
   any other variant) and lose everywhere else. h1 is a third of the test, so this is worth
   re-checking once layout C exists.

## The finding that changed the code: a large, systematic OVER-prediction
Every one of the seven experiments reports a positive bias -- mean prediction minus truth:

  e1  A +0.0136   B +0.0625        e4  A +0.0110   B +0.0555
  e2  A +0.0167   B +0.0601        e7  A +0.0029   B +0.0517

Layout B's +0.055 is not small: it is about 1% of the RMSE in the mean alone, and no feature or
capacity change moves it much (uniform weights shave it from 0.0625 to 0.0517 and nothing else
touches it). postcal.py fitted only a SCALE through the origin, which cannot remove an offset at
all, so this error was structurally invisible to every correction in the pipeline.

postcal now fits both forms and lets the held-out layout choose:

  scale    p' = k + a_h * (p - k)
  affine   p' = k + a_h * (p - k) + b_h

The offset is not obviously transferable -- it is five times larger on B than on A, so it is a
property of the era rather than of the model -- which is precisely why the decision is made by a
leave-one-layout-out score rather than by judgement. Verified on fixtures in three regimes:
identical bias on both layouts (affine adopted, offset = half the bias, as LAM=0.5 intends), the
real same-sign divergence +0.011 / +0.055 (affine still adopted, offset shrunk to about a third of
B's), and opposite signs +0.05 / -0.05 (affine REJECTED, scale kept, FINAL_CALIB_B empty).

## Measured timings, so the orchestrator's estimates stop being guesses
build_mats 8 min (A), 4.5 (B), 11.5 (FINAL); one grid experiment 100-215 s; a FINAL seed 1.6 min
for lgb and 3.2 for xgb; the whole old run_all.sh 77 minutes. run_night.sh is therefore about
3.5 h by default, 5 h with WIDE=1, 7.5 h with DEEP=1 -- not the 4/11 h previously written.

## Next action for the entrant
Submit out/sub_p_final.csv. It is the first file that contains the anomaly features, its
validation gain is 10-20x the noise threshold for this pair, and whether that gain appears on the
public board decides everything else: if it transfers, the same features under the session-9m
pipeline should go further; if it does not, the problem is the 2015-18 regime and the effort
belongs in calibration and the ensemble rather than in more features.

# Session 9o — measured on a full synthetic integration run: 38 features are dead at h=1

Found while running the whole orchestrator end to end in the cloud container against a
structurally faithful synthetic dataset (same column names, .5-centred grid, the real month
calendar with GRACE gaps, and the six real test blocks at their real dates). Measured on the
layout-A matrix, not reasoned about:

  horizon   rows    the 14 '_acc' features      the 24 '_d' features
    h=1     8000    100% NULL                   100% EXACTLY ZERO
    h>=2     ...      0% NULL                     0% zero (sd ~0.6)

Why, and why it is not a bug: a row's accumulation window is (t_known, t] and its difference is
value(t) - value(t_known). At h=1, horizon = mdiff(t, t_known) + 1 = 1 means t_known == t, so the
window is empty and every difference is identically zero. That is the truth about the data --
at h=1 there is no gap to describe -- but the CONSEQUENCE was never noticed:

  h=1 is 33.3% of the test, the largest single slice, and there 38 of the 201 features carry
  no information whatsoever.

Three things follow.

1. THE HORIZON-1 SPECIALIST IS MUCH BETTER MOTIVATED THAN THE ARGUMENT IN SESSION 9m. That
   argument was about h=1 being "a different problem". The real reason is mechanical: with
   feature_fraction=0.6 a pooled model offers roughly 23 of those 38 dead features at every
   split it makes on an h=1 row, so a fifth of its candidate set at the largest test slice is
   guaranteed waste. A specialist trained with HFILT=1 never sees them.
2. THE ANTECEDENT WINDOWS ADDED IN SESSION 9m ARE THE ONLY COVARIATE-DYNAMICS INFORMATION THE
   MODEL HAS AT h=1. The 24 '_w3/_w6/_w12/_m3/_m6/_m12' features are 0% null at every horizon
   including h=1, because their window ends at t and does not depend on t_known at all. They
   were built for drought memory; it turns out they also fill a hole that had been open since
   session 2 and that nothing else covers.
3. It is consistent with the grid's per-horizon table, where uniform weights (e7) were the best
   variant at h=1 on layout A by a wide margin (-0.0224 against the base) and lost everywhere
   else. Whatever helps at h=1 is not what helps elsewhere.

No code change: the features are correct, and the two mechanisms that exploit this (the
specialist and the windows) are already in the pipeline and already gated. This is recorded so
the h=1 specialist is read as a structural fix rather than a hunch, and so nobody later "fixes"
the empty window by making it inclusive of t_known -- that would double-count the anchor month
at every horizon above 1 to buy a single month of flux at h=1.

# Session 9o results — the integration run earned its keep: two real bugs

Ran the whole orchestrator end to end in the cloud container against a structurally faithful
synthetic dataset (real column names, .5-centred grid, the real month calendar with GRACE gaps,
the six real test blocks at their real dates, 1600 cells so it finishes in two hours). All
thirteen phases completed. Everything below was found by that run and nothing else would have
found it before the deadline.

## Bug 1 (submission-blocking): horizon is Float32, and a float cannot index an array
`build_mats.py` casts every feature to Float32 on write, and `horizon` is a feature. So
`va["horizon"].to_numpy()` is float32 in EVERY matrix -- A, B, C and FINAL. Both postcal.py and
final_assemble.py applied the per-horizon calibration as `a[np.clip(h, 1, 7)]`, which raises

    IndexError: arrays used as indices must be of integer (or boolean) type

postcal crashed outright, so no calibration was produced. Worse, final_assemble carries the same
line: the moment postcal DID adopt a calibration, assembling the submission would have died --
and it would have died at the very end of a long run, after the training was done. Fixed with an
explicit .astype(int) in both, verified by assembling with the scale-only form and with the
affine form. Everything else that touches horizon (hsplice, stack, analyze, smooth.horizon_w,
run_models' HFILT) compares or casts rather than indexing, so nothing else was affected.

## Bug 2: layout C only runs the finalists, but phase 3b asked it for the whole grid
eval_mix raises on the first missing tag, so the C table was lost entirely rather than partly.
Phase 3b now asks C for e1/e4/e8 only. Cosmetic -- the step is `|| true` -- but the C table is
the whole reason C exists, and it was silently absent from the report.

## What the run confirms works
- all three layouts build with the full feature set (201 features: 44 anomaly, 7 zonal, 12 anchor
  columns including the 1500/2500 km radii)
- validation_c reproduces the test horizon mix EXACTLY: h1 .333 h2 .222 h3 .167 h4 .111
- data_report derives the block structure from Test.csv and cross-checks eval_mix: MATCH
- select_config chose DROPF='' / lgb / ramp -- the same configuration the entrant's real run chose
- stack.py fitted a genuine four-family ensemble: lgb .375, xgb .213, lgbm .160, cat .252,
  which is the first time this project has used more than two families with fitted weights
- smooth_scan turned smoothing OFF on this data (w=0) -- correct, the synthetic residuals are
  near-white, and it shows the scan is capable of rejecting the incumbent rather than only tuning it
- postcal's three-way choice works: after the fix it adopts the scale form over the affine one on
  the held-out average and leaves FINAL_CALIB_B empty
- compliance.py: every check passed, including clim_next reproducing a history-only climatology
- four submission files written, and report_night produced a 35 kB RUN_REPORT.md in 23 sections

## Consequence for a run already in flight
Neither bug aborts a run: both steps are `|| true`. A run that hit them completes and produces
valid submissions, just WITHOUT the calibration stage and without the layout-C table. Recovery is
free because failed steps never write a marker: `git pull` and rerun the same command. Only the
two failed steps and the assembles repeat; every training run is skipped.

# Session 9p — IT TRANSFERRED. 0.709259 -> 0.696326 on the public leaderboard

`out/sub_q_main.csv`, the first submission carrying the covariate anomalies, scored
**0.696326352** against a previous best of 0.709259.

  gain                    -0.012933
  SE of the gap           0.00046      (lb_se.py, for this exact pair of files)
  in standard errors      28
  vs the noise threshold  11x

## The number that matters more than the gain
Validation predicted the anomalies would be worth between -0.0128 (layout A) and -0.0184
(layout B). The leaderboard delivered -0.0129 -- inside the predicted range, essentially 1:1
with layout A.

After session 4's re-base (+0.0069) and the climatology pull (+0.0208) both transferred
BACKWARDS, this project had good reason to distrust its own validation, and several sessions
were spent hedging against it. That distrust is now resolved for this class of change:
a feature that wins on both layouts under the test horizon mix wins on the leaderboard, by
about the amount validation says. Every remaining idea gated by xfit.py is therefore worth
more than it was yesterday, because the gate has been shown to predict rather than merely
to filter.

It also retires the session-5 E2 conclusion for good. E2 said no legal covariate can predict
the residual; ceiling.py showed that rested on a raw-vs-anomaly encoding artifact; the
leaderboard has now paid out 0.0129 on exactly the features E2 said could not exist.

## What this was, precisely
The submitted file is not the anomalies alone. It is the whole session-9 stack: covariate
anomalies, 3/6/12-month antecedent windows, the modelled-TWS composite, 1500/2500 km
continental anchors, zonal context, the configuration select_config chose (DROPF='' / lgb /
ramp), a fitted multi-family ensemble, tuned spatial smoothing, and 16 seeds per family --
but NOT the per-horizon calibration, because postcalscan crashed on the Float32 horizon bug
(fixed in 47750f7, after this run). So the calibration stage is still unspent.

## Position
0.696326 is 0.006326 above the 0.69 target the entrant set. At the measured cluster density
of 0.00047 RMSE per rank it is worth roughly 28 places from rank 82.

## What it makes urgent
diag2_ceiling_A now decides ERA5-Land on arithmetic rather than hope: with validation shown to
transfer, its measured headroom is a forecast of leaderboard gain, and ERA5-Land is the only
remaining single idea large enough to close 0.0063. Read it first.
