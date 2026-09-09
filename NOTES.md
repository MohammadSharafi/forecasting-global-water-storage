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

# Session 9q — rank 82 -> 55, what the three controls settled, and three real defects

Public leaderboard, same models and same seeds, differing only in post-processing:

  sub_q_main       tuned smoothing + calibration   0.696326352
  sub_q_main_nocal tuned smoothing, no calibration 0.695964865   <- best
  sub_q_main_nosm  no smoothing, calibration       0.699149222

  smoothing   removing it COSTS  0.002823   -> keep it
  calibration removing it GAINS  0.000361   -> drop it

Rank 82 -> 55 on one submission. The controls paid for themselves the first time they were used:
without them the calibrated file would have been submitted as the best and 0.00036 thrown away.

## Defect 1: the calibration is the one stage whose held-out gate did not transfer
postcal's leave-one-layout-out gate ADOPTED the scale form (a = 1.130, 1.105, 1.029, 0.972,
0.988, 0.982, 1.000) and the leaderboard says it is worse. Everything else this session predicted
its own leaderboard result closely; this did not. The likely mechanism is visible in analyze_A:
the model's bias at h=1 is +0.0917, by far the largest of any horizon, and the calibration
MULTIPLIES the h=1 predicted change by 1.13 -- amplifying a quantity that is already too large.
A scale through the origin cannot see a bias, and the affine form that could was rejected by the
same gate. Calibration is now OFF by default (USE_CALIB=1 to apply) and the calibrated file is
produced as the control instead of the uncalibrated one.

## Defect 2: layout C was structurally broken and voted anyway
On the real Train.csv the hardcoded 40-month shift put block months inside GRACE's gaps, so blocks
were truncated: 140,491 rows instead of ~280,000, horizons 3, 5, 6 and 7 EMPTY, and a horizon mix
of .445/.110/0/.111/0/0/0 against the test's .333/.222/.167/.111/.056/.056/.056. select_config
used it as a third vote, and it changed the answer: A+B alone had chosen DROPF='' with no horizon
reweighting, while A+B+C chose DROPF='scale,bigsa' with HMIX='test'. A layout that cannot score
four of seven horizons is not a weaker vote, it is a wrong one.

validation_c.py now SEARCHES every placement of the [1,3,4,7,1,2] pattern at the test's exact
spacing and keeps the latest one whose months train actually has, so the geometry is exact by
construction. If no gap-free placement exists it says so loudly, and it EXITS NON-ZERO when any
horizon is empty so the layout can never silently become a vote again. Verified on a fixture:
planting a gap inside the chosen placement's 7-month block makes it slide from 2012-05 to 2011-11
and stay exact, rather than truncate.

Note the configuration that scored 0.695965 was chosen WITH the broken C in the vote. It is a
good configuration; it is not necessarily the one A+B+working-C would pick.

## Defect 3: globalshift measured the largest prize in the project and never tested reachability
The band-offset decomposition on the real data says 13.0% of total MSE is per-latitude-band offset
error, with an oracle of -0.0423 RMSE. That is almost exactly the 0.0400 gap to the top ten. But an
oracle is not an opportunity -- it needs the answer to compute the correction. globalshift tested
reachability for the GLOBAL offset (persistence corr -0.081, dead) and never for the band offset,
which is four times larger. It now runs the same test per band, plus a second one asking whether
the offset is predictable from what the band itself has recently been OBSERVED doing, which would
be legal since it uses only months <= t_known.

## What analyze_A says about where the error actually is
- residual spatial correlation: lag 1 +0.967, lag 2 +0.899, lag 3 +0.809, lag 5 +0.621, lag 8
  +0.447. The error field is extremely coherent -- it is REGIONAL, not per-cell noise. Two
  consequences: smoothing can only ever take a little (a neighbour's error is nearly the same
  error), which matches the measured 0.0028; and higher-resolution forcing such as ERA5-Land is
  the least likely thing to help, because the error does not live at fine scales.
- h=1 carries 31.7% of the test-weighted MSE, more than any other horizon, and has bias +0.0917.
- the model is worse than plain persistence on 38-45% of rows at every horizon.
- 25% of the error comes from 8.7% of cells; the 20 worst cells carry only 0.67%, so there is no
  small set of pathological cells to fix.
- ceiling.py on the real data: raw e5PER_acc r = 0.004, anomaly encoding r = 0.177; raw e5P_acc
  r = -0.032, anomaly r = 0.235. The session-5 E2 artifact is confirmed on real data at a factor
  of roughly 60. The linear ridge bound on the whole anomaly block is R^2 = 0.050 on the change.

## Standing
0.695965 at rank 55. The 0.69 target is 0.0060 away and reachable. The top ten begins at
0.655935, which is 0.0400 away -- three times everything session 9 has gained. That gap will not
close on incremental feature work, and the honest reading of the diagnostics is that the
remaining large-scale error is regional and, so far, not shown to be predictable.

# Session 9r — the band offset is closed, and a caching bug that would have corrupted the rerun

## The largest identified prize is unreachable. Measured, not assumed.
globalshift on the real layout-A predictions (`_bw`, the submitted blend):

  band-offset error        12.9% of total MSE
  ORACLE, perfect band correction   RMSE 0.5847  (-0.0418)

  reachability, applying the PREVIOUS month's band offset:
    a=0.25  +0.0003     a=0.50  +0.0055     a=0.75  +0.0155     a=1.00  +0.0302
    corr(previous band offset, this band offset) = +0.190  over 85 band-months
    corr(band's own recent observed change, its offset) = +0.062  over 85 band-months

Every weight tried makes it WORSE, including the smallest. The arithmetic agrees: a linear
predictor with correlation r removes r^2 of the component it targets, so the ceiling here is
0.190^2 x 0.129 = 0.0047 of MSE, or -0.0015 RMSE, and that is fitted in sample. The second row is
worse still: 0.062^2 x 0.129 = 0.0005 of MSE, -0.0002 RMSE. The oracle of -0.0418 is real and
completely out of reach, because it needs the answer in order to compute the correction.

Both offset avenues are now closed on evidence rather than on judgement: global (persistence corr
-0.110, oracle -0.0090) and band (persistence corr +0.190, oracle -0.0418). Session 3 identified
the regional coherent error as the largest error component and it has taken until now to
establish that no part of it is predictable from information at <= t. That is a negative result,
and it is worth as much as a positive one: it stops the remaining days being spent there.

Taken with analyze_A's residual variogram (lag 1 +0.967, lag 2 +0.899, lag 3 +0.809), the picture
is consistent and unwelcome. The dominant remaining error is a large-scale, spatially coherent,
temporally white field. Spatial smoothing can only shave it (a neighbour has the same error,
which is why the leaderboard paid exactly 0.0028 for it). Higher-resolution forcing cannot reach
it, so ERA5-Land is not the answer either. And it does not persist month to month, so it cannot
be corrected from history.

## The caching bug: markers do not encode the configuration
A step's marker is its NAME, and a FINAL training step's name carries the tag, the family and the
seed -- but not the configuration it was trained under. So when select_config changes its answer
between runs, which is EXACTLY what fixing layout C will do, the markers still match, every
training step is skipped, and the submission is assembled from models trained under the OLD
configuration with nothing anywhere to say so.

This was about to happen: the 0.695965 configuration (DROPF='scale,bigsa', HMIX='test') was chosen
with the broken layout C voting, and the rerun with C fixed will very likely choose differently.

`cfg_guard` now records what each tag was trained with -- dropf, weights, hmix, the family list and
the seed count -- and discards that tag's markers when any of it changes, printing the old and new
signatures. Verified on a fixture: identical configuration keeps all three markers, a changed one
discards them and says why.

# Session 9s — the offset that IS reachable, and a training-set size nobody ever swept

## The seasonal bias: -0.0060 on the table, and periodic by construction
analyze_A's target-month table, on the submitted blend:

  month  6   bias +0.2912   16.1% of MSE        month  2   bias +0.0600   18.0% of MSE
  month 10   bias -0.1262    3.4% of MSE        month  9   bias +0.0585    6.2% of MSE
  month 11   bias -0.0774    6.3% of MSE        month 12   bias +0.0549   11.8% of MSE

Row-weighted mean squared bias 0.00747. Removing all of it: RMSE 0.6260 -> 0.6200, i.e. -0.0060 --
four times the best case the band offset could ever have given, and the same size as the gap to
the 0.69 target.

WHY THIS ONE IS DIFFERENT FROM THE TWO CLOSED AVENUES. The global and band offsets failed because
they are temporally white: nothing at time <= t predicts them (persistence corr -0.110 and +0.190).
A calendar-month effect is periodic by construction -- June recurs every year -- so an estimate
built from other years IS a legal prediction for this one. That is a property of the quantity, not
an assumption about it.

IT ALSO EXPLAINS WHY postcal FAILED ON THE LEADERBOARD. postcal fitted per HORIZON, and inside one
of these layouts horizon is very nearly a proxy for calendar month: layout A's 7-month block
supplies its h7 rows from exactly one target month, its h6 rows from one other, and so on. Fitting
"per horizon" on layout A is really fitting "per those particular months", which the test then
applies to entirely different months. Its leave-one-layout-out gate could not catch this because
both layouts share the confound. Indexing by the month itself removes it.

WHY THE MODEL HAS NOT ABSORBED IT ALREADY. `m_next` IS in FEATS2, so this is not a missing input.
But a gradient-boosted tree optimises local squared error, and a small additive offset spread
evenly across every cell in a month buys almost no split gain anywhere while costing real MSE in
aggregate. One additive number expresses what the tree would need many splits to approximate.

seasonal.py estimates the bias per target calendar month on the OTHER layouts and scores it on the
held-out one, shrinks it, clips it at 0.25, zeroes any month the fitting layouts do not cover, and
prints how many distinct month-years back each coefficient -- because a "seasonal" effect measured
from one June is not a seasonal effect. Verified on two fixtures:
  * a planted seasonal bias (June +0.29, Oct -0.13, Feb +0.06) is recovered almost exactly
    (+0.298, -0.116, +0.085) and adopted, with held-out gains on all three layouts;
  * a per-layout constant offset that is NOT seasonal is REJECTED -- and it fails in exactly the
    shape that killed postcal, helping two layouts and hurting the third badly.
final_assemble applies it to the TARGET month (t+1, not t); verified that the shift lands on
precisely the intended months and nowhere else.

`sub_q_main_noseason` is produced as a control, so the leaderboard judges this the same way it
judged smoothing and calibration rather than validation having the last word.

## PER_ROW: the training set has been a third smaller all session, by accident
build_mats.py draws PER_ROW (t, t_known) pairs per cell-month, and its own default is 3.
run_night.sh passes 2. That was not a decision; it was a value typed into the orchestrator, and it
means every run this session trained on about a third fewer rows than the pipeline that produced
the earlier submissions. It has never been swept -- not in any session, on any layout.

More rows per cell-month is not free (build time and training time both scale with it) and it is
not obviously better either: the extra pairs reuse the same underlying observations, so they add
coverage of the (horizon, staleness) space rather than independent information. But a 200-feature
model on a third more rows is exactly the kind of change that is worth an hour to measure, and
nobody has. PER_ROW=3 and PER_ROW=4 on layouts A and B, scored under the test mix, is the next
cheap experiment.

# Session 9t — the organisers' clarifications: what changes, what does not, and one urgent gap

## URGENT, and nothing to do with score: emissions must be measured DURING training
"The sustainability criterion asks for an estimate of training emissions using tools such as
CodeCarbon, which measure consumption while training runs and cannot be applied afterwards.
Instrument your training runs if you intend to compete for a top-10 placing."

This project had `carbon_final.py`, which is called from NO pipeline, measures featsets
(`allnoll`, `v5x_noll`) and families (cat, mlp) that the current configuration does not use, and
re-runs training separately just to measure it -- which is exactly the after-the-fact
reconstruction the organisers say does not count.

run_models.py is now instrumented directly: CARBON=1 starts a CodeCarbon tracker around the run
that actually produces the submitted model and writes one row per run to out/carbon/emissions.csv.
run_night.sh exports CARBON=1 by default, so from the next run the measurement is a sum over the
work that really happened. It can never break training -- a missing codecarbon or a platform that
withholds power counters prints one line and continues; verified both ways, including a full run
that trained and saved normally with codecarbon absent. carbon_report.py totals it, splits it by
phase (validation versus FINAL) and writes out/carbon/summary.md. Verified end to end on three
real runs.

## Recursive forecasting is now explicitly permitted -- and the arithmetic says it would LOSE
"Permitted - recursive forecasting, feeding your own prediction forward as an input to the next
month. The restriction is on future observations, not on model output."

This was the largest structurally-different idea left, so it deserved a real answer rather than an
attempt. The decisive number is already in analyze_A:

  our one-step error in the LEVEL of TWS(t+1)   0.6220
  the direct model's overall RMSE               0.6260

Recursive forecasting pays off when one-step prediction is much more accurate than multi-step.
Here it is not more accurate AT ALL -- it is the same. And a recursive prediction at horizon k
inherits that level error whole as its anchor, so

  err_recursive(k)^2 = err_anchor^2 + err_step^2 >= 0.6220^2

even with a perfect step model. Against the direct model's 0.5345 at h2 and 0.5491 at h3 -- 39% of
the test by weight -- recursion loses before any compounding at all. The horizons where the floor
looks favourable (h4-h6, 22% of weight) are the ones where the direct error is inflated by calendar
month rather than by staleness: layout A's h6 is one specific June, which carries a +0.29 bias and
an RMSE of 1.07, and feeding predictions forward does not fix June.

CORRECTION (session 9w). The sentence that stood here said "Not built", and that is wrong: the same
session added `recursive.py`, a 160-line measurement script that chains horizon-1 predictions
through build_mats.prepare's own feature builder and prints direct-versus-recursive RMSE per
horizon under the test mix. It does not touch the submission. Having both is right and the note was
simply inaccurate -- but the distinction matters, because the argument above is a BOUND derived
from numbers that are confounded with calendar month (layout A's h6 is one specific June), while
the script is the measurement that would settle it. The bound predicts recursion loses; it has not
yet been run on real data to confirm that. Run `python recursive.py A` (and B, C) if there is time;
if the measurement disagrees with the bound, the measurement wins.

## Seasonal forecasts: do NOT use them, the rules contradict each other on exactly this
A competitor asked whether GDO seasonal forecasts -- issued in month t, covering t+1 to t+6 --
are allowed, noting the "source date <= t" language was written for reanalysis. The published
answer lists what is permitted and prohibited and does not address forecasts, so the question is
open. Meanwhile the two governing clauses conflict for this product and nothing else:

  permitted      "external non-TWS covariates, provided every source date is at or before t"
  prohibited     "any feature encoding month t+1 or later, from any source, remains grounds
                  for disqualification"

A forecast issued at t has source date t and ENCODES t+1. Under a top-ten code review the second
clause is the one that gets applied, and the cost of being wrong is the placing, not a few
thousandths. Treat as prohibited until the organisers answer that specific question in writing.

## What the clarifications confirm that we already do
- t+1 is the next calendar month for every row; horizon up to seven. Matches.
- only TWS_t is masked; SPEI and SOIL_MOISTURE are present for every test row at its own month t.
  Matches -- add_era5/add_ncep/add_anom already read covariates at t for every row.
- the strictly-backward anchor is the permitted fill. That is exactly `t_known`.
- filling a masked row with the cell's mean over observed TEST months is PROHIBITED. We have never
  done this; compliance.py's independent reconstruction of tws_known from Train plus unmasked Test
  rows proves it.
- lat/lon as a lookup key is permitted, as predictive features is not. compliance.py audits the
  used-feature list for exactly this and passes.
- GRACE-derived products of any kind are prohibited absolutely. external/ holds NCEP reanalysis
  and ONI only; compliance.py inventories it.

## The one genuinely new opening: longer and different covariates
The permitted list explicitly includes GDO products whose source date is <= t. Two are worth the
remaining time and are NOT resolution refinements -- the objection that closed ERA5-Land was that
the error does not live at fine scales, which says nothing about new VARIABLES:
  * SPI at 9, 24 and 48 months. The feature set stops at 12. TWS includes groundwater, which
    integrates over years, and nothing in the model sees beyond one year of accumulated deficit.
  * fAPAR and its anomaly. Vegetation is an OBSERVATION of how much water the surface actually
    had, integrated over weeks, and it is not derived from the same reanalysis as everything else
    in the feature set. It is the only candidate that is independent information rather than
    another function of P, E and R.
The measured ceiling for the existing anomaly block is a ridge R^2 of 0.050 on the change; that is
what a new covariate has to beat.

# Session 9u — REPORT.md rewritten; it described a pipeline that no longer exists

The report is half the top-ten outcome and it had drifted badly out of date. It described two
stacks of four families blended 50/50, a public score of 0.712, and feature lists of 168 and 166 --
none of which is now true. It did not mention the covariate anomaly encoding at all, which is the
single change that produced the entire leaderboard gain, nor layout C, nor the leave-one-layout-out
gate, nor any of the avenues closed on evidence.

Worse, its sustainability section described exactly the measurement the organisers have since said
does not count: a separate run of `carbon_final.py` after the fact. That section now describes the
live instrumentation in run_models.py.

Rewritten around what is actually established, with seven explicit **[pending]** markers for the
numbers the final run will produce, so nothing is invented and finishing it is a fill-in rather
than a rewrite. New material a code reviewer will care about:

  * section 4 lists the five avenues CLOSED on evidence, with the measurement that closed each. A
    negative result honestly established is evidence of method, and this project now has five of
    them: the global offset, the band offset, higher-resolution forcing, recursive forecasting
    (explicitly permitted, and shown to lose on arithmetic we already had), and the hindcast.
  * section 5 describes compliance.py auditing ARTEFACTS rather than intent -- in particular
    recomputing clim_next from history alone and comparing it to the matrix column, which measures
    the leakage rule rather than asserting it -- and records the deliberate abstention from GDO
    seasonal forecasts, where the permitted and prohibited clauses conflict and the question is
    unanswered.
  * section 3.3 records the two post-processing stages the LEADERBOARD ruled on, in both
    directions: smoothing kept (removing it costs 0.0028), calibration dropped (it costs 0.00036
    despite the held-out gate adopting it).

The old version is not kept as a file; it is in git history at 2f8caf6.

# Session 9v — a rewritten script was skipped, and a seven-hour run finished in five seconds

The entrant pulled the layout-C repair and started `DEEP=1 ./run_night.sh`. It completed in five
seconds having changed nothing of substance:

  [skip] val_C      validation_c.py had been REWRITTEN, but the marker from the previous run
  [skip] build_C    survived, so layout C was never rebuilt and is still the broken one with four
  [skip] x_C_*      empty horizons
  [skip] select     so the configuration was never re-chosen; it is still the one the broken C
                    voted for -- DROPF='scale,bigsa', HMIX='test'

cfg_guard did not fire, correctly: the configuration did not change, because the step that chooses
it never ran. seasonscan DID run (it had no marker, being new) and rejected the seasonal
correction -- but it ran with the broken layout C in the vote, so that decision is not trustworthy
either. carbon failed because no training happened at all.

## The gap
A marker records that a step SUCCEEDED. It does not record WHAT it succeeded at. cfg_guard covers
the configuration; nothing covered the code. Any script rewritten between runs was therefore
silently ignored for every step already marked done -- which is the worst possible failure mode for
a resumable orchestrator, because it looks exactly like a successful run.

`src_guard <marker prefix> <sources...>` fingerprints the sources a family of steps depends on and
drops those markers when the fingerprint moves. Every family is now covered: the layouts by their
builder and every feature module, the anchors by their builder, every training step by
run_models.py, select by select_config/eval_mix/xfit, and each scan by its own script plus xfit.
Verified on a fixture: an unchanged source keeps the markers, an edited one discards them and says
so.

## Bootstrap: this does not fix the CURRENT stale state
src_guard can only compare against a fingerprint it has recorded, and none exists yet, so the first
run after this change records the fingerprints and discards nothing. The already-stale markers have
to be cleared once by hand; the command is in the session notes and covers everything downstream of
layout C -- val_C, build_C, anchor_C, the C grid and family runs, eval_grid_C, analyze_C, select,
and all six scans. FINAL training then takes care of itself: if select's answer changes, cfg_guard
discards it; if it does not change, the models are still valid.

One consequence worth stating: if the configuration does NOT change, FINAL training is skipped
again and no emissions are recorded, because CodeCarbon measures training that actually happens.
Getting the sustainability number therefore requires at least one configuration to be trained under
CARBON=1 -- which the config change will force, and which FORCE=1 can force otherwise.

# Session 9w — GDO wired end to end, and a correction to the record

## features_gdo.py -> build_mats
`load_gdo` reads `external/gdo/<product>/*.nc` and hands the result to `features_anom.build`, the
same per-cell standardisation that produced this project's only large leaderboard gain. That is the
whole point: SPI and fAPAR are LEVELS, and a model with no location features cannot use a level --
which is exactly the mistake that cost four sessions with the ERA5 block.

Products recognised: spi09, spi24, spi48, fapar, fapanom, smanom. Each yields six features -- the
z-scored level at t and at t_known, their difference, and 3/6/12-month antecedent windows ending
at t.

Verified on synthetic files built to have the quirks the real ones have, each case separately:
several files per product concatenated in time; `latitude`/`longitude` as well as `lat`/`lon`; 0-360
longitudes folded; a bounds variable beside the data variable; a stray singleton dimension; and
DEKADAL data averaged to monthly. Then end to end through build_mats:

  with two products present   201 -> 213 features, the 12 named exactly as expected
  DROPF=gdo                   213 -> 201, removing exactly those 12 and nothing else
  directory absent            "gdo: not present", 201 features, a clean no-op

The dedicated `DROPF=gdo` switch exists because the block shares the `an_` prefix with the ERA5 and
NCEP anomalies; without it, its contribution could never be measured separately from theirs.
`GDO_SETUP.md` documents what to download, what is deliberately NOT downloaded (the GRACE layer,
and the seasonal forecasts) and how to ablate it. `src_guard build_` already covers
features_gdo.py, so adding the data invalidates the cached matrices automatically.

## Correction: recursive forecasting WAS built, as a measurement
Session 9t's note said "Not built", and the same session added `recursive.py` -- 160 lines that
chain horizon-1 predictions through build_mats.prepare's own feature builder and print
direct-versus-recursive RMSE per horizon under the test mix, touching no submission. The note was
simply wrong, and it mattered: a reviewer reading the repo would have found a script contradicting
the log.

The distinction is worth keeping straight. The argument in 9t is a BOUND, and it rests on numbers
confounded with calendar month -- layout A's h6 is one specific June with a +0.29 bias. The script
is the MEASUREMENT. The bound predicts recursion loses; it has not been run on real data to confirm
that. If the measurement disagrees with the bound, the measurement wins. NOTES and REPORT now say
so, and REPORT carries it as a [pending] rather than as a settled result.

# Session 9x — the leaderboard settled the diversity question, and a silent under-training

## What the two new submissions settled
  sub_q_base  pre-session-9 feature set        0.708852   +0.012887 vs best
  sub_q_alt   31-leaf LightGBM, other capacity 0.698048   +0.002083 vs best

`sub_q_base` confirms the session-9 feature work end to end: the old feature set scores almost
exactly the old score, so the 0.0129 came from the features and not from seeds, smoothing or luck.

`sub_q_alt` is the interesting one. stack.py fitted the ensemble weights on validation and gave the
31-leaf model a weight of **0.000** -- validation says it adds nothing. The leaderboard says that
model, alone, is two thousandths off the best. Two points apart is not "adds nothing"; it is a
strong, genuinely different model that the fitted weight zeroed out. Validation has been wrong
about exactly this shape of question before -- it adopted the per-horizon calibration the
leaderboard then refuted -- so `blend_subs.py` makes the blend directly and spends one submission
to settle it rather than trusting the fitted zero. Verified to produce an exact weighted average
and to report the RMS distance to each input, which bounds how far the score can move.

## PRIVATE SLOTS: the current selection is the same bet twice
`sub_q_main_nocal` (0.695965) and `sub_q_main` (0.696326) are ticked. They differ only by the
calibration -- 0.00036 apart, essentially the same file. That is not a hedge. The second slot
should hold something different in KIND, and `sub_q_alt` at 0.698048 is the only competitive
candidate that qualifies.

## rounds.py: FINAL has probably been under-trained in every run this project has made
`rd()` took each family's boosting rounds from what early stopping chose on layout A. Layout A's
history is ~122 months; FINAL trains on ~160, about 30% more data. The number of trees a boosted
model wants grows with the training set, so FINAL was being given a count fitted to a set a third
smaller -- and under-training is invisible: there is no validation set for FINAL, nothing errors,
and the submission looks normal.

The layouts have different history lengths (B ~90 months, C ~120, A ~122), so their early-stopped
counts are three points on the curve of rounds against training size. rounds.py reads them from the
orchestrator's own logs, measures each layout's history from the parquet it was built from, fits
log(rounds) against log(months) and extrapolates to FINAL. It is deliberately conservative: the
power law is used only with three layouts and a fit explaining more than 60% with an exponent in
[0,1]; otherwise it falls back to the plain ratio of history lengths. The recommendation is never
below the incumbent, because the failure being corrected is under-training, and never above 1.6x.

Verified on fixtures: a planted exponent of 0.5 is recovered exactly and gives 1.15x; layouts that
disagree wildly (300/900/120 rounds) produce an exponent of -5.26, which is rejected in favour of
the ratio, and the result is still capped.

The round count is now part of cfg_guard's signature, so changing it invalidates the cached FINAL
models rather than silently reusing models trained for a different number of trees. Verified.

## What this does NOT do
None of it closes the gap to 0.65. sub_q_base and sub_q_alt bracket the current work at 0.7089 and
0.6980; the blend and the round correction are each worth thousandths at best. The honest position
is in REPORT.md section 4: five avenues closed on measurement, and the dominant remaining error is
large-scale, spatially coherent and temporally white.

# Session 9y — check_FINAL was mangled by eval, and it gates ALL final training

## The bug
`step` passes its command string to `eval`, which re-parses the quoting. The check_FINAL step was
an inline Python heredoc, and after that second parse it arrived as

    have=set(pl.scan_parquet(n')))
    SyntaxError: unterminated string literal

so the step failed. And check_FINAL GATES phase 7:

    if done_ check_FINAL; then   ... all FINAL training ...   fi

In the clean integration run on this container the log goes straight from PHASE 6 to PHASE 9: no
FINAL training happened at all, and the run still wrote five submission files from whatever
predictions were already on disk, reported "steps that did not complete: check_FINAL", and looked
otherwise normal. That is the most expensive failure this orchestrator can have -- it produces a
plausible submission built from nothing that ran.

The entrant's runs escaped it only because check_FINAL carried a `.done` marker from an older run
where the string happened to survive. A fresh machine, a FORCE=1 run, or any src_guard clear would
have hit it.

Moved to `check_final.py`. A file cannot be mangled by eval. Verified through the identical
`( eval ... )` path: exit 0, "FINAL carries all 201 features".

## Confirmed working in the same run
CodeCarbon end to end inside the orchestrator: 39 training runs measured, 1.86 hours, 0.0264 kWh,
split by phase, written to out/carbon/summary.md. This is the live measurement the sustainability
criterion asks for, on the runs that actually happened.

## The entrant's run, with layout C repaired
The repair changed the answer, which is the whole reason it mattered:

  configuration   DROPF='scale,bigsa' -> DROPF='bigsa'
                  the zonal features are now KEPT; the broken C had voted them out
  cfg_guard       fired on both _f1 and _f2, discarded 32 cached training steps each, retrained
  h=1 specialist  ADOPTED at beta=0.50 -- hsplice had rejected it every time the broken layout C
                  was in the vote. 16 seeds trained in phase 7b.
  seasonal        still rejected, now on trustworthy evidence
  stack           lgb 0.500 / xgb 0.500
  smoothing       incumbent kept (0.7, r=1, it=1)
  elapsed         3 h

So `out/sub_q_main.csv` from that run is a genuinely different model from the 0.695965 file: a
different feature set, retrained from scratch, plus a horizon-1 specialist spliced in at half
weight. It has not been scored yet.

# Session 9z — an overnight run for a day with no submissions left

The allowance is spent, so tonight has to buy ANSWERS and CANDIDATE FILES rather than another copy
of what already exists. `run_tonight.sh`, resumable the same way run_night.sh is, cheapest and most
certain first:

  T1  blends of the existing submission files, needing no training at all
  T2  recursive.py on A, B and C -- the measurement, where NOTES so far has only a bound
  T3  the training-rows-per-cell-month sweep, never run in any session
  T4  the full pipeline with whatever T3 decided, plus rounds.py's boosting-round correction
  T5  a ranked list of candidates for tomorrow

## The sweep is safe by construction
A layout may now carry its own PER_ROW in its name: `Ap3` builds `out/mats/Ap3_*.parquet` from
layout A's pseudo-test with per_row=3 and leaves `A_*` untouched. Verified -- after building Ap2,
`A_tr.parquet` was byte-identical and four new Ap2 files existed. That property is what makes it
safe to sweep a parameter that changes every matrix, on a machine holding a working submission.

perrow_scan.py scores each variant under the real test horizon mix and adopts one only if it wins
on EVERY layout swept. Verified on fixtures: a p3 that wins on both A and B is adopted; a p4 that
wins on A and loses on B is rejected with both numbers printed.

PER_ROW is now part of the build_ source fingerprint, so changing it invalidates the cached
matrices instead of silently mixing a p2 matrix with a p3 decision.

## Honest expectation
T1 costs nothing and tests something the fitted ensemble weight got wrong before. T2 closes or
reopens an avenue. T3 is a genuine unknown -- more pairs reuse the same observations, so they add
coverage of the (horizon, staleness) space rather than independent information, and the answer
could be zero. T4's round correction is the one with a clear mechanism: FINAL has been trained on
~30% more history than the layout whose early stopping set its round count.

None of it closes the gap to 0.65. The five avenues closed on measurement in REPORT.md section 4
still stand, and the dominant remaining error is still large-scale, spatially coherent and
temporally white.

# Session 10 — recursion is settled, the training-row sweep is a clean zero

Both were run overnight by `run_tonight.sh` on a day with no submissions left, and both bought an
ANSWER rather than another copy of a file that already exists.

## T2 — recursive forecasting: the measurement, and it agrees with the bound

`recursive.py` trains a horizon-1 model, chains it month by month through `build_mats.prepare`'s
own feature builder (49/41/45 monthly steps on A/B/C) and scores direct against recursive per
horizon. All three layouts, every horizon, the same direction:

              layout A            layout B            layout C
    h   direct  recur   r-d   direct  recur   r-d   direct  recur   r-d
    1   0.6232 0.6238 +0.0006 0.4974 0.4985 +0.0011 0.4666 0.4860 +0.0194
    2   0.5349 0.5658 +0.0309 0.5491 0.5985 +0.0494 0.5349 0.5697 +0.0348
    3   0.5484 0.5803 +0.0319 0.5544 0.6212 +0.0668 0.5390 0.5650 +0.0260
    4   0.7362 0.7967 +0.0605 0.5386 0.6314 +0.0929 0.5595 0.5763 +0.0168
    5   0.7974 0.8260 +0.0286 0.5652 0.6609 +0.0957 0.5914 0.6823 +0.0909
    6   0.9906 1.0182 +0.0276 0.5994 0.7270 +0.1275 0.6390 0.7222 +0.0832
    7   0.5059 0.6016 +0.0957 0.5397 0.6641 +0.1244 0.6158 0.7128 +0.0969

  test-mix RMSE   A  direct 0.6383  recursive 0.6657      B  0.5356 / 0.5909      C  0.5315 / 0.5692
  a 50/50 blend of the two is also worse than direct everywhere: 0.6434, 0.5523, 0.5364

Recursion loses at 21 of 21 (layout, horizon) cells, it loses by MORE as the horizon grows -- the
opposite of the only shape that would have justified building it out -- and even the free half-and-
half blend loses. The bound in session 9t argued this from arithmetic on numbers confounded with
calendar month; the measurement now says the same thing on real data with the confound removed, so
the avenue is closed on evidence rather than on an argument. REPORT.md section 4 carries the
measured numbers and no longer carries a [pending].

Worth being precise about WHY, because it is the same reason the bound gave: the h=1 model is not
materially better than the direct model at h=1 (0.6232 vs 0.6238 on A). Recursion therefore pays
the full h=1 error as an anchor and then compounds it, while the direct model gets to see the true
last observation. There is no error to save, only error to accumulate.

The one caveat, stated so the conclusion survives it: the chain's h=1 model is deliberately cheap
-- non-anchor features, a flat 300 rounds, no ramp weighting -- while `direct` is the chosen
configuration with its early-stopped round count. So the chain is handicapped at its own anchor,
which is most visible on C (+0.0194 at h=1). It does not matter. Grant recursion a PERFECT anchor,
equal to direct's own h=1 error, and the h>=2 gaps it would still have to overcome are +0.03 to
+0.13 -- one to two orders of magnitude larger than the handicap. The project's own tuned h=1
specialist is the same evidence from the other side: hsplice adopted it at beta=0.50, i.e. worth
half a vote at horizon 1, not the step change a chain would need to pay for seven of them.

## T3 — training rows per cell-month: a clean, well-measured zero

`build_mats` defaults to 3 training pairs per cell-month; the orchestrator has passed 2 since it
was written, and no session had ever checked which is right.

    test-mix RMSE     p2 (incumbent)      p3
      layout A            0.6383       0.6407   +0.0024
      layout B            0.5356       0.5362   +0.0006
    p3 does not clear 0.0003 on either layout -> PER_ROW=2 kept

p3 is worse on both, so this is not even a close call needing the every-layout rule to break it.
The mechanism was predicted in 9z and holds: extra pairs reuse the same observations, so they add
(horizon, staleness) coverage rather than independent information, and here they add correlated
rows that dilute the ones that matter. The sweep was safe by construction -- `Ap3` builds
`out/mats/Ap3_*.parquet` and leaves `A_*` untouched -- so a working submission was never at risk.

## T4 — the pipeline reran with the boosting-round correction

PER_ROW did not move, so every cached matrix, layout and configuration decision was correctly
reused (phases 1-5c all skipped, src_guard fingerprints unchanged). The one thing that DID change
is rounds.py's correction, and it changed cfg_guard's signature for all three FINAL tags, which
discarded 32 cached training steps each and retrained them -- which is exactly what that guard was
built for.

    FINAL trains on 138 history months against layout A's 111
    every family: power law REJECTED (fit 0-8%), plain ratio used, 1.24x
    lgb 330->410   lgbs 511->635   lgbm 411->511   xgb 185->230   cat 364->453

The power law being rejected for all five families is itself a result. Layout C has FEWER months
than A (75 vs 111) but early-stops LATER (361 vs 330 for lgb), so the round count is not a function
of training size alone -- layout difficulty and horizon mix move it too. Three points that do not
lie on a curve is precisely the case rounds.py was written to refuse to fit, and it fell back to
the conservative ratio, capped, never below the incumbent.

## A defect in the audit trail: cfg_guard's round list

    sig="dropf=$2 weights=$3 hmix=$4 fams=$(shift 4; echo "$@") seeds=$SEEDS"
    for m in "$@"; do sig="$sig r_$m=$(rd "$m")"; done

`shift 4` runs inside a command substitution, so it shifts a COPY of the positional parameters and
the caller's `$@` is untouched. The loop therefore walks the tag, dropf, weights and hmix as if
they were model families, and `rd` returns its 300 default for each:

    r__f1=300 r_bigsa=300 r_ramp=300 r_test=300 r_lgb=410 r_xgb=230
                                                ^^^^^^^^^^^^^^^^^^^ the only two that are real

The signature is still a deterministic function of its inputs, so the guard cannot MISS a change
and no run has been mistrained by this. What it corrupts is the log line that tells a reader what
changed and why 32 models were retrained -- and that line is the audit trail for the single most
expensive decision the orchestrator makes. Reproduced in a fixture, fixed by shifting in the
function body, verified to produce `r_lgb=410 r_xgb=230` and nothing else. Not edited while the
run was in flight: /bin/sh reads a script by byte offset as it executes, so editing a running
shell script can corrupt it mid-run.
# Session 10a -- ERA5 has never been in a single model

Chasing the 0.6495 leaderboard cluster, the first thing checked was the cheapest: whether the
covariates already provisioned are actually reaching the models. They are not.

`build_mats.py:52` reads ERA5 only `if glob.glob("external/era5/*.nc")`, and that directory does
not exist. Three independent confirmations:

  * all four `out/night/build_*.log` print `era5: None` (and `gdo: not present`);
  * `out/mats/feats.json` holds 201 features and not one of them starts with `e5`;
  * the run report's external inventory lists ncep, ncep2, cpc and oni -- 15 files, no ERA5.

So `features_era5.py`, `cds_download.py` and `ERA5_SETUP.md` have been dead code since session 8.
Every leaderboard number this project has ever posted was produced without ERA5.

What that costs is not a rounding error. The domain is 40x40 one-degree cells over tropical South
America (lat -19.5..19.5, lon -79.5..-40.5) -- the Amazon, the eastern Andes and the Nordeste.
Against that grid:

  * NCEP-R1/R2 is ~1.9-2.5 degrees, so a single reanalysis cell covers four to six target cells,
    and it carries latent heat flux rather than evaporation;
  * ERA5 is one degree and lands on the target grid exactly, and carries evaporation, runoff and
    a four-layer soil column (0-7, 7-28, 28-100, 100-289 cm) directly;
  * the family that produced the entire session-9 leaderboard gain was the per-cell standardised
    anomalies of exactly these fields. `features_anom.build` is already called for ERA5 at
    `build_mats.py:79`; it has simply always received `None`.

Two guards were wrong in a way that would have hidden the fix as well:

  * `src_guard build_` fingerprinted the feature SOURCE files only. Downloading ERA5 changes no
    source file, so the cached `A_tr.parquet` would have been reused and the new data would never
    have reached a model. The external files are now fingerprinted too (`$S/extinv.txt`, name and
    byte count of every .nc/.parquet/.data under external/), so arriving data invalidates the
    matrices exactly as edited code does.
  * `run_tonight.sh` had no ERA5 step at all. T0 now downloads it when `~/.cdsapirc` exists,
    and otherwise says plainly that it is absent and points at ERA5_SETUP.md, without failing
    the run.

Note for whoever adds variables later: snow is dead weight in this domain. `e5SWE` (from `sd`) and
NCEP's `weasd` are ~zero over a box that stops at 19.5N, so they cost features and buy nothing
outside a few Andean cells. The four `swvl` layers are currently collapsed into one `SW` column by
a fixed-thickness weighted sum; the profile shape (fast top layer against slow bottom layer) is
thrown away and is worth carrying separately once ERA5 is actually present.

## Recursive forecasting: measured, and it loses

`recursive.py` needs build_mats and therefore the external archives, so it had never been run.
`fastval.py` rebuilds the same framing from Train.csv alone and settles it in a minute:

```
placement          direct   recursive   delta
2010-08            0.3061     0.3716    +0.0655
2012-02            0.3076     0.3749    +0.0673
2013-08            0.3119     0.3874    +0.0755

per-horizon (2013-08)   h1     h2     h3     h4     h5     h6     h7
  direct               0.274  0.323  0.325  0.326  0.329  0.355  0.345
  recursive            0.270  0.358  0.413  0.463  0.484  0.531  0.544
```

h=1 is the same model in both columns, so the two agree there and the gap is pure compounding.
The bound in REPORT.md §4 said recursion inherits the one-step error as an anchor; the
measurement is worse than the bound. Closed.

Two calibration points fall out of the same harness, and they matter for reading every future
number. Under the test's horizon mix, on the real data:

  * persistence (last observed TWS) scores **1.166** -- at h>=2 it is worse than predicting
    nothing, because a stale anchor is anticorrelated with a standardised anomaly seven months
    later. The starter notebook calls it "a very strong baseline"; in this framing it is the
    weakest thing available.
  * per-cell per-calendar-month climatology scores **0.502**, and climatology plus the anchor's
    anomaly ("anomaly persistence") scores **0.407** -- three lines of arithmetic, no model.
  * the full pipeline scores **0.334** on layout A, and this 40-feature harness scores 0.306.

So validation sits near 0.31-0.33 while the leaderboard sits at 0.696 and its leader at 0.6495:
the ratio is about 2.1. The test window is genuinely harder than any window available for
validation -- it is the record 2015-16 El Nino Amazon drought plus the post-GRACE-gap months of
2018 -- and that is a level shift, not a broken pipeline. What transfers is the RATIO, which is
what session 9 already observed when validation predicted -0.0128/-0.0184 and the leaderboard
paid -0.0129. Read every candidate below as a relative change: closing 0.696 -> 0.6495 needs a
6.6% relative gain, which is about 0.022 in validation, and e8_bigsa alone was worth 0.0265.

## Four structural feature families, all rejected

Run in `fastval.py` on three block placements each, mean of the three, test horizon mix:

```
base                       0.3061 0.3076 0.3119   0.3085
+mem     12/24-month lags of the cell's own anomaly, plus a 24-month anomaly trend
                           0.3067 0.3075 0.3122   0.3088   (+0.0002)
+box     mean anomaly over 7x7 and 13x13 boxes at the anchor month
                           0.3069 0.3080 0.3127   0.3092   (+0.0006)
+dir     the same 13x13 box split into its west and east halves
                           0.3067 0.3079 0.3125   0.3090   (+0.0005)
+upcov   standardised soil moisture and SPEI_03 averaged over the cells to the west,
         at the row's own month
                           0.3065 0.3083 0.3112   0.3087   (+0.0001)
+box+mem                   0.3080 0.3076 0.3120   0.3092   (+0.0007)
all four                   0.3079 0.3087 0.3118   0.3095   (+0.0009)
```

Every one is worse than base, by an amount consistent with pure dilution rather than harm. The
motivations were sound and are recorded so they are not re-proposed:

  * **mem** -- groundwater in the Amazon has multi-year memory, so the anomaly a year or two
    before the anchor should carry information the anchor does not. It does not, on top of a
    per-cell per-calendar-month climatology, which already encodes the cell's slow state.
  * **box / dir** -- the residual is known to be large-scale and spatially coherent (variogram
    +0.967 at lag 1), and the pipeline's isotropic smoothed anchors at 300-2500 km were the
    single largest feature gain in the grid (e8_bigsa, -0.0265). Splitting a large box into
    upstream and downstream halves tests whether the coherence is *directional*, since the
    Amazon drains west to east. It is not: the west and east halves buy nothing over the
    isotropic mean, which is a real answer about the error's geometry.
  * **upcov** -- a cell's storage integrates rainfall over its whole upstream basin, so the
    forcing that matters is not the forcing in the cell. Averaging the released covariates
    upstream buys nothing either.

Read together with the anomaly result from session 9, this says the missing ingredient is not a
smarter transform of the released columns -- four independent attempts at one found nothing --
but a forcing field the model has never seen. Which is exactly what ERA5 is (session 10a).
Caveat: four families is suggestive of saturation, not proof of it, and all four were tested in
the 40-feature harness rather than on top of the pipeline's 201.

# Session 10b — the ERA5 contradiction, resolved the way this project resolves things

Session 10a and session 10 ran in parallel on different machines and reached opposite conclusions
about the same question, which is worth resolving in writing rather than letting the next reader
find two contradictory sections.

10a: "ERA5 has never been in a single model. `external/era5` does not exist, all four build logs
print `era5: None`, feats.json holds 201 features and none starts with `e5`."

That is true of the container it was written on. It is false of the machine that produces the
submissions, and the artefacts settle it — not a reading of the code, which is exactly the
distinction 10a's own last paragraph draws:

    external/era5/                       19 files, downloaded 09-05 07:31-07:48
    out/night/build_{A,B,C,FINAL}.log    era5: (14774400, 10)   -- all four
    out/mats/feats.json                  266 features, 65 of them ERA5
    out/mats/used_FINAL_*.json           n=261, e5=21, an_e5=44  -- every tag trained tonight
    out/night/compliance.log             lists the 19 ERA5 files in the external inventory

`used_*.json` is written by run_models.py per run for precisely this purpose: it records what a
model was actually given, so no one has to infer it from a glob and a log from another day. Every
FINAL model behind tonight's submissions carries 21 raw ERA5 features and 44 ERA5 anomaly features.

Nothing in 10a is wasted. Its two guards are real fixes and are merged: `src_guard build_` now
fingerprints the external archives as well as the source files -- downloading data changes no .py
file, so without it an arriving covariate would have been silently ignored by every cached matrix
-- and `run_tonight.sh` grows a T0 that downloads ERA5 when `~/.cdsapirc` exists. Both would have
mattered on this machine too, four days ago.

## The box download is redundant here, and should not be moved into external/era5

`~/era5_dl` holds 19 box-cut files (24.5N/-84.5W/-24.5S/-35.5E). `external/era5` already holds 19
global files covering the same box, and those are what every current matrix was built from.
`load_era5` globs the directory, concatenates and then `.unique(["lat","lon","time"])`, so mixing
the two sets would not duplicate rows -- but it would make which file supplied a cell-month
arbitrary, and it would change `extinv.txt`, invalidating every cached matrix and forcing a full
rebuild and retrain for no gain: the two products are the same fields on the same grid.

Leave them where they are. The box cut is the right request for a machine that has to download
ERA5 fresh (a twentieth of the bytes), and it is what `cds_download.py` now asks for; it is not a
reason to disturb data that is already on disk and already in the models.

# Session 10c — the fast harness was measuring the wrong thing, and the residual anchor is a no

## fastval's placements do not survive on this Train.csv

Before spending the harness on a new idea, it was run on the incumbent settings to see whether it
reproduced its own recorded numbers. It does not, and the reason is structural.

`fastval` counts a horizon in INDEX steps of the months present in Train.csv: `h = ti - ki + 1`,
the covariate window accumulates over `ki+1 .. ti`, and the answer is `T[:, ti+1]`. GRACE is
missing 12 months of this record, so where a gap falls, one index step is two or three calendar
months and a block laid across it asks for a multi-month lead while labelling it h=1.

    record on this machine   149 months, 2002-05 .. 2015-09, 12 missing
    start=100 (2010-12)      5 calendar months missing inside the 25-month window
    start=118 (2012-09)      5 missing
    start=136                reaches index 160, PAST THE END of a 149-month record

All three defaults. And nothing measured on them reproduces here:

                        recorded (10a)      this Train.csv
    persistence             1.166               0.711
    climatology             0.502               1.019
    start=100 is            2010-08             2010-12
    direct                  0.306               0.736

The month index alone settles where those numbers came from: for MON[100] to be 2010-08 the record
has to start 2002-04 and have no gaps, i.e. 161 contiguous months. This one has 149. They were
produced on a different Train.csv, not on this one -- which also explains why 10a read persistence
as the weakest baseline available when REPORT §1, the leaderboard's own 0.886 persistence line and
this harness on real data all say it is the strongest simple one.

The edits were ruled out first, not assumed innocent: the unmodified file from 20d46bd was run on
start=100 and produced byte-identical output to the patched one.

This is layout C's bug in a second place. `validation_c.py` searches for a placement the record can
carry gap-free because a hardcoded offset silently truncated blocks; `fastval` now does the same --
`check()` refuses a placement that straddles a gap or runs off the end and prints the ones that
work (starts 12..77, 2003-08..2009-01), and training rows are held to the same standard. Verified:
both old defaults are now refused with the reason and the alternatives.

One limitation to state plainly: every gap-free window on this record lies in 2003-2009, so the
harness cannot validate anything on the 2015-16 El Nino era that the test actually covers.

## The residual anchor: measured, and rejected

The idea was the best of the untested ones -- the model predicts `target - tws_known`, and under
the test's horizon mix `tws_known` is a stale anchor, so the model spends capacity undoing it.
Two replacements were tried, each trained on identical rows and scored on an identical mask so
the anchor is the only difference:

    ap   clim_next + (tws_known - clim_known)                      raw departure carried forward
    apz  clim_next + z(tws_known) * sd_next                        the same in standardised units

                   p1 (2003-12)   p2 (2006-04)   p3 (2008-08)
    tws (incumbent)    0.7348         0.6312         0.6189
    ap                 0.7294         0.6260         0.6205
                      -0.0055        -0.0052        +0.0016      REJECTED: loses on p3
    apz                0.7504         0.6426         0.6332
                      +0.0156        +0.0114        +0.0143      REJECTED: loses everywhere

`ap` is the interesting failure. It wins by 0.005 on two placements out of three, which is ten
times the adoption threshold, and then loses on the third. The standing rule -- win on every
layout -- rejects it, and the rule is right here: a change that is worth 0.005 twice and -0.002
once is era-dependent, and the era the test covers (2015-16) is not one the harness can reach.

`apz` is a clean loss and the reason is worth keeping: standardising divides by the anchor month's
sd and multiplies by the target month's, so it carries the ratio of two noisy per-cell estimates
into every prediction. The raw departure has no such term.

Neither is dead, but neither is adoptable on this evidence. The cheap next measurement, if anyone
returns to it, is `ap` inside the real pipeline rather than the 40-feature harness -- where
`anom_persist` is already the single highest-gain feature (9.4% on layout A), which is itself
evidence that the pipeline has largely found this signal through the feature rather than the anchor.

# Session 10d — the leaderboard: a new best, and the blend question answered against the blend

Three files scored:

    sub_q_main    0.692657311   NEW BEST   (previous best 0.695965, -0.003308)
    sub_blend73   0.693283      +0.000626 vs main   (0.7 main + 0.3 alt)
    sub_blend55   0.693725      +0.001068 vs main   (0.5 main + 0.5 alt)

## The blend is refuted, and stack.py was right all along

Session 9x doubted the fitted ensemble weight: stack.py's NNLS gave the 31-leaf model a weight of
**0.000**, while the leaderboard said that model alone scored 0.698048, within 0.002 of the best.
"Two points apart is not adds-nothing" was the argument, and `blend_subs.py` was written to spend
a submission settling it rather than trusting the fit.

It is settled, and the fit wins. The response is monotone in the alt weight:

    alt weight   0.0        0.3        0.5
    score        0.692657   0.693283   0.693725
    delta        --         +0.000626  +0.001068

Both gaps clear the readability threshold for these files (RMS difference 0.0057 and 0.0095 -> a
gap under about 0.00005 and 0.00007 would be noise), so this is a real, ordered effect and not a
coin flip. Every unit of the alt model makes the file worse in proportion.

The lesson is not "trust validation" -- validation adopted the per-horizon calibration the
leaderboard then refuted. It is narrower and more useful: **a fitted zero from a non-negative least
squares over test-mix-weighted rows is a measurement, not an artifact of the optimiser**, and the
intuition that a model scoring well ALONE must add something to an ensemble is simply wrong when
that model is correlated with what is already in it. sub_q_alt shares the feature set, the training
matrix and the anchor with the main model; only its capacity differs. It is not a different bet.

## What produced the 0.003308

Three changes landed together in the run that made this file and cannot be separated without
spending submissions on the ablation:

  * the layout-C repair changed the chosen configuration, `DROPF='scale,bigsa'` -> `'bigsa'`,
    so the zonal features are kept -- the broken C had voted them out;
  * the h=1 specialist was adopted at beta=0.50, which hsplice had rejected every time the broken
    layout C was in the vote;
  * rounds.py's correction for FINAL training on 138 history months against layout A's 111:
    lgb 330->410, xgb 185->230.

All three trace back to session 9's finding that a validation layout was silently broken. The
honest summary is that repairing the measurement was worth 0.0033 on the board, which is roughly a
quarter of what the covariate-anomaly encoding was worth and was bought with no new information.

## Private slots
`sub_q_main` and `sub_blend73` are ticked. That is defensible even though blend73 is 0.0006 worse
publicly: the two files differ by an RMS of 0.0057, blending two capacities is the classic
variance-reduction hedge, and the public set is ~84k rows against a private set that is scored
separately. Nothing else on hand is both competitive and genuinely different in kind -- sub_q_base
is the pre-session-9 feature set and scored 0.708852.

# Session 10e — the dead features at h=1 are harmless, and the reason is worth knowing

The backlog's idea 4: at horizon 1 the accumulation window (t_known, t] is empty, so the `_acc`
features are null and the `_d` differences are exactly zero. Give the h=1 specialist its own
feature list with those removed.

The premise is real, and larger than the estimate. Measured on `A_tr.parquet` directly rather
than assumed: at h=1, **31 features are entirely null and 33 are identically zero — 64 of 266**,
a quarter of the set, on the slice that is 33.3% of the test by weight. With the anchor block it
is 65 of 261 in the chosen configuration.

`DEADF=1` in run_models.py drops any feature that is constant on the rows the run actually trains
on. It is computed from the data, not from a stored list, so it cannot go stale when the feature
set changes, and it is safe by construction -- a column with one distinct value has no information
to lose. Control and treatment, lgb, `DROPF=bigsa`, two seeds on each layout:

    layout  seed   control  DEADF=1     delta
    A          0    0.6617   0.6619   +0.0002
    A          1    0.6630   0.6629   -0.0001
    B          0    0.5859   0.5865   +0.0006
    B          1    0.5891   0.5859   -0.0032
    C          0    0.6066   0.6113   +0.0047
    C          1    0.6086   0.6065   -0.0021

    mean effect over six pairs   +0.0000
    sd of the effect              0.0027
    seed-to-seed spread of the control within a layout   0.0022

The effect is exactly zero and its spread is larger than itself, and larger than the adoption
threshold by an order of magnitude. **Not adopted.**

## Why the plausible mechanism does not bite

The argument was that `feature_fraction=0.6` offers the sampler ~39 dead candidates at every split,
so a quarter of its draws are wasted. The arithmetic is right and the conclusion does not follow.
With 261 features of which 65 are dead, a 0.6 draw yields ~157 candidates of which ~118 are live.
With the 196 live features alone, a 0.6 draw yields ~118 candidates, all live. **The number of live
candidates per split is the same either way.** Dropping the dead columns does not give the tree
more to choose from; it only changes the effective subsampling rate of the live pool, from 118/196
to 118/196 -- which is to say, it changes nothing that matters, and what is left is the seed noise
the table shows. A constant column cannot be split on, so it is not a competitor for a split, only
a name the sampler passes over.

This is the same shape of error as the raw-millimetres correlation and the recursion bound: an
argument that sounds mechanical, is arithmetically correct in its premise, and does not survive
being run.

## What it is worth: 25% of the training time
Six matched pairs, 432 s of control against 323 s. That is real and it is free, since accuracy is
unchanged, and it applies to every specialist run. Recorded under §6.4 rather than as an accuracy
change, and left OFF by default so the submitted configuration is the one that was gated.
