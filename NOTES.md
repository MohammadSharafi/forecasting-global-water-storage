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
