# Experiment log (validation layout A unless stated; persistence = 0.757)

| id | change | RMSE | verdict |
|---|---|---|---|
| v1 | LGB, level target, in-sample climatology | 0.712 | train/val mismatch (early stop @37) |
| v2 | LOYO climatology, neighbours, slope; residual target | 0.669 | keep residual target |
| v3 | + per-cell covariate proxies, drop gap-target rows | 0.672 | no gain, dropped |
| B  | regularised params / huber | 0.669–0.671 | plateau -> features, not params |
| v4 | + AR lags 1/2/3/6/12, 24-mo trend/dev | **0.661** (layout B 0.593, −13.7%) | keep |
| v5 | + 5x5 neighbourhood means of dynamic features | **0.6555** (best_iter 201) | keep; wide SPEI-6/12 deltas rank 3rd/4th |
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
