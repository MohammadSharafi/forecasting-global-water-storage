# Scored submission files, preserved

`run_night.sh` phase 9 rewrites out/sub_q_main.csv, sub_q_alt.csv and sub_q_base.csv from
whatever models are current, so a file that already carries a public score must be kept
somewhere the pipeline does not touch.

| file | public score | what it is |
|---|---|---|
| sub_q_main.csv | **0.692657** | best. 261 features, corrected round counts, beta=0.50 horizon-1 splice |
| sub_blend73.csv | 0.693283 | 0.7 main + 0.3 alt |
| sub_blend55.csv | 0.693725 | 0.5 main + 0.5 alt |
| sub_q_alt.csv | not scored in this form | second capacity (an earlier build of it scored 0.698048) |
| sub_q_base.csv | not scored in this form | pre-session-9 feature set (an earlier build scored 0.708852) |
| sub_q_main_nosm.csv | not scored | no smoothing, control |

Copied 2026-09-09 before rebuilding with the ERA5 soil profile and GDO SPI.
| sub_x_lb2.csv | **0.683712087** | L1<=2 blend over the 16-file ledger (lb_blend.py). Predicted 0.683244; the 0.00047 miss is the D^2(all rows) vs D^2(public 30%) proxy error |
| sub_y_lb.csv | **0.681717090** | L1<=4 refit. Predicted 0.676475 -- the oracle over-shot by 0.0052, transfer only 28% |
| probe_clim.csv | 1.279955747 | pure climatology probe; a dimension, not a model |
| sub_z_lb.csv | **0.681692450** | L1<=2 refit over 19 files. Predicted 0.677391; transfer 0.6% -- the oracle is exhausted |
| sub_goal065.csv | **0.68284693** | sub_x_lb2 + the marginal AR correction. -0.000865 against its base, an 18% transfer of -0.0048 validation. Built on the private-safer base, so it does not beat sub_z_lb's display |
