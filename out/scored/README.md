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
