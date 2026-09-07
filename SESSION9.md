# Session 9 — where the remaining score actually is

No competition data was available in this environment (`out/`, `external/` and the CSVs are
gitignored), so nothing here is a measured result. Everything below is either an analysis of
the leaderboard record already in `NOTES.md`, or code that is ready to run and has been
smoke-tested on synthetic data with the same structure. Run order and expected value are at
the bottom.

## 1. Two facts about the competition worth acting on

- **The leaderboard RMSE is only 50% of the final score.** Trustworthiness is 30% and
  Innovation & Practicality 20%, judged by rubric for the top 10. `REPORT.md` already covers
  both, and the compliance record (two rulings applied at a cost in score) is a genuine asset
  in that half. A 0.002 RMSE gain is worth less than one weak section of the report.
- **5 submissions/day, 200 overall.** Probes are not scarce; the constraint is calendar days
  left, not submission budget. Spend them on probes that separate hypotheses, not on
  re-uploading near-identical blends.
- **Close: 13 September 21:59; code review of the top TWENTY; results by 4 October.** The
  review threshold moved from 10 to 20 (organiser email, 7 Sep), so the odds that this entry
  is code-reviewed went up substantially. Every compliance decision in §4 of `REPORT.md` is
  now more likely to be read, and the repository itself has to be runnable by a stranger.

I could not read the competition page or the discussion board directly: `zindi.world`,
`zindi.africa` and `qiita.com` are all blocked by this session's egress policy. The 50/30/20
split and the submission limits come from search-engine extracts of the competition page; the
close date and the top-20 review come from the organiser email.

## 2. Reading the leaderboard record properly

For two submissions scored on the same public rows, the standard error of the observed RMSE
gap reduces to a one-liner (derivation in `lb_se.py`):

    SE(gap) ~= RMS(pB - pA) / sqrt(N_public),    N_public ~= 84,288,  sqrt(N) ~= 290

Applying it to this project's own numbers, using the prediction distances recorded in
`NOTES.md`:

| comparison | RMS(diff) | gap | in SE units | verdict |
|---|---|---|---|---|
| recent-anchor stack (0.7191) vs long-term (0.7110) | ~0.11 | 0.0081 | ~21 | **certain** |
| 50/50 blend (0.71159) vs long-term (0.7110) | ~0.055 | 0.0006 | ~3 | probably real |
| `sub_h_l7_rb` (0.71133) vs `sub_g_longterm` (0.71096) | ~0.04 | 0.00037 | ~2.7 | marginal |

Two consequences:

- The **anchor-philosophy axis is the only large effect anyone has found on this test set**
  — 0.008, against ±0.001 for everything else since session 2. It is worth pushing further
  along that axis (§3.2) rather than hunting for new features.
- The current two private-leaderboard selections (`sub_h_l7_rb` at w_long=0.7 and
  `sub_g_longterm` at w_long=1.0) are separated by ~2.7 SE. That is a reasonable
  regime hedge, but the two files are ~97% the same prediction, so the second slot is
  buying very little. Advice in §5.

Run `python lb_se.py out/subA.csv out/subB.csv 0.711 <gap>` before believing any future
sub-0.001 leaderboard difference.

## 3. What is untried, ranked

### 3.1 The post-processing operator is the wrong shape (highest value, zero retraining)

`final_assemble.py` ends with `smooth(va, p, w=0.7, radius=1)`. That operator has two defects:

1. **It is a grid box, not a physical radius.** At 60°N a 1° longitude step is 55 km and at
   the equator 111 km, so how much smoothing a cell gets depends on its latitude. Session 7
   established that the scale that matters here is 300–500 km — the smoothed anchor as a bare
   persistence forecast went 0.757 → 0.724 (300 km) → 0.717 (500 km) on layout A.
2. **It smooths the residual `p - tws_known`, so the entire fine-scale noise of `tws_known`
   is still carried into every prediction.** Session 7 measured that only ~20% of a cell's
   departure from its 300 km neighbourhood survives into the next month. The post-hoc re-base
   `p + a*(sa300 - tws_known)` was a partial fix for exactly this — and `a` was pinned at 0.3
   and never scanned, at one radius, even though the same session's own numbers say 500 km
   beats 300 km and that the surviving fraction is nearer 0.2 than 0.7.

`postproc2.py` replaces both with one operator applied to the **prediction field** of each
target month:

    p_out = M_r(p) + beta * (p - M_r(p))

`M_r` is the great-circle mean already used for the anchor features (longitude window scaled
by 1/cos(lat)), `beta` is the fraction of fine-scale content worth keeping. `beta` has a
closed form — `beta* = sum(d*(y-s)) / sum(d*d)` with `s = M_r(p)`, `d = p - s` — so there is
no grid search, though a grid is printed so the curvature is visible. It subsumes both the
grid smoothing and the re-base, and it is scanned over radius on **both** validation layouts,
which the re-base never was.

Compliance is unchanged: the operator mixes predictions of the *same target month across
space only*, never across horizons. Mixing across horizons is what made the session-6
trajectory smoothing non-causal, and it is not reintroduced here.

Smoke test (synthetic smooth field + independent white noise on target and prediction, so the
right answer is known): the closed form matched the grid minimum at every radius and drove
RMSE 0.4252 → 0.3198 against a 0.3002 oracle, recovering `beta* ≈ 0.05`. The plumbing —
row alignment, per-month grouping, the kernel — is correct.

Expect the real gain to be much smaller than the synthetic one, because the models already
carry `sa300/500/800` and `dsa*` as features and have learned part of this. But the scan is
minutes of compute and settles the question with a number instead of a guess.

### 3.2 Train against a climatology anchor, not only a persistence anchor

The leaderboard has said twice that this test period rewards long-term anchors over recent
ones. So far that has only ever been expressed through the **feature set** (`v5x` = LONGTERM
in, RECENT out). But what a model is trained *against* controls its shrinkage far more
directly than any feature can: with `y = target - tws_known`, a tree that predicts 0 returns
persistence; with `y = target - clim_next`, a tree that predicts 0 returns climatology.

`run_models.py` now takes `ANCHOR_TARGET`:

- `tws` (default) — unchanged, `y = target - tws_known`
- `clim` — `y = target - clim_next`
- `decay` — `y = target - (lam*tws_known + (1-lam)*clim_next)`, `lam = ANCHOR_RHO**horizon`

`decay` is the one to try first (`ANCHOR_RHO=0.85` → λ = 0.85 at h=1 down to 0.32 at h=7). It
leaves h=1 — 33% of test rows — almost untouched while pulling long horizons toward the
long-term normal, which is the direction the test period has twice rewarded. `clim` is the
aggressive end and may well lose at h=1; run it only if `decay` wins.

`clim_next` is leave-one-year-out for training rows (`features2.assemble2`), so this
introduces no leakage, and it is already a column in every cached matrix — no rebuild needed.

This is a genuinely new axis of ensemble diversity, aimed at the one axis with a measured
0.008 of leaderboard signal, and it costs one LightGBM run per layout to test.

### 3.3 Free variance reduction: more seeds

The long-term stack is 12 models (lgb 3, xgb 3, cat 2, mlp 4). The MLP is a single one-cycle
epoch and is the highest-variance member at weight 0.25. Going to 8–10 MLP seeds and 5 tree
seeds costs only compute, cannot make the expected score worse, and is worth an estimated
0.001–0.002 of ensemble noise. This is the only item on the list with no downside risk.

### 3.4 One methodological bug worth knowing about

In the MLP branch of `run_models.py`, `pbest` is selected by validation RMSE using `yv`. Any
layout-A/B MLP number from a run with `EPOCHS>1` is therefore optimistically biased, and the
blend weights derived from those numbers — notably the 0.7 MLP weight in the recent-anchor
stack — rest on an inflated estimate. In practice the pipelines mostly use `EPOCHS=1`, where
the loop runs once and no selection happens, so the damage is probably small and the affected
stack is the one being dropped. I have left the behaviour alone rather than change results
mid-competition, but the report should not quote an `EPOCHS>1` MLP validation figure.

### 3.5 Deprioritised, with reasons

- **Gap-month target structure.** Reconstructing a gap-filled target from its neighbours needs
  the months *after* t+1, which is prohibited. The legal half (previous real month only) is
  what the model already does, and `sub_s2_v2_fillshrink` tested the shrinkage version and
  lost (0.7217 vs 0.7191).
- **ERA5-Land instead of ERA5.** Higher resolution, better land hydrology — but session 5's
  E2 test found next-month water balance explains R²≈0.006 of the monthly TWS change, and
  ERA5 as a whole bought 0.0016 on validation and nothing on the leaderboard. Neither ERA5
  nor ERA5-Land carries groundwater, which is what the residual variance mostly is.
- **Recursive forecasting** (permitted by the organisers): direct multi-horizon with `horizon`
  as a feature is normally the stronger of the two, and feeding a noisy h=1 prediction forward
  compounds error over seven steps. Not worth the days.
- **Seasonal forecast covariates (C3S SEAS5).** Ruled out by the same E2 result.

## 4. Schedule to 13 September

Six days, ~30 submissions available, and the binding constraint is the machine: the memory
rule is one heavy process at a time, and a FINAL 12-model retrain is hours, not minutes. So
the ordering below front-loads everything that needs no retraining and puts the one retrain
early enough that its leaderboard answer arrives with days to spare.

| day | compute | uploads |
|---|---|---|
| **Mon 7** | `postproc2.py` scan on A and B (minutes). Then start `ANCHOR_TARGET=decay` lgb on A, then B, sequentially. Extra long-term-stack seeds overnight. | `sub_k_gc` at the (radius, beta) that wins both layouts, plus one more conservative beta to bracket it — 2 probes that isolate §3.1 |
| **Tue 8** | If `decay` beat `lgb v5x_noll_sa` (A 0.6463 / B 0.5630) on **both** layouts, launch the FINAL 12-model train with `TAG=_ct`. | the extra-seed rebuild of the best recipe |
| **Wed 9** | finish the `_ct` train | `_ct` stack alone, and 50/50 with the long-term stack — these two separate "is the climatology anchor better" from "does it blend well" |
| **Thu 10** | — | interpolate the blend weight from Wed's two results; 2–3 probes, each checked with `lb_se.py` before believing it |
| **Fri 11** | last new ideas; then **freeze** the feature and model set | 1–2 confirmations |
| **Sat 12** | final maximum-seed rebuild of the chosen recipe; re-run CodeCarbon on it; update `REPORT.md` §3 and §5.4 and `NOTES.md` | the final file — upload it and check it scores where you expect |
| **Sun 13** | nothing new | pick the two private-leaderboard entries with hours to spare |

Freeze on Friday rather than Saturday if anything slips. A recipe that is 0.001 better but
whose report and code are rushed is a bad trade when the leaderboard is half the score and
the top twenty get read.

## 4b. Commands

```sh
# 1. post-processing scan, both layouts, no retraining (minutes)
python postproc2.py A lgb_v5x_noll:0.3:_sa xgb_v5x_noll:0.3:_sa cat_v5x_noll:0.15:_sa mlp_v5x_noll:0.25:_e1sa
python postproc2.py B ...same...
#    -> pick (radius, beta) that wins on BOTH layouts, as for every other adopted change
RADIUS=500 BETA=0.55 python final_assemble2.py sub_k_gc \
    lgb_v5x_noll:0.3 xgb_v5x_noll:0.3 cat_v5x_noll:0.15 mlp_v5x_noll:0.25
#    upload: this is sub_g_longterm with a better-shaped final filter, so it isolates §3.1

# 2. climatology-anchored target, validation only first (one lgb per layout)
ANCHOR_TARGET=decay ANCHOR_RHO=0.85 TAG=_ct python run_models.py A lgb v5x_noll_sa
ANCHOR_TARGET=decay ANCHOR_RHO=0.85 TAG=_ct python run_models.py B lgb v5x_noll_sa
#    adopt only if it beats lgb v5x_noll_sa (A 0.6463 / B 0.5630) on BOTH layouts,
#    then train the FINAL family with TAG=_ct and blend it in as a third stack

# 3. more seeds on the long-term stack (no leaderboard risk, run whenever the machine is free)
```

Exact prediction-file stems and tags depend on what is on disk — `ls out/mats/pred_A_*` will
show them; the spec syntax is `stem:weight[:tag]`, matching `final_assemble.py`.

## 5. Private-leaderboard selection

Assuming the 30/70 split is by row (Zindi's default), the public ordering transfers well and
the large-gap conclusions hold. The current picks are defensible but redundant — the two files
are ~97% identical, so the hedge buys little.

- **Slot 1:** the best compliant public score. Today `sub_g_longterm` (0.71096); replace it if
  §3.1 or §3.2 produces something better on the public leaderboard by more than `lb_se.py`'s
  threshold for that pair.
- **Slot 2:** spend it on something *different in kind*, not on a neighbouring blend weight.
  Either the §3.1 post-processed file or the §3.2 climatology-anchored blend. If neither pans
  out, keeping `sub_h_l7_rb` (w_long=0.7) as a regime hedge is reasonable — it protects
  against the split being by block rather than by row, which is the one scenario where the
  public ordering could mislead.

Do **not** select `sub_s4_blend_v5s_era5` (0.71068) even though it is the best public score on
record: it uses lat/lon as features and the removed trajectory smoothing, both explicitly
ruled out. Top-10 entries get a code review, and RMSE is only half the score.
