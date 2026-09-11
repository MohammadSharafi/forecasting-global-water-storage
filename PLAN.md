# Plan: public leaderboard below 0.65, legally, without hurting the private 70%

Depth: tree 5   Mode: orchestrated
Budget note: realistic single pass is research (~1 h) + a shared harness (~1 h) + ~1.5 h of
sequential training per trained candidate on five layouts + handover (~1 h). The binding
constraint is not time but compute: ONE training process at a time on a 16 GB machine.

## Contract

### Rules (breaking any = goal failed, whatever the score)
- R1 no lat/lon as features. R2 nothing at or after t+1 (no t+1 covariates, no seasonal
  forecasts, no smoothing along the horizon). R3 neighbour TWS at <= t is allowed.
  R4 external covariates only with source date <= t. R5 no GRACE or GRACE-derived product of any
  kind, including land-surface models that assimilate GRACE (e.g. GLDAS-2.2 DA) and GDO's TWS layer.
- Never fit weights or choose configs on the public leaderboard; validation labels only.
- No dataset download without the user's explicit approval in chat (the connection is poor).
  Research may look up metadata (URLs, licences, dates) but must not fetch data files.

### Closed avenues (do not retry without a written "New reason:" the old measurement does not apply)
public-board blending, post-processing tweaks (smoothing, calibration, splice), recursive
forecasting, EOF/mode forecasting, U-Net, per-cell reliability, ensemble-disagreement weighting,
analogue/ENSO weights, variance-weighted loss, directional neighbourhoods, second-stage residual
model, groundwater memory via fastval.py.

### Environment
- Python: ./.venv/bin/python. Layouts: Avn2 Bvn2 Cvn2 D E; final matrix FINALvn2. Never bare A/B/C
  (269-column stale builds; run_models.py fails fast on them).
- Every run_models.py call exports PER_ROW=2 and CARBON=1.
- Shipped control configuration, seeds 0 and 1:
  `DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test ./.venv/bin/python run_models.py <L> lgb v5x_noll_sa`
  Training is bit-deterministic (max|a0-d0| = 0.000e+00), so existing `_d0` predictions ARE the
  control for any candidate whose only change is an additive side-car switch.

### Measurement
- Score per layout: test-mix RMSE (eval_mix.py weighting h1..h7 = .3333 .2222 .1667 .1111 .0556
  .0556 .0556), rows with horizon > 7 ignored. delta = treatment - control; negative is better.
- Trained candidates: control and treatment differ ONLY in the candidate's switch, same seeds,
  each layout measured independently. Linear-correction candidates: weights fitted on four
  layouts, scored on the fifth (as ar_blend.py).
- Early stop: once two layouts have delta > 0, stop the candidate; verdict ABANDON-EARLY.
- Provenance (trained): feature counts from out/mats/used_<L>_lgb_v5x_noll*<tag>.json;
  treatment minus control must equal the number of columns the candidate adds.
- Summary line, exact format, one line:
  `<cand> | Avn2 <d> Bvn2 <d> Cvn2 <d> D <d> E <d> | wins <n>/5 | mean <d> | worst <d> | clears-0.003 <n>/5 | <VERDICT>`
  deltas printed %+.4f; unmeasured layouts printed `n/a`.
- VERDICT: ADOPT-0.65 iff wins 5/5 and mean <= -0.041 and every layout <= -0.003.
  HELPS-NOT-0.65 iff wins 5/5 and every layout <= -0.0003 but not ADOPT-0.65.
  ABANDON-EARLY as above. REJECT otherwise.

### Data ownership (no two leaves share a file)
- 1.1 research: RESEARCH.md only.
- 1.2.1 harness: goal065_eval.py (new) and the ONLY edit to run_models.py: a generic additive switch
  `XF=<name>[,<name>]` that appends columns from out/mats/{L}_{part}_<name>.parquet.
- each candidate leaf <cand>: goal065_<cand>.py, out/mats/{L}_{tr,va}_<cand>.parquet,
  out/goal065/<cand>.txt (full per-layout log; last line = the summary line). Never run_models.py.
- driver only: out/goal065/heldout.txt (append-only: one summary line per finished candidate, and
  the ADOPTED SET line last), PLAN.md status log, gates/*.md checkboxes.
- 1.3 handover: out/sub_goal065.csv, the Session 12 section of NOTES.md, REPORT.md §4 and §8.

### Known issue awaiting the user
SampleSubmission.csv in the repo belongs to a different competition (header GEOID,
coverage_gap_score,...), so top-level G4 cannot pass as written. G4 is NOT edited; the user decides.

## Tree

- 1 goal 0.65 ........................................ GATES.md
  - 1.1 research ..................................... gates/leaf-1.1-research.md
  - 1.2 measure (branch) ............................. gates/node-1.2.md
    - 1.2.1 shared harness + XF switch ............... gates/leaf-1.2.1-harness.md
    - 1.2.2 AR family already in flight (branch) ..... gates/node-1.2.2.md
      - 1.2.2.1 ARF: AR rescaling as features ........ gates/leaf-1.2.2.1-arf.md
      - 1.2.2.2 seasonal AR / AR(2) corrections ...... gates/leaf-1.2.2.2-arext.md
    - 1.2.3 new candidates from RESEARCH.md (branch) . gates/node-1.2.3.md
      - 1.2.3.k one leaf per candidate, best expected first, gates written once research ranks them
  - 1.3 handover (branch) ............................ gates/node-1.3.md
    - 1.3.1 joint adopted-set measurement + build .... gates/leaf-1.3.1-build.md
    - 1.3.2 NOTES Session 12 + REPORT §4/§8 .......... gates/leaf-1.3.2-record.md

## Status log

- step 0: GATES.md written before any work; plan written, contract fixed. ARF (artest.sh) and
  seasonal AR/AR(2) (ar_ext.py) were already running when the goal arrived; folded in as 1.2.2.
- step 0: found SampleSubmission.csv is from another competition; G4 flagged to the user, not edited.
- step 1: dispatched 1.1 research as a fresh subagent (owns RESEARCH.md only, no downloads).
- step 1: 1.2.2.2 measured (ar_ext.py, budget 2.0, LOO): seasonal AR mean +0.0001 wins 0/5 REJECT; AR(2) mean +0.0000 wins 2/5 REJECT, and structurally unavailable (prior month observed for 0-6% of rows; 1 of 6 test anchors). Two lines appended to heldout.txt.
- step 2: CORRECTION to step 1: ar2's line said wins 0/5 because -0.0000 parsed as -0.0 and was not counted; ar_ext.py counted wins on unrounded deltas = 2/5. heldout.txt and out/goal065/ar2.txt corrected to the measurement. Verdict unchanged (REJECT: every delta rounds to 0.0000, worst +0.0001).
- step 3: gate-check.mjs bug found: without --timeout, `i !== tIdx + 1` is `i !== 0`, so the FIRST file argument is dropped and every gates file is processed. It prematurely ticked GATES.md G3 (compliance, before any final artefact exists), research L8 (research still running) and node-1.2 N2. All three reset to pending, plus arext L4 whose evidence quoted the pre-correction wins count. Workaround for every invocation: pass `--timeout N` first. Skill script left unmodified; flagged to the user.
- step 4: dispatched 1.2.1 harness as a fresh subagent (owns goal065_eval.py, goal065_run.sh, the single XF edit to run_models.py; no training or run_models.py edit until ARF finishes). Handover gates 1.3.1 and 1.3.2 written BEFORE any candidate result exists, so they cannot be shaped around the outcome. 1.2.2.2 committed.
