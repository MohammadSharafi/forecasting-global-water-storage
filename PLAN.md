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
- step 5: 1.2.2.1 ARF partial: Avn2 scored with eval_mix.py (harness not ready): shipped 0.6343 (identical to d0 in Session 11g, as bit-identity requires), +AR features 0.6334, delta -0.0009 -> not an early stop. For contrast the post-hoc linear form was -0.0085 on Avn2: the model recovers ~a tenth of it as features. Control arms bit-identical to d0 on Avn2 s0/s1, Bvn2 s0/s1; feature counts 340 control / 343 treatment in all 7 finished logs.
- step 6: rebuild recipe VERIFIED for leaf 1.3.1 — re-assembling from the prediction files on disk reproduces out/sub_v_anwide.csv exactly (max|diff| 0.00e+00, 280961 rows): FLAYOUT=FINALvn2 TAG=_w1, lgb_v5x_noll:0.5 xgb_v5x_noll:0.5 (16 seeds each, rounds 410/230, DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test), H1SPEC=lgb_v5x_noll H1TAG=_h1g H1BETA=0.50 (16 seeds, HFILT=1, 353 rounds), SMOOTH_W1=0.7 SMOOTH_W7=0.7 SMOOTH_R=1 SMOOTH_IT=1 SMOOTH_WRAP=0. Measured cost of that set from out/carbon/emissions.csv: 48 runs, 6.90 h, 0.1210 kg CO2e lgb general: 16 runs x 4.0 min; lgb h1 specialist: 16 runs x 1.5 min; xgb general: 16 runs x 20.4 min. Any adopted TRAINED change therefore costs a full FINALvn2 retrain of that size before a submission exists.
- step 7: node-1.2.3 integration gates written (ids c<n> from RESEARCH headings; each candidate measured or `deferred c<n>: <reason>`).
- step 7 CONTRACT ADDENDUM (driver): per-candidate leaf gates, instantiated as gates/leaf-1.2.3.<n>-<cand>.md once research ranks the candidates. Kept here, not in gates/, because gate-check counts every file in gates/ and a template would register as permanently unmet.
    C1  builder goal065_<cand>.py writes out/mats/{L}_{tr,va}_<cand>.parquet for Avn2 Bvn2 Cvn2 D E, row counts equal to {L}_{tr,va}.parquet
    C2  leakage: builder reads only history (pseudo_hist / months <= t / Train.csv for FINAL) — manual, quote the lines
    C3  rules R1-R5 each stated for this candidate — manual
    C4  FEATS_ONLY=1 XF=<cand> on D prints n = 340 + k, k = side-car column count
    C5  goal065_run.sh <cand> finished or stopped ABANDON-EARLY; out/goal065/<cand>.txt ends in a contract summary line
    C6  provenance: every treatment used_*.json has 340 + k features
    C7  driver appended exactly one "<cand> |" line to heldout.txt
    C8  if verdict is HELPS-NOT-0.65 or better: re-measured on top of the ar_blend correction, leave-one-layout-out
  Linear-correction candidates (no training) replace C4-C6 with a leave-one-layout-out fit that imports goal065_eval's verdict/summary functions.
- step 7 RISK NOTE: validation trains LightGBM only, but the shipped file is 0.5 LightGBM + 0.5 XGBoost, and XGBoost is 5.44 of the 6.90 h FINALvn2 retrain (16 runs x 20.4 min, measured). An adopted TRAINED change therefore enters the submission through an XGBoost half never validated with it.
- step 7 HANDOVER RULE (cost-aware, decided before results): a trained change whose verdict is HELPS-NOT-0.65 with mean > -0.003 is NOT retrained into FINALvn2 — the project's own transfer table says gains under ~0.003 flip sign on the board, which cannot justify a 6.9 h retrain and the one daily submission. Such changes are recorded, not shipped.
- step 8: ABANDON: G4 added to GATES.md with the reason (foreign SampleSubmission.csv); reversible on the user's choice. REPORT.md §4 gained rows for arseas and ar2, numbers read from heldout.txt. NOTE FOR FINAL VERIFICATION: gate-check re-runs only UNMET gates, so leaf-1.3.2 R2/R3 (and node-1.2 N2) can go stale as candidates are appended — reset them to pending and re-run before handover.
- step 5: session resumed after a rate limit killed both subagents. Measured state rather than trusting memory: RESEARCH.md complete (8 candidates, BUDGET TOTAL 0.0000), harness present and working (selftest ok; reproduces the Session 11g directional line exactly), but run_models.py had NO XF switch - the harness agent died before that edit. Driver finished the XF switch and FEATS_ONLY exit itself.
- step 5: ARF scored: arf | Avn2 -0.0009 Bvn2 +0.0026 Cvn2 +0.0027 D -0.0003 E -0.0003 | wins 3/5 | mean +0.0008 | REJECT. Giving the model the AR quantity as features is WORSE than applying it as a post-hoc linear correction (-0.0048, 5/5).
- step 5: REGEX ANCHOR FIX, not a threshold change: arf L4 and build B1 ended their regex with \S+$ while the CHECK output ends in a newline, so they could never match. Changed to \S+\s*$. Required line content unchanged.
- step 6: candidate disposition. c1 is being measured (top of RESEARCH.md's recommended order). The rest are deferred, with the reason and the number:
- deferred c2: expected validation gain +0.0000. RESEARCH.md's read-only proxy says the tree already gets this from b_spei1, resp_sm and csd, and RESEARCH.md itself recommends not paying 2 h of training for a verdict the proxy already gives. Carries the only "New reason:" line (vs closed per-cell reliability) and that reason survives, but the practical verdict does not change.
- deferred c3: expected validation gain +0.0000. LightGBM linear_tree leaves; same reason as c2, RESEARCH.md recommends not training it.
- deferred c5: expected validation gain -0.0020, needs a ~100 MB GLDAS-2.1 CLSM download behind an Earthdata login. Downloads need the user's explicit approval (poor connection) and were not approved. Board expectation ~0.0015 at the 0.77 ratio, against the 0.0317 the goal needs.
- deferred c6: expected validation gain -0.0025, needs a 0.3-1.5 GB WaterGAP 2.2e download. Not approved. Board expectation ~0.0019.
- deferred c7: expected validation gain -0.0010, needs a ~20 MB GPCC download (the smallest). Not approved. Board expectation ~0. RESEARCH.md names this one first if only a single download is possible.
- deferred c8: expected validation gain -0.0009, needs a 2-5 GB ERA5 daily download. Not approved. Board expectation ~0.
- step 6: order note for node-1.2.3 N3: measurement followed RESEARCH.md's recommended order, which puts c1 first and downloads last; c2 and c3 are skipped on RESEARCH.md's own recommendation rather than reordered.
- deferred c4: expected validation gain -0.0003, which is at this project's adoption bar and two orders below the -0.041 G2 needs. It is not a side-car: matching the test's history masking changes training-row construction, so it needs build_mats rebuilt on five layouts (~20 min each) plus 20 training runs, several hours, for an expected board gain of ~0. Deferred rather than run, with the number stated.
- step 7: STALE PASS caught. R3 (REPORT §4 names every rejected candidate) was ticked in the earlier session when heldout.txt held only arseas and ar2; gate-check only runs UNMET gates, so it never re-ran after arf was added, and arf was missing from §4. Row added, R3 reset and re-checked, now genuinely passing. Consequence for the handover: before the final report, reset EVERY ticked gate and re-run the whole ledger, because a pass recorded against an earlier state is not evidence about the current one.
- step 8: re-verification pass. Reset every tick in leaf-1.1, leaf-1.2.2.1, leaf-1.2.2.2 and node-1.2 and re-ran them against today's files: arf 5/5, arext 4/4, research 8/8, node-1.2 2/4 (N1 waits on children, N4 manual). L6's manual evidence was wrong twice before it was right: first it claimed only C2 mentions a closed avenue (a substring scan flags C1, C5, C7, C8 too), then it claimed all 8 blocks carry a "Closed-list check:" line (C2 does not; it carries the "New reason:" line instead). The third version states both facts correctly. Manual evidence is where this discipline is weakest, because nothing executes it.
- step 9: SELF-INFLICTED DEADLOCK, found and cleared. goal065_run.sh's wait_free greps the process table with `pgrep -f "[r]un_models\.py"`. The [r] trick stops the pattern matching ITSELF, but it does not stop it matching any OTHER shell whose command line contains the text "run_models.py" - and both of my background helpers did: the queued H8 run (which literally invokes it) and the c1 monitor (whose condition greps for it). The runner therefore sat in wait_free from 08:33 with NO python training alive, showing 3/10 files and looking like slow progress. Killed both helpers; the runner resumes.
- step 9: the lesson generalises beyond this script. A pgrep guard over a shared machine is only safe if the watchers stay OUT of the process table's text, so helper commands now live in FILES (out/prof/h8.sh, out/prof/watch_c1.sh) and are invoked by path. Diagnostic one-liners that mention a training script by name are also enough to stall a runner for one 50 s cycle, so they are avoided while a runner is live.
- step 9: this also invalidated my earlier "no concurrent training" reasoning in the other direction: the log-mtime overlap test I tried was unsound on macOS (ctime updates on write, so ctime == mtime), and the real evidence for node-1.2 N4 is the runners' own printed start times plus wait_free, not file timestamps.
- step 10: paused out/prof/final_gpcc.sh at 1/12 to let the h=1 experiment have the machine. Reason, stated so the priority is auditable: GPCC's value is already MEASURED at -0.0020 and the file it builds will display worse than the 0.679786 record because its base is a single model rather than the blend; the h=1 question is worth -0.0140 (h1 to 0.45) or -0.0336 (h1 matching our long-horizon edge over AR) and is unmeasured. final_gpcc.sh skips existing predictions, so it resumes from seed 1 with nothing lost.
- step 11: the h=1 launcher was killed by my own pkill. `pkill -f "[f]inal_gpcc"` matched it because the launcher's command line CONTAINED the heredoc that writes h1spec.sh, and that heredoc quotes "final_gpcc". Same class as step 9: a pattern over the process table hitting my own helper because the helper's text mentions the pattern. The rule now is not only "helpers live in files" but "pkill patterns must be checked against pgrep FIRST, and scripts are written in one call and launched in another".
- step 12: CORRECTION to step 11's framing. I ranked horizons by our MARGIN OVER THE AR BASELINE and called h=1 "where we lose". That is a false equivalence: the margin measures how badly the baseline degrades, not how much headroom we have. At h=7 the AR line collapses to 0.7352 because a seven-month-old observation says little, while our model has six months of forcing since the anchor; at h=1 the AR line is strong at 0.5390 because one-month persistence really is strong. A narrow margin over a strong baseline is not a weak model. In absolute terms our h=1 of 0.4963 already beats train-era persistence (0.5737) and the best global AR shrinkage (0.5444). The -0.0336 "prize" was computed by assuming h=1 could beat AR by the same margin we manage at h=7, and there is no reason that margin should be constant across horizons. The h=1 specialist measurement is the test of it, and on the first layout it went the wrong way (+0.0206 at h=1).
- step 13: last day, one submission. Stopped the WaterGAP layout run at 3 of 5 layouts (Cvn2 -0.0025, D -0.0052, E -0.0042, mean -0.0040, no layout losing) to free the machine for the training-structure ablation, because the ablation DECIDES the FINAL configuration and running a 2.2-hour FINAL build before knowing it risks wasting the remaining time. Stated as a deliberate trade of validation breadth for a config decision, not as an omission: wgap has three winning layouts and none losing.
