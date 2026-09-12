#!/bin/sh
# Train the FINAL models with the adopted GPCC block and assemble a submission.
# Same configuration as the shipped file (DROPF=bigsa,gdo, WEIGHTS=ramp, HMIX=test, rounds 410/230,
# smoothing 0.7/r1/it1) with XF=gpcc added, so the ONLY difference is the gauge precipitation.
# Six seeds per family rather than sixteen: the seed-to-seed spread is far below the -0.0020 the
# block is worth, and six keeps this inside an hour.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2 CARBON=1
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Single-instance lock: two copies of a runner raced once and trained the same run twice on a
# 16 GB machine, both writing one file. mkdir is atomic, so a second copy exits instead.
LOCK=out/prof/.final_gpcc.lock
mkdir "$LOCK" 2>/dev/null || { say "another instance holds $LOCK -- exiting"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
for S in 0 1 2 3 4 5; do
  for fam in "lgb:410" "xgb:230"; do
    M=${fam%%:*}; R=${fam#*:}
    P=out/mats/pred_FINALvn2_${M}_v5x_noll_s${S}_gp1.npy
    [ -f "$P" ] && { say "$M s$S exists"; continue; }
    say "$M s$S rounds=$R"
    XF=gpcc DROPF="bigsa,gdo" WEIGHTS=ramp HMIX=test SEED=$S TAG=_gp1 \
      $PY run_models.py FINALvn2 $M v5x_noll_sa $R > out/prof/fg_${M}_s${S}.log 2>&1 \
      || { say "  FAILED"; tail -6 out/prof/fg_${M}_s${S}.log; exit 4; }
  done
done
say "assembling"
FLAYOUT=FINALvn2 TAG=_gp1 SMOOTH_W1=0.7 SMOOTH_W7=0.7 SMOOTH_R=1 SMOOTH_IT=1 SMOOTH_WRAP=0 \
  $PY final_assemble.py sub_gpcc_base lgb_v5x_noll:0.5 xgb_v5x_noll:0.5 || exit 5
say "done -> out/sub_gpcc_base.csv"
