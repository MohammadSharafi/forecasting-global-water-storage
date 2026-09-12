#!/bin/sh
# goal065_run.sh CAND -- train side-car candidate CAND against the shipped control, one layout at a time.
#   control   = the existing _d0 predictions (shipped config; training is bit-deterministic)
#   treatment = the same config + XF=CAND, TAG=_${CAND}1
# Needs out/mats/<L>_tr_CAND.parquet and <L>_va_CAND.parquet for each layout. Layouts run fastest
# first, and after each one the layouts done so far are scored: two or more losing layouts stop the
# run (ABANDON-EARLY) before the slow layouts are paid for. The final score goes to out/goal065/CAND.txt.
cd "$(cd "$(dirname "$0")" && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2
export CARBON=1   # or the runs are invisible to carbon_report.py
CAND="$1"
case "$CAND" in
  ''|*[!A-Za-z0-9_]*) echo "usage: sh goal065_run.sh CAND   (CAND: letters, digits and _ only)"; exit 2;;
esac
STEM=lgb_v5x_noll
TT="_${CAND}1"
SCORE=out/prof/g065_${CAND}_score.txt
mkdir -p out/prof out/goal065
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

# 16 GB machine, OOM-killed before: one training process at a time. The [r] keeps pgrep from
# matching a shell whose own command line merely mentions the script; the second look 20 s later
# covers the gap between two runs of another sequential runner.
wait_free(){
  while :; do
    until ! pgrep -f "[r]un_models\.py" >/dev/null 2>&1; do sleep 30; done
    sleep 20
    pgrep -f "[r]un_models\.py" >/dev/null 2>&1 || return 0
  done
}

DONE=""
for X in D E Cvn2 Bvn2 Avn2; do
  for f in out/mats/${X}_tr_${CAND}.parquet out/mats/${X}_va_${CAND}.parquet; do
    [ -f "$f" ] || { say "MISSING side-car $f -- build it before running $CAND on $X"; exit 3; }
  done
  for S in 0 1; do
    [ -f out/mats/pred_${X}_${STEM}_s${S}_d0.npy ] \
      || { say "MISSING control out/mats/pred_${X}_${STEM}_s${S}_d0.npy -- cannot score $X"; exit 3; }
  done
  for S in 0 1; do
    P=out/mats/pred_${X}_${STEM}_s${S}${TT}.npy
    if [ -f "$P" ]; then say "$X s$S: $P exists, not retraining"; continue; fi
    wait_free
    LOG=out/prof/g065_${CAND}_${X}_s${S}.log
    say "$X s$S: training XF=$CAND TAG=$TT -> $LOG"
    XF="$CAND" DROPF="bigsa,gdo" WEIGHTS=ramp HMIX=test SEED=$S TAG="$TT" \
      $PY run_models.py $X lgb v5x_noll_sa > "$LOG" 2>&1 \
      || { say "$X s$S: TRAINING FAILED"; tail -8 "$LOG"; exit 4; }
    [ -f "$P" ] || { say "$X s$S: run exited 0 but did not write $P"; tail -8 "$LOG"; exit 4; }
  done
  DONE="${DONE:+$DONE,}$X"
  $PY goal065_eval.py trained --cand "$CAND" --control _d0 --treat "$TT" --layouts "$DONE" --no-append > "$SCORE" 2>&1 \
    || { say "scoring failed after $DONE"; tail -8 "$SCORE"; exit 5; }
  LINE=$(tail -1 "$SCORE")
  say "after $DONE: $LINE"
  case "$LINE" in
    *"| ABANDON-EARLY")
      # record the early verdict in out/goal065/CAND.txt, then stop before the remaining layouts
      $PY goal065_eval.py trained --cand "$CAND" --control _d0 --treat "$TT" --layouts "$DONE"
      say "$CAND: ABANDON-EARLY -- two or more layouts lose after $DONE; remaining layouts not trained"
      exit 0;;
  esac
done
$PY goal065_eval.py trained --cand "$CAND" --control _d0 --treat "$TT"
