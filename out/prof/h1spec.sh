#!/bin/sh
# A specialist trained on the h=1 slice only. HFILT=1 keeps the h=1 training rows; the control is
# the shipped model's own h=1 predictions, so the comparison is on identical rows.
# Waits for the machine: one training process at a time on 16 GB.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2 CARBON=1
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

# A single-instance lock. Two copies of this script raced once and started two trainings of the
# same run on a 16 GB machine, both writing the same output file. mkdir is atomic, so the second
# copy exits instead of racing.
LOCK=out/prof/.h1spec.lock
mkdir "$LOCK" 2>/dev/null || { say "another instance holds $LOCK -- exiting"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
for X in D E Cvn2 Bvn2 Avn2; do
  for S in 0 1; do
    P=out/mats/pred_${X}_lgb_v5x_noll_s${S}_h1x.npy
    [ -f "$P" ] && continue
    while pgrep -f "[f]inal_gpcc" >/dev/null 2>&1; do sleep 60; done
    sleep 20
    say "$X s$S h=1 specialist"
    HFILT=1 DROPF="bigsa,gdo" WEIGHTS=ramp HMIX=test SEED=$S TAG=_h1x \
      $PY run_models.py $X lgb v5x_noll_sa 700 > out/prof/h1x_${X}_s${S}.log 2>&1 \
      || { say "  FAILED"; tail -5 out/prof/h1x_${X}_s${S}.log; exit 4; }
  done
done
say "h1 specialist training done"
