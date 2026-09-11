#!/bin/sh
# Cheap falsification of analogue weighting: same subsample, same seeds, paired arms.
# If the idea is worth >0.003 it should be visible at SUB=0.4; if it is invisible here it is
# not worth an hour of full training.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export CARBON=1   # or the runs are invisible to carbon_report.py
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in Avn Bvn Cvn; do
  for arm in "a0:ramp" "a1:ramp,analog"; do
    tag=${arm%%:*}; wt=${arm#*:}
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S (WEIGHTS=$wt)"
      DROPF='bigsa,gdo' WEIGHTS="$wt" HMIX=test SEED=$S SUB=0.4 TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa 300 > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "--- $X"
  $PY eval_mix.py $X ramp=lgb_v5x_noll:_a0 analog=lgb_v5x_noll:_a1 2>&1 | grep -E "^  (set|ramp|analog) "
done
say "exp2 done"
