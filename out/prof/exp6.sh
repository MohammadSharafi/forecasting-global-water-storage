#!/bin/sh
# Variance-weighted training: spend capacity where the metric's error actually is.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in Bvn2 Cvn2 Avn2; do
  for arm in "v0:ramp" "v1:ramp,var"; do
    tag=${arm%%:*}; wt=${arm#*:}
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S (WEIGHTS=$wt)"
      DROPF='bigsa,gdo' WEIGHTS="$wt" HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "--- $X"
  $PY eval_mix.py $X ramp=lgb_v5x_noll:_v0 varwt=lgb_v5x_noll:_v1 2>&1 | grep -E "^  (set|ramp|varwt) "
done
say "exp6 done"
