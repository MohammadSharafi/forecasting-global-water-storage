#!/bin/sh
# Screen: regional means of the covariate-anomaly block. One layout first.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export PER_ROW=2
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in ${LAYOUTS:-Bvn2}; do
  if [ ! -f out/mats/${X}_tr.parquet ]; then
    say "build $X"
    $PY build_mats.py $X > out/prof/build_$X.log 2>&1 || { say "BUILD FAILED"; tail -12 out/prof/build_$X.log; exit 1; }
    tail -1 out/prof/build_$X.log
  fi
  [ -f out/mats/${X}_tr_anchor.parquet ] || $PY add_anchor_feats.py $X > out/prof/anchor_$X.log 2>&1 || { say "ANCHOR FAILED"; exit 1; }
  for arm in "q0:bigsa,gdo,anwide" "q1:bigsa,gdo"; do
    tag=${arm%%:*}; d=${arm#*:}
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S"
      DROPF="$d" WEIGHTS=ramp HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "--- $X  (q0 = without regional anomaly means, q1 = with)"
  $PY eval_mix.py $X base=lgb_v5x_noll:_q0 anwide=lgb_v5x_noll:_q1 2>&1 | grep -E "^  (set|base|anwide) "
done
say "exp3 done"
