#!/bin/sh
# Does offering the anomaly block at several spatial scales beat the single r=4?
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export CARBON=1   # or the runs are invisible to carbon_report.py
export PER_ROW=2 ANWIDE_R=2,4,8
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in Avn3 Bvn3 Cvn3; do
  if [ ! -f out/mats/${X}_tr.parquet ]; then
    say "build $X"
    $PY build_mats.py $X > out/prof/build_$X.log 2>&1 || { say "BUILD FAILED"; tail -12 out/prof/build_$X.log; exit 1; }
    tail -1 out/prof/build_$X.log
  fi
  [ -f out/mats/${X}_tr_anchor.parquet ] || $PY add_anchor_feats.py $X > out/prof/anchor_$X.log 2>&1 || { say "ANCHOR FAILED"; exit 1; }
done
say "matrices ready"
for arm in "m0:bigsa,gdo,aw2,aw8" "m1:bigsa,gdo"; do
  tag=${arm%%:*}; d=${arm#*:}
  for X in Avn3 Bvn3 Cvn3; do
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S"
      DROPF="$d" WEIGHTS=ramp HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
done
for X in Avn3 Bvn3 Cvn3; do
  say "--- $X   (m0 = single radius 4, m1 = radii 2, 4 and 8)"
  $PY eval_mix.py $X r4=lgb_v5x_noll:_m0 multi=lgb_v5x_noll:_m1 2>&1 | grep -E "^  (set|r4|multi) "
done
say "exp4 done"
