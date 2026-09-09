#!/bin/sh
# Experiment 1: anomaly-encode the three blocks that never received it.
# One matrix per layout, four arms on it, so the only difference between arms is the feature set.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export PER_ROW=2
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in Avn Bvn Cvn; do
  if [ ! -f out/mats/${X}_tr.parquet ]; then
    say "build $X"
    $PY build_mats.py $X > out/prof/build_$X.log 2>&1 || { say "BUILD FAILED"; tail -15 out/prof/build_$X.log; exit 1; }
    tail -1 out/prof/build_$X.log
  fi
  [ -f out/mats/${X}_tr_anchor.parquet ] || $PY add_anchor_feats.py $X > out/prof/anchor_$X.log 2>&1 \
    || { say "ANCHOR FAILED"; exit 1; }
done
say "matrices ready"
for arm in "n0:bigsa,gdo,e5prof,r2anom,cpcanom,speianom" "n1:bigsa,gdo,e5prof,speianom" \
           "n2:bigsa,gdo,e5prof,r2anom,cpcanom" "n3:bigsa,gdo,e5prof"; do
  tag=${arm%%:*}; d=${arm#*:}
  for X in Avn Bvn Cvn; do
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S"
      DROPF="$d" WEIGHTS=ramp HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "=== after arm $tag ==="
  for X in Avn Bvn Cvn; do
    [ -f out/mats/pred_${X}_lgb_v5x_noll_s0_n0.npy ] || continue
    $PY eval_mix.py $X base=lgb_v5x_noll:_n0 spei=lgb_v5x_noll:_n1 \
        r2cpc=lgb_v5x_noll:_n2 all=lgb_v5x_noll:_n3 2>&1 | grep -E "^  (set|base|spei|r2cpc|all) " || true
  done
done
say "experiment 1 done"
