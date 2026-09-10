#!/bin/sh
# Re-gate the two adopted representation fixes on layouts the decision never saw.
#   g0 = the feature set BEFORE both fixes      g1 = the shipped configuration
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export PER_ROW=2
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in D E; do
  if [ ! -f out/mats/${X}_tr.parquet ]; then
    say "build $X"
    $PY build_mats.py $X > out/prof/build_$X.log 2>&1 || { say "BUILD FAILED"; tail -12 out/prof/build_$X.log; exit 1; }
    tail -1 out/prof/build_$X.log
  fi
  [ -f out/mats/${X}_tr_anchor.parquet ] || $PY add_anchor_feats.py $X > out/prof/anchor_$X.log 2>&1 \
    || { say "ANCHOR FAILED"; exit 1; }
  for arm in "g0:bigsa,gdo,anwide,r2anom,cpcanom,speianom" "g1:bigsa,gdo"; do
    tag=${arm%%:*}; d=${arm#*:}
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S"
      DROPF="$d" WEIGHTS=ramp HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "--- $X   (g0 = before both fixes, g1 = shipped)"
  $PY eval_mix.py $X before=lgb_v5x_noll:_g0 shipped=lgb_v5x_noll:_g1 2>&1 | grep -E "^  (set|before|shipped) "
done
say "regate done"
