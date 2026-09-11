#!/bin/sh
# Two untested things at once, each with its own switch:
#   - regional aggregates of the TWS-derived quantities that had none (d1, d3, d12,
#     trend_persist, anom_persist), aimed at h=1 where a third of the test weight is
#   - three radii instead of one
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export CARBON=1   # or the runs are invisible to carbon_report.py
export PER_ROW=2 ANWIDE_R=2,4,8
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
for X in Bvn4 Cvn4 Avn4; do
  if [ ! -f out/mats/${X}_tr.parquet ]; then
    say "build $X"
    $PY build_mats.py $X > out/prof/build_$X.log 2>&1 || { say "BUILD FAILED"; tail -12 out/prof/build_$X.log; exit 1; }
    tail -1 out/prof/build_$X.log
  fi
  [ -f out/mats/${X}_tr_anchor.parquet ] || $PY add_anchor_feats.py $X > out/prof/anchor_$X.log 2>&1 || { say "ANCHOR FAILED"; exit 1; }
  # p0 = the shipped set (single radius 4, no TWS-momentum aggregates is not separable here, so
  # p0 keeps only radius 4); p1 = everything
  for arm in "p0:bigsa,gdo,aw2,aw8" "p1:bigsa,gdo"; do
    tag=${arm%%:*}; d=${arm#*:}
    for S in 0 1; do
      [ -f out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy ] && continue
      say "  $X $tag s$S"
      DROPF="$d" WEIGHTS=ramp HMIX=test SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 || say "  FAILED"
    done
  done
  say "--- $X   (p0 = radius 4 only, p1 = radii 2/4/8 incl. the TWS-momentum aggregates)"
  $PY eval_mix.py $X r4=lgb_v5x_noll:_p0 multi=lgb_v5x_noll:_p1 2>&1 | grep -E "^  (set|r4|multi) "
done
say "exp5 done"
