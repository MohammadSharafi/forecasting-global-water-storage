#!/bin/sh
# 1. build the FINAL profile matrix (longest single step, needed regardless)
# 2. check the profile on XGBoost too -- it was adopted on LightGBM alone, and the submitted
#    blend is half XGBoost, so the family it was never measured on carries half the weight
# 3. early-stopped round counts for both families on the new feature set
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
# build_mats defaults PER_ROW to 3; the value gated by perrow_scan.py and used by every
# matrix the pipeline ships is 2. Set it explicitly -- an unset PER_ROW here silently
# builds a third more training rows than the configuration being shipped, and on this
# machine that is also the difference between a build that finishes and one the kernel kills.
export PER_ROW=${PER_ROW:-2}
FD='bigsa,gdo'; L=FINALe
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

if [ ! -f out/mats/${L}_tr.parquet ]; then
  say "build $L"
  $PY build_mats.py $L > out/prof/build_$L.log 2>&1 || { say "BUILD FAILED"; tail -20 out/prof/build_$L.log; exit 1; }
  tail -1 out/prof/build_$L.log
fi
if [ ! -f out/mats/${L}_tr_anchor.parquet ]; then
  say "anchors $L"
  $PY add_anchor_feats.py $L > out/prof/anchor_$L.log 2>&1 || { say "ANCHOR FAILED"; tail -10 out/prof/anchor_$L.log; exit 1; }
fi
say "FINALe ready"

for X in Ae Be Ce; do
  for arm in "x0:bigsa,e5prof,gdo" "x1:bigsa,gdo"; do
    tag=${arm%%:*}; d=${arm#*:}
    f=out/mats/pred_${X}_xgb_v5x_noll_s0_${tag}.npy
    [ -f "$f" ] && continue
    say "xgb $X $tag"
    DROPF="$d" WEIGHTS=ramp HMIX=test SEED=0 TAG=_$tag \
      $PY run_models.py $X xgb v5x_noll_sa > out/prof/${X}_xgb_${tag}.log 2>&1 || say "  FAILED"
  done
  say "--- $X xgb"
  $PY eval_mix.py $X base=xgb_v5x_noll:_x0 prof=xgb_v5x_noll:_x1 2>&1 | grep -E "^  (set|base|prof) "
done

say "early-stopped rounds on the new feature set"
for X in Ae Be Ce; do
  for m in lgb xgb; do
    l=out/prof/${X}_${m}_x1.log; [ "$m" = lgb ] && l=out/prof/${X}_x1_s0.log
    printf '  %-3s %-4s %s\n' "$X" "$m" "$(grep -o 'best_iter=[0-9]*' "$l" 2>/dev/null | tail -1)"
  done
done
say "stage 3 done"
