#!/bin/sh
# Finish the attribution table, then re-gate the horizon-1 splice on the NEW feature set --
# beta=0.50 was adopted on the old one, and an adopted stage should not be carried across a
# feature change without being re-measured.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
export CARBON=1   # or the runs are invisible to carbon_report.py
FD='bigsa,gdo'      # what the ablation adopted: soil profile in, GDO not on top of it
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

for S in 0 1; do
  f=out/mats/pred_Ce_lgb_v5x_noll_s${S}_x2.npy
  [ -f "$f" ] && continue
  say "Ce x2 s$S (completing the attribution table)"
  DROPF='bigsa,e5prof' WEIGHTS=ramp HMIX=test SEED=$S TAG=_x2 \
    $PY run_models.py Ce lgb v5x_noll_sa > out/prof/Ce_x2_s${S}.log 2>&1 || say "  FAILED"
done

for X in Ae Be Ce; do
  for S in 0 1; do
    f=out/mats/pred_${X}_lgb_v5x_noll_s${S}_h1e.npy
    [ -f "$f" ] && continue
    say "h1 specialist $X s$S"
    DROPF="$FD" WEIGHTS=ramp HFILT=1 SEED=$S TAG=_h1e \
      $PY run_models.py $X lgb v5x_noll_sa > out/prof/h1_${X}_s${S}.log 2>&1 || say "  FAILED"
  done
done

say "hsplice on the new feature set"
LAYOUTS=Ae,Be,Ce $PY hsplice.py 'lgb_v5x_noll:_x1' 'lgb_v5x_noll:_h1e' > out/prof/h1beta.sh 2> out/prof/hsplice.log
cat out/prof/hsplice.log; cat out/prof/h1beta.sh
say "stage 2 done"
