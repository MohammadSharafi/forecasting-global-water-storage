#!/bin/sh
# Four arms on ONE matrix, so the only difference between them is which features the model is
# offered. Arms run in decision order: base and both first, across all three layouts, which
# answers "do the new features help at all"; the two separating arms follow.
#   x0 base   incumbent 261 features        DROPF=bigsa,e5prof,gdo
#   x3 both   + soil profile + GDO SPI      DROPF=bigsa
#   x1 prof   + soil profile only           DROPF=bigsa,gdo
#   x2 gdo    + GDO SPI only                DROPF=bigsa,e5prof
# Two seeds each; eval_mix averages seeds for a stem, which is how the pipeline scores anyway.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
FW=ramp; FH=test; FM=lgb
for arm in "x0:bigsa,e5prof,gdo" "x3:bigsa" "x1:bigsa,gdo" "x2:bigsa,e5prof"; do
  tag=${arm%%:*}; d=${arm#*:}
  for X in Ae Be Ce; do
    for S in 0 1; do
      f=out/mats/pred_${X}_${FM}_v5x_noll_s${S}_${tag}.npy
      [ -f "$f" ] && { echo "  [skip] $X $tag s$S"; continue; }
      echo "  [run ] $X $tag s$S  (DROPF=$d)  $(date '+%H:%M:%S')"
      DROPF="$d" WEIGHTS=$FW HMIX="$FH" SEED=$S TAG=_$tag \
        $PY run_models.py $X $FM v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 \
        || { echo "  [FAIL] $X $tag s$S"; tail -5 out/prof/${X}_${tag}_s${S}.log; }
    done
  done
done
for X in Ae Be Ce; do
  echo "=== $X ==="
  $PY eval_mix.py $X base=${FM}_v5x_noll:_x0 both=${FM}_v5x_noll:_x3 \
                     prof=${FM}_v5x_noll:_x1 gdo=${FM}_v5x_noll:_x2 2>&1 | tee out/prof/eval_$X.txt
done
