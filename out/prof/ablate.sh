#!/bin/sh
# Four arms on ONE matrix, so the only difference between them is which features the model is
# offered: the incumbent set, plus the ERA5 soil profile, plus the GDO long-window SPI, plus both.
# Two seeds each; eval_mix averages seeds for a stem, which is how the pipeline scores anyway.
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
FW=ramp; FH=test; FM=lgb
for X in Ae Be Ce; do
  for arm in "x0:bigsa,e5prof,gdo" "x1:bigsa,gdo" "x2:bigsa,e5prof" "x3:bigsa"; do
    tag=${arm%%:*}; d=${arm#*:}
    for S in 0 1; do
      f=out/mats/pred_${X}_${FM}_v5x_noll_s${S}_${tag}.npy
      [ -f "$f" ] && { echo "  [skip] $X $tag s$S"; continue; }
      echo "  [run ] $X $tag s$S  (DROPF=$d)  $(date '+%H:%M:%S')"
      DROPF="$d" WEIGHTS=$FW HMIX="$FH" SEED=$S TAG=_$tag \
        $PY run_models.py $X $FM v5x_noll_sa > out/prof/${X}_${tag}_s${S}.log 2>&1 \
        || { echo "  [FAIL] $X $tag s$S"; tail -5 out/prof/${X}_${tag}_s${S}.log; }
    done
  done
  echo "=== $X ==="
  $PY eval_mix.py $X base=${FM}_v5x_noll:_x0 prof=${FM}_v5x_noll:_x1 \
                     gdo=${FM}_v5x_noll:_x2 both=${FM}_v5x_noll:_x3 2>&1 | tee out/prof/eval_$X.txt
done
