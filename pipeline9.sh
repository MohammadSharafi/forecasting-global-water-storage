#!/bin/sh
cd "/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU"; PY=./.venv/bin/python
for FS in allnoll_sa v5x_noll_sa; do
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb $FS 560 > out/mats/FINAL_lgb_${FS}_s$s.log 2>&1; done
  for s in 0 1 2 3; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp $FS > out/mats/FINAL_mlp_${FS}_s$s.log 2>&1; done
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb $FS 400 > out/mats/FINAL_xgb_${FS}_s$s.log 2>&1; done
  for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat $FS 975 > out/mats/FINAL_cat_${FS}_s$s.log 2>&1; done
done
echo "final done" > out/mats/p9.final
