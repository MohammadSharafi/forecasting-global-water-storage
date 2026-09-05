#!/bin/sh
cd "/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU"; PY=./.venv/bin/python
until [ -f out/mats/e5only.done ]; do sleep 30; done
for L in A B; do $PY run_models.py $L lgb e5only_noll > out/mats/${L}_lgb_e5only_noll.log 2>&1; SEED=0 EPOCHS=1 TAG=_e1 $PY run_models.py $L mlp e5only_noll > out/mats/${L}_mlp_e5only_noll_e1.log 2>&1; $PY run_models.py $L lgb e5only_v5x_noll > out/mats/${L}_lgb_e5only_v5x_noll.log 2>&1; done
echo "val done" > out/mats/p7.val
for FS in e5only_noll e5only_v5x_noll; do
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb $FS 420 > out/mats/FINAL_lgb_${FS}_s$s.log 2>&1; done
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb $FS 360 > out/mats/FINAL_xgb_${FS}_s$s.log 2>&1; done
  for s in 0 1 2 3; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp $FS > out/mats/FINAL_mlp_${FS}_s$s.log 2>&1; done
  for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat $FS 975 > out/mats/FINAL_cat_${FS}_s$s.log 2>&1; done
done
echo "final done" > out/mats/p7.final
