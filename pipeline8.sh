#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
for L in A B; do $PY run_models.py $L lgb allnoll > out/mats/${L}_lgb_allnoll.log 2>&1; $PY run_models.py $L lgb v5x_noll > out/mats/${L}_lgb_v5x_noll.log 2>&1; done
echo "val done" > out/mats/p8.val
for FS in allnoll v5x_noll; do
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb $FS 420 > out/mats/FINAL_lgb_${FS}_s$s.log 2>&1; done
  for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb $FS 360 > out/mats/FINAL_xgb_${FS}_s$s.log 2>&1; done
  for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat $FS 975 > out/mats/FINAL_cat_${FS}_s$s.log 2>&1; done
done
for s in 2 3; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp allnoll > out/mats/FINAL_mlpnoll_s$s.log 2>&1; done
for s in 0 1 2 3; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp v5x_noll > out/mats/FINAL_mlp_v5x_noll_s$s.log 2>&1; done
echo "final done" > out/mats/p8.final
