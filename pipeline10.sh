#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
for L in A B; do TAG=_sa $PY run_models.py $L lgb allL_noll_sa > out/mats/${L}_lgb_allL_noll_sa.log 2>&1; SEED=0 EPOCHS=1 TAG=_e1sa $PY run_models.py $L mlp allL_noll_sa > out/mats/${L}_mlp_allL_noll_sa.log 2>&1; done
echo "val done" > out/mats/p10.val
FS=allL_noll_sa
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb $FS 500 > out/mats/FINAL_lgb_${FS}_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb $FS 400 > out/mats/FINAL_xgb_${FS}_s$s.log 2>&1; done
for s in 0 1 2 3; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp $FS > out/mats/FINAL_mlp_${FS}_s$s.log 2>&1; done
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat $FS 975 > out/mats/FINAL_cat_${FS}_s$s.log 2>&1; done
echo "final done" > out/mats/p10.final
