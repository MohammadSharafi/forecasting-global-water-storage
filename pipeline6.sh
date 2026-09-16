#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
for L in A B; do $PY run_models.py $L lgb allL > out/mats/${L}_lgb_allL.log 2>&1; $PY run_models.py $L xgb allL > out/mats/${L}_xgb_allL.log 2>&1; done
echo "val done" > out/mats/p6.val
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb allL 420 > out/mats/FINAL_lgb_allL_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb allL 360 > out/mats/FINAL_xgb_allL_s$s.log 2>&1; done
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat allL 975 > out/mats/FINAL_cat_allL_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp allL > out/mats/FINAL_mlp_allL_s$s.log 2>&1; done
echo "final done" > out/mats/p6.final
