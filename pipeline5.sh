#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
for L in A B; do $PY run_models.py $L lgb v5x > out/mats/${L}_lgb_v5x.log 2>&1; SEED=0 EPOCHS=1 TAG=_e1 $PY run_models.py $L mlp v5x > out/mats/${L}_mlp_v5x_e1.log 2>&1; done
echo "val done" > out/mats/p5.val
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb v5x 420 > out/mats/FINAL_lgb_v5x_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb v5x 360 > out/mats/FINAL_xgb_v5x_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp v5x > out/mats/FINAL_mlp_v5x_s$s.log 2>&1; done
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat v5x 975 > out/mats/FINAL_cat_v5x_s$s.log 2>&1; done
echo "final done" > out/mats/p5.final
