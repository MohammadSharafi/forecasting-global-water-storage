#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
until [ -f out/mats/p3.val ]; do sleep 30; done
pkill -f pipeline3.sh; sleep 2; pkill -f "run_models.py FINAL"; sleep 5
PER_ROW=2 $PY build_mats.py FINAL > out/mats_FINAL_era5.log 2>&1
grep -q "^tr " out/mats_FINAL_era5.log || { echo "FINAL build failed" > out/mats/p4.err; exit 1; }
echo "build done" > out/mats/p4.build
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb all 420 > out/mats/FINAL_lgb_s$s.log 2>&1; done
for s in 0 1 2 3 4; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp all > out/mats/FINAL_mlp_s$s.log 2>&1; done
for s in 0 1; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp allnoll > out/mats/FINAL_mlpnoll_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb all 360 > out/mats/FINAL_xgb_s$s.log 2>&1; done
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat all 975 > out/mats/FINAL_cat_s$s.log 2>&1; done
echo "final done" > out/mats/p4.final
