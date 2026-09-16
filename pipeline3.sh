#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
until grep -q "all done" out/era5_download.log; do sleep 30; done
mkdir -p out/mats_noera && cp out/mats/*.parquet out/mats/feats.json out/mats_noera/ 2>/dev/null
for L in A B FINAL; do $PY build_mats.py $L > out/mats_${L}_era5.log 2>&1; done
echo "build done" > out/mats/p3.build
for L in A B; do
  $PY run_models.py $L lgb noera > out/mats/${L}_lgb_noera.log 2>&1
  $PY run_models.py $L lgb all > out/mats/${L}_lgb_all.log 2>&1
  for s in 0 1 2; do SEED=$s EPOCHS=1 TAG=_e1 $PY run_models.py $L mlp all > out/mats/${L}_mlp_all_e1_s$s.log 2>&1; done
  SEED=0 EPOCHS=1 TAG=_e1 $PY run_models.py $L mlp noera > out/mats/${L}_mlp_noera_e1.log 2>&1
  $PY run_models.py $L xgb all > out/mats/${L}_xgb_all.log 2>&1
  CATLR=0.03 TAG=_lr03 $PY run_models.py $L cat all > out/mats/${L}_cat_all_lr03.log 2>&1
done
echo "val done" > out/mats/p3.val
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb all 420 > out/mats/FINAL_lgb_s$s.log 2>&1; done
for s in 0 1 2 3 4; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp all > out/mats/FINAL_mlp_s$s.log 2>&1; done
for s in 0 1; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp allnoll > out/mats/FINAL_mlpnoll_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb all 360 > out/mats/FINAL_xgb_s$s.log 2>&1; done
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat all 975 > out/mats/FINAL_cat_s$s.log 2>&1; done
echo "final done" > out/mats/p3.final
