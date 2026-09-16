#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-./.venv/bin/python}
while pgrep -f "build_mats.py FINAL" >/dev/null; do sleep 20; done
[ -f out/mats/FINAL_tr.parquet ] || { echo "FINAL matrix missing" > out/mats/pipeline2.err; exit 1; }
EPOCHS=1 TAG=_e1 $PY run_models.py A mlp all > out/mats/A_mlp_all_e1.log 2>&1
EPOCHS=2 TAG=_e2 $PY run_models.py A mlp all > out/mats/A_mlp_all_e2.log 2>&1
EPOCHS=1 TAG=_e1 $PY run_models.py B mlp all > out/mats/B_mlp_all_e1.log 2>&1
EPOCHS=3 $PY run_models.py B mlp all > out/mats/B_mlp_all.log 2>&1
CATLR=0.03 TAG=_lr03 $PY run_models.py A cat all > out/mats/A_cat_all_lr03.log 2>&1
CATLR=0.03 TAG=_lr03 $PY run_models.py B cat all > out/mats/B_cat_all_lr03.log 2>&1
echo "stage1 done" > out/mats/pipeline2.stage1
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL lgb all 420 > out/mats/FINAL_lgb_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s $PY run_models.py FINAL xgb all 360 > out/mats/FINAL_xgb_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s EPOCHS=1 $PY run_models.py FINAL mlp all > out/mats/FINAL_mlp_s$s.log 2>&1; done
CB=$(grep -o "best_iter=[0-9]*" out/mats/A_cat_all_lr03.log | cut -d= -f2); CB=$(( ${CB:-450} * 11 / 10 ))
for s in 0 1; do SEED=$s CATLR=0.03 $PY run_models.py FINAL cat all $CB > out/mats/FINAL_cat_s$s.log 2>&1; done
echo "stage2 done" > out/mats/pipeline2.stage2
$PY run_cnn.py A 40 > out/mats/A_cnn.log 2>&1
echo "stage3 done" > out/mats/pipeline2.stage3
