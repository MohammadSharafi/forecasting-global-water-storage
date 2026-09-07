#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
until [ -f out/mats/p10.final ]; do sleep 60; done
export WEIGHTS=uniform; FS=v5x_noll_sa; T=_u
for L in A B; do TAG=$T $PY run_models.py $L lgb $FS > out/mats/${L}_lgb_${FS}_u.log 2>&1; done
for s in 0 1 2; do SEED=$s TAG=$T $PY run_models.py FINAL lgb $FS 560 > out/mats/FINAL_lgb_${FS}_u_s$s.log 2>&1; done
for s in 0 1 2; do SEED=$s TAG=$T $PY run_models.py FINAL xgb $FS 400 > out/mats/FINAL_xgb_${FS}_u_s$s.log 2>&1; done
for s in 0 1 2 3; do SEED=$s TAG=$T EPOCHS=1 $PY run_models.py FINAL mlp $FS > out/mats/FINAL_mlp_${FS}_u_s$s.log 2>&1; done
for s in 0 1; do SEED=$s TAG=$T CATLR=0.03 $PY run_models.py FINAL cat $FS 975 > out/mats/FINAL_cat_${FS}_u_s$s.log 2>&1; done
echo "final done" > out/mats/p11.final
