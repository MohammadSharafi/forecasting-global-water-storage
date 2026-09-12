#!/bin/sh
# H8: with the XF switch in place and XF unset, one D seed must reproduce the shipped control
# exactly. Kept in a FILE, not in a command line: goal065_run.sh's wait_free greps the process
# table for the script's name, so any shell whose command line merely CONTAINS that text is
# mistaken for a running training and deadlocks the runner. That is what happened here.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
while pgrep -f "[g]oal065_run" >/dev/null 2>&1; do sleep 30; done
sleep 30
DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test SEED=0 TAG=_xfnull PER_ROW=2 CARBON=1 \
  $PY run_models.py D lgb v5x_noll_sa > out/prof/h8_xfnull_D_s0.log 2>&1 || { echo "H8 TRAINING FAILED"; tail -5 out/prof/h8_xfnull_D_s0.log; exit 4; }
$PY -c "import numpy as np;a=np.load('out/mats/pred_D_lgb_v5x_noll_s0_xfnull.npy');b=np.load('out/mats/pred_D_lgb_v5x_noll_s0_d0.npy');print('maxdiff=%.3e'%abs(a-b).max())"
