#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
mkdir -p out
log() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date '+%m-%d %H:%M:%S')"; }

log "STATE"
echo "branch : $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "commit : $(git rev-parse --short HEAD 2>/dev/null)"
for f in eval_mix.py features_scale.py select_config.py pipeline13.sh pipeline14.sh; do
  [ -f "$f" ] || { echo "MISSING $f -- the checkout did not complete, stopping"; exit 1; }
done
echo "all session-9d files present"

log "SESSION-9C ABLATION, PER HORIZON"
for L in A B; do
  $PY eval_mix.py $L base=lgb_v5x_noll_noanom anom=lgb_v5x_noll 2>&1 | tee "out/eval_9c_$L.txt" || \
    echo "  (skipped layout $L: predictions not found)"
done

log "EXPERIMENT GRID (pipeline13)"
if ./pipeline13.sh 2>&1 | tee out/pipeline13.log; then
  :
else
  echo "pipeline13 failed -- see out/pipeline13.log. Stopping before FINAL."
  echo "out/sub_n_anom_lgbxgb.csv from the earlier run is still valid and uploadable."
  exit 1
fi

log "CONFIGURATION CHOICE"
$PY select_config.py > out/config.sh || { echo "select_config failed"; exit 1; }
cat out/config.sh
. ./out/config.sh

log "FINAL (pipeline14)"
FINAL_DROPF="$FINAL_DROPF" FINAL_MODEL="$FINAL_MODEL" FINAL_WEIGHTS="$FINAL_WEIGHTS" \
  FINAL_HMIX="$FINAL_HMIX" ./pipeline14.sh 2>&1 | tee out/pipeline14.log

log "SUMMARY"
echo "chosen configuration:"; sed 's/^/  /' out/config.sh
echo
echo "files to upload, best first:"
ls -1t out/sub_p_final.csv out/sub_p_final_nosm.csv out/sub_n_anom_lgbxgb.csv \
        out/sub_n_anom_lgbxgb_nosm.csv 2>/dev/null | sed 's/^/  /'
echo
echo "previous best preserved: out/backup_pre_anom/sub_i_lt_lgbxgb.csv (public 0.709259)"
echo
echo "readability of each new file against it:"
for f in out/sub_p_final.csv out/sub_n_anom_lgbxgb.csv; do
  [ -f "$f" ] && { echo "  --- $f"; $PY lb_se.py out/backup_pre_anom/sub_i_lt_lgbxgb.csv "$f" | sed 's/^/    /'; }
done
echo
echo "logs: out/eval_9c_A.txt out/eval_9c_B.txt out/pipeline13.log out/pipeline14.log"
echo "Paste out/pipeline13.log's two eval_mix tables and out/config.sh into the chat."
