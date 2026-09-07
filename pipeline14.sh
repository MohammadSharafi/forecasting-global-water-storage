#!/bin/sh
# Build the final submission with a chosen configuration (session 9d).
#
#   FINAL_MODEL=lgb FINAL_DROPF=scale FINAL_WEIGHTS=ramp ./pipeline14.sh
#
# FINAL_MODEL    lgb (127 leaves, default) | lgbm (63) | lgbs (31)
# FINAL_DROPF    feature groups to drop: "" | anom | scale | anom,scale
# FINAL_WEIGHTS  ramp (default) | uniform
# SEEDS          how many seeds per family (default 5)
# XGB            1 to also train xgb and blend 50/50 (default 1; 0 for the single family)
#
# Pick these from pipeline13.sh's testmix table. Everything is trained on the FINAL matrix,
# which this script rebuilds so the feature set matches what the experiments measured.
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
M=out/mats; mkdir -p "$M"
log() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }

FM=${FINAL_MODEL:-lgb}; FD=${FINAL_DROPF:-}; FW=${FINAL_WEIGHTS:-ramp}; FH=${FINAL_HMIX:-}
NS=${SEEDS:-5}; USE_XGB=${XGB:-1}
TAG="_f"
echo "configuration: model=$FM dropf='$FD' weights=$FW hmix='$FH' seeds=$NS xgb=$USE_XGB"

B=out/backup_pre_anom; mkdir -p "$B"
cp out/*.csv "$B"/ 2>/dev/null
echo "submissions backed up to $B"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  log "build_mats FINAL"
  PER_ROW=${PER_ROW:-2} $PY build_mats.py FINAL > "$M/build_FINAL.log" 2>&1 || { tail -20 "$M/build_FINAL.log"; exit 1; }
  $PY add_anchor_feats.py FINAL >> "$M/build_FINAL.log" 2>&1 || { tail -20 "$M/build_FINAL.log"; exit 1; }
  tail -1 "$M/build_FINAL.log"
fi

# build_mats writes feats.json for whichever layout ran last. If FINAL was not rebuilt in this
# run (SKIP_BUILD=1) its matrix can predate that list, and run_models would fail deep into
# training on a missing column. Check it here, in one second, instead.
$PY - <<'PYCHK' || exit 1
import json, sys, polars as pl
want=set(json.load(open("out/mats/feats.json")))
have=set(pl.scan_parquet("out/mats/FINAL_va.parquet").collect_schema().names())
have|=set(pl.scan_parquet("out/mats/FINAL_va_anchor.parquet").collect_schema().names())
miss=sorted(want-have)
if miss:
    print(f"FINAL matrix is missing {len(miss)} features that feats.json expects, e.g. {miss[:6]}")
    print("Rerun without SKIP_BUILD=1 so the FINAL matrix matches the experiments.")
    sys.exit(1)
print(f"FINAL matrix carries all {len(want)} features")
PYCHK

# rounds from the matching layout-A experiment if it is on disk, else the historical default
R=$(grep -o 'best_iter=[0-9]*' "$M"/x_A_e8.log 2>/dev/null | tail -1 | cut -d= -f2)
LR=${LGB_ROUNDS:-${R:-560}}; XR=${XGB_ROUNDS:-400}
echo "rounds: $FM=$LR xgb=$XR"

s=0
while [ "$s" -lt "$NS" ]; do
  log "FINAL $FM seed $s"
  SEED=$s DROPF="$FD" WEIGHTS="$FW" HMIX="$FH" TAG="$TAG" \
    $PY run_models.py FINAL "$FM" v5x_noll_sa "$LR" > "$M/F_${FM}_s$s.log" 2>&1 \
    || { tail -20 "$M/F_${FM}_s$s.log"; exit 1; }
  s=$((s+1))
done
if [ "$USE_XGB" = "1" ]; then
  s=0
  while [ "$s" -lt "$NS" ]; do
    log "FINAL xgb seed $s"
    SEED=$s DROPF="$FD" WEIGHTS="$FW" HMIX="$FH" TAG="$TAG" \
      $PY run_models.py FINAL xgb v5x_noll_sa "$XR" > "$M/F_xgb_s$s.log" 2>&1 \
      || { tail -20 "$M/F_xgb_s$s.log"; exit 1; }
    s=$((s+1))
  done
fi

log "assembling"
if [ "$USE_XGB" = "1" ]; then SPEC="${FM}_v5x_noll:0.5 xgb_v5x_noll:0.5"; else SPEC="${FM}_v5x_noll:1.0"; fi
TAG="$TAG" $PY final_assemble.py sub_p_final $SPEC
TAG="$TAG" SMOOTH_W=0 $PY final_assemble.py sub_p_final_nosm $SPEC

log "DONE"
echo "out/sub_p_final.csv        <- upload this"
echo "out/sub_p_final_nosm.csv   <- same models, grid smoothing off (never LB-ablated)"
echo
echo "Is the difference from your current best readable on the public LB?"
echo "  $PY lb_se.py $B/sub_i_lt_lgbxgb.csv out/sub_p_final.csv"
