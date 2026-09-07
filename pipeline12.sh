#!/bin/sh
# Session 9c, end to end: rebuild the matrices with the per-cell covariate anomalies,
# ablate them on both validation layouts, retrain lgb+xgb only, and write the submission.
#
#   ./pipeline12.sh            # everything (~2 h on the 16 GB machine)
#   PER_ROW=2 ./pipeline12.sh  # PER_ROW is already the default; lower it to 1 only on OOM
#
# One heavy process at a time throughout, per the memory rule. Output CSVs land in out/.
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
M=out/mats; mkdir -p "$M"
log() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }

# ---------------------------------------------------------------- preflight
fail=0
for f in Train.csv Test.csv out/pseudo_test.parquet out/pseudo_hist.parquet \
         out/pseudo_test_B.parquet out/pseudo_hist_B.parquet; do
  [ -f "$f" ] || { echo "MISSING: $f"; fail=1; }
done
ls external/era5/*.nc >/dev/null 2>&1 || { echo "MISSING: external/era5/*.nc (see ERA5_SETUP.md)"; fail=1; }
[ "$fail" = 0 ] || { echo "preflight failed, nothing run"; exit 1; }

# The rebuild overwrites out/mats. Keep a copy of what produced the 0.709259 submission so a
# failure here cannot cost the best file you already have.
log "backing up current predictions and submissions"
B=out/backup_pre_anom; mkdir -p "$B"
cp out/*.csv "$B"/ 2>/dev/null
cp "$M"/pred_FINAL_*.npy "$B"/ 2>/dev/null
cp "$M"/feats.json "$B"/ 2>/dev/null
echo "backed up $(ls "$B" | wc -l) files to $B"

# ---------------------------------------------------------------- 1. matrices
for X in A B FINAL; do
  log "build_mats $X"
  PER_ROW=${PER_ROW:-2} $PY build_mats.py $X > "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
  tail -2 "$M/build_$X.log"
  log "add_anchor_feats $X"
  $PY add_anchor_feats.py $X >> "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
done
echo "features now in the matrices: $($PY -c 'import json;f=json.load(open("out/mats/feats.json"));print(len(f),"of which anomaly:",sum(1 for x in f if x.startswith("an_")))')"

# ------------------------------------------------- 2. ablation on both layouts
# Adopt the anomalies only if they win on A AND B, the rule every other change was held to.
for X in A B; do
  for FS in v5x_noll_noanom_sa v5x_noll_sa; do
    log "validate $X lgb $FS"
    $PY run_models.py $X lgb $FS > "$M/abl_${X}_${FS}.log" 2>&1 || { tail -20 "$M/abl_${X}_${FS}.log"; exit 1; }
    grep RMSE= "$M/abl_${X}_${FS}.log" | tail -1
  done
done

log "ABLATION VERDICT"
r() { grep -o 'RMSE=[0-9.]*' "$1" | tail -1 | cut -d= -f2; }
i() { grep -o 'best_iter=[0-9]*' "$1" | tail -1 | cut -d= -f2; }
A0=$(r "$M/abl_A_v5x_noll_noanom_sa.log"); A1=$(r "$M/abl_A_v5x_noll_sa.log")
B0=$(r "$M/abl_B_v5x_noll_noanom_sa.log"); B1=$(r "$M/abl_B_v5x_noll_sa.log")
echo "layout A   without anomalies $A0   with $A1"
echo "layout B   without anomalies $B0   with $B1"
$PY - "$A0" "$A1" "$B0" "$B1" <<'EOF'
import sys
a0,a1,b0,b1=map(float,sys.argv[1:5])
ok=a1<a0 and b1<b0
print(f"  A {a1-a0:+.4f}   B {b1-b0:+.4f}")
print("  VERDICT:", "ADOPT - wins on both layouts" if ok else
      "DO NOT ADOPT - does not win on both; submit the backup file instead")
EOF

# --------------------------------------------- 3. FINAL: lgb + xgb only, 5 seeds each
# The leaderboard family ladder (lgb+xgb 0.70926 < +cat 0.71038 < +cat+mlp 0.71096) says drop
# CatBoost and the MLP; the seeds freed up go into the two families that are left.
LR=${LGB_ROUNDS:-$(i "$M/abl_A_v5x_noll_sa.log")}; LR=${LR:-560}
XR=${XGB_ROUNDS:-400}
echo "FINAL rounds: lgb=$LR (from layout A early stopping) xgb=$XR"
for s in 0 1 2 3 4; do
  log "FINAL lgb seed $s"
  SEED=$s $PY run_models.py FINAL lgb v5x_noll_sa $LR > "$M/FINAL_lgb_s$s.log" 2>&1 || { tail -20 "$M/FINAL_lgb_s$s.log"; exit 1; }
done
for s in 0 1 2 3 4; do
  log "FINAL xgb seed $s"
  SEED=$s $PY run_models.py FINAL xgb v5x_noll_sa $XR > "$M/FINAL_xgb_s$s.log" 2>&1 || { tail -20 "$M/FINAL_xgb_s$s.log"; exit 1; }
done

# ---------------------------------------------------------------- 4. submissions
log "assembling"
$PY final_assemble.py sub_n_anom_lgbxgb lgb_v5x_noll:0.5 xgb_v5x_noll:0.5
SMOOTH_W=0 $PY final_assemble.py sub_n_anom_lgbxgb_nosm lgb_v5x_noll:0.5 xgb_v5x_noll:0.5

log "DONE"
echo "Upload out/sub_n_anom_lgbxgb.csv first."
echo "out/sub_n_anom_lgbxgb_nosm.csv is the same models with the grid smoothing off --"
echo "  that filter is in every submission you have ever made and has never been LB-ablated."
echo "Your previous best is preserved at $B/sub_i_lt_lgbxgb.csv (public 0.709259)."
echo
echo "Sanity check before uploading:"
echo "  $PY lb_se.py $B/sub_i_lt_lgbxgb.csv out/sub_n_anom_lgbxgb.csv"
