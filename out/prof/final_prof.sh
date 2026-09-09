#!/bin/sh
# Train FINAL on the soil-profile + GDO matrix and assemble a submission, resumably.
#   DROPF is passed in, so the caller decides what the ablation adopted.
#   sh out/prof/final_prof.sh 'bigsa'            both new families kept
#   sh out/prof/final_prof.sh 'bigsa,e5prof'     GDO only
#   sh out/prof/final_prof.sh 'bigsa,gdo'        profile only
cd "$(cd "$(dirname "$0")/../.." && pwd)"; PY=./.venv/bin/python
FD=${1:-bigsa}; L=FINALe; TAG=_g1; SEEDS=${SEEDS:-16}
LGBR=${LGBR:-410}; XGBR=${XGBR:-230}; H1R=${H1R:-353}; HB=${HB:-0.50}
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

say "FINAL soil-profile run: DROPF='$FD' seeds=$SEEDS rounds lgb=$LGBR xgb=$XGBR h1=$H1R beta=$HB"

if [ ! -f out/mats/${L}_tr.parquet ]; then
  say "build $L"; $PY build_mats.py $L > out/prof/build_$L.log 2>&1 || { say "BUILD FAILED"; tail -20 out/prof/build_$L.log; exit 1; }
  tail -1 out/prof/build_$L.log
fi
if [ ! -f out/mats/${L}_tr_anchor.parquet ]; then
  say "anchors $L"; $PY add_anchor_feats.py $L > out/prof/anchor_$L.log 2>&1 || { say "ANCHOR FAILED"; tail -10 out/prof/anchor_$L.log; exit 1; }
fi

# the 261-feature FINAL carried 201 of them; check the new matrix really has the new families
$PY - <<'PY'
import polars as pl
c = pl.scan_parquet("out/mats/FINALe_va.parquet").collect_schema().names()
prof = [x for x in c if x.startswith(("e5SW1","e5SW2","e5SW3","e5SW4","an_e5SW1","an_e5SW2","an_e5SW3","an_e5SW4"))]
gdo  = [x for x in c if x.startswith(("an_spi24","an_spi48"))]
print(f"  FINALe: {len(c)} columns, {len(prof)} soil-profile, {len(gdo)} GDO")
assert prof and gdo, "the new feature families are missing from FINALe"
PY
[ $? -eq 0 ] || exit 1

for m in lgb xgb; do
  r=$LGBR; [ "$m" = xgb ] && r=$XGBR
  s=0
  while [ "$s" -lt "$SEEDS" ]; do
    f=out/mats/pred_${L}_${m}_v5x_noll_s${s}${TAG}.npy
    if [ -f "$f" ]; then s=$((s+1)); continue; fi
    say "  train $m s$s"
    SEED=$s DROPF="$FD" WEIGHTS=ramp HMIX=test TAG=$TAG CARBON=1 \
      $PY run_models.py $L $m v5x_noll_sa $r > out/prof/F_${m}_s${s}.log 2>&1 \
      || { say "  FAILED $m s$s"; tail -5 out/prof/F_${m}_s${s}.log; }
    s=$((s+1))
  done
done

# the horizon-1 specialist, spliced at the beta the scan adopted
s=0
while [ "$s" -lt "$SEEDS" ]; do
  f=out/mats/pred_${L}_lgb_v5x_noll_s${s}_h1g.npy
  if [ -f "$f" ]; then s=$((s+1)); continue; fi
  say "  train h1 s$s"
  SEED=$s DROPF="$FD" WEIGHTS=ramp HFILT=1 TAG=_h1g CARBON=1 \
    $PY run_models.py $L lgb v5x_noll_sa $H1R > out/prof/FH1_s${s}.log 2>&1 \
    || { say "  FAILED h1 s$s"; tail -5 out/prof/FH1_s${s}.log; }
  s=$((s+1))
done

say "assemble"
FLAYOUT=$L TAG=$TAG SMOOTH_W1=0.7 SMOOTH_W7=0.7 SMOOTH_R=1 SMOOTH_IT=1 SMOOTH_WRAP=0 \
  H1SPEC=lgb_v5x_noll H1TAG=_h1g H1BETA=$HB \
  $PY final_assemble.py sub_r_prof lgb_v5x_noll:0.5 xgb_v5x_noll:0.5 2>&1 | tee out/prof/assemble.log
FLAYOUT=$L TAG=$TAG SMOOTH_W=0 H1SPEC=lgb_v5x_noll H1TAG=_h1g H1BETA=$HB \
  $PY final_assemble.py sub_r_prof_nosm lgb_v5x_noll:0.5 xgb_v5x_noll:0.5 > out/prof/assemble_nosm.log 2>&1
say "done"
