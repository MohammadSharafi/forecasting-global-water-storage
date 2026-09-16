#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
M=out/mats; mkdir -p "$M"
log() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }

[ -f Train.csv ] || { echo "MISSING Train.csv"; exit 1; }
ls external/era5/*.nc >/dev/null 2>&1 || { echo "MISSING external/era5/*.nc"; exit 1; }

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  for X in A B; do
    log "build_mats $X"
    PER_ROW=${PER_ROW:-2} $PY build_mats.py $X > "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
    log "add_anchor_feats $X (radii ${ANCHOR_RADII:-300,500,800,1500,2500})"
    $PY add_anchor_feats.py $X >> "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
    tail -1 "$M/build_$X.log"
  done
fi
$PY - <<'EOF'
import json, polars as pl
from features_scale import SCALE
f=json.load(open("out/mats/feats.json"))
sa=pl.scan_parquet("out/mats/A_tr_anchor.parquet").collect_schema().names()
print(f"matrix features {len(f)}: anomaly {sum(1 for x in f if x.startswith('an_'))}, "
      f"zonal {sum(1 for x in f if x in SCALE)} | anchor columns {len(sa)}: {sa}")
EOF

run() {  # run <id> <model> <dropf> <weights> <hmix>
  for X in A B; do
    log "$1 $2 dropf='$3' weights=$4 hmix='$5'  layout $X"
    DROPF="$3" WEIGHTS="$4" HMIX="$5" TAG="_$1" SUB="${SUB:-1.0}" \
      $PY run_models.py $X "$2" v5x_noll_sa > "$M/x_${X}_$1.log" 2>&1 \
      || { tail -20 "$M/x_${X}_$1.log"; exit 1; }
    grep -m1 RMSE= "$M/x_${X}_$1.log"
  done
}
run e1 lgb  anom,scale,bigsa ramp    ""
run e2 lgb  scale,bigsa      ramp    ""
run e3 lgb  anom,bigsa       ramp    ""
run e4 lgb  bigsa            ramp    ""
run e8 lgb  ""               ramp    ""
run e5 lgbm ""               ramp    ""
run e6 lgbs ""               ramp    ""
run e7 lgb  ""               uniform ""
run e9 lgb  ""               ramp    test

for X in A B; do
  log "layout $X scored under the REAL TEST horizon mix"
  $PY eval_mix.py $X \
    e1_base=lgb_v5x_noll:_e1 \
    e2_anom=lgb_v5x_noll:_e2 \
    e3_zonal=lgb_v5x_noll:_e3 \
    e4_anom_zonal=lgb_v5x_noll:_e4 \
    e8_plus_bigsa=lgb_v5x_noll:_e8 \
    e5_lgbm63=lgbm_v5x_noll:_e5 \
    e6_lgbs31=lgbs_v5x_noll:_e6 \
    e7_uniform=lgb_v5x_noll:_e7 \
    e9_hmix=lgb_v5x_noll:_e9
done

log "HOW TO READ THIS"
cat <<'EOT'
Use the "testmix" column, not "plain": layout A over-weights h2/h3 by 5.6 points each and
under-weights h1/h4 by the same, so plain RMSE has been flattering mid-horizon methods.

Adopt only what wins on BOTH layouts -- the rule every accepted change has been held to, and
the one the MLP would have failed.

The per-horizon columns are the point. Every transfer failure so far (the MLP, the 500 km
re-base, the climatology pull) should show as a method that wins at long horizons and loses at
h1, because h1 is a third of the test.
EOT
