#!/bin/sh
# Session 9d experiment grid: run every queued hypothesis on BOTH validation layouts and
# report each one under the real test's horizon mix.  Nothing is adopted here -- this
# prints the evidence, you decide, then pipeline14.sh builds the submission.
#
#   ./pipeline13.sh              # full run, ~4-6 h
#   SUB=0.5 ./pipeline13.sh      # half the training rows, ~2-3 h, same ordering, noisier
#
# Run this AFTER pipeline12.sh has finished; it rebuilds the matrices again because the
# zonal-scale features (features_scale.py) were added after pipeline12 started.
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
M=out/mats; mkdir -p "$M"
log() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }

[ -f Train.csv ] || { echo "MISSING Train.csv"; exit 1; }
ls external/era5/*.nc >/dev/null 2>&1 || { echo "MISSING external/era5/*.nc"; exit 1; }

# ---------------------------------------------------------------- matrices (A and B only)
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  for X in A B; do
    log "build_mats $X"
    PER_ROW=${PER_ROW:-2} $PY build_mats.py $X > "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
    $PY add_anchor_feats.py $X >> "$M/build_$X.log" 2>&1 || { tail -20 "$M/build_$X.log"; exit 1; }
    tail -1 "$M/build_$X.log"
  done
fi
$PY -c 'import json;f=json.load(open("out/mats/feats.json"));from features_scale import SCALE;print(f"features {len(f)}  anomaly {sum(1 for x in f if x.startswith(chr(97)+chr(110)+chr(95)))}  scale {sum(1 for x in f if x in SCALE)}")'

# ---------------------------------------------------------------- the experiment grid
# id | model | features dropped | training weights      what it tests
# e1 | lgb   | anom,scale       | ramp     the pre-session-9 baseline, the thing to beat
# e2 | lgb   | scale            | ramp     per-cell covariate anomalies (features_anom)
# e3 | lgb   | anom             | ramp     zonal-scale context (features_scale)
# e4 | lgb   | -                | ramp     both feature groups together
# e5 | lgbm  | -                | ramp     63 leaves: does less capacity generalise better?
# e6 | lgbs  | -                | ramp     31 leaves, stronger L2: further down the ladder
# e7 | lgb   | -                | uniform  drop the recent-year ramp
run() {  # run <id> <model> <dropf> <weights>
  for X in A B; do
    log "$1 $2 dropf='$3' weights=$4  layout $X"
    DROPF="$3" WEIGHTS="$4" TAG="_$1" SUB="${SUB:-1.0}" \
      $PY run_models.py $X "$2" v5x_noll_sa > "$M/x_${X}_$1.log" 2>&1 \
      || { tail -20 "$M/x_${X}_$1.log"; exit 1; }
    grep -m1 RMSE= "$M/x_${X}_$1.log"
  done
}
run e1 lgb  anom,scale ramp
run e2 lgb  scale      ramp
run e3 lgb  anom       ramp
run e4 lgb  ""         ramp
run e5 lgbm ""         ramp
run e6 lgbs ""         ramp
run e7 lgb  ""         uniform

# ---------------------------------------------------------------- report
for X in A B; do
  log "layout $X scored under the REAL TEST horizon mix"
  $PY eval_mix.py $X \
    e1_base=lgb_v5x_noll:_e1 \
    e2_anom=lgb_v5x_noll:_e2 \
    e3_scale=lgb_v5x_noll:_e3 \
    e4_both=lgb_v5x_noll:_e4 \
    e5_lgbm=lgbm_v5x_noll:_e5 \
    e6_lgbs=lgbs_v5x_noll:_e6 \
    e7_uniform=lgb_v5x_noll:_e7
done

log "HOW TO READ THIS"
cat <<'EOT'
Use the "testmix" column, not "plain": layout A over-weights h2/h3 by 5.6 points each and
under-weights h1/h4 by the same, so plain RMSE has been flattering mid-horizon methods.

Adopt a change only if it wins on BOTH layouts under testmix -- the rule every accepted
change in this project has been held to, and the one the MLP would have failed.

The per-horizon columns are the point of the exercise. Every transfer failure so far
(the MLP, the 500 km re-base, the climatology pull) should show up as a method that wins
at long horizons and loses at h1, because h1 is a third of the test.

Then build the submission with the winning configuration:
  FINAL_MODEL=lgb FINAL_DROPF=scale FINAL_WEIGHTS=ramp ./pipeline14.sh
EOT
