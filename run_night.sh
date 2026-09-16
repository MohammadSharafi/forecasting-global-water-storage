#!/bin/sh
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
export CARBON=${CARBON:-1}
M=out/mats; S=out/night; mkdir -p "$M" "$S"
SEEDS=${SEEDS:-8}; [ "${DEEP:-0}" = 1 ] && SEEDS=${SEEDS_DEEP:-16}
WIDE=${WIDE:-${DEEP:-0}}                  # extra model families, so the stack has something to fit
LAYOUT_C=${LAYOUT_C:-${DEEP:-0}}          # the third validation layout: the test's own block shape
BUDGET_H=${BUDGET_H:-$([ "${DEEP:-0}" = 1 ] && echo 24 || echo 12)}
T0=$(date +%s)
R=$S/report.txt

say()  { printf '%s\n' "$*" | tee -a "$R"; }
head1(){ printf '\n=========== %s  (%s) ===========\n' "$1" "$(date '+%m-%d %H:%M')" | tee -a "$R"; }
elapsed_h() { echo $(( ( $(date +%s) - T0 ) / 3600 )); }
have_time() {
  used=$(( ( $(date +%s) - T0 ) / 60 ))
  [ $(( used + ${1:-1} * 60 )) -le $(( BUDGET_H * 60 )) ] && return 0
  say "  [budget] ${used}m used of $(( BUDGET_H * 60 ))m -- skipping a phase that needs ~${1}h"
  return 1
}

step() {
  if [ -f "$S/$1.done" ] && [ "${FORCE:-0}" != 1 ]; then printf '  [skip] %s\n' "$1"; return 0; fi
  printf '  [run ] %-38s %s\n' "$1" "$(date '+%H:%M:%S')"
  if ( eval "$2" ) > "$S/$1.log" 2>&1; then touch "$S/$1.done"; return 0; fi
  printf '  [FAIL] %s   (tail of %s)\n' "$1" "$S/$1.log"; tail -12 "$S/$1.log"; return 1
}
astep() {
  printf '  [asm ] %-38s %s\n' "$1" "$(date '+%H:%M:%S')"
  if ( eval "$2" ) > "$S/$1.log" 2>&1; then touch "$S/$1.done"; return 0; fi
  printf '  [FAIL] %s   (tail of %s)\n' "$1" "$S/$1.log"; tail -12 "$S/$1.log"; return 1
}
done_() { [ -f "$S/$1.done" ]; }

src_guard() {   # src_guard <marker prefix> <source files...>
  t=$1; shift
  sig=$(cat "$@" 2>/dev/null | cksum | cut -d' ' -f1)
  f="$S/src_$t.txt"
  if [ -f "$f" ] && [ "$(cat "$f")" != "$sig" ]; then
    n=$(ls "$S"/${t}*.done 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && say "  [source changed] $t: discarding $n cached step(s) so they re-run"
    rm -f "$S"/${t}*.done
  fi
  printf '%s' "$sig" > "$f"
}

: > "$R"
head1 "RUN START   seeds=$SEEDS wide=$WIDE deep=${DEEP:-0} layoutC=$LAYOUT_C budget=${BUDGET_H}h"
say "branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null)  commit $(git rev-parse --short HEAD 2>/dev/null)"

step backup 'mkdir -p out/backup_pre_anom && cp out/*.csv out/backup_pre_anom/ 2>/dev/null; ls out/backup_pre_anom | wc -l'
say "submissions backed up: $(ls out/backup_pre_anom 2>/dev/null | wc -l) files"

head1 "PHASE 1  the data, and diagnostics on the current state"
step data_report   "$PY data_report.py" || true
done_ data_report && sed -n '/TEST GEOMETRY/,/^$/p' "$S/data_report.log" >> "$R"
step diag_evalmix_A "$PY eval_mix.py A base=lgb_v5x_noll_noanom anom=lgb_v5x_noll" || true
step diag_evalmix_B "$PY eval_mix.py B base=lgb_v5x_noll_noanom anom=lgb_v5x_noll" || true
step diag_ceiling_A "$PY ceiling.py A"     || true
step diag_ceiling_B "$PY ceiling.py B"     || true
step diag_shift_A   "$PY globalshift.py A lgb_v5x_noll" || true
step diag_shift_B   "$PY globalshift.py B lgb_v5x_noll" || true

src_guard val_C      validation_c.py
printf 'PER_ROW=%s PROF=%s ANWIDE_R=%s' "${PER_ROW:-2}" "${PROF:-0}" "${ANWIDE_R:-4}" > "$S/perrow.txt"   # a different PER_ROW or soil-profile setting is a different matrix
find external -type f \( -name '*.nc' -o -name '*.parquet' -o -name '*.data' \) \
     -exec wc -c {} \; 2>/dev/null | sort > "$S/extinv.txt"
src_guard build_     "$S/extinv.txt" "$S/perrow.txt" build_mats.py features.py features2.py features4.py features5.py features6.py \
                     features_ncep.py features_era5.py features_x.py features_anom.py \
                     features_scale.py features_gdo.py
src_guard anchor_    add_anchor_feats.py anchor.py
src_guard x_         run_models.py
src_guard cv_        run_models.py
src_guard h1_        run_models.py
src_guard eval_grid_ eval_mix.py
src_guard select     select_config.py eval_mix.py xfit.py
src_guard analyze_   analyze.py
src_guard roundscan  rounds.py
src_guard check_FINAL check_final.py
src_guard F          run_models.py
src_guard blendscan   blend_scan.py xfit.py
src_guard stackscan   stack.py xfit.py
src_guard smoothscan  smooth_scan.py smooth.py xfit.py
src_guard hsplicescan hsplice.py xfit.py
src_guard seasonscan  seasonal.py xfit.py
src_guard postcalscan postcal.py xfit.py

head1 "PHASE 2  rebuild the validation layouts (covariate anomalies, windows, modelled TWS, zonal, 5 anchor radii)"
LAYOUTS="A B"
if [ "$LAYOUT_C" = 1 ]; then
  step val_C "$PY validation_c.py" && LAYOUTS="A B C"
  done_ val_C && cat "$S/val_C.log" >> "$R"
fi
export LAYOUTS
for X in $LAYOUTS; do
  step "build_$X"  "PER_ROW=${PER_ROW:-2} $PY build_mats.py $X" || true
  done_ "build_$X" && { step "anchor_$X" "$PY add_anchor_feats.py $X" || true; }
done
done_ build_A && say "$($PY - <<'EOF'
import json,polars as pl
from features_scale import SCALE
f=json.load(open("out/mats/feats.json"))
sa=pl.scan_parquet("out/mats/A_tr_anchor.parquet").collect_schema().names()
print(f"features {len(f)}: anomaly {sum(1 for x in f if x.startswith('an_'))}, zonal {sum(1 for x in f if x in SCALE)}; anchors {len(sa)}")
EOF
)"

head1 "PHASE 3  experiment grid, each run resumable on its own"
grid='e1 lgb  anom,scale,bigsa  ramp     -
e2 lgb  scale,bigsa       ramp     -
e3 lgb  anom,bigsa        ramp     -
e4 lgb  bigsa             ramp     -
e8 lgb  ""                ramp     -
e5 lgbm ""                ramp     -
e6 lgbs ""                ramp     -
e7 lgb  ""                uniform  -
e9 lgb  ""                ramp     test'
echo "$grid" | while read id model dropf weights hmix; do
  [ -n "$id" ] || continue
  [ "$dropf" = '""' ] && dropf=""
  [ "$hmix"  = '-'  ] && hmix=""
  for X in A B; do
    step "x_${X}_$id" "DROPF='$dropf' WEIGHTS=$weights HMIX='$hmix' TAG=_$id SUB=${SUB:-1.0} $PY run_models.py $X $model v5x_noll_sa" || true
  done
done

if [ "$LAYOUT_C" = 1 ] && done_ build_C && have_time 2; then
  for id in e1 e4 e8; do
    d=""; [ "$id" = e1 ] && d="anom,scale,bigsa"; [ "$id" = e4 ] && d="bigsa"
    step "x_C_$id" "DROPF='$d' WEIGHTS=ramp TAG=_$id SUB=${SUB:-1.0} $PY run_models.py C lgb v5x_noll_sa" || true
  done
fi

head1 "PHASE 3b  grid scored under the REAL TEST horizon mix"
FULL="e1_base=lgb_v5x_noll:_e1 e2_anom=lgb_v5x_noll:_e2 e3_zonal=lgb_v5x_noll:_e3 e4_anom_zonal=lgb_v5x_noll:_e4 e8_bigsa=lgb_v5x_noll:_e8 e5_lgbm63=lgbm_v5x_noll:_e5 e6_lgbs31=lgbs_v5x_noll:_e6 e7_uniform=lgb_v5x_noll:_e7 e9_hmix=lgb_v5x_noll:_e9"
FINALISTS="e1_base=lgb_v5x_noll:_e1 e4_anom_zonal=lgb_v5x_noll:_e4 e8_bigsa=lgb_v5x_noll:_e8"
for X in $LAYOUTS; do
  sets=$FULL; [ "$X" = C ] && sets=$FINALISTS
  step "eval_grid_$X" "$PY eval_mix.py $X $sets" || true
  done_ "eval_grid_$X" && cat "$S/eval_grid_$X.log" >> "$R"
done

head1 "PHASE 4  configuration choice"
step select "$PY select_config.py > out/config.sh" || true
if [ -s out/config.sh ]; then cat "$S/select.log" >> "$R"; cat out/config.sh | tee -a "$R"; . ./out/config.sh; fi
FD=${FINAL_DROPF-}; FM=${FINAL_MODEL:-lgb}; FW=${FINAL_WEIGHTS:-ramp}; FH=${FINAL_HMIX-}

step diag2_ceiling_A "$PY ceiling.py A" || true
step diag2_shift_A   "$PY globalshift.py A ${FM}_v5x_noll _e8" || true

head1 "PHASE 5  the chosen configuration, every family, tag _bw"
CV="$FM xgb"
if [ "$WIDE" = 1 ]; then
  for m in lgbs lgbm cat; do [ "$m" = "$FM" ] || CV="$CV $m"; done
fi
say "families on validation: $CV"
for X in $LAYOUTS; do
  for m in $CV; do
    case $m in cat|lgbm|lgbs) have_time 2 || continue;; esac
    step "cv_${X}_${m}" "DROPF='$FD' WEIGHTS=$FW HMIX='$FH' TAG=_bw $PY run_models.py $X $m v5x_noll_sa" || true
  done
done
for X in $LAYOUTS; do
  step "h1_$X" "DROPF='$FD' WEIGHTS=$FW HFILT=1 TAG=_h1 $PY run_models.py $X $FM v5x_noll_sa" || true
done

head1 "PHASE 5b  everything fitted on top of the predictions, by leave-one-layout-out"
step blendscan "$PY blend_scan.py ${FM}_v5x_noll xgb_v5x_noll _bw > out/blendw.sh" || true
if [ -s out/blendw.sh ]; then cat "$S/blendscan.log" >> "$R"; . ./out/blendw.sh; fi
WL=${FINAL_WLGB:-0.5}; WX=$(awk "BEGIN{printf \"%.2f\", 1-$WL}")

XTRA=""; for m in $CV; do case $m in "$FM"|xgb) ;; *) XTRA="$XTRA,${m}_v5x_noll";; esac; done
XTRA=${XTRA#,}
step stackscan "$PY stack.py '${FM}_v5x_noll:$WL,xgb_v5x_noll:$WX' _bw '$XTRA' > out/stack.sh" || true
if [ -s out/stack.sh ]; then cat "$S/stackscan.log" >> "$R"; . ./out/stack.sh; fi
STK=${FINAL_STACK-}
if [ -n "$STK" ]; then
  MEMBERS=$(echo "$STK" | tr ',' '\n' | awk -F: '$2+0>0.005 {printf "%s:%s ", $1, $2}')
  FAMS=$(echo "$STK" | tr ',' '\n' | awk -F: '$2+0>0.005 {sub(/_v5x_noll$/,"",$1); printf "%s ", $1}')
else
  MEMBERS="${FM}_v5x_noll:$WL xgb_v5x_noll:$WX"; FAMS="$FM xgb"
fi
say "ensemble members: $MEMBERS"

BLEND=$(echo "$MEMBERS" | tr ' ' '\n' | awk -F: 'NF{printf "%s:_bw,", $1}'); BLEND=${BLEND%,}
step smoothscan "$PY smooth_scan.py '$BLEND' > out/smoothw.sh" || true
if [ -s out/smoothw.sh ]; then cat "$S/smoothscan.log" >> "$R"; . ./out/smoothw.sh; fi
SW1=${FINAL_SMOOTH_W1:-0.7}; SW7=${FINAL_SMOOTH_W7:-0.7}
SR=${FINAL_SMOOTH_R:-1}; SI=${FINAL_SMOOTH_IT:-1}; SP=${FINAL_SMOOTH_WRAP:-0}
export SMOOTH_W1=$SW1 SMOOTH_W7=$SW7 SMOOTH_R=$SR SMOOTH_IT=$SI SMOOTH_WRAP=$SP

step hsplicescan "$PY hsplice.py '$BLEND' '${FM}_v5x_noll:_h1' > out/h1beta.sh" || true
if [ -s out/h1beta.sh ]; then cat "$S/hsplicescan.log" >> "$R"; . ./out/h1beta.sh; fi
HB=${FINAL_H1BETA:-0}

step seasonscan "$PY seasonal.py '$BLEND' > out/season.sh" || true
if [ -s out/season.sh ]; then cat "$S/seasonscan.log" >> "$R"; . ./out/season.sh; fi
SE=${FINAL_SEASON-}
[ -n "$SE" ] && say "  seasonal bias correction adopted: $SE" || say "  no seasonal bias correction"

step postcalscan "$PY postcal.py '$BLEND' > out/calib.sh" || true
if [ -s out/calib.sh ]; then cat "$S/postcalscan.log" >> "$R"; . ./out/calib.sh; fi
CB=""; CBB=""
if [ "${USE_CALIB:-0}" = 1 ]; then CB=${FINAL_CALIB-}; CBB=${FINAL_CALIB_B-}
else say "  calibration computed but NOT applied (refuted on the leaderboard; USE_CALIB=1 to apply)"; fi
say "post-processing: smoothing w1=$SW1 w7=$SW7 r=$SR it=$SI wrap=$SP | h1 beta=$HB"
say "  calibration scale='${CB:-none}' offset='${CBB:-none}'"

head1 "PHASE 5c  where the remaining error is"
for X in $LAYOUTS; do
  step "analyze_$X" "$PY analyze.py $X '$BLEND'" || true
done

head1 "PHASE 6  FINAL matrix"
step build_FINAL  "PER_ROW=${PER_ROW:-2} $PY build_mats.py FINAL" || true
done_ build_FINAL && { step anchor_FINAL "$PY add_anchor_feats.py FINAL" || true; }
step check_FINAL "$PY check_final.py" || true

step roundscan "$PY rounds.py $S > out/rounds.sh" || true
if [ -s out/rounds.sh ]; then cat "$S/roundscan.log" >> "$R"; . ./out/rounds.sh; fi

rd() {
  case $1 in
    lgb) r=${LGB_ROUNDS-};; lgbs) r=${LGBS_ROUNDS-};; lgbm) r=${LGBM_ROUNDS-};;
    xgb) r=${XGB_ROUNDS-};; cat) r=${CAT_ROUNDS-};;   *) r=;;
  esac
  [ -n "$r" ] || r=$(grep -o 'best_iter=[0-9]*' "$S/cv_A_$1.log" 2>/dev/null | tail -1 | cut -d= -f2)
  [ -n "$r" ] || case $1 in xgb) r=400;; cat) r=1500;; *) r=300;; esac
  echo "$r"
}
H1R=$(grep -o 'best_iter=[0-9]*' "$S/h1_A.log" 2>/dev/null | tail -1 | cut -d= -f2); H1R=${H1R:-$(rd "$FM")}
say "rounds: $(for m in $FAMS; do printf '%s=%s ' "$m" "$(rd $m)"; done) h1=$H1R"
say "config: dropf='$FD' weights=$FW hmix='$FH'"

cfg_guard() {   # cfg_guard <tag> <dropf> <weights> <hmix> <families...>
  t=$1; d=$2; w=$3; x=$4; shift 4
  sig="dropf=$d weights=$w hmix=$x fams=$* seeds=$SEEDS"
  for m in "$@"; do sig="$sig r_$m=$(rd "$m")"; done
  f="$S/cfg$t.txt"
  if [ -f "$f" ] && [ "$(cat "$f")" != "$sig" ]; then
    n=$(ls "$S"/F${t}_*.done 2>/dev/null | wc -l)
    say "  [config changed for $t] was: $(cat "$f")"
    say "                          now: $sig"
    say "  discarding $n cached training step(s) for $t so they are retrained"
    rm -f "$S"/F${t}_*.done
  fi
  printf '%s' "$sig" > "$f"
}

train_final() {   # train_final <tag> <dropf> <weights> <hmix> <families...>
  cfg_guard "$@"
  t=$1; d=$2; w=$3; x=$4; shift 4
  for m in "$@"; do
    r=$(rd "$m"); s=0
    while [ "$s" -lt "$SEEDS" ]; do
      step "F${t}_${m}_s$s" "SEED=$s DROPF='$d' WEIGHTS=$w HMIX='$x' TAG=$t $PY run_models.py FINAL $m v5x_noll_sa $r" || true
      s=$((s+1))
    done
  done
}

asm() {  # asm <name> <tag> <members> <smoothing override, empty = the tuned one>
  env="TAG=$2"
  if [ -n "$4" ]; then env="$env SMOOTH_W=$4 SMOOTH_W1=$4 SMOOTH_W7=$4 SMOOTH_R=1 SMOOTH_IT=1 SMOOTH_WRAP=0"; fi
  [ -n "$SE" ] && env="$env SEASON=$SE"
  [ -n "$CB" ] && env="$env CALIB=$CB"
  [ -n "$CBB" ] && env="$env CALIB_B=$CBB"
  if [ "$HB" != "0" ] && [ "$HB" != "0.00" ] && done_ "FH1_${FM}_s0"; then
    env="$env H1SPEC=${FM}_v5x_noll H1TAG=_h1 H1BETA=$HB"
  fi
  astep "asm_$1" "$env $PY final_assemble.py $1 $3" || true
}

if done_ check_FINAL; then
  head1 "PHASE 7  FINAL config 1 (the grid winner), $SEEDS seeds x $(echo $FAMS | wc -w) families"
  train_final _f1 "$FD" "$FW" "$FH" $FAMS
  asm sub_q_main _f1 "$MEMBERS" ""

  if [ "$HB" != "0" ] && [ "$HB" != "0.00" ]; then
    head1 "PHASE 7b  FINAL horizon-1 specialist (beta=$HB), $SEEDS seeds"
    s=0
    while [ "$s" -lt "$SEEDS" ]; do
      step "FH1_${FM}_s$s" "SEED=$s DROPF='$FD' WEIGHTS=$FW HFILT=1 TAG=_h1 $PY run_models.py FINAL $FM v5x_noll_sa $H1R" || true
      s=$((s+1))
    done
    asm sub_q_main _f1 "$MEMBERS" ""
  else
    say "no horizon-1 specialist adopted -- phase 7b skipped"
  fi

  if have_time 3; then
    head1 "PHASE 8  FINAL config 2 (a deliberately different capacity, for the second slot)"
    ALT=lgbs; [ "$FM" = lgbs ] && ALT=lgb
    train_final _f2 "$FD" "$FW" "$FH" "$ALT" xgb
  fi

  if [ "${DEEP:-0}" = 1 ] && have_time 3; then
    head1 "PHASE 8b  FINAL config 3 (DEEP): the pre-session-9 feature set, as an insurance file"
    train_final _f3 "anom,scale,bigsa" ramp "" lgb xgb
  fi
else
  say "FINAL matrix not ready -- skipping training. Fix build_FINAL and rerun."
fi

head1 "PHASE 9  submissions, audit, report"
if done_ "F_f1_${FM}_s0"; then
  asm sub_q_main      _f1 "$MEMBERS" ""     # the tuned post-processing
  asm sub_q_main_nosm _f1 "$MEMBERS" 0      # no smoothing at all, as a control
  CBK=$CB; CBBK=$CBB; CB=${FINAL_CALIB-}; CBB=${FINAL_CALIB_B-}
  [ -n "$CB" ] && asm sub_q_main_cal _f1 "$MEMBERS" ""
  CB=$CBK; CBB=$CBBK
  if [ -n "$SE" ]; then SEK=$SE; SE=""; asm sub_q_main_noseason _f1 "$MEMBERS" ""; SE=$SEK; fi
fi
ALT=lgbs; [ "$FM" = lgbs ] && ALT=lgb
done_ "F_f2_${ALT}_s0" && asm sub_q_alt _f2 "${ALT}_v5x_noll:$WL xgb_v5x_noll:$WX" ""
[ "${DEEP:-0}" = 1 ] && done_ "F_f3_lgb_s0" && asm sub_q_base _f3 "lgb_v5x_noll:$WL xgb_v5x_noll:$WX" ""

step compliance "$PY compliance.py FINAL" || true
astep carbon "$PY carbon_report.py" || say "  emissions not summarised (was CARBON=1 set for the training runs?)"
done_ compliance || say "COMPLIANCE AUDIT FAILED OR INCOMPLETE -- read $S/compliance.log before submitting"

head1 "REPORT"
say "elapsed: $(elapsed_h) h of a ${BUDGET_H} h budget"
say "files ready to upload:"
for f in out/sub_q_main.csv out/sub_q_main_nosm.csv out/sub_q_main_cal.csv out/sub_q_main_noseason.csv out/sub_q_alt.csv \
         out/sub_q_base.csv out/sub_n_anom_lgbxgb.csv out/backup_pre_anom/sub_i_lt_lgbxgb.csv; do
  [ -f "$f" ] && say "  $f"
done
say ""
say "how different is each from the 0.709259 reference (gaps below the threshold are noise):"
for f in out/sub_q_main.csv out/sub_q_alt.csv; do
  [ -f "$f" ] && { say "  --- $f"; $PY lb_se.py out/backup_pre_anom/sub_i_lt_lgbxgb.csv "$f" 2>/dev/null | sed 's/^/    /' | tee -a "$R"; }
done
say ""
say "steps that did not complete:"
for l in "$S"/*.log; do b=$(basename "$l" .log); [ -f "$S/$b.done" ] || say "  $b   ($l)"; done
$PY report_night.py "$S" | tee -a "$R"
say ""
say "FULL REPORT: out/RUN_REPORT.md   (raw log: $R)"
