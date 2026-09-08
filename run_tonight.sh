#!/bin/sh
# One overnight run that buys ANSWERS and CANDIDATE FILES, for a day when the submission
# allowance is already spent.
#
#   ./run_tonight.sh
#
# Resumable in exactly the way run_night.sh is: every step drops a marker in out/tonight/ and a
# rerun skips what is done. Interrupt it and run the same line again. It never touches a cached
# matrix or prediction that something else depends on -- the training-set sweep writes to its own
# layout names (Ap3, Bp3, ...) precisely so nothing already computed is at risk.
#
# What it does, cheapest and most certain first:
#   T1  blend the existing submission files            seconds, needs no training
#   T2  run the recursive-forecasting MEASUREMENT      the bound says it loses; this settles it
#   T3  sweep training rows per cell-month             never swept in any session
#   T4  the full pipeline with whatever T3 decided, plus the boosting-round correction
#   T5  a ranked list of what to submit tomorrow
cd "$(cd "$(dirname "$0")" && pwd)"; PY=${PY:-./.venv/bin/python}
S=out/tonight; mkdir -p "$S" out/mats; R=$S/report.txt; T0=$(date +%s)

say()  { printf '%s\n' "$*" | tee -a "$R"; }
head1(){ printf '\n=========== %s  (%s) ===========\n' "$1" "$(date '+%m-%d %H:%M')" | tee -a "$R"; }
step() {
  if [ -f "$S/$1.done" ] && [ "${FORCE:-0}" != 1 ]; then printf '  [skip] %s\n' "$1"; return 0; fi
  printf '  [run ] %-34s %s\n' "$1" "$(date '+%H:%M:%S')"
  if ( eval "$2" ) > "$S/$1.log" 2>&1; then touch "$S/$1.done"; return 0; fi
  printf '  [FAIL] %s   (tail of %s)\n' "$1" "$S/$1.log"; tail -8 "$S/$1.log"; return 1
}
done_() { [ -f "$S/$1.done" ]; }

: > "$R"
head1 "TONIGHT   branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null)  commit $(git rev-parse --short HEAD 2>/dev/null)"

# ---------------------------------------------------------------- T0  the missing covariate
head1 "T0  ERA5 -- wired into build_mats since session 8, never actually downloaded"
# The run report's external inventory lists ncep, ncep2, cpc and oni and nothing else, and
# build_mats reads ERA5 only `if glob.glob("external/era5/*.nc")`. Every ERA5 feature has
# therefore been absent from every model this project has trained. NCEP-R1/R2 is ~2 degrees
# and carries no evaporation; ERA5 is 1 degree, matches the target grid exactly, and adds
# evaporation and a four-layer soil column -- and it is the per-cell standardised anomalies
# of exactly this kind of field that produced the whole leaderboard gain in session 9.
if [ -n "$(ls external/era5/*.nc 2>/dev/null)" ]; then
  say "  ERA5 present: $(ls external/era5/*.nc | wc -l | tr -d ' ') files"
elif [ -f "$HOME/.cdsapirc" ]; then
  step era5 "$PY cds_download.py" && say "  ERA5 downloaded" \
    || say "  ERA5 download failed -- see $S/era5.log; the run continues without it"
else
  say "  ERA5 ABSENT and ~/.cdsapirc is missing, so tonight's run cannot use it."
  say "  Five minutes of setup buys it for every future run: see ERA5_SETUP.md."
  say "  The run continues on ncep/ncep2/cpc, exactly as before."
fi

# ---------------------------------------------------------------- T1  free candidates
head1 "T1  blends of the files you already have (no training)"
# stack.py fitted the 31-leaf model's weight to 0.000 on validation, and the leaderboard says that
# model alone is within 0.002 of the best. Validation has been wrong about this shape of question
# before -- it adopted the calibration the leaderboard then refuted -- so make the blend and let
# tomorrow's submissions decide instead of the fitted zero.
if [ -f out/sub_q_main.csv ] && [ -f out/sub_q_alt.csv ]; then
  step blend73 "$PY blend_subs.py out/sub_blend73.csv out/sub_q_main.csv:0.7 out/sub_q_alt.csv:0.3" || true
  step blend55 "$PY blend_subs.py out/sub_blend55.csv out/sub_q_main.csv:0.5 out/sub_q_alt.csv:0.5" || true
  done_ blend73 && cat "$S/blend73.log" >> "$R"
else
  say "  sub_q_main.csv or sub_q_alt.csv missing -- run run_night.sh first"
fi

# ---------------------------------------------------------------- T2  settle recursion
head1 "T2  recursive forecasting: the measurement, not the bound"
# Explicitly permitted by the organisers. The arithmetic in NOTES says it loses, but that argument
# rests on per-horizon numbers confounded with calendar month, so it is a bound and not a result.
for X in A B C; do
  step "recursive_$X" "$PY recursive.py $X" || true
  done_ "recursive_$X" && { say ""; tail -20 "$S/recursive_$X.log" >> "$R"; }
done

# ---------------------------------------------------------------- T3  training-set size
head1 "T3  training rows per cell-month -- never swept, in any session"
# build_mats' own default is 3; the orchestrator has been passing 2 since it was written. A layout
# may carry its own PER_ROW in its name, so these build to Ap3_*.parquet and leave A_* untouched.
. ./out/config.sh 2>/dev/null || true
FD=${FINAL_DROPF-}; FM=${FINAL_MODEL:-lgb}; FW=${FINAL_WEIGHTS:-ramp}; FH=${FINAL_HMIX-}
say "  sweeping with the chosen configuration: dropf='$FD' model=$FM weights=$FW hmix='$FH'"
SWEEP=${SWEEP:-3}
for v in $SWEEP; do
  for X in A B; do
    step "build_${X}p${v}"  "$PY build_mats.py ${X}p${v}"      || continue
    step "anchor_${X}p${v}" "$PY add_anchor_feats.py ${X}p${v}" || continue
    step "cv_${X}p${v}" "DROPF='$FD' WEIGHTS=$FW HMIX='$FH' TAG=_bw $PY run_models.py ${X}p${v} $FM v5x_noll_sa" || true
  done
done
step perrowscan "$PY perrow_scan.py '${FM}_v5x_noll:_bw' 2 $SWEEP > out/perrow.sh" || true
if [ -s out/perrow.sh ]; then cat "$S/perrowscan.log" >> "$R"; . ./out/perrow.sh; fi
PR=${FINAL_PER_ROW:-2}
say "  PER_ROW for the final build: $PR"

# ---------------------------------------------------------------- T4  the real pipeline
head1 "T4  full pipeline with tonight's decisions"
say "  PER_ROW=$PR, plus rounds.py's correction for FINAL training on ~30% more history than"
say "  the layout whose early stopping set its round count. Both change cfg_guard's signature,"
say "  so the FINAL models retrain rather than being silently reused."
step night "PER_ROW=$PR DEEP=1 ./run_night.sh" || true
tail -40 "$S/night.log" >> "$R" 2>/dev/null

# ---------------------------------------------------------------- T5  what to submit
head1 "T5  candidates for tomorrow, best evidence first"
say ""
say "  file                        what it is"
for f in out/sub_q_main.csv "the pipeline's answer with tonight's decisions" \
         out/sub_blend73.csv "0.7 main + 0.3 alt -- tests the ensemble weight the fit zeroed" \
         out/sub_blend55.csv "0.5/0.5 of the same two" \
         out/sub_q_alt.csv   "different capacity -- the honest second private slot" \
         out/sub_q_main_nosm.csv "no smoothing, as a control"; do
  case $f in out/*) [ -f "$f" ] && printf '  %-27s ' "$f" | tee -a "$R";; *) say "$f";; esac
done
say ""
say "  how far each sits from the current best on the public board:"
for f in out/sub_blend73.csv out/sub_blend55.csv out/sub_q_main.csv; do
  [ -f "$f" ] && [ -f out/backup_pre_anom/sub_i_lt_lgbxgb.csv ] && {
    say "  --- $f"; $PY lb_se.py out/backup_pre_anom/sub_i_lt_lgbxgb.csv "$f" 2>/dev/null | sed 's/^/    /' | tee -a "$R"; }
done
say ""
say "  steps that did not complete:"
for l in "$S"/*.log; do b=$(basename "$l" .log); [ -f "$S/$b.done" ] || say "    $b   ($l)"; done
say ""
say "  elapsed: $(( ( $(date +%s) - T0 ) / 60 )) minutes"
say "  send me: out/tonight/report.txt, out/perrow.sh, out/rounds.sh, out/config.sh,"
say "           out/carbon/summary.md, and out/tonight/recursive_A.log"
