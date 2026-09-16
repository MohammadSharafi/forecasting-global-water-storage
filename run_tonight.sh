#!/bin/sh
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

head1 "T0  ERA5 -- wired into build_mats since session 8, never actually downloaded"
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

head1 "T1  blends of the files you already have (no training)"
if [ -f out/sub_q_main.csv ] && [ -f out/sub_q_alt.csv ]; then
  step blend73 "$PY blend_subs.py out/sub_blend73.csv out/sub_q_main.csv:0.7 out/sub_q_alt.csv:0.3" || true
  step blend55 "$PY blend_subs.py out/sub_blend55.csv out/sub_q_main.csv:0.5 out/sub_q_alt.csv:0.5" || true
  done_ blend73 && cat "$S/blend73.log" >> "$R"
else
  say "  sub_q_main.csv or sub_q_alt.csv missing -- run run_night.sh first"
fi

head1 "T2  recursive forecasting: the measurement, not the bound"
for X in A B C; do
  step "recursive_$X" "$PY recursive.py $X" || true
  done_ "recursive_$X" && { say ""; tail -20 "$S/recursive_$X.log" >> "$R"; }
done

head1 "T3  training rows per cell-month -- never swept, in any session"
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

head1 "T4  full pipeline with tonight's decisions"
say "  PER_ROW=$PR, plus rounds.py's correction for FINAL training on ~30% more history than"
say "  the layout whose early stopping set its round count. Both change cfg_guard's signature,"
say "  so the FINAL models retrain rather than being silently reused."
step night "PER_ROW=$PR DEEP=1 ./run_night.sh" || true
tail -40 "$S/night.log" >> "$R" 2>/dev/null

head1 "T5  candidates for tomorrow, best evidence first"
if done_ night && [ -f out/sub_q_main.csv ] && [ -f out/sub_q_alt.csv ]; then
  say "  remaking the blends from the files T4 produced (T1 blended the previous ones)"
  rm -f "$S/blend73.done" "$S/blend55.done"
  step blend73 "$PY blend_subs.py out/sub_blend73.csv out/sub_q_main.csv:0.7 out/sub_q_alt.csv:0.3" || true
  step blend55 "$PY blend_subs.py out/sub_blend55.csv out/sub_q_main.csv:0.5 out/sub_q_alt.csv:0.5" || true
  done_ blend73 && cat "$S/blend73.log" >> "$R"
fi
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
