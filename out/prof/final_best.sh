#!/bin/sh
# The last-submission pipeline. Trains the FINAL models with both adopted blocks, assembles,
# applies the AR correction in its validated form, then mixes with the record.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2 CARBON=1
LOCK=out/prof/.final_best.lock
mkdir "$LOCK" 2>/dev/null || { echo "another instance holds the lock"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
while pgrep -f "[a]blate_structure|[w]gap_run" >/dev/null 2>&1; do sleep 30; done
say "machine free; training FINAL with XF=gpcc,wgap"
for S in 0 1 2 3 4 5; do
  for fam in "lgb:410" "xgb:230"; do
    M=${fam%%:*}; R=${fam#*:}
    P=out/mats/pred_FINALvn2_${M}_v5x_noll_s${S}_bw1.npy
    [ -f "$P" ] && { say "$M s$S exists"; continue; }
    say "$M s$S rounds=$R"
    XF=gpcc,wgap DROPF="bigsa,gdo" WEIGHTS=ramp HMIX="" SEED=$S TAG=_bw1 \
      $PY run_models.py FINALvn2 $M v5x_noll_sa $R > out/prof/fb_${M}_s${S}.log 2>&1 \
      || { say "  FAILED"; tail -6 out/prof/fb_${M}_s${S}.log; exit 4; }
  done
done
say "assembling"
FLAYOUT=FINALvn2 TAG=_bw1 SMOOTH_W1=0.7 SMOOTH_W7=0.7 SMOOTH_R=1 SMOOTH_IT=1 SMOOTH_WRAP=0 \
  $PY final_assemble.py sub_best_base lgb_v5x_noll:0.5 xgb_v5x_noll:0.5 || exit 5
say "done -> out/sub_best_base.csv"
