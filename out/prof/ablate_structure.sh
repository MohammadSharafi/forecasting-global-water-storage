#!/bin/sh
# Does our accumulated training structure help or hurt? Three arms against the shipped control,
# on the two fastest layouts. Each removes ONE thing the pipeline has carried for many sessions:
#   nh  = no HMIX=test horizon-mix reweighting
#   nw  = no WEIGHTS=ramp recency weighting
#   nb  = neither
# The control (_gpcc1) already carries the adopted GPCC block, so this isolates the reweighting.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2 CARBON=1
LOCK=out/prof/.ablate.lock
mkdir "$LOCK" 2>/dev/null || { echo "locked"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
while pgrep -f "[w]gap_run" >/dev/null 2>&1; do sleep 20; done
say "machine free; ablating the training structure"
for X in D E; do
  for arm in "nh:ramp:" "nw::test" "nb::"; do
    tag=$(echo "$arm" | cut -d: -f1); W=$(echo "$arm" | cut -d: -f2); H=$(echo "$arm" | cut -d: -f3)
    for S in 0 1; do
      P=out/mats/pred_${X}_lgb_v5x_noll_s${S}_${tag}.npy
      [ -f "$P" ] && continue
      say "$X $tag s$S  WEIGHTS='$W' HMIX='$H'"
      XF=gpcc DROPF="bigsa,gdo" WEIGHTS="$W" HMIX="$H" SEED=$S TAG=_$tag \
        $PY run_models.py $X lgb v5x_noll_sa > out/prof/ab_${X}_${tag}_s${S}.log 2>&1 \
        || { say "  FAILED"; tail -4 out/prof/ab_${X}_${tag}_s${S}.log; }
    done
  done
done
for tag in nh nw nb; do
  say "--- $tag vs the shipped control"
  $PY goal065_eval.py trained --cand ab_$tag --control _gpcc1 --treat _$tag --layouts D,E --no-append 2>&1 | tail -1
done
say "ablation done"
