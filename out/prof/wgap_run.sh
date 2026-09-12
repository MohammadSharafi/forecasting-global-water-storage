#!/bin/sh
# Measure gpcc+wgap against gpcc alone, so the control already has the adopted block and the only
# difference is WaterGAP. Fastest layouts first; stop after two losing layouts.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
PY=./.venv/bin/python
export PER_ROW=2 CARBON=1
LOCK=out/prof/.wgap.lock
mkdir "$LOCK" 2>/dev/null || { echo "another instance holds the lock"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
say(){ printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }
while [ "$(ls out/mats/*_va_wgap.parquet 2>/dev/null | wc -l | tr -d ' ')" -lt 6 ]; do sleep 20; done
say "all side-cars present"
for X in D E Cvn2 Bvn2 Avn2; do
  for S in 0 1; do
    P=out/mats/pred_${X}_lgb_v5x_noll_s${S}_wg1.npy
    [ -f "$P" ] && continue
    say "$X s$S  XF=gpcc,wgap"
    XF=gpcc,wgap DROPF="bigsa,gdo" WEIGHTS=ramp HMIX=test SEED=$S TAG=_wg1 \
      $PY run_models.py $X lgb v5x_noll_sa > out/prof/wg_${X}_s${S}.log 2>&1 \
      || { say "  FAILED"; tail -5 out/prof/wg_${X}_s${S}.log; exit 4; }
  done
  $PY goal065_eval.py trained --cand wgap --control _gpcc1 --treat _wg1 --no-append > out/prof/wgap_score.txt 2>&1
  LINE=$(tail -1 out/prof/wgap_score.txt); say "after $X: $LINE"
  case "$LINE" in *"| ABANDON-EARLY") say "stopping early"; break;; esac
done
$PY goal065_eval.py trained --cand wgap --control _gpcc1 --treat _wg1
say "wgap done"
