#!/bin/sh
# Watch the FINAL build. Covers failure and death, not just success, so silence cannot look like progress.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
prev=""
while :; do
  n=$(find out/mats -name 'pred_FINALvn2_*_bw1.npy' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" != "$prev" ] && { echo "FINAL build: $n/12 models trained"; prev=$n; }
  [ -f out/sub_best_base.csv ] && { echo "FINAL build: assembled out/sub_gpcc_base.csv"; exit 0; }
  grep -q "FAILED" out/prof/final_best_run.log 2>/dev/null && { echo "FINAL build FAILED: $(tail -3 out/prof/final_gpcc_run.log | tr '\n' ' ')"; exit 0; }
  [ -d out/prof/.final_best.lock ] || { echo "FINAL build runner exited at $n/12"; exit 0; }
  sleep 60
done
