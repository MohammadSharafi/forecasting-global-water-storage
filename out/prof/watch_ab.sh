#!/bin/sh
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
prev=""
while :; do
  n=$(find out/mats -name 'pred_*_s[01]_n[hwb].npy' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" != "$prev" ] && { echo "ablation: $n/12"; prev=$n; }
  grep -q "ablation done" out/prof/ablate_run.log 2>/dev/null && { echo "ABLATION DONE"; grep -E "^ab_n" out/prof/ablate_run.log | tail -3; exit 0; }
  grep -q "FAILED" out/prof/ablate_run.log 2>/dev/null && { echo "ablation FAILED: $(tail -2 out/prof/ablate_run.log)"; exit 0; }
  [ -d out/prof/.ablate.lock ] || { echo "ablation runner exited at $n/12"; exit 0; }
  sleep 45
done
