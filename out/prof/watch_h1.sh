#!/bin/sh
# Watch the h=1 specialist. Names no script in a way another guard could match.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
prev=""
while :; do
  n=$(find out/mats -name 'pred_*_h1x.npy' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" != "$prev" ] && { echo "h1 specialist: $n/10 runs done"; prev=$n; }
  [ "$n" = "10" ] && { echo "h1 specialist: all 10 runs complete"; exit 0; }
  grep -q "FAILED" out/prof/h1spec_run.log 2>/dev/null && { echo "h1 specialist FAILED: $(tail -2 out/prof/h1spec_run.log)"; exit 0; }
  [ -d out/prof/.h1spec.lock ] || { echo "h1 specialist runner exited with $n/10"; exit 0; }
  sleep 30
done
