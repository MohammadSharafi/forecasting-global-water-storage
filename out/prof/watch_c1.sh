#!/bin/sh
# Watch the c1 candidate. Deliberately does NOT mention the training script by name anywhere in a
# command line: goal065_run.sh greps the process table for that text and would mistake this watcher
# for a running training, which is exactly how the run deadlocked once already.
cd "$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
prev=""
while :; do
  n=$(find out/mats -name 'pred_*_c11.npy' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" != "$prev" ] && { echo "c1: $n/10 prediction files"; prev=$n; }
  if [ -f out/goal065/c1.txt ]; then echo "c1 SCORED: $(tail -1 out/goal065/c1.txt)"; exit 0; fi
  grep -q "ABANDON-EARLY\|TRAINING FAILED\|MISSING" out/prof/g065_c1_score.txt 2>/dev/null && { echo "c1 stopped early: $(tail -1 out/prof/g065_c1_score.txt)"; exit 0; }
  pgrep -f "[g]oal065_run" >/dev/null 2>&1 || { sleep 30; pgrep -f "[g]oal065_run" >/dev/null 2>&1 || { echo "c1 runner EXITED with $n/10 files"; exit 0; }; }
  sleep 30
done
