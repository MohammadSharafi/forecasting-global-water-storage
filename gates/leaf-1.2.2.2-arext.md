# Gates: 1.2.2.2 seasonal AR / AR(2) as extra vectors in the ar_blend correction

Scope: measure whether a month-modulated AR slope or an AR(2) forecast adds to the four-vector correction, leave-one-layout-out.

- [x] L1: the measurement ran to completion on all five layouts
  CHECK: grep -c "ar2 had a prior month" /private/tmp/claude-501/-Users-moe-Programming-Forecasting-Global-Water-Storage-Challenge-by-ITU/985a7837-906e-460a-bbf4-24f6291ddbc4/tasks/b7ld8nud8.output | sed 's/^/layouts=/'
  EXPECT: layouts=5
  EVIDENCE: layouts=5

- [x] L2: per-candidate logs exist and end in a contract summary line
  CHECK: tail -qn1 out/goal065/arseas.txt out/goal065/ar2.txt | grep -c "| clears-0.003 [0-5]/5 | " | sed 's/^/lines=/'
  EXPECT: lines=2
  EVIDENCE: lines=2

- [x] L3: seasonal AR verdict recorded from the measured numbers
  CHECK: grep "^arseas " out/goal065/heldout.txt
  EXPECT: /wins 0\/5 \| mean \+0\.0001 .*REJECT/
  EVIDENCE: arseas | Avn2 +0.0003 Bvn2 +0.0001 Cvn2 +0.0000 D +0.0000 E +0.0000 | wins 0/5 | mean +0.0001 | worst +0.0003 | clears-0.003 0/5 | REJECT

- [x] L4: AR(2) verdict recorded, with the structural-availability caveat in its log
  CHECK: grep -c "structurally unavailable" out/goal065/ar2.txt | sed 's/^/caveat=/' ; grep "^ar2 " out/goal065/heldout.txt
  EXPECT: /caveat=1[\s\S]*REJECT/
  EVIDENCE: caveat=1 | ar2 | Avn2 -0.0000 Bvn2 +0.0000 Cvn2 +0.0001 D +0.0001 E -0.0000 | wins 2/5 | mean +0.0000 | worst +0.0001 | clears-0.003 0/5 | REJECT
