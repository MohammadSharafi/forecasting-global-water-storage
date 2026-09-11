# Gates: 1.2.2.1 ARF — the per-cell AR anchor rescaling offered to the model as features

Scope: out/prof/artest.sh on five layouts (a0 = shipped, a1 = + ar_slope/ar_pred/ar_dev), scored with the shared harness.

- [ ] L1: all 20 prediction files exist (5 layouts x 2 arms x 2 seeds)
  CHECK: ls out/mats/pred_{Avn2,Bvn2,Cvn2,D,E}_lgb_v5x_noll_s[01]_a[01].npy 2>/dev/null | wc -l | tr -d ' ' | sed 's/^/files=/'
  EXPECT: files=20
  EVIDENCE: pending

- [ ] L2: provenance: every treatment run used 343 features and every control 340
  CHECK: grep -h "Xv (" out/prof/ar_*_a1_s*.log | grep -c ", 343)" | sed 's/^/t343=/'; grep -h "Xv (" out/prof/ar_*_a0_s*.log | grep -c ", 340)" | sed 's/^/c340=/'
  EXPECT: /t343=10[\s\S]*c340=10/
  EVIDENCE: pending

- [ ] L3: the control arm is bit-identical to the shipped d0 predictions on all 10 (layout, seed) pairs
  CHECK: ./.venv/bin/python -c "import numpy as np;m=max(abs(np.load(f'out/mats/pred_{L}_lgb_v5x_noll_s{s}_a0.npy')-np.load(f'out/mats/pred_{L}_lgb_v5x_noll_s{s}_d0.npy')).max() for L in ('Avn2','Bvn2','Cvn2','D','E') for s in (0,1));print('maxdiff=%.3e'%m)"
  EXPECT: maxdiff=0.000e+00
  EVIDENCE: pending

- [ ] L4: scored with the shared harness; log ends in a contract summary line
  CHECK: tail -1 out/goal065/arf.txt
  EXPECT: /^arf \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d\/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d\/5 \| \S+$/
  EVIDENCE: pending

- [ ] L5: the summary line is in heldout.txt
  CHECK: grep -c "^arf " out/goal065/heldout.txt | sed 's/^/n=/'
  EXPECT: n=1
  EVIDENCE: pending
