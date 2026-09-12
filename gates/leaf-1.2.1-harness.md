# Gates: 1.2.1 shared measurement harness and the generic side-car switch

Scope: goal065_eval.py (scoring + verdict + summary line), goal065_run.sh (sequential treatment runner with early stop), and the XF switch in run_models.py.

- [x] H1: goal065_eval.py parses and exposes the trained and selftest modes
  CHECK: ./.venv/bin/python goal065_eval.py --help 2>&1 | grep -c -E "trained|selftest" | sed 's/^/modes=/'
  EXPECT: /modes=[2-9]/
  EVIDENCE: modes=4

- [x] H2: verdict logic passes a self-test covering ADOPT-0.65, HELPS-NOT-0.65, REJECT, ABANDON-EARLY and n/a layouts
  CHECK: ./.venv/bin/python goal065_eval.py selftest 2>&1 | tail -1
  EXPECT: selftest ok
  EVIDENCE: selftest ok

- [x] H3: the harness reproduces the Session 11g directional result exactly from existing predictions
  CHECK: ./.venv/bin/python goal065_eval.py trained --cand dirrepro --control _d0 --treat _d1 --no-append 2>&1 | tail -1
  EXPECT: /Avn2 \+0\.000[78] Bvn2 -0\.000[34] Cvn2 -0\.00(09|10) D -0\.000[34] E -0\.0001/
  EVIDENCE: dirrepro | Avn2 +0.0008 Bvn2 -0.0004 Cvn2 -0.0010 D -0.0003 E -0.0001 | wins 4/5 | mean -0.0002 | worst +0.0008 | clears-0.003 0/5 | REJECT

- [x] H4: that reproduction line is in the exact contract format
  CHECK: ./.venv/bin/python goal065_eval.py trained --cand dirrepro --control _d0 --treat _d1 --no-append 2>&1 | tail -1 | ./.venv/bin/python -c "import re,sys;l=sys.stdin.read().strip();print('ok' if re.match(r'^\S+ \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d/5 \| (ADOPT-0\.65|HELPS-NOT-0\.65|REJECT|ABANDON-EARLY)$',l) else 'bad: '+l)"
  EXPECT: ok
  EVIDENCE: ok

- [x] H5: goal065_eval.py never writes heldout.txt (driver-owned)
  CHECK: grep -c "heldout" goal065_eval.py goal065_run.sh | awk -F: '{s+=$2} END {print "refs="s}'
  EXPECT: refs=0
  EVIDENCE: refs=0

- [ ] H6: XF switch is additive: with XF unset the shipped feature list is unchanged (340)
  CHECK: FEATS_ONLY=1 DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test PER_ROW=2 ./.venv/bin/python run_models.py D lgb v5x_noll_sa 2>&1 | grep "^FEATS_ONLY"
  EXPECT: /FEATS_ONLY D n=340\b/
  EVIDENCE: pending

- [ ] H7: XF appends exactly the side-car's columns (dummy 2-column side-car on D gives 342)
  CHECK: FEATS_ONLY=1 XF=xftest DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test PER_ROW=2 ./.venv/bin/python run_models.py D lgb v5x_noll_sa 2>&1 | grep "^FEATS_ONLY"
  EXPECT: /FEATS_ONLY D n=342\b/
  EVIDENCE: pending

- [ ] H8: XF unset stays bit-identical to the shipped control after the edit (one D seed retrained)
  CHECK: ./.venv/bin/python -c "import numpy as np;a=np.load('out/mats/pred_D_lgb_v5x_noll_s0_xfnull.npy');b=np.load('out/mats/pred_D_lgb_v5x_noll_s0_d0.npy');print('maxdiff=%.3e'%abs(a-b).max())"
  EXPECT: maxdiff=0.000e+00
  EVIDENCE: pending

- [x] H9: goal065_run.sh parses, trains layouts in the order D E Cvn2 Bvn2 Avn2 and stops after two losing layouts
  CHECK: sh -n goal065_run.sh && grep -c "for X in D E Cvn2 Bvn2 Avn2" goal065_run.sh | sed 's/^/order=/' && grep -c "ABANDON-EARLY" goal065_run.sh | sed 's/^/early=/'
  EXPECT: /order=1[\s\S]*early=[1-9]/
  EVIDENCE: order=1 | early=3

- [ ] H10: no training started while another run_models.py was running (manual: quote the pgrep guard and the H8 run's start time)
  EVIDENCE: pending
