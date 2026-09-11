# Gates: 1.3.1 joint adopted-set measurement and submission build

Scope: measure the adopted changes TOGETHER (not as a sum), decide whether the ar_blend correction still helps on top, build out/sub_goal065.csv, audit it.

- [ ] B1: the jointly measured set ends in a contract summary line named ADOPTED-SET
  CHECK: tail -1 out/goal065/adopted_set.txt
  EXPECT: /^ADOPTED-SET \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d\/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d\/5 \| (ADOPT-0\.65|HELPS-NOT-0\.65|REJECT|ABANDON-EARLY)$/
  EVIDENCE: pending

- [ ] B2: the set's members are listed, each with its own heldout.txt line
  CHECK: grep -c "^members:" out/goal065/adopted_set.txt | sed 's/^/members=/'
  EXPECT: members=1
  EVIDENCE: pending

- [ ] B3: the ar_blend marginal correction was re-validated on top of the adopted set, leave-one-layout-out, or excluded with a reason
  CHECK: grep "^arblend-on-top:" out/goal065/adopted_set.txt
  EXPECT: /arblend-on-top: (Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d\/5.*|not included: .+)/
  EVIDENCE: pending

- [ ] B4: the final matrix used is named and passes the compliance audit
  CHECK: M=$(grep '^final-matrix:' out/goal065/adopted_set.txt | awk '{print $2}'); echo "matrix=$M"; .venv/bin/python compliance.py "$M" 2>&1 | tail -1
  EXPECT: /matrix=FINAL\S*[\s\S]*every check passed/
  EVIDENCE: pending

- [ ] B5: out/sub_goal065.csv covers every Test.csv ID exactly once with finite values
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_goal065.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and a['ID'].n_unique()==a.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] B6: the base file and its public-specific share are stated, so the private risk is explicit
  CHECK: grep -E "^base: \S+" out/goal065/adopted_set.txt && grep -c "public-specific" out/goal065/adopted_set.txt | sed 's/^/risk=/'
  EXPECT: /base: \S+[\s\S]*risk=[1-9]/
  EVIDENCE: pending

- [ ] B7: expected board score stated with its reasoning (validation mean x transfer ratio, and the range) (manual)
  EVIDENCE: pending
