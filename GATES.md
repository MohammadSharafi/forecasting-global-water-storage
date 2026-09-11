# Gates: public leaderboard below 0.65, legally, without hurting the private 70%

Scope: research legal candidates, measure them leave-one-layout-out on Avn2/Bvn2/Cvn2/D/E, build
out/sub_goal065.csv from what is adopted, hand over. Rules R1-R5 and the no-board-fitting rule apply.

- [ ] G1: RESEARCH.md has >= 6 legal candidates, each checked against R1–R5, and a budget total
  CHECK: grep -q "^BUDGET TOTAL:" RESEARCH.md && echo ok
  EXPECT: ok
  EVIDENCE: pending

- [ ] G2: an adopted change set wins 5/5 held-out layouts with mean gain <= -0.041
  CHECK: tail -1 out/goal065/heldout.txt
  EXPECT: /wins 5\/5 .*mean -0\.(0[4-9]|[1-9])/
  EVIDENCE: pending

- [ ] G3: the final artefacts pass the compliance audit
  CHECK: .venv/bin/python compliance.py FINALvn2 2>&1 | tail -1
  EXPECT: every check passed
  EVIDENCE: pending

- [ ] G4: out/sub_goal065.csv matches SampleSubmission.csv in shape, columns and IDs, no nulls
  CHECK: .venv/bin/python -c "import polars as pl;a=pl.read_csv('out/sub_goal065.csv');b=pl.read_csv('SampleSubmission.csv');k=b.columns[0];print('ok' if a.columns==b.columns and a.height==b.height and set(a[k])==set(b[k]) and a.null_count().sum_horizontal()[0]==0 else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] G5: NOTES.md has a "Session 12 — goal 0.65" section with every measurement, including failures; REPORT.md §4/§8 updated
  CHECK: grep -q "Session 12 — goal 0.65" NOTES.md && echo ok
  EXPECT: ok
  EVIDENCE: pending

- [ ] G6: public leaderboard score of out/sub_goal065.csv is below 0.65 (manual: the user pastes the score)
  EVIDENCE: pending
