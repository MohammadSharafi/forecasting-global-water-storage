# Gates: public leaderboard below 0.65, legally, without hurting the private 70%

Scope: research legal candidates, measure them leave-one-layout-out on Avn2/Bvn2/Cvn2/D/E, build
out/sub_goal065.csv from what is adopted, hand over. Rules R1-R5 and the no-board-fitting rule apply.

- [x] G1: RESEARCH.md has >= 6 legal candidates, each checked against R1–R5, and a budget total
  CHECK: grep -q "^BUDGET TOTAL:" RESEARCH.md && echo ok
  EXPECT: ok
  EVIDENCE: ok

- [ ] G2: an adopted change set wins 5/5 held-out layouts with mean gain <= -0.041
  CHECK: tail -1 out/goal065/heldout.txt
  EXPECT: /wins 5\/5 .*mean -0\.(0[4-9]|[1-9])/
  EVIDENCE: pending

- [x] G3: the final artefacts pass the compliance audit
  CHECK: .venv/bin/python compliance.py FINALvn2 2>&1 | tail -1
  EXPECT: every check passed
  EVIDENCE: every check passed

- [ ] G4: out/sub_goal065.csv matches SampleSubmission.csv in shape, columns and IDs, no nulls
  CHECK: .venv/bin/python -c "import polars as pl;a=pl.read_csv('out/sub_goal065.csv');b=pl.read_csv('SampleSubmission.csv');k=b.columns[0];print('ok' if a.columns==b.columns and a.height==b.height and set(a[k])==set(b[k]) and a.null_count().sum_horizontal()[0]==0 else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [x] G5: NOTES.md has a "Session 12 — goal 0.65" section with every measurement, including failures; REPORT.md §4/§8 updated
  CHECK: grep -q "Session 12 — goal 0.65" NOTES.md && echo ok
  EXPECT: ok
  EVIDENCE: ok

- [ ] G6: public leaderboard score of out/sub_goal065.csv is below 0.65 (manual: the user pastes the score)
  EVIDENCE: pending

ABANDON: G4 SampleSubmission.csv in this repo belongs to a different competition (9378 rows x 17 columns keyed on GEOID: coverage_gap_score, region, transport_gap, ...), so any valid submission for this challenge (ID,Target, 280961 rows) makes the check print "bad" by construction. The gate is not edited. Reversible: if the user approves pointing it at Test.csv, or supplies the real sample file, this line is removed and G4 re-checked. The equivalent substantive check against Test.csv is leaf-1.3.1-build B5.
ABANDON: G2 The goal needs a change set winning 5/5 with a mean held-out gain <= -0.041. The best that exists wins 5/5 at mean -0.0048 (Avn2 -0.0086, Bvn2 -0.0024, Cvn2 -0.0038, D -0.0029, E -0.0061), which is 8.5x too small, and it was already measured before this goal began. Everything measured under the goal failed: arseas mean +0.0001 (wins 0/5), ar2 +0.0000 (2/5), arf +0.0008 (3/5), c1 +0.0010 (0/5, stopped early after D +0.0012 and E +0.0008). RESEARCH.md said this in advance with BUDGET TOTAL: 0.0000 against the 0.0317 required; its largest single estimate was -0.0025 and the largest validation gain in this project's whole history is -0.0156. The threshold was never moved and no layout was hidden. Reversible: if a future candidate clears -0.041 on 5/5, delete this line and re-run the gate.
ABANDON: G6 Not impossible, and not mine: G6 is the manual gate that records the public score of out/sub_goal065.csv, which only the user can obtain because only they can submit (one submission per day). Handing over with it open. To close it: submit out/sub_goal065.csv, then replace this line with the score and tick G6. Expected 0.6797-0.6835, most likely ~0.680.
