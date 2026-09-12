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

- [x] G6: public leaderboard score of out/sub_goal065.csv is below 0.65 (manual: the user pastes the score)
  EVIDENCE: submitted by the user; public score 0.68284693. That is NOT below 0.65, so the goal is not reached. Against its own base (sub_x_lb2, 0.683712087) the adopted correction gained -0.000865, a realised transfer of 18% of its -0.0048 validation gain. It also did not beat the displayed best (sub_z_lb, 0.681692450): it is +0.001155 worse than that file.

ABANDON: G4 SampleSubmission.csv in this repo belongs to a different competition (9378 rows x 17 columns keyed on GEOID: coverage_gap_score, region, transport_gap, ...), so any valid submission for this challenge (ID,Target, 280961 rows) makes the check print "bad" by construction. The gate is not edited. Reversible: if the user approves pointing it at Test.csv, or supplies the real sample file, this line is removed and G4 re-checked. The equivalent substantive check against Test.csv is leaf-1.3.1-build B5.

REOPENED: G2 was abandoned on 12 Sep with BUDGET TOTAL 0.0000 and twelve rejected candidates. It is re-opened because gauge precipitation (GPCC, candidate c7) then won on all five layouts at mean -0.0020, the first genuinely new information since the NCEP-R2/CPC block. The threshold is unchanged at -0.041.
ABANDON: G2 Re-opened on the GPCC evidence and re-measured; it still cannot be met. The adopted set is now GPCC gauge precipitation inside the model plus the marginal AR anchor rescaling applied after it, measured JOINTLY (not summed) by re-fitting the correction against the GPCC model's own predictions: Avn2 -0.0108, Bvn2 -0.0100, Cvn2 -0.0102, D -0.0032, E -0.0083, mean -0.0085, winning 5/5 and clearing -0.003 on all five, which no earlier change managed. It is the largest held-out gain in the project's history and it is still 4.8x short of the -0.041 G2 requires. Reaching 0.65 from 0.679786 needs a board gain of 0.029786, i.e. 0.0359 on validation at the best transfer ratio ever observed here (0.83) and 0.0745 at the ratio the AR correction actually achieved (0.40); the largest validation gain ever measured in this project is 0.0156. Even if every remaining untested idea worked and the gains simply added (gpcc 0.0020 measured, plus c5 0.0020, c6 0.0025, c8 0.0009 estimated), the total is 0.0074 on validation, about 0.0061 on the board, leaving 0.0236 missing. The threshold was never moved.
