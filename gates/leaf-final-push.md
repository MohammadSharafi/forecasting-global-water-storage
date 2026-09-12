# Gates: last-submission push

Scope: find and exploit whatever separates a five-submission entrant at 0.65 from our thirty-submission pipeline at 0.679, and ship one file.

- [x] FP1: the web was searched for this competition and for GRACE TWS forecasting method, findings written to RESEARCH2.md
  CHECK: test -s RESEARCH2.md && grep -c "^## " RESEARCH2.md | sed 's/^/sections=/'
  EXPECT: /sections=[2-9]/
  EVIDENCE: sections=4

- [x] FP2: the training-structure ablation is measured on two layouts (horizon-mix weights, recency ramp, both)
  CHECK: grep -hc "^ab_n" out/goal065/heldout.txt out/prof/ablate_run.log 2>/dev/null | head -1 | sed 's/^/arms=/'
  EXPECT: /arms=[1-9]/
  EVIDENCE: arms=3

- [x] FP3: WaterGAP measured against the GPCC control
  CHECK: tail -1 out/goal065/wgap.txt
  EXPECT: /wins \d\/5/
  EVIDENCE: wgap | Avn2 n/a Bvn2 n/a Cvn2 -0.0025 D -0.0052 E -0.0042 | wins 3/5 | mean -0.0040 | worst -0.0025 | clears-0.003 2/5 | REJECT

- [ ] FP4: the final file exists, covers every Test ID once, finite
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_final.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] FP5: its expected score is computed from scored files by the affine identity, not guessed (manual)
  EVIDENCE: pending
