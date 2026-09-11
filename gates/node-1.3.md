# Gates: 1.3 handover (integration)

Scope: the adopted set measured jointly, the submission built, compliance and records done.

- [ ] N1: both handover leaves fully met or ABANDONed
  CHECK: node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status gates/leaf-1.3.1-build.md gates/leaf-1.3.2-record.md
  EXPECT: ALL MET
  EVIDENCE: pending

- [ ] N2: the last line of heldout.txt is the jointly measured ADOPTED SET, not an individual candidate
  CHECK: tail -1 out/goal065/heldout.txt | cut -d'|' -f1
  EXPECT: ADOPTED-SET
  EVIDENCE: pending

- [ ] N3: sub_goal065.csv covers every Test.csv ID exactly once with finite values
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_goal065.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and a['ID'].n_unique()==a.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] N4: private-risk statement: sub_goal065 is not built on a public-fitted base without saying so (manual)
  EVIDENCE: pending
