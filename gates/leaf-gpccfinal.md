# Gates: the GPCC submission file

Scope: train the FINAL models with the adopted GPCC block, assemble a submission, and state what it is worth against the record.

- [ ] GF1: twelve FINAL predictions exist, six LightGBM and six XGBoost
  CHECK: find out/mats -name 'pred_FINALvn2_*_v5x_noll_s*_gp1.npy' | wc -l | tr -d ' ' | sed 's/^/files=/'
  EXPECT: files=12
  EVIDENCE: pending

- [ ] GF2: provenance -- the FINAL runs really carried the GPCC block (355 features, not 340)
  CHECK: grep -h "Xv (" out/prof/fg_*.log 2>/dev/null | grep -c ", 355)" | sed 's/^/n355=/'
  EXPECT: /n355=1[0-2]/
  EVIDENCE: pending

- [ ] GF3: the assembled file covers every Test.csv ID exactly once with finite values
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_gpcc_base.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] GF4: the AR correction is applied on top and the result validated the same way
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_gpcc_ar.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] GF5: expected board score stated with its reasoning, and compared honestly with the 0.679786 record (manual)
  EVIDENCE: pending
