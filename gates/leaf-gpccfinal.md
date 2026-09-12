# Gates: the GPCC submission file

Scope: train the FINAL models with the adopted GPCC block, assemble a submission, and state what it is worth against the record.

- [x] GF1: twelve FINAL predictions exist, six LightGBM and six XGBoost
  CHECK: find out/mats -name 'pred_FINALvn2_*_v5x_noll_s*_gp1.npy' | wc -l | tr -d ' ' | sed 's/^/files=/'
  EXPECT: files=12
  EVIDENCE: files=12

- [x] GF2: provenance -- the FINAL runs really carried the GPCC block (355 features, not 340)
  CHECK: grep -h "Xv (" out/prof/fg_*.log 2>/dev/null | grep -c ", 355)" | sed 's/^/n355=/'
  EXPECT: /n355=1[0-2]/
  EVIDENCE: n355=12

- [x] GF3: the assembled file covers every Test.csv ID exactly once with finite values
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_gpcc_base.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: ok

- [x] GF4: the AR correction is applied on top and the result validated the same way
  CHECK: ./.venv/bin/python -c "import polars as pl,numpy as np;a=pl.read_csv('out/sub_gpcc_ar.csv');t=pl.read_csv('Test.csv');print('ok' if a.columns==['ID','Target'] and a.height==t.height and set(a['ID'])==set(t['ID']) and np.isfinite(a['Target'].to_numpy()).all() else 'bad')"
  EXPECT: ok
  EVIDENCE: ok

- [x] GF5: expected board score stated with its reasoning, and compared honestly with the 0.679786 record (manual)
  EVIDENCE: the file replaces the single-model sub_v_anwide (public 0.692189463), not the blend. Its two changes measure -0.0020 (gpcc) and -0.0065 (AR on top of gpcc, re-fitted and re-measured at build time, 5/5). At the transfer ratios this project has actually observed for the AR correction, 0.18 and 0.40, the expected board score is 0.688789 to 0.690659. Against the blend-based record of 0.679786 that is +0.0090 to +0.0109 WORSE on display. It is offered for the private 70% only: the record is a blend fitted to the public board, of whose incremental gain 99% was public-specific for sub_z_lb, while this file contains no board-fitted component at all. Caveats: six seeds per family rather than sixteen, and no horizon-1 splice (worth -0.0005), so it is a slightly weaker build than the shipped recipe.
