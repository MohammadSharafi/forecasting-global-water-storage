# Gates: 1.2.3.1 candidate c1 — regional forcing-anomaly means at a second and third radius

Scope: build the c1 side-cars, train the treatment arm against the shipped control on five layouts, score under the contract.

- [ ] C1G1: side-cars exist for all five layouts, 22 columns each
  CHECK: ./.venv/bin/python -c "import polars as pl;L=['Avn2','Bvn2','Cvn2','D','E'];n=[len(pl.scan_parquet(f'out/mats/{x}_va_c1.parquet').collect_schema().names()) for x in L];print('cols=%s'%set(n))"
  EXPECT: cols={22}
  EVIDENCE: pending

- [ ] C1G2: the XF switch really adds them (D: 340 without, 362 with)
  CHECK: for v in "" c1; do FEATS_ONLY=1 XF=$v DROPF=bigsa,gdo WEIGHTS=ramp HMIX=test PER_ROW=2 ./.venv/bin/python run_models.py D lgb v5x_noll_sa 2>&1 | grep FEATS_ONLY; done
  EXPECT: /n=340[\s\S]*n=362/
  EVIDENCE: pending

- [ ] C1G3: scored under the contract, log ends in a summary line
  CHECK: tail -1 out/goal065/c1.txt
  EXPECT: /^c1 \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d\/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d\/5 \| \S+\s*$/
  EVIDENCE: pending

- [ ] C1G4: the summary line is in heldout.txt
  CHECK: grep -c "^c1 " out/goal065/heldout.txt | sed 's/^/n=/'
  EXPECT: n=1
  EVIDENCE: pending
