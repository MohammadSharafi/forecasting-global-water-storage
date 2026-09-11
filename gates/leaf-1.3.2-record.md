# Gates: 1.3.2 NOTES Session 12 and REPORT §4/§8

Scope: every measurement of this goal recorded, failures included; REPORT closed-avenue table and results table updated.

- [ ] R1: NOTES.md has the Session 12 section
  CHECK: grep -c "Session 12 — goal 0.65" NOTES.md | sed 's/^/n=/'
  EXPECT: /n=[1-9]/
  EVIDENCE: pending

- [ ] R2: every heldout.txt line appears verbatim in NOTES.md (numbers copied, not retyped)
  CHECK: ./.venv/bin/python -c "n=open('NOTES.md').read();L=[l.strip() for l in open('out/goal065/heldout.txt') if l.strip()];m=[l for l in L if l not in n];print('ok n=%d'%len(L) if L and not m else 'missing: '+' || '.join(m))"
  EXPECT: ok
  EVIDENCE: pending

- [ ] R3: REPORT.md §4 names every rejected or abandoned candidate from heldout.txt
  CHECK: ./.venv/bin/python -c "import re;r=open('REPORT.md').read();s=r[r.index('## 4.'):r.index('## 5.')];c=[l.split(' |')[0] for l in open('out/goal065/heldout.txt') if re.search(r'\| (REJECT|ABANDON-EARLY)$',l.strip())];m=[x for x in c if x not in s];print('ok n=%d'%len(c) if not m else 'missing: '+','.join(m))"
  EXPECT: ok
  EVIDENCE: pending

- [ ] R4: REPORT.md §8 has a row for sub_goal065
  CHECK: ./.venv/bin/python -c "r=open('REPORT.md').read();print('ok' if 'sub_goal065' in r[r.index('## 8.'):] else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] R5: NOTES Session 12 records the gate-check.mjs file-argument bug and the foreign SampleSubmission.csv
  CHECK: ./.venv/bin/python -c "n=open('NOTES.md').read();s=n[n.index('Session 12 — goal 0.65'):];print('ok' if 'gate-check' in s and 'SampleSubmission' in s else 'bad')"
  EXPECT: ok
  EVIDENCE: pending
