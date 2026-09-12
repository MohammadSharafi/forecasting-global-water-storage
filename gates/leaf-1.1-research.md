# Gates: 1.1 research

Scope: RESEARCH.md — gap hypotheses, >= 6 legal candidates not on the closed list, a budget table and total.

- [x] L1: RESEARCH.md states the gap to the 0.56-0.62 entries as numbered hypotheses
  CHECK: grep -c "^H[0-9]" RESEARCH.md | sed 's/^/hyp=/'
  EXPECT: /hyp=([3-9]|[1-9]\d)\b/
  EVIDENCE: hyp=7

- [x] L2: at least 6 candidates, each under a "### C<n>:" heading
  CHECK: grep -c "^### C[0-9]" RESEARCH.md | sed 's/^/cands=/'
  EXPECT: /cands=([6-9]|[1-9]\d)\b/
  EVIDENCE: cands=8

- [x] L3: every candidate carries a rules line covering R1 to R5
  CHECK: ./.venv/bin/python -c "import re;t=open('RESEARCH.md').read();b=re.split(r'^### C\d',t,flags=re.M)[1:];print('ok' if b and all(all(f'R{i}' in re.search(r'^Rules:.*$',x,re.M).group(0) for i in range(1,6)) if re.search(r'^Rules:.*$',x,re.M) else False for x in b) else 'bad')"
  EXPECT: ok
  EVIDENCE: ok

- [x] L4: every candidate states expected validation gain, how it was estimated, and cost in hours
  CHECK: ./.venv/bin/python -c "import re;t=open('RESEARCH.md').read();b=re.split(r'^### C\d',t,flags=re.M)[1:];k=['Expected validation gain:','Estimated by:','Cost (hours):'];print('ok' if b and all(all(re.search('^'+re.escape(x),y,re.M) for x in k) for y in b) else 'bad')"
  EXPECT: ok
  EVIDENCE: ok

- [x] L5: every external-covariate candidate names Source, URL, Licence and date availability
  CHECK: ./.venv/bin/python -c "import re;t=open('RESEARCH.md').read();b=[x for x in re.split(r'^### C\d',t,flags=re.M)[1:] if re.search(r'^Type: external',x,re.M)];k=['Source:','URL:','Licence:','Available through:'];print('ok n=%d'%len(b) if all(all(re.search('^'+re.escape(x),y,re.M) for x in k) for y in b) else 'bad')"
  EXPECT: ok
  EVIDENCE: ok n=4

- [ ] L6: no candidate is on the closed list unless it carries a "New reason:" line (manual, quote the check)
  EVIDENCE: pending

- [x] L7: budget table and a numeric BUDGET TOTAL line, compared plainly against 0.0317
  CHECK: grep -E "^BUDGET TOTAL: -?[0-9]*\.[0-9]+" RESEARCH.md && grep -c "0\.0317" RESEARCH.md | sed 's/^/ref=/'
  EXPECT: /BUDGET TOTAL: .*\n.*ref=[1-9]/
  EVIDENCE: BUDGET TOTAL: 0.0000 | ref=4

- [x] L8: research downloaded no dataset (external/ unchanged since GATES.md was written)
  CHECK: find external -type f -newer GATES.md | wc -l | tr -d ' ' | sed 's/^/newfiles=/'
  EXPECT: newfiles=0
  EVIDENCE: newfiles=0
