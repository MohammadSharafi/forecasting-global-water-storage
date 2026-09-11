# Gates: 1.2.3 new candidates from RESEARCH.md (integration)

Scope: every RESEARCH.md candidate is either measured under the contract or explicitly deferred with a reason. Candidate ids are "c<n>", lowercased from the RESEARCH heading "### C<n>:".

- [ ] N1: every dispatched candidate leaf is fully met or carries ABANDON lines
  CHECK: F=$(find gates -maxdepth 1 -name 'leaf-1.2.3.*.md' | sort | tr '\n' ' '); [ -n "$F" ] && node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status $F 2>&1 | tail -1 || echo "no candidate leaves yet"
  EXPECT: ALL MET
  EVIDENCE: pending

- [ ] N2: every RESEARCH candidate has a heldout.txt line or a "deferred c<n>:" line in PLAN.md
  CHECK: ./.venv/bin/python -c "import re;c=sorted({'c'+x for x in re.findall(r'^### C(\d+):',open('RESEARCH.md').read(),re.M)},key=lambda s:int(s[1:]));h=open('out/goal065/heldout.txt').read();p=open('PLAN.md').read();m=[x for x in c if not re.search(rf'^{x} \|',h,re.M) and ('deferred '+x+':') not in p];print('ok n=%d'%len(c) if c and not m else 'open: '+','.join(m) if c else 'no candidates')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] N3: candidates were measured in RESEARCH.md's recommended order, or PLAN.md logs why not (manual)
  EVIDENCE: pending
