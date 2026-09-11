# Gates: 1.2 measure (integration)

Scope: harness, in-flight AR family and new candidates, all reported in one comparable format.

- [ ] N1: every child gates file under 1.2 is fully met or carries ABANDON lines
  CHECK: node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status gates/leaf-1.2.1-harness.md gates/node-1.2.2.md gates/node-1.2.3.md
  EXPECT: ALL MET
  EVIDENCE: pending

- [ ] N2: every line of heldout.txt parses in the contract format
  CHECK: ./.venv/bin/python -c "import re;L=[l for l in open('out/goal065/heldout.txt') if l.strip()];p=re.compile(r'^\S+ \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d/5 \| (ADOPT-0\.65|HELPS-NOT-0\.65|REJECT|ABANDON-EARLY|ADOPTED-SET\S*)$');print('ok n=%d'%len(L) if L and all(p.match(l.strip()) for l in L) else 'bad')"
  EXPECT: ok
  EVIDENCE: pending

- [ ] N3: the harness reproduces a known result exactly (directional family d0/d1 from Session 11g)
  CHECK: ./.venv/bin/python goal065_eval.py trained --cand dirrepro --control _d0 --treat _d1 --no-append 2>&1 | tail -1
  EXPECT: /Avn2 \+0\.000[78] Bvn2 -0\.000[34] Cvn2 -0\.00(09|10) D -0\.000[34] E -0\.0001/
  EVIDENCE: pending

- [ ] N4: no training process was run concurrently with another (manual: quote start/end times from the logs)
  EVIDENCE: pending
