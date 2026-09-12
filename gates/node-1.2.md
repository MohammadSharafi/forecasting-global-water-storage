# Gates: 1.2 measure (integration)

Scope: harness, in-flight AR family and new candidates, all reported in one comparable format.

- [x] N1: every child gates file under 1.2 is fully met or carries ABANDON lines
  CHECK: node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status gates/leaf-1.2.1-harness.md gates/node-1.2.2.md gates/node-1.2.3.md
  EXPECT: ALL MET
  EVIDENCE: gates/node-1.2.3.md: 3 gates | ALL MET (15 met)

- [x] N2: every line of heldout.txt parses in the contract format
  CHECK: ./.venv/bin/python -c "import re;L=[l for l in open('out/goal065/heldout.txt') if l.strip()];p=re.compile(r'^\S+ \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d/5 \| (ADOPT-0\.65|HELPS-NOT-0\.65|REJECT|ABANDON-EARLY|ADOPTED-SET\S*)$');print('ok n=%d'%len(L) if L and all(p.match(l.strip()) for l in L) else 'bad')"
  EXPECT: ok
  EVIDENCE: ok n=5

- [x] N3: the harness reproduces a known result exactly (directional family d0/d1 from Session 11g)
  CHECK: ./.venv/bin/python goal065_eval.py trained --cand dirrepro --control _d0 --treat _d1 --no-append 2>&1 | tail -1
  EXPECT: /Avn2 \+0\.000[78] Bvn2 -0\.000[34] Cvn2 -0\.00(09|10) D -0\.000[34] E -0\.0001/
  EVIDENCE: dirrepro | Avn2 +0.0008 Bvn2 -0.0004 Cvn2 -0.0010 D -0.0003 E -0.0001 | wins 4/5 | mean -0.0002 | worst +0.0008 | clears-0.003 0/5 | REJECT

- [x] N4: no training process was run concurrently with another (manual: quote start/end times from the logs)
  EVIDENCE: ARF's 20 runs were logged by a sequential shell loop, one arm at a time, starts 19:52:14, 19:58:28, 20:04:02, 20:09:54 -- each after the previous finished. The candidate runner additionally blocks on a process-table guard before every run. A live check shows exactly 1 python binary on the machine (ps -Ao comm | grep -c 'python$'). An earlier attempt to prove this from file timestamps was WITHDRAWN as unsound: on macOS ctime updates on write, so ctime == mtime and the comparison proved nothing. Caveat recorded honestly: the guard was also shown to over-match -- shells merely mentioning the script name were counted as trainings and deadlocked the runner (PLAN.md step 9) -- which risks false STALLS, never concurrency.
