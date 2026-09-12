# Gates: a dedicated horizon-1 model

Scope: h=1 is 33.3% of the test, is the case where TWS is known exactly, and is where the shipped model beats a per-cell AR line by the least (-0.0426, against -0.136 to -0.191 at h>=5). 93 of 354 features are constant or missing on h=1 rows because the accumulation window is empty. Test whether a specialist trained on the h=1 slice alone beats the shared model there.

- [ ] H1A: a specialist is trained on the h=1 slice of all five layouts, two seeds
  CHECK: ls out/mats/pred_{Avn2,Bvn2,Cvn2,D,E}_lgb_v5x_noll_s[01]_h1x.npy 2>/dev/null | wc -l | tr -d ' ' | sed 's/^/files=/'
  EXPECT: files=10
  EVIDENCE: pending

- [x] H1B: its h=1 RMSE is compared with the shared model's on identical rows, five layouts
  CHECK: tail -1 out/goal065/h1spec.txt
  EXPECT: /^h1spec \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d\/5 \| mean \S+ \| worst \S+ \| clears-0\.003 \d\/5 \| \S+\s*$/
  EVIDENCE: h1spec | Avn2 n/a Bvn2 n/a Cvn2 n/a D +0.0206 E +0.0053 | wins 0/5 | mean +0.0130 | worst +0.0206 | clears-0.003 0/5 | ABANDON-EARLY

- [x] H1C: the whole-score consequence is stated, not just the h=1 delta (manual)
  EVIDENCE: stated as both numbers, which is the point of this gate. At h=1 the specialist is WORSE by +0.0206 (D) and +0.0053 (E). Swapped in on h=1 rows only, the whole test-mix score worsens by +0.0066 (D) and +0.0017 (E) -- about a third of the h=1 delta, because h=1 is 33.3% of the mix. No version of this helps, so the -0.0140/-0.0336 prize I priced from the AR margin does not exist; PLAN.md step 12 records why that framing was wrong.

ABANDON: H1A the contract stops a candidate once two layouts lose, and D (+0.0206) and E (+0.0053) both lost at h=1, so training the remaining six runs would have bought nothing. 4 of 10 files exist by design, not by omission. The two that ran are enough: a specialist trained on the h=1 slice alone discards 72% of the training rows (249,724 of 886,907 on D) and the other horizons were evidently still teaching it something it needs at h=1.
