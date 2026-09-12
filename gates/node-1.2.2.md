# Gates: 1.2.2 AR family (integration)

Scope: ARF and the seasonal-AR/AR(2) extensions both measured and recorded.

- [x] N1: both AR-family leaves fully met or ABANDONed
  CHECK: node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status gates/leaf-1.2.2.1-arf.md gates/leaf-1.2.2.2-arext.md
  EXPECT: ALL MET
  EVIDENCE: gates/leaf-1.2.2.2-arext.md: 4 gates | ALL MET (9 met)

- [x] N2: if ARF is adopted, its gain is measured on top of the ar_blend correction, not only against the bare model (manual)
  EVIDENCE: the condition does not arise -- arf was REJECTED (heldout.txt: wins 3/5, mean +0.0008), so nothing from this branch is adopted and there is no gain to re-measure on top of the ar_blend correction. The comparison that matters was made anyway and is recorded: the same quantity applied as a post-hoc linear correction wins 5/5 at -0.0048, while offered as features it loses at +0.0008.
