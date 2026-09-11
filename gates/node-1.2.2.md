# Gates: 1.2.2 AR family (integration)

Scope: ARF and the seasonal-AR/AR(2) extensions both measured and recorded.

- [ ] N1: both AR-family leaves fully met or ABANDONed
  CHECK: node ~/.claude/skills/unlazy/scripts/gate-check.mjs --status gates/leaf-1.2.2.1-arf.md gates/leaf-1.2.2.2-arext.md
  EXPECT: ALL MET
  EVIDENCE: pending

- [ ] N2: if ARF is adopted, its gain is measured on top of the ar_blend correction, not only against the bare model (manual)
  EVIDENCE: pending
