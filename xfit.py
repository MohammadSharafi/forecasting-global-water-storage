"""Leave-one-layout-out selection, shared by every post-processing scan (session 9m).

The rule this project has used since session 4 -- "adopt only if it wins on BOTH layouts" -- is a
consistency check, not an out-of-sample one: the winner is still chosen by looking at both. With
three layouts (A, B and C) something stricter is affordable and is what these scans use:

  * choose the configuration on the OTHER layouts, score it on the held-out one, and report that
    gain. That number is honest -- the held-out layout took no part in choosing.
  * adopt the overall best only if every held-out gain clears the threshold.

The threshold is not cosmetic. At ~84k public rows a leaderboard difference below about 0.00014
RMSE is unreadable (see lb_se.py), and a validation difference of 0.0003 between two smoothing
weights is well inside what a different seed would produce. Anything under it is not a finding.
"""
import os


def layouts(default="A,B,C"):
    """Validation layouts that actually have a built matrix, in order. LAYOUTS overrides."""
    ls = os.environ.get("LAYOUTS", default).replace(" ", ",").split(",")
    return [L for L in ls if L and os.path.exists(f"out/mats/{L}_va.parquet")]


def loo_grid(scores, base, thresh=-0.0003):
    """scores: {layout: {cfg: rmse}} over a shared grid of cfgs, `base` the incumbent cfg.

    Returns (chosen_cfg, {layout: held-out gain}, adopted, lines) where `lines` is a short
    human-readable trace. With a single layout there is nothing to hold out, so the choice is
    in-sample and is reported as such."""
    lines = []
    Ls = list(scores)
    keys = set.intersection(*[set(s) for s in scores.values()])
    assert base in keys, "the incumbent configuration must be in every layout's grid"
    mean = lambda k, sub: sum(scores[o][k] for o in sub) / len(sub)
    cand = min(keys, key=lambda k: mean(k, Ls))
    if len(Ls) == 1:
        L = Ls[0]
        g = {L: scores[L][cand] - scores[L][base]}
        lines.append(f"  only layout {L}: choice is IN sample, treat with suspicion")
        return cand, g, g[L] < thresh, lines
    held = {}
    for L in Ls:
        others = [o for o in Ls if o != L]
        pick = min(keys, key=lambda k: mean(k, others))
        held[L] = scores[L][pick] - scores[L][base]
        lines.append(f"  chosen on {'+'.join(others)} -> {pick}; on held-out {L}: {held[L]:+.5f}")
    gains = {L: scores[L][cand] - scores[L][base] for L in Ls}
    ok = all(g < thresh for g in held.values()) and all(g < 0 for g in gains.values())
    lines.append(f"  best overall {cand}: on each layout {[f'{gains[L]:+.5f}' for L in Ls]}")
    if not ok:
        lines.append(f"  does not clear {abs(thresh):.4f} out of sample everywhere -- "
                     f"keeping the incumbent {base}")
    return (cand if ok else base), held, ok, lines
