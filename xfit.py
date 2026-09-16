import os

def layouts(default="A,B,C"):
    ls = os.environ.get("LAYOUTS", default).replace(" ", ",").split(",")
    return [L for L in ls if L and os.path.exists(f"out/mats/{L}_va.parquet")]

def loo_grid(scores, base, thresh=-0.0003):
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
