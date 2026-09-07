"""Score validation predictions the way the leaderboard scores them (session 9d).

The problem this fixes
----------------------
Layout A's blocks are [3,2,3,7,3] months, so its horizon mix is

    h1 27.8%  h2 27.8%  h3 22.2%  h4 5.6%  h5 5.6%  h6 5.6%  h7 5.6%

while the real test's blocks are [1,3,4,7,1,2], giving

    h1 33.3%  h2 22.2%  h3 16.7%  h4 11.1%  h5 5.6%  h6 5.6%  h7 5.6%

RMSE varies strongly with horizon, so a plain average over layout-A rows silently
over-weights h2 and h3 by ~5.5 points each and under-weights h1 and h4 by the same.
Every method that trades h1 accuracy for mid-horizon accuracy has therefore been
scored more kindly on validation than the leaderboard will score it.  Layout B
(blocks [1,3,4,3,7,2]) is much closer to the test, which is part of why the two
layouts disagree.

This module reports three things for every prediction set:
  * plain    - the unweighted RMSE this project has always quoted
  * testmix  - RMSE reweighted to the real test's horizon mix, the comparable number
  * per-horizon RMSE and bias, which is how you see *where* a method wins or loses

usage
  python eval_mix.py A lgbxgb=lgb_v5x_noll:0.5,xgb_v5x_noll:0.5 base=lgb_v5x_noll_noanom
  python eval_mix.py B ...
Each argument is NAME=stem[:tag],stem[:tag],...  Stems are averaged over seeds and then
over the listed members with equal weight; add ':tag' for a prediction-file suffix.
"""
import polars as pl, numpy as np, sys, glob, re, os

# real test blocks: 2015-09 | 2016-01..03 | 2016-06..09 | 2016-12..2017-06 | 2018-07 | 2018-11..12
TEST_BLOCKS = (1, 3, 4, 7, 1, 2)


def test_mix(blocks=TEST_BLOCKS, max_h=7):
    """Share of test rows at each horizon: a block of L consecutive months contributes
    one month at each horizon 1..L, and every month covers the whole globe."""
    n = np.zeros(max_h + 1)
    for L in blocks:
        for h in range(1, min(L, max_h) + 1):
            n[h] += 1
    return n[1:] / n[1:].sum()


def layout_mix(h, max_h=7):
    return np.array([(h == i).mean() for i in range(1, max_h + 1)])


def load(L, members):
    acc = None
    for m in members:
        f = m.split(":")
        stem = f[0]; tag = f[1] if len(f) > 1 else ""
        pat = re.compile(rf"pred_{re.escape(L)}_{re.escape(stem)}_s\d+{re.escape(tag)}\.npy$")
        fs = sorted(x for x in glob.glob(f"out/mats/pred_{L}_{stem}_s*{tag}.npy")
                    if pat.search(os.path.basename(x)))
        if not fs:
            raise SystemExit(f"no prediction files for {stem!r} (tag {tag!r}) in layout {L}")
        p = np.mean([np.load(x) for x in fs], 0)
        acc = p if acc is None else acc + p
    return acc / len(members)


def main():
    L = sys.argv[1]
    sets = []
    for a in sys.argv[2:]:
        name, spec = a.split("=", 1)
        sets.append((name, spec.split(",")))
    va = pl.read_parquet(f"out/mats/{L}_va.parquet", columns=["horizon", "tws_known", "target"])
    y = va["target"].to_numpy(); h = va["horizon"].to_numpy(); k = va["tws_known"].to_numpy()
    w = test_mix()
    lm = layout_mix(h)
    print(f"layout {L}: {len(va)} rows")
    print("  horizon share   " + "  ".join(f"h{i}" for i in range(1, 8)))
    print("    this layout   " + "  ".join(f"{x:.3f}" for x in lm))
    print("    real test     " + "  ".join(f"{x:.3f}" for x in w))
    print("    difference    " + "  ".join(f"{x:+.3f}" for x in lm - w))

    def scores(p):
        mse_h = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                          for i in range(1, 8)])
        plain = float(np.sqrt(np.mean((y - p) ** 2)))
        mix = float(np.sqrt(np.nansum(w * np.where(np.isnan(mse_h), 0, mse_h))
                            / np.sum(w[~np.isnan(mse_h)])))
        return plain, mix, np.sqrt(mse_h)

    print(f"\n  {'set':22} {'plain':>8} {'testmix':>8}   " + "  ".join(f"h{i:<5}" for i in range(1, 8)))
    base = None
    for name, members in [("persistence", None)] + sets:
        p = k if members is None else load(L, members)
        plain, mix, rh = scores(p)
        if base is None and members is not None:
            base = mix
        d = "" if members is None or base is None else f"  ({mix - base:+.4f})"
        print(f"  {name:22} {plain:8.4f} {mix:8.4f}   " + "  ".join(f"{x:.4f}" for x in rh) + d)

    if len(sets) >= 2:
        print("\n  per-horizon difference vs the first set (negative = better):")
        p0 = load(L, sets[0][1]); _, _, r0 = scores(p0)
        for name, members in sets[1:]:
            _, _, r = scores(load(L, members))
            print(f"  {name:22} " + "  ".join(f"{x:+.4f}" for x in r - r0))


if __name__ == "__main__":
    main()
