"""Blend submission CSVs that already have public scores (session 9x).

Why this is worth a slot
------------------------
stack.py fitted the ensemble weights on validation and gave the 31-leaf LightGBM a weight of
0.000 -- validation says it adds nothing. The leaderboard disagrees about how good that model is:
`sub_q_alt`, built from it, scores 0.698048 against the main file's 0.695965. Two points apart is
not "adds nothing"; it is a strong, genuinely different model.

Validation has been wrong about exactly this kind of question before. It adopted the per-horizon
calibration, which the leaderboard then refuted. So rather than trust the fitted zero, this makes
the blend directly and spends one submission to settle it. Averaging two models of different
capacity is the single most reliable source of a small gain in this kind of problem, and it costs
no training at all -- the files already exist.

Read the result the same way as everything else: the gap has to clear what lb_se.py says is
readable for the pair, or it is noise.

usage: python blend_subs.py out/sub_blend.csv out/sub_q_main.csv:0.7 out/sub_q_alt.csv:0.3
"""
import sys
import numpy as np
import polars as pl


def main():
    out = sys.argv[1]
    spec = [a.rsplit(":", 1) for a in sys.argv[2:]]
    if not spec:
        raise SystemExit(__doc__)
    acc, wsum, ids = None, 0.0, None
    for path, w in spec:
        d = pl.read_csv(path)
        col = "Target" if "Target" in d.columns else d.columns[-1]
        d = d.sort("ID")
        if ids is None:
            ids = d["ID"]
        elif not (d["ID"] == ids).all():
            raise SystemExit(f"{path}: IDs do not match {spec[0][0]}")
        v = d[col].to_numpy().astype(float)
        print(f"  {path:34} w={float(w):.2f}  mean {v.mean():+.5f}  sd {v.std():.5f}")
        acc = float(w) * v if acc is None else acc + float(w) * v
        wsum += float(w)
    p = acc / wsum
    assert np.isfinite(p).all() and len(p) == 280961, f"{len(p)} rows, expected 280961"
    pl.DataFrame({"ID": ids, "Target": np.round(p, 6)}).write_csv(out, float_precision=6)
    print(f"  -> {out}  ({len(p)} rows, mean {p.mean():+.5f}, sd {p.std():.5f})")
    # how far the blend sits from each input, which bounds how much the score can move
    for path, _ in spec:
        d = pl.read_csv(path).sort("ID")
        v = d["Target" if "Target" in d.columns else d.columns[-1]].to_numpy().astype(float)
        rms = float(np.sqrt(np.mean((p - v) ** 2)))
        print(f"     RMS difference from {path}: {rms:.4f}"
              f"   -> SE of the public gap ~= {rms/np.sqrt(84288):.5f}")


if __name__ == "__main__":
    main()
