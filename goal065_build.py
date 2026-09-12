"""Build the goal-0.65 handover: measure the adopted set jointly, write it down, emit the submission.

Nothing here is adopted on a public-leaderboard number. The only change that cleared this project's
every-layout bar is the marginal AR anchor rescaling from ar_blend.py, which is fitted on validation
labels across five layouts and scored on the held-out one. This script re-measures it (never quotes
it), states the base it is added to and how much of that base's edge was public-specific, and writes
out/goal065/adopted_set.txt plus out/sub_goal065.csv.

Base choice: sub_x_lb2 rather than the better-displaying sub_z_lb. Of sub_x_lb2's public gain 5%
evaporated against the all-rows prediction, against 99% for sub_z_lb, and that evaporated share is
by construction the part driven by public-versus-private row structure. The goal says "without
hurting the private 70%", so the less public-fitted base wins even though it displays worse.

usage: python goal065_build.py
"""
import numpy as np
import polars as pl

from ar_blend import LAYOUTS, fit, layout_data, mixmse

# ar_blend keeps these inside main(), so restate them here rather than reaching into it
K3, K4 = ["p", "k", "c"], ["p", "k", "c", "ar"]
from goal065_eval import summary_line

BASE = "out/scored/sub_x_lb2.csv"
BASE_PUBLIC = 0.683712087
BASE_EVAPORATED = "5%"
FINAL_MATRIX = "FINALvn2"


def load(ref, path):
    d = pl.read_csv(path)
    col = [c for c in d.columns if c != "ID"][0]
    v = ref.join(d.rename({col: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
    assert np.isfinite(v).all(), path
    return v


def main():
    D = {L: layout_data(L) for L in LAYOUTS}
    deltas = {}
    for L in D:
        tr = [D[x] for x in D if x != L]
        w3, w4 = fit(tr, K3), fit(tr, K4)
        d = D[L]
        a = np.sqrt(mixmse(d["y"], d["h"], w3 @ np.stack([d[x] for x in K3])))
        b = np.sqrt(mixmse(d["y"], d["h"], w4 @ np.stack([d[x] for x in K4])))
        deltas[L] = b - a
    line = summary_line("ADOPTED-SET", deltas)

    w3, w4 = fit(list(D.values()), K3), fit(list(D.values()), K4)
    ref = pl.read_csv("Test.csv").select("ID")
    model = load(ref, "out/sub_v_anwide.csv"); pers = load(ref, "out/probe_persistence.csv")
    clim = load(ref, "out/probe_clim.csv"); ar = load(ref, "out/probe_ar.csv")
    base = load(ref, BASE)
    corr = w4 @ np.stack([model, pers, clim, ar]) - w3 @ np.stack([model, pers, clim])
    out = base + corr
    pl.DataFrame({"ID": ref["ID"], "Target": out}).write_csv("out/sub_goal065.csv")

    per = " ".join(f"{L} {deltas[L]:+.4f}" for L in LAYOUTS)
    with open("out/goal065/adopted_set.txt", "w") as f:
        f.write("members: the marginal AR anchor rescaling only (ar_blend.py). No trained candidate "
                "was adopted: arf, arseas, ar2 and c1 are all in heldout.txt with their numbers.\n")
        f.write(f"arblend-on-top: {per} | wins {sum(v < 0 for v in deltas.values())}/5 | "
                f"measured on top of a persistence/climatology-corrected base, leave-one-layout-out\n")
        f.write(f"final-matrix: {FINAL_MATRIX}\n")
        f.write(f"base: {BASE} (public {BASE_PUBLIC}); {BASE_EVAPORATED} of its incremental public "
                f"gain was public-specific, against 99% for the better-displaying sub_z_lb, so this "
                f"base carries less private risk\n")
        f.write(f"weights: w3 {np.round(w3, 4).tolist()}  w4 {np.round(w4, 4).tolist()}\n")
        f.write(f"correction: mean {corr.mean():+.4f} std {corr.std():.4f} "
                f"range [{corr.min():.2f},{corr.max():.2f}]\n")
        f.write(line + "\n")
    print(line)
    print(f"wrote out/sub_goal065.csv  std={out.std():.4f}  "
          f"RMS distance from the base={np.sqrt(np.mean(corr ** 2)):.4f}")


if __name__ == "__main__":
    main()
