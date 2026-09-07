"""Is a public-leaderboard difference real, or is it noise?  (session 9)

For two submissions A and B scored on the same public rows, write d = pB - pA and
e = y - pA.  Then the per-row difference in squared error is

    dMSE_i = (y-pA)^2 - (y-pB)^2 = d_i * (2 e_i - d_i)

so SD(dMSE_i) ~= 2 * RMS(d) * RMS(e), and after converting MSE to RMSE by dividing by
2*RMSE the standard error of the observed RMSE gap collapses to a one-liner:

    SE(dRMSE) ~= RMS(d) / sqrt(N_public)

N_public ~= 0.30 * 280961 ~= 84288, so sqrt(N) ~= 290.  Everything therefore hinges on how
different the two submissions actually are, which is computable from the CSVs alone --
no labels needed.  Rule of thumb from this project's numbers:

  two stacks with different anchor philosophies differ by RMS(d) ~ 0.11  -> SE ~ 3.8e-4
  a blend-weight tweak or a post-processing change   RMS(d) ~ 0.03       -> SE ~ 1.0e-4

which is why the recent-vs-long-term gap (0.7191 vs 0.7110 = 81 SE units) is certain,
while gaps of a few 1e-4 between neighbouring blends need this check before being believed.

usage: python lb_se.py out/sub_g_longterm.csv out/sub_h_l7_rb.csv [public_rmse] [gap]
"""
import polars as pl, numpy as np, sys

a = pl.read_csv(sys.argv[1]).sort("ID")
b = pl.read_csv(sys.argv[2]).sort("ID")
assert a["ID"].to_list() == b["ID"].to_list(), "submissions cover different IDs"
d = b["Target"].to_numpy() - a["Target"].to_numpy()
rms_d = float(np.sqrt(np.mean(d ** 2)))
n_pub = int(round(0.30 * len(d)))
se = rms_d / np.sqrt(n_pub)
print(f"rows            {len(d)}   (public ~{n_pub})")
print(f"RMS(difference) {rms_d:.4f}   max |diff| {np.abs(d).max():.4f}")
print(f"SE of the public RMSE gap  ~= {se:.5f}")
if len(sys.argv) > 4:
    gap = abs(float(sys.argv[4]))
    print(f"observed gap {gap:.5f} = {gap/se:.1f} SE"
          f"  -> {'real' if gap > 2.5*se else 'NOT distinguishable from noise'}")
else:
    print(f"a gap below {2.5*se:.5f} between these two is not distinguishable from noise")
