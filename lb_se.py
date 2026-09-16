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
