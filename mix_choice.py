import numpy as np
import polars as pl

REC, HON = "out/scored/sub_mix25.csv", "out/sub_w65_ar.csv"
REC_PUBLIC = 0.679248865

def load(ref, path):
    d = pl.read_csv(path); c = [x for x in d.columns if x != "ID"][0]
    v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
    assert np.isfinite(v).all(), path
    return v

def main():
    ref = pl.read_csv("Test.csv").select("ID")
    rec, hon = load(ref, REC), load(ref, HON)
    D2 = float(np.mean((rec - hon) ** 2))
    print(f"RMS distance between the two files: {np.sqrt(D2):.4f}   D^2={D2:.5f}")

    recs = [0.679, 0.684, 0.689]
    hons = [0.676, 0.680, 0.684, 0.688]
    ts = np.arange(0, 1.001, 0.05)
    worst = []
    for t in ts:
        w = max((1 - t) * r ** 2 + t * hh ** 2 - (1 - t) * t * D2 for r in recs for hh in hons)
        worst.append(np.sqrt(w))
    worst = np.array(worst)
    i = int(worst.argmin())
    print(f"\nminimax weight on the honest file: t={ts[i]:.2f}, worst case {worst[i]:.6f}")
    print("\n  t     worst-case RMSE   best-case RMSE")
    for t, w in zip(ts, worst):
        b = np.sqrt(min((1 - t) * r ** 2 + t * hh ** 2 - (1 - t) * t * D2 for r in recs for hh in hons))
        mark = "  <-- minimax" if abs(t - ts[i]) < 1e-9 else ""
        if abs(t * 20 - round(t * 20)) < 1e-9 and round(t * 20) % 2 == 0:
            print(f" {t:.2f}      {w:.6f}        {b:.6f}{mark}")
    print(f"\nrecord's public mark for reference: {REC_PUBLIC:.6f}")

if __name__ == "__main__":
    main()
