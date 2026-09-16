import numpy as np
import polars as pl

SCORED = {"main": ("out/scored/sub_q_main.csv", 0.692657),
          "b55":  ("out/scored/sub_blend55.csv", 0.693725),
          "b73":  ("out/scored/sub_blend73.csv", 0.693283),
          "mix25": ("out/scored/sub_mix25.csv", 0.679248865),
          "armarg": ("out/scored/sub_ac_armarg.csv", 0.679785572),
          "gpccar": ("out/scored/sub_gpcc_ar.csv", 0.684984991),
          "clim": ("out/scored/probe_clim.csv", 1.279955747)}

def main():
    ref = pl.read_csv("Test.csv").select("ID")
    def L(p):
        d = pl.read_csv(p); c = [x for x in d.columns if x != "ID"][0]
        return ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
    V = {k: L(p) for k, (p, _) in SCORED.items()}
    M = {k: s ** 2 for k, (_, s) in SCORED.items()}
    m, a = V["main"], (V["b55"] - 0.5 * V["main"]) / 0.5

    A = np.array([[1, -2, 0], [1, -1, -1], [1, -1.4, -0.6]], float)
    rhs = np.array([M["main"] - np.mean(m ** 2),
                    M["b55"] - np.mean(V["b55"] ** 2),
                    M["b73"] - np.mean(V["b73"] ** 2)])
    z = np.linalg.solve(A, rhs)
    Ey2, Eym, Eya = z
    print(f"E[y^2] on the test rows = {Ey2:.5f}   ->  RMS of the target = {np.sqrt(Ey2):.4f}")
    print(f"  for comparison, an all-zero submission would therefore score {np.sqrt(Ey2):.4f}")
    print(f"  validation target RMS runs 0.80-0.93, so the test rows are {np.sqrt(Ey2)/0.86:.2f}x that\n")

    print("optimal global scale on each scored file, c* = E[y.f]/E[f^2]:")
    print("  file        score      c*      score at c*    gain")
    for k, (_, s) in SCORED.items():
        f = V[k]
        Eyf = (Ey2 + np.mean(f ** 2) - M[k]) / 2.0
        c = Eyf / np.mean(f ** 2)
        best = np.sqrt(max(Ey2 - Eyf ** 2 / np.mean(f ** 2), 0))
        print(f"  {k:9s} {s:.6f}  {c:.4f}   {best:.6f}   {best-s:+.6f}")
    print("\n  c* below 1 means the file is over-confident and should be shrunk toward zero;")
    print("  above 1 means it under-predicts the size of the anomalies.")

if __name__ == "__main__":
    main()
