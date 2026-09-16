import numpy as np
import polars as pl

T = 0.55
REC, HON, OUT = "out/scored/sub_mix25.csv", "out/sub_w65_ar.csv", "out/sub_submit.csv"

def load(ref, path):
    d = pl.read_csv(path); c = [x for x in d.columns if x != "ID"][0]
    v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
    assert np.isfinite(v).all(), path
    return v

def main():
    ref = pl.read_csv("Test.csv").select("ID")
    rec, hon = load(ref, REC), load(ref, HON)
    out = (1 - T) * rec + T * hon
    assert np.isfinite(out).all() and len(out) == 280961
    pl.DataFrame({"ID": ref["ID"], "Target": out}).write_csv(OUT)
    print(f"wrote {OUT}  rows={len(out)}  mean={out.mean():+.4f}  sd={out.std():.4f}")
    print(f"  weight {T:.2f} honest / {1-T:.2f} record")
    print(f"  RMS distance from the record {np.sqrt(np.mean((out-rec)**2)):.4f}"
          f"   from the honest file {np.sqrt(np.mean((out-hon)**2)):.4f}")

if __name__ == "__main__":
    main()
