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

    df = pl.DataFrame({"ID": ids, "Target": np.round(p, 6)})
    try:
        order = pl.read_csv("Test.csv", columns=["ID"])
        df = order.join(df, on="ID", how="left")
        assert df["Target"].null_count() == 0 and len(df) == 280961, "ID mismatch vs Test.csv"
    except FileNotFoundError:
        print("  Test.csv not found -- writing in sorted-ID order")
    df.write_csv(out, float_precision=6)
    print(f"  -> {out}  ({len(p)} rows, mean {p.mean():+.5f}, sd {p.std():.5f})")

    for path, _ in spec:
        d = pl.read_csv(path).sort("ID")
        v = d["Target" if "Target" in d.columns else d.columns[-1]].to_numpy().astype(float)
        rms = float(np.sqrt(np.mean((p - v) ** 2)))
        print(f"     RMS difference from {path}: {rms:.4f}"
              f"   -> SE of the public gap ~= {rms/np.sqrt(84288):.5f}")

if __name__ == "__main__":
    main()
