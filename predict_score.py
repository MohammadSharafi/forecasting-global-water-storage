import numpy as np
import polars as pl

REC_PUBLIC = 0.679248865
GPCC_AR_PUBLIC = 0.684984991
T = 0.55

ADOPTED = [("WaterGAP storage block", -0.0040, "3/3"),
           ("HMIX removal",           -0.0007, "2/2"),
           ("255 leaves (lgbd)",      -0.0017, "4/5"),
           ("900 rounds",             -0.0011, "1 layout, Avn2")]

def load(ref, path):
    d = pl.read_csv(path); c = [x for x in d.columns if x != "ID"][0]
    v = ref.join(d.rename({c: "v"}), on="ID", how="left")["v"].to_numpy().astype(float)
    assert np.isfinite(v).all(), path
    return v

def mix_rmse(m_rec, m_hon, d2, t=T):
    return float(np.sqrt((1 - t) * m_rec ** 2 + t * m_hon ** 2 - (1 - t) * t * d2))

def main(hon_path="out/sub_lgbd_ar.csv"):
    ref = pl.read_csv("Test.csv").select("ID")
    rec, hon = load(ref, "out/scored/sub_mix25.csv"), load(ref, hon_path)
    d2 = float(np.mean((rec - hon) ** 2))
    tot = sum(g for _, g, _ in ADOPTED)
    print(f"honest file: {hon_path}")
    print(f"D^2 between the record and it: {d2:.5f}  (RMS {np.sqrt(d2):.4f})\n")
    print("validation gains adopted since the scored honest anchor:")
    for n, g, w in ADOPTED:
        print(f"   {n:24s} {g:+.4f}   {w}")
    print(f"   {'total':24s} {tot:+.4f}\n")

    print("PUBLIC")
    print(f"  anchor: sub_gpcc_ar scored {GPCC_AR_PUBLIC:.6f}, no board-fitted component")
    rows = []
    for tr in (0.60, 0.85, 1.00):
        mh = GPCC_AR_PUBLIC + tot * tr
        rows.append((tr, mh, mix_rmse(REC_PUBLIC, mh, d2)))
        print(f"   transfer {tr:.0%}: honest file {mh:.4f}  ->  submission {rows[-1][2]:.4f}")
    print(f"  central estimate (85% transfer): {rows[1][2]:.4f}")
    print(f"  and the last honest forecast ran 0.004-0.006 pessimistic, so treat that as a ceiling")

    print("\nPRIVATE")
    print("  the honest file carries no public-fitted weights, so private ~ public")
    print("  the record is 0.75*sub_ac_armarg + 0.25*sub_gpcc_ar; sub_ac_armarg descends from")
    print("  sub_z_lb whose incremental public gain was 99% public-specific")
    mh = GPCC_AR_PUBLIC + tot * 0.85
    for infl in (0.000, 0.004, 0.008):
        mr = REC_PUBLIC + infl
        print(f"   record inflated by {infl:.3f} -> record private {mr:.4f}, "
              f"submission {mix_rmse(mr, mh, d2):.4f}")
    print(f"  central estimate: record inflated ~0.004-0.005 -> submission ~{mix_rmse(REC_PUBLIC+0.0045, mh, d2):.4f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "out/sub_lgbd_ar.csv")
