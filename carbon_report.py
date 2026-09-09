"""Total the emissions CodeCarbon recorded during the runs that produced the submission.

The sustainability criterion asks for training emissions measured while training runs. run_models.py
writes one CodeCarbon row per training run into out/carbon/emissions.csv when CARBON=1, so this is
a sum over the runs that actually happened rather than a one-seed extrapolation -- which is the
distinction the organisers drew when they said it "cannot be applied afterwards".

Reports the total, the split by phase (validation grid, chosen-configuration families, FINAL
training), and the per-family cost, so the report can say where the energy went and not only how
much there was.

usage: python carbon_report.py   ->  out/carbon/summary.md
"""
import glob
import os
import sys
import polars as pl

CSV = "out/carbon/emissions.csv"


def main():
    if not os.path.exists(CSV):
        print("no out/carbon/emissions.csv -- training was not instrumented (run with CARBON=1)")
        sys.exit(1)
    d = pl.read_csv(CSV, infer_schema_length=None)
    need = [c for c in ("project_name", "emissions", "duration", "energy_consumed") if c in d.columns]
    d = d.select(need)
    d = d.with_columns(pl.col("emissions").cast(pl.Float64, strict=False),
                       pl.col("duration").cast(pl.Float64, strict=False))
    # project_name is LAYOUT_MODEL_FEATSET[TAG]_sSEED
    d = d.with_columns(pl.col("project_name").str.split("_").list.first().alias("layout"))
    d = d.with_columns(
        # A layout may carry its variant in its name -- 'FINALe' is a FINAL matrix built with the
        # soil profile -- so matching the literal "FINAL" put 32 runs that trained the submitted
        # models into the validation bucket, understating the cost of the thing being submitted.
        pl.when(pl.col("layout").str.starts_with("FINAL")).then(pl.lit("FINAL training"))
          .otherwise(pl.lit("validation")).alias("phase"))
    tot = float(d["emissions"].sum())
    hrs = float(d["duration"].sum()) / 3600.0
    lines = ["# Training emissions", "",
             f"Measured with CodeCarbon during the runs themselves, over {len(d)} training runs.", "",
             f"- **total: {tot:.4f} kg CO2e**",
             f"- total training time: {hrs:.2f} hours",
             f"- mean per run: {tot/max(len(d),1)*1000:.2f} g CO2e", ""]
    g = d.group_by("phase").agg(pl.col("emissions").sum().alias("kg"),
                                (pl.col("duration").sum() / 3600).alias("hours"),
                                pl.len().alias("runs")).sort("kg", descending=True)
    lines += ["| phase | runs | hours | kg CO2e |", "|---|---|---|---|"]
    for r in g.iter_rows(named=True):
        lines.append(f"| {r['phase']} | {r['runs']} | {r['hours']:.2f} | {r['kg']:.4f} |")
    lines.append("")
    if "energy_consumed" in d.columns:
        e = d["energy_consumed"].cast(pl.Float64, strict=False).sum()
        lines.append(f"Energy consumed: {float(e):.4f} kWh.")
        lines.append("")
    lines.append("Runs that were skipped because a checkpoint already existed are not counted, so "
                 "this is the cost of the work actually done in the measured runs. Earlier "
                 "sessions' training was not instrumented and is not included.")
    os.makedirs("out/carbon", exist_ok=True)
    open("out/carbon/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
