import glob
import os
import sys
import datetime as dt

S = sys.argv[1] if len(sys.argv) > 1 else "out/night"
OUT = "out/RUN_REPORT.md"

SECTIONS = [
    ("__config__",    "Configuration chosen", None),
    ("select",        "Why that configuration: the experiment grid", None),
    ("eval_grid_A",   "Grid scored under the real test horizon mix -- layout A", None),
    ("eval_grid_B",   "Grid scored under the real test horizon mix -- layout B", None),
    ("eval_grid_C",   "Grid scored under the real test horizon mix -- layout C", None),
    ("blendscan",     "Ensemble weight (lgb vs xgb)", None),
    ("stackscan",     "Ensemble weights over every family", None),
    ("smoothscan",    "Spatial smoothing", 40),
    ("hsplicescan",   "Horizon-1 specialist", None),
    ("postcalscan",   "Per-horizon calibration", None),
    ("data_report",   "What is in the data", None),
    ("analyze_A",     "Where the error is -- layout A", None),
    ("analyze_B",     "Where the error is -- layout B", None),
    ("diag_ceiling_A", "How much the covariates can explain (before the rebuild)", None),
    ("diag2_ceiling_A", "How much the covariates can explain (after the rebuild)", None),
    ("diag_shift_A",  "The global month-to-month shift", None),
    ("diag2_shift_A", "The global month-to-month shift (after the rebuild)", None),
    ("compliance",    "Compliance audit", None),
    ("check_FINAL",   "Feature completeness of the FINAL matrix", None),
]

def read(stem, tail=None):
    p = os.path.join(S, f"{stem}.log")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read().rstrip()
    if not t:
        return None
    if tail:
        lines = t.splitlines()
        if len(lines) > tail:
            t = f"... ({len(lines)-tail} earlier lines omitted)\n" + "\n".join(lines[-tail:])
    return t

def main():
    parts = [f"# Overnight run report\n\nGenerated {dt.datetime.now():%Y-%m-%d %H:%M}."]
    done = len(glob.glob(os.path.join(S, "*.done")))
    logs = glob.glob(os.path.join(S, "*.log"))
    failed = [os.path.basename(l)[:-4] for l in sorted(logs)
              if not os.path.exists(l[:-4] + ".done")]
    parts.append(f"{done} steps completed, {len(failed)} did not.")
    if failed:
        parts.append("Incomplete steps: " + ", ".join(f"`{f}`" for f in failed))

    subs = sorted(glob.glob("out/sub_q_*.csv"))
    if subs:
        parts.append("\n## Submission files\n\n"
                     + "\n".join(f"- `{s}` ({os.path.getsize(s)/1e6:.1f} MB)" for s in subs))

    for stem, title, tail in SECTIONS:
        if stem == "__config__":
            body = ""
            for f in ("out/config.sh", "out/blendw.sh", "out/stack.sh", "out/smoothw.sh",
                      "out/h1beta.sh", "out/calib.sh"):
                if os.path.exists(f) and os.path.getsize(f):
                    body += open(f).read()
            if body.strip():
                parts.append(f"\n## {title}\n\n```sh\n{body.strip()}\n```")
            continue
        t = read(stem, tail)
        if t:
            parts.append(f"\n## {title}\n\n```\n{t}\n```")

    os.makedirs("out", exist_ok=True)
    open(OUT, "w").write("\n".join(parts) + "\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} kB, {len(parts)} sections)")

if __name__ == "__main__":
    main()
