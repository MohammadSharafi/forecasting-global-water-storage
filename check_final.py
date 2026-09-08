"""Does the FINAL matrix carry every feature the models were selected on? (session 9y)

This gates FINAL training in run_night.sh: if the FINAL matrix is missing features that the
validation layouts had, the models would train on a different feature set than the one every
decision was made with, and nothing else would notice.

It lives in a file rather than in a heredoc inside the orchestrator because that heredoc was being
mangled. `step` passes its command to `eval`, which re-parses the quoting, and the Python came out
as `have=set(pl.scan_parquet(n')))` -- a SyntaxError. The step then failed, and because it gates
phase 7, an entire run trained nothing at all and still wrote submissions from whatever
predictions happened to be on disk. That is the most expensive kind of bug this orchestrator can
have, and a file cannot be mangled.

exit 0 = every feature present; exit 1 = something is missing, and it says what.
"""
import json
import os
import sys

import polars as pl


def main():
    f = "out/mats/feats.json"
    if not os.path.exists(f):
        print(f"{f} missing -- build a validation layout first"); return 1
    want = set(json.load(open(f)))
    have = set()
    for p in ("out/mats/FINAL_va.parquet", "out/mats/FINAL_va_anchor.parquet"):
        if not os.path.exists(p):
            print(f"{p} missing -- run build_mats.py FINAL and add_anchor_feats.py FINAL")
            return 1
        have |= set(pl.scan_parquet(p).collect_schema().names())
    missing = sorted(want - have)
    if missing:
        print(f"MISSING {len(missing)} of {len(want)} features, e.g. {missing[:6]}")
        return 1
    print(f"FINAL carries all {len(want)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
