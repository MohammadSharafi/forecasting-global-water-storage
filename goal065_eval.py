"""goal065 measurement harness: score a candidate against the shipped control on five layouts.

modes
  trained   score control vs treatment prediction files per layout (test-mix RMSE) and print
            the one-line verdict summary as the LAST line of stdout
  selftest  assert the verdict logic, n/a handling, seed averaging, the test-mix formula and the
            exact summary-line format on synthetic inputs; prints `selftest ok` last

Scoring is eval_mix.py's, not a re-derivation: the horizon weights are eval_mix.test_mix()
(real test blocks 1,3,4,7,1,2), the per-horizon MSE / nan-horizon renormalisation is the same
expression as eval_mix.main().scores(), and each arm is the plain mean over its per-seed
prediction files before anything is scored.  delta = treatment - control, negative = better.

  python goal065_eval.py trained --cand NAME --control _d0 --treat _NAME1 [--layouts D,E] [--no-append]

Reusable from other scripts (linear-correction candidates score arrays, not files):
  from goal065_eval import testmix_rmse, verdict, summary_line, load_arm, LAYOUTS
"""
import argparse, glob, json, os, re, sys
import numpy as np

LAYOUTS = ("Avn2", "Bvn2", "Cvn2", "D", "E")
STEM = "lgb_v5x_noll"            # pred_<L>_lgb_v5x_noll_s<seed><TAG>.npy, used_<L>_lgb_v5x_noll<TAG>.json
SEEDS = (0, 1)
MATS = "out/mats"
OUTDIR = "out/goal065"
ADOPT_MEAN = -0.041              # ADOPT-0.65 needs mean <= this ...
ADOPT_EACH = -0.003              # ... and every layout <= this (also the clears-0.003 threshold)
HELPS_EACH = -0.0003             # HELPS-NOT-0.65 needs every layout <= this
ADOPT, HELPS, REJECT, ABANDON = "ADOPT-0.65", "HELPS-NOT-0.65", "REJECT", "ABANDON-EARLY"
SUMMARY_RE = re.compile(r"^\S+ \| Avn2 \S+ Bvn2 \S+ Cvn2 \S+ D \S+ E \S+ \| wins \d/5 \| mean \S+ \| "
                        r"worst \S+ \| clears-0\.003 \d/5 \| (ADOPT-0\.65|HELPS-NOT-0\.65|REJECT|ABANDON-EARLY)$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HEADER = "goal065_eval"          # first token of every file this tool writes


# ----------------------------------------------------------------------------- scoring
def testmix_rmse(y, h, p):
    """Test-mix RMSE exactly as eval_mix.py computes `testmix`: per-horizon MSE for h=1..7,
    weighted by eval_mix.test_mix(); a horizon with no rows is dropped and the weights of
    the present horizons renormalised."""
    from eval_mix import test_mix
    w = test_mix()
    y = np.asarray(y); h = np.asarray(h); p = np.asarray(p)
    mse_h = np.array([np.mean((y[h == i] - p[h == i]) ** 2) if (h == i).any() else np.nan
                      for i in range(1, 8)])
    return float(np.sqrt(np.nansum(w * np.where(np.isnan(mse_h), 0, mse_h))
                         / np.sum(w[~np.isnan(mse_h)])))


def pred_paths(L, tag, stem=STEM, seeds=SEEDS, mats=MATS):
    """Per-seed prediction files for one arm, or None if any requested seed is missing.
    seeds=None reproduces eval_mix.load's glob (every s<digits> file with this exact tag)."""
    if seeds is None:
        pat = re.compile(rf"pred_{re.escape(L)}_{re.escape(stem)}_s\d+{re.escape(tag)}\.npy")
        fs = sorted(x for x in glob.glob(os.path.join(mats, f"pred_{L}_{stem}_s*{tag}.npy"))
                    if pat.fullmatch(os.path.basename(x)))
        return fs or None
    fs = [os.path.join(mats, f"pred_{L}_{stem}_s{s}{tag}.npy") for s in seeds]
    return fs if all(os.path.exists(f) for f in fs) else None


def load_arm(L, tag, stem=STEM, seeds=SEEDS, mats=MATS):
    """Seed-averaged prediction for one arm (eval_mix.load's np.mean over seeds), or None."""
    fs = pred_paths(L, tag, stem, seeds, mats)
    if fs is None:
        return None, []
    return np.mean([np.load(x) for x in fs], 0), fs


def used_count(L, tag, stem=STEM, mats=MATS):
    """Feature count from used_<L>_<stem>*<tag>.json (key 'features', or the list itself)."""
    exact = os.path.join(mats, f"used_{L}_{stem}{tag}.json")
    cands = [exact] if os.path.exists(exact) else sorted(glob.glob(os.path.join(mats, f"used_{L}_{stem}*{tag}.json")))
    if len(cands) != 1:
        return None
    try:
        d = json.load(open(cands[0]))
    except (OSError, ValueError):
        return None
    feats = d.get("features") if isinstance(d, dict) else d
    return len(feats) if isinstance(feats, list) else None


def sidecar_cols(L, xf, mats=MATS):
    """Columns a candidate adds = columns of out/mats/<L>_tr_<name>.parquet summed over XF names.
    None (provenance '?') when a side-car is absent or unreadable -- scoring must not die on it."""
    import polars as pl
    n = 0
    for name in [x for x in xf.split(",") if x]:
        f = os.path.join(mats, f"{L}_tr_{name}.parquet")
        if not os.path.exists(f):
            return None
        try:
            n += len(pl.scan_parquet(f).collect_schema().names())
        except Exception:
            return None
    return n


# ----------------------------------------------------------------------------- verdict
def _measured(deltas):
    return [float(deltas[L]) for L in LAYOUTS if deltas.get(L) is not None]


def verdict(deltas):
    """deltas: {layout: float or None (n/a)}. Every comparison is on unrounded values."""
    m = _measured(deltas)
    if len(m) == len(LAYOUTS):
        mean = sum(m) / len(m)
        if all(d < 0 for d in m) and mean <= ADOPT_MEAN and all(d <= ADOPT_EACH for d in m):
            return ADOPT
        if all(d < 0 for d in m) and all(d <= HELPS_EACH for d in m):
            return HELPS
        return REJECT
    if sum(d > 0 for d in m) >= 2:
        return ABANDON
    return REJECT


def _fmt(d):
    return "n/a" if d is None else "%+.4f" % d


def summary_line(cand, deltas):
    m = _measured(deltas)
    wins = sum(d < 0 for d in m)
    clears = sum(d <= ADOPT_EACH for d in m)
    mean = sum(m) / len(m) if m else None
    worst = max(m) if m else None
    lay = " ".join(f"{L} {_fmt(None if deltas.get(L) is None else float(deltas[L]))}" for L in LAYOUTS)
    return (f"{cand} | {lay} | wins {wins}/5 | mean {_fmt(mean)} | worst {_fmt(worst)} | "
            f"clears-0.003 {clears}/5 | {verdict(deltas)}")


# ----------------------------------------------------------------------------- output file
def safe_to_write(path, cand):
    """Only overwrite a file that is this candidate's own: refuse if an existing file carries a
    summary line for any OTHER candidate (a driver-owned aggregate file does)."""
    if not os.path.exists(path):
        return True
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if SUMMARY_RE.match(line) and line.split(" ", 1)[0] != cand:
            return False
    return True


def score_layouts(cand, control, treat, layouts, stem=STEM, seeds=SEEDS, xf=None, mats=MATS):
    import polars as pl
    rows, deltas, notes = [], {L: None for L in LAYOUTS}, []
    for L in layouts:
        r = {"L": L, "rows": None, "ctl": None, "trt": None, "d": None, "nc": None, "nt": None,
             "add": None, "prov": "?", "why": ""}
        rows.append(r)
        r["nc"] = used_count(L, control, stem, mats); r["nt"] = used_count(L, treat, stem, mats)
        r["add"] = sidecar_cols(L, xf or cand, mats)
        if r["nc"] is not None and r["nt"] is not None and r["add"] is not None:
            r["prov"] = "ok" if r["nt"] - r["nc"] == r["add"] else "MISMATCH"
            if r["prov"] == "MISMATCH":
                notes.append(f"PROVENANCE MISMATCH {L}: treatment {r['nt']} - control {r['nc']} "
                             f"= {r['nt'] - r['nc']} features, side-car adds {r['add']}")
        va_f = os.path.join(mats, f"{L}_va.parquet")
        if not os.path.exists(va_f):
            r["why"] = f"no {L}_va.parquet"; continue
        pc, fc = load_arm(L, control, stem, seeds, mats)
        pt, ft = load_arm(L, treat, stem, seeds, mats)
        if pc is None or pt is None:
            r["why"] = "missing " + " and ".join(a for a, p in (("control", pc), ("treatment", pt)) if p is None)
            continue
        va = pl.read_parquet(va_f, columns=["horizon", "target"])
        y = va["target"].to_numpy(); h = va["horizon"].to_numpy(); r["rows"] = len(va)
        if len(pc) != len(y) or len(pt) != len(y):
            r["why"] = f"length mismatch va={len(y)} control={len(pc)} treatment={len(pt)}"
            notes.append(f"{L}: {r['why']}"); continue
        if not (np.isfinite(pc).all() and np.isfinite(pt).all()):
            r["why"] = "non-finite predictions"; notes.append(f"{L}: non-finite predictions"); continue
        r["ctl"] = testmix_rmse(y, h, pc); r["trt"] = testmix_rmse(y, h, pt)
        r["d"] = r["trt"] - r["ctl"]; deltas[L] = r["d"]
        r["why"] = f"seeds {len(fc)}/{len(ft)}"
    return rows, deltas, notes


def render(cand, control, treat, stem, seeds, rows, deltas, notes):
    out = [f"{HEADER} {cand}: control {control} vs treatment {treat}  stem {stem}  "
           f"seeds {'all' if seeds is None else ','.join(map(str, seeds))}  score test-mix RMSE (eval_mix.test_mix)"]
    out.append(f"  {'layout':6} {'rows':>7} {'control':>8} {'treat':>8} {'delta':>9}  "
               f"{'nF_ctl':>6} {'nF_trt':>6} {'added':>5} {'prov':>8}  note")
    s = lambda v, f: "n/a" if v is None else f % v
    for r in rows:
        out.append(f"  {r['L']:6} {s(r['rows'], '%d'):>7} {s(r['ctl'], '%.4f'):>8} {s(r['trt'], '%.4f'):>8} "
                   f"{s(r['d'], '%+.6f'):>9}  {s(r['nc'], '%d'):>6} {s(r['nt'], '%d'):>6} {s(r['add'], '%d'):>5} "
                   f"{r['prov']:>8}  {r['why']}")
    out += ["  " + n for n in notes]
    out.append(summary_line(cand, deltas))
    return out


def cmd_trained(a):
    if not NAME_RE.match(a.cand):
        raise SystemExit(f"--cand must match {NAME_RE.pattern}: {a.cand!r}")
    layouts = [x for x in a.layouts.split(",") if x]
    bad = [x for x in layouts if x not in LAYOUTS]
    if bad:
        raise SystemExit(f"unknown layout(s) {bad}; allowed {','.join(LAYOUTS)}")
    seeds = None if a.seeds == "all" else tuple(int(x) for x in a.seeds.split(","))
    rows, deltas, notes = score_layouts(a.cand, a.control, a.treat, layouts, a.stem, seeds, a.xf)
    out = render(a.cand, a.control, a.treat, a.stem, seeds, rows, deltas, notes)
    if not a.no_append:
        os.makedirs(OUTDIR, exist_ok=True)
        path = os.path.join(OUTDIR, f"{a.cand}.txt")
        if not safe_to_write(path, a.cand):
            raise SystemExit(f"refusing to overwrite {path}: it holds summary lines for other candidates")
        with open(path, "w") as f:
            f.write("\n".join(out) + "\n")
        out.insert(-1, f"  wrote {path}")
    print("\n".join(out))


# ----------------------------------------------------------------------------- selftest
def cmd_selftest(_a):
    import tempfile
    Z = dict(zip(LAYOUTS, [None] * 5))
    D = lambda *v: dict(zip(LAYOUTS, v))
    cases = [
        # (deltas, verdict, exact summary line or None)
        (D(-0.05, -0.04, -0.045, -0.035, -0.05), ADOPT,
         "c | Avn2 -0.0500 Bvn2 -0.0400 Cvn2 -0.0450 D -0.0350 E -0.0500 | wins 5/5 | mean -0.0440 | worst -0.0350 | clears-0.003 5/5 | ADOPT-0.65"),
        (D(-0.01, -0.01, -0.01, -0.01, -0.01), HELPS, None),                 # 5/5 but mean > -0.041
        (D(-0.1, -0.1, -0.1, -0.002, -0.1), HELPS,                            # mean ok, one layout > -0.003
         "c | Avn2 -0.1000 Bvn2 -0.1000 Cvn2 -0.1000 D -0.0020 E -0.1000 | wins 5/5 | mean -0.0804 | worst -0.0020 | clears-0.003 4/5 | HELPS-NOT-0.65"),
        (D(-0.003, -0.003, -0.003, -0.003, -0.003), HELPS, None),             # boundary: -0.003 clears, mean > -0.041
        (D(-0.0003, -0.001, -0.001, -0.001, -0.001), HELPS, None),            # boundary -0.0003 inclusive
        (D(-0.00029, -0.01, -0.01, -0.01, -0.01), REJECT, None),              # a win, but not <= -0.0003
        (D(-0.05, -0.05, -0.05, -0.05, +0.001), REJECT, None),                # one loss, all measured
        (D(+0.01, +0.01, -0.05, -0.05, -0.05), REJECT, None),                 # two losses but all 5 measured
        (D(-0.05, -0.05, -0.05, -0.05, 0.0), REJECT, None),                   # exactly zero is not a win
        (dict(Z, D=+0.001, E=+0.0005), ABANDON,
         "c | Avn2 n/a Bvn2 n/a Cvn2 n/a D +0.0010 E +0.0005 | wins 0/5 | mean +0.0008 | worst +0.0010 | clears-0.003 0/5 | ABANDON-EARLY"),
        (dict(Z, D=+0.00001, E=+0.00002, Cvn2=-0.00001), ABANDON,             # rounds to +0.0000 but still a loss
         "c | Avn2 n/a Bvn2 n/a Cvn2 -0.0000 D +0.0000 E +0.0000 | wins 1/5 | mean +0.0000 | worst +0.0000 | clears-0.003 0/5 | ABANDON-EARLY"),
        (dict(Z, D=-0.00001, E=+0.001, Cvn2=-0.05), REJECT,                  # -0.0000 is a win: one loss only
         "c | Avn2 n/a Bvn2 n/a Cvn2 -0.0500 D -0.0000 E +0.0010 | wins 2/5 | mean -0.0163 | worst +0.0010 | clears-0.003 1/5 | REJECT"),
        (dict(Z, D=-0.05, E=-0.05, Cvn2=-0.05, Bvn2=-0.05), REJECT, None),     # 4/5 winning but unfinished
        (dict(Z, D=+0.001, E=0.0), REJECT, None),                              # zero is not a loss either
        (Z, REJECT,
         "c | Avn2 n/a Bvn2 n/a Cvn2 n/a D n/a E n/a | wins 0/5 | mean n/a | worst n/a | clears-0.003 0/5 | REJECT"),
        (D(-0.05, -0.05, -0.05, -0.05, None), REJECT, None),                  # 4 measured, 0 losses
        (dict(Z, Avn2=+0.01, Bvn2=+0.02, Cvn2=+0.03, D=+0.04), ABANDON, None),
        (D(-0.0409, -0.0409, -0.0409, -0.0409, -0.0409), HELPS, None),         # mean just above -0.041
        (D(-0.0411, -0.0411, -0.0411, -0.0411, -0.0411), ADOPT, None),
    ]
    for i, (d, v, line) in enumerate(cases):
        got = verdict(d)
        assert got == v, f"case {i}: verdict {got} != {v} for {d}"
        s = summary_line("c", d)
        assert SUMMARY_RE.match(s), f"case {i}: summary regex fails: {s}"
        assert s.endswith("| " + v), f"case {i}: {s}"
        if line is not None:
            assert s == line, f"case {i}:\n got  {s}\n want {line}"
    # numpy scalars behave like floats
    assert verdict({L: np.float64(-0.05) for L in LAYOUTS}) == ADOPT
    assert SUMMARY_RE.match(summary_line("my_cand.v2", D(-0.05, -0.05, -0.05, -0.05, -0.05)))
    assert not SUMMARY_RE.match(summary_line("bad name", D(-0.05, -0.05, -0.05, -0.05, -0.05)))

    # test-mix RMSE: identical error on every horizon -> that error; h1-only error -> sqrt(w1);
    # a missing horizon renormalises; and equal to eval_mix's formula written out independently
    from eval_mix import test_mix
    w = test_mix()
    h = np.repeat(np.arange(1, 8), 10); y = np.zeros(len(h))
    assert abs(testmix_rmse(y, h, y + 0.5) - 0.5) < 1e-12
    p = np.where(h == 1, 1.0, 0.0)
    assert abs(testmix_rmse(y, h, p) - np.sqrt(w[0])) < 1e-12
    k = h != 7
    assert abs(testmix_rmse(y[k], h[k], p[k]) - np.sqrt(w[0] / w[:6].sum())) < 1e-12
    rng = np.random.default_rng(0)
    hh = rng.integers(1, 8, 5000); yy = rng.normal(size=5000); pp = yy + rng.normal(size=5000) * hh / 7
    ref = np.sqrt(sum(w[i - 1] * np.mean((yy[hh == i] - pp[hh == i]) ** 2) for i in range(1, 8)))
    assert abs(testmix_rmse(yy, hh, pp) - ref) < 1e-12
    assert abs(w.sum() - 1) < 1e-12 and np.allclose(w, np.array([6, 4, 3, 2, 1, 1, 1]) / 18)

    # seed averaging, missing seed -> n/a, eval_mix-style glob does not pick up other tags
    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, "pred_D_lgb_v5x_noll_s0_x1.npy"), np.array([1.0, 1.0]))
        np.save(os.path.join(td, "pred_D_lgb_v5x_noll_s1_x1.npy"), np.array([3.0, 5.0]))
        np.save(os.path.join(td, "pred_D_lgb_v5x_noll_s0_zx1.npy"), np.array([99.0, 99.0]))
        np.save(os.path.join(td, "pred_Dvn2_lgb_v5x_noll_s0_x1.npy"), np.array([99.0, 99.0]))
        np.save(os.path.join(td, "pred_E_lgb_v5x_noll_s0_x1.npy"), np.array([1.0, 1.0]))
        p, fs = load_arm("D", "_x1", mats=td)
        assert np.array_equal(p, [2.0, 3.0]) and len(fs) == 2, (p, fs)
        p, fs = load_arm("D", "_x1", seeds=None, mats=td)
        assert np.array_equal(p, [2.0, 3.0]) and len(fs) == 2, (p, fs)
        assert load_arm("E", "_x1", mats=td)[0] is None           # s1 missing -> unmeasured
        assert load_arm("D", "_x2", mats=td)[0] is None
        json.dump({"features": ["a", "b", "c"]}, open(os.path.join(td, "used_D_lgb_v5x_noll_x1.json"), "w"))
        json.dump(["a"], open(os.path.join(td, "used_D_lgb_v5x_noll_x0.json"), "w"))
        assert used_count("D", "_x1", mats=td) == 3 and used_count("D", "_x0", mats=td) == 1
        assert used_count("D", "_x9", mats=td) is None
        # an existing file holding other candidates' summary lines is never overwritten
        f = os.path.join(td, "agg.txt")
        open(f, "w").write(summary_line("other", Z) + "\n")
        assert not safe_to_write(f, "agg")
        open(f, "w").write("header\n" + summary_line("agg", Z) + "\n")
        assert safe_to_write(f, "agg") and safe_to_write(os.path.join(td, "new.txt"), "new")
    print(f"{len(cases)} verdict cases, test-mix formula, seed averaging, provenance counts, write guard")
    print("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", metavar="{trained,selftest}")
    t = sub.add_parser("trained", help="trained: score control vs treatment predictions per layout, print the summary line last")
    t.add_argument("--cand", required=True, help="candidate name (first field of the summary line)")
    t.add_argument("--control", required=True, help="control prediction tag, e.g. _d0")
    t.add_argument("--treat", required=True, help="treatment prediction tag, e.g. _NAME1")
    t.add_argument("--layouts", default=",".join(LAYOUTS), help="comma list, default all five")
    t.add_argument("--no-append", action="store_true", help="do not write out/goal065/<cand>.txt")
    t.add_argument("--stem", default=STEM, help="prediction stem MODEL_FEATSET (default %(default)s)")
    t.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                   help="seeds to average, all required (default %(default)s); 'all' = eval_mix glob")
    t.add_argument("--xf", default=None, help="side-car name(s) for the added-column provenance check (default: --cand)")
    sub.add_parser("selftest", help="selftest: assert verdict logic, n/a handling and summary-line format")
    a = ap.parse_args(argv)
    if a.mode == "trained":
        cmd_trained(a)
    elif a.mode == "selftest":
        cmd_selftest(a)
    else:
        ap.print_help(); sys.exit(2)


if __name__ == "__main__":
    main()
