# Getting the ERA5 covariates

ERA5 has been wired into `build_mats.py` since session 8 and has never once been downloaded, so
every leaderboard number so far was produced without it (three confirmations in NOTES.md, session
10a). This is the setup that fixes that.

## Why it is worth the twenty minutes

The competition grid is 40x40 one-degree cells over tropical South America — the Amazon, the
eastern Andes and the Nordeste. What the models have been fed instead is NCEP-R1/R2 at roughly
two degrees, where one reanalysis cell covers four to six target cells, with latent heat flux
standing in for evaporation and a two-layer soil column. ERA5 is one degree, lands on the target
grid exactly, and carries evaporation, runoff and a four-layer soil column directly. The feature
family that produced the entire session-9 leaderboard gain was the per-cell standardised
anomalies of exactly these fields; `features_anom.build` is already called for ERA5 at
`build_mats.py:79` and has simply always been handed `None`.

## 1. Account and licence (browser, five minutes)

1. Register, free: https://cds.climate.copernicus.eu → log in → click your name → **API token**.
2. Accept the licence **once**, or every request fails with a 403:
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download
   → scroll to the bottom → tick *accept terms*.
3. Create `~/.cdsapirc` with exactly two lines, your own token in the second:

       url: https://cds.climate.copernicus.eu/api
       key: <your-token>

## 2. Download

`cdsapi` is deliberately **not** in `requirements.txt`: it is needed once, to fetch data, and
never on the training path. Installing it into the project venv while a run is in flight risks
upgrading a dependency underneath that run, so use a throwaway environment:

    python3 -m venv /tmp/cdsenv
    /tmp/cdsenv/bin/pip install -q cdsapi polars

Then, from the project folder:

    ERA5_DIR=external/.era5_dl /tmp/cdsenv/bin/python cds_download.py

19 files, 2001..2019, cut to the competition's own box plus five degrees — about a twentieth of
the global request this script used to make. CDS queues requests; usually minutes. Re-running
resumes: a file already on disk and larger than 100 kB is skipped.

## 3. Move it into place — after any running job has finished

    mkdir -p external/era5 && mv external/.era5_dl/*.nc external/era5/

`ERA5_DIR` and the staging directory exist for one reason: the external files are now part of
`src_guard build_`'s fingerprint, so files arriving mid-run would invalidate the matrices while
some had already been built. A matrix built before the download and one built after it would
carry different feature sets, and nothing downstream would notice. Stage, then move.

## 4. Confirm it actually reached a model

Two checks, and the second is the one that matters — the first only proves the files parse:

    ./.venv/bin/python -c "from features_era5 import load_era5; print(load_era5().shape)"
    grep -h '^era5:' out/night/build_*.log        # must NOT say None
    ./.venv/bin/python -c "import json; f=json.load(open('out/mats/feats.json')); \
      print(len([x for x in f if x.startswith('e5')]), 'ERA5 features')"

If the third prints 0 after a rebuild, the matrices were reused rather than rebuilt: check that
`out/night/src_build_.txt` changed, and if it did not, `rm -f out/night/build_*.done`.

## Compliance

ERA5 is a public reanalysis, not a GRACE product, and ERA5T is released about five days after
each month ends — so every value used at a row is available at or before that row's month, which
is the rule the competition actually imposes. Recorded in REPORT.md section 5.

## For whoever adds variables later

Snow is dead weight here. `e5SWE` (from `sd`) and NCEP's `weasd` are essentially zero over a box
that stops at 19.5N, so they cost features and buy nothing outside a few Andean cells. And the
four `swvl` layers are currently collapsed into a single `SW` column by a fixed-thickness
weighted sum: the profile shape — a fast top layer against a slow bottom one — is thrown away,
and is worth carrying separately once ERA5 is actually present.
