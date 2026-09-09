# Copernicus GDO covariates — what to download and where to put it

Optional. The pipeline runs unchanged without any of this: `load_gdo` skips a product whose
directory is absent, `build_mats` prints `gdo: not present`, and the feature count is unchanged.

## Why these two products

The residual error in this model is large-scale and spatially coherent — correlation +0.967 with
the neighbouring cell, +0.899 at two cells (`analyze.py`). That is why a finer-resolution version
of the *same* forcing (ERA5-Land) was ruled out: the error does not live at fine scales. It says
nothing about new **variables**, and two GDO products are genuinely new information rather than
another function of the P, E and R already in the feature set:

| product | directory | why |
|---|---|---|
| SPI at 24 and 48 months | `external/gdo/spi24/`, `external/gdo/spi48/` | the feature set stops at a 12-month window. TWS includes groundwater, which integrates over years, so nothing in the model currently sees beyond one year of accumulated deficit. |
| fAPAR, or its anomaly | `external/gdo/fapar/`, `external/gdo/fapanom/` | vegetation greenness is an *observation* of how much water the surface actually had. It is not derived from the same reanalysis as everything else here, so its errors are not the same errors. |

`spi09` and `smanom` are also recognised if you want them.

## What is deliberately NOT used

- **GDO's GRACE TWS anomaly layer.** Prohibited outright.
- **GDO seasonal forecasts.** A forecast issued in month *t* has a source date of *t* but encodes
  *t+1*. The rules permit the first and prohibit the second; the question was raised publicly and
  is unanswered, so under a code review the prohibition is the clause that applies.

## Getting it

The products are published by the JRC through the Copernicus European/Global Drought Observatory,
which the challenge's own Overview page links to. Download the **global (GDO)** monthly NetCDF for
the full period **2002-01 to 2019-01** — the model needs history well before the test era to build
each cell's climatology — and drop the files in the directories above. One directory per product;
multiple files per directory are fine and are concatenated in time.

### What was actually retrieved (2026-09-09)

    root  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DROUGHTOBS/Drought_Observatories_datasets

    external/gdo/spi24/   GDO_GPCC_Standardized_Precipitation_Index_SPI24/ver1-2-0/
                          spg24_m_wld_<YYYY>0101_<YYYY>1201_m.nc, 2001..2019, 19 files, 14 MB
    external/gdo/spi48/   GDO_GPCC_Standardized_Precipitation_Index_SPI48/ver1-2-0/
                          spg48_m_wld_<YYYY>0101_<YYYY>1201_m.nc, 2001..2019, 19 files, 14 MB

**GPCC, not ERA5.** The observatory publishes each SPI window three times over — from ERA5, from
GPCC and from MARSMet. The GPCC version is the one taken here, and the reason is the same reason
fAPAR is on the list: ERA5 precipitation is *already* in this feature set, so an SPI computed from
ERA5 is a rescaling of a column the model has, while GPCC is an independent gauge analysis whose
errors are not the same errors. Taking the ERA5 version would have added twelve features and no
information.

Downloaded into `~/gdo_dl` first and copied into `external/gdo` only when no build was running:
files arriving under `external/` change `src_guard`'s inventory fingerprint, and a matrix built
half-way through a download would carry a different feature set from the one built before it.

Retrieved as one NetCDF per year; `load_gdo` concatenates them. Verified on the competition's own
grid before use: 222 months (2001-01…2019-06), **95.5% of cell-months non-null**, 14,946 of 15,715
cells carrying 200 or more months, and a sane distribution (median +0.06, 1st and 99th percentiles
−2.97 and +2.85 for SPI-24).

**Not taken:** `GDO_GRACE_Total_Water_Storage_Anomaly`, which sits in the same directory listing and
is prohibited outright; and the seasonal forecasts, for the reason in the section above.

## Format tolerance

You should not have to reshape anything. `features_gdo.py` handles what these files actually look
like, each case verified against a synthetic file built to have that quirk:

- several files per product, concatenated along time
- `latitude`/`longitude` as well as `lat`/`lon`
- longitudes on 0–360, folded to −180–180
- a bounds variable alongside the data variable
- a stray singleton dimension (`lev`, and similar)
- **dekadal** data (three values per month), averaged to monthly — a no-op for monthly products

It then interpolates to the competition's 1° grid and hands the result to `features_anom.build`,
the same per-cell standardisation that produced this project's only large leaderboard gain. That
is the point: SPI and fAPAR are levels, and a model with no location features cannot use a level.

## Checking it worked

```sh
python build_mats.py A          # expect: gdo: ['spi24', 'fapar', ...]
python - <<'EOF'
import json; F=json.load(open("out/mats/feats.json"))
print([f for f in F if f.startswith(("an_spi","an_fapa","an_smanom"))])
EOF
```

Each product yields six features: the z-scored level at *t* and at the last observed month, their
difference, and 3/6/12-month antecedent windows ending at *t*.

## Measuring whether it helped

```sh
DROPF=gdo TAG=_nogdo python run_models.py A lgb v5x_noll_sa
TAG=_gdo             python run_models.py A lgb v5x_noll_sa
python eval_mix.py A without=lgb_v5x_noll:_nogdo with=lgb_v5x_noll:_gdo
```

Repeat on B and C. The standing rule applies: adopt only if it wins on every layout under the test
horizon mix. `DROPF=gdo` exists precisely so this block can be ablated on its own — it shares the
`an_` prefix with the ERA5 and NCEP anomalies, so without a dedicated switch its contribution could
never be separated from theirs.
