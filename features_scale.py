"""Zonal-scale context for the anchor field (session 9d).

The gap this closes
-------------------
Session 3 measured that this test period's unpredictable component is *regional and
spatially coherent* -- month-to-month global offsets of +-0.3 in 2015 -- and not white
per-cell noise.  That is the single largest identified error term.  But the widest
spatial aggregation anywhere in the feature set is a radius-4 grid convolution (~4 deg,
440 km at the equator and less toward the poles) plus the 800 km smoothed anchor.  There
is nothing at zonal, continental or global scale, so the model has no way to see "this
whole latitude band has been draining for three months".

What this builds
----------------
A zonal decomposition of the observed TWS field at the last observed month:

  zm      mean observed TWS over the cell's latitude band (default 5 deg, all longitudes)
  zn      how many cells that mean is built from
  dz      the cell's departure from its band -- the quantity whose persistence the model
          should learn separately from the band's own level
  zd3     the band's change over the 3 months before t_known   (zonal momentum)
  zd12    the same over 12 months
  dzd3    d3 - zd3: the cell's own 3-month change with the band's change removed
  dzd12   the same at 12 months

Why zonal and not global: session 3's ONI experiment showed a single global index per
month is heavily used by the trees and makes both layouts worse, because a handful of
ENSO cycles is not enough to learn from and the index doubles as a month identifier.
A band-month value has 36 bands x ~160 months behind it, and the deviations (dz, dzd*)
are cell-relative rather than time-indexed, so they carry far less of that risk.  No raw
global level is exposed here for exactly that reason.

Compliance: only observed TWS at months <= t_known enters, which the organisers' 24 Aug
ruling permits ("neighbouring cells' TWS at months <= t, including spatial filtering and
aggregation").  Latitude is used to group a cell with its band, never as a feature.
"""
import polars as pl

BAND = 5.0
LAGS = (3, 12)
SCALE = ["zm", "zn", "dz", "zd3", "zd12", "dzd3", "dzd12"]


def zonal_table(obs, band=BAND):
    """obs: lat, lon, time, TWS_t (observed only).  Mean per (month, latitude band)."""
    return (obs.with_columns(((pl.col("lat") + 90) // band).cast(pl.Int32).alias("zb"))
               .group_by(["time", "zb"])
               .agg(pl.col("TWS_t").mean().alias("zm"), pl.len().alias("zn")))


def add_scale(r, obs, band=BAND, lags=LAGS):
    """r must have lat, lon, t_known, tws_known and the AR features d3 / d12."""
    zt = zonal_table(obs, band)
    r = r.with_columns(((pl.col("lat") + 90) // band).cast(pl.Int32).alias("zb"))
    r = r.join(zt.rename({"time": "t_known"}), on=["t_known", "zb"], how="left")
    for L in lags:
        r = r.with_columns(pl.col("t_known").dt.offset_by(f"-{L}mo").alias("tz"))
        r = r.join(zt.select(["time", "zb", "zm"]).rename({"time": "tz", "zm": f"_zm{L}"}),
                   on=["tz", "zb"], how="left").drop("tz")
        r = r.with_columns((pl.col("zm") - pl.col(f"_zm{L}")).alias(f"zd{L}"))
        # the cell's own change with the band's change removed
        r = r.with_columns((pl.col(f"d{L}") - pl.col(f"zd{L}")).alias(f"dzd{L}"))
    r = r.with_columns((pl.col("tws_known") - pl.col("zm")).alias("dz"))
    return r.drop([f"_zm{L}" for L in lags] + ["zb"])
