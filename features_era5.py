"""ERA5 water-balance features. Loads external/era5/*.nc -> long table (lat, lon, time, vars),
then per row: values at t, at t_known, and accumulated P-E-R over (t_known, t].
Units: tp/e/ro are monthly-mean m/day in monthly-means product -> multiply by days in month
for monthly totals (mm equivalent x1000). swvl are m3/m3; sd is m of water equivalent."""
import polars as pl, numpy as np, glob, xarray as xr, os
from features import mdiff
VARS={"tp":"tp","e":"e","ro":"ro","sd":"sd","swvl1":"swvl1","swvl2":"swvl2","swvl3":"swvl3","swvl4":"swvl4","t2m":"t2m"}
def load_era5(pattern="external/era5/era5_monthly_1deg_*.nc", prof=False):
    """Each CDS file is a zip with two netcdf members (accumulated-mean stream: tp,e,ro; instantaneous-mean stream:
    sd, swvl1-4, t2m). Each member becomes its own (lat, lon, month) table; the two are joined on cell-month."""
    import zipfile
    tabs={}
    for f in sorted(glob.glob(pattern)):
        d_=f[:-3]+"_x"; os.makedirs(d_,exist_ok=True); z=zipfile.ZipFile(f); z.extractall(d_)
        for n in z.namelist():
            if not n.endswith(".nc"): continue
            ds=xr.open_dataset(os.path.join(d_,n),engine="netcdf4").load()
            ds=ds.drop_vars([v for v in ("number","expver") if v in ds.variables])
            tcoord="valid_time" if "valid_time" in ds.coords else "time"
            df=ds.to_dataframe().reset_index().rename(columns={tcoord:"time","latitude":"lat","longitude":"lon"})
            df=pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))
            key=tuple(sorted(c for c in df.columns if c not in ("lat","lon","time")))
            tabs.setdefault(key,[]).append(df.select(["lat","lon","time"]+list(key)))
    parts=[pl.concat(v).unique(["lat","lon","time"]) for v in tabs.values()]
    df=parts[0]
    for p in parts[1:]: df=df.join(p,on=["lat","lon","time"],how="inner")
    df=df.with_columns(pl.col("time").dt.month_end().dt.day().alias("dim"))
    # monthly totals in mm; soil water column (m) = sum(layer thickness * volumetric)
    df=df.with_columns([(pl.col("tp")*1000*pl.col("dim")).alias("P"),(-pl.col("e")*1000*pl.col("dim")).alias("E"),(pl.col("ro")*1000*pl.col("dim")).alias("R"),
                        (0.07*pl.col("swvl1")+0.21*pl.col("swvl2")+0.72*pl.col("swvl3")+1.89*pl.col("swvl4")).alias("SW"), (pl.col("sd")*1000).alias("SWE"), pl.col("t2m").alias("T2M")])
    df=df.with_columns((pl.col("P")-pl.col("E")-pl.col("R")).alias("PER"))
    keep=["P","E","R","PER","SW","SWE","T2M"]
    if prof:
        # The four ERA5 soil layers, kept apart instead of summed. Every soil product in this
        # pipeline collapses its profile -- ERA5's four swvl layers into one SW here, and NCEP-R1
        # and R2's 0-10cm and 10-200cm into one SW in features_ncep/features_x -- so the shallow
        # and deep stores appear nowhere as separate quantities. That is the drainage timescale:
        # the 7 cm top layer answers a month of rain, the 189 cm bottom layer integrates seasons,
        # and TWS is the integral. A fixed-weight sum is exactly the operation that destroys the
        # contrast between them. Each is metres of water, so they sum back to SW by construction.
        df=df.with_columns([(0.07*pl.col("swvl1")).alias("SW1"),(0.21*pl.col("swvl2")).alias("SW2"),
                            (0.72*pl.col("swvl3")).alias("SW3"),(1.89*pl.col("swvl4")).alias("SW4")])
        keep+=["SW1","SW2","SW3","SW4"]
    df=df.select(["lat","lon","time"]+keep).rename({c:"e5"+c for c in keep})
    return df
E5=["e5P","e5E","e5R","e5PER","e5SW","e5SWE","e5T2M"]
E5PROF=["e5SW1","e5SW2","e5SW3","e5SW4"]
E5DIFF=["e5SW","e5SWE","e5T2M"]


def e5_cols(era):
    """The ERA5 columns actually present, so the soil-profile split is additive and optional."""
    return [c for c in E5+E5PROF if c in era.columns]


def era5_feats(era):
    c=e5_cols(era); d=[x for x in E5DIFF+E5PROF if x in era.columns]
    return [x+"_t" for x in c]+[x+"_k" for x in c]+[x+"_d" for x in d]+["e5PER_acc","e5P_acc","e5E_acc","e5R_acc"]
def add_era5(r, era):
    """r must have lat, lon, time, t_known, horizon."""
    E=e5_cols(era)
    r=r.join(era.rename({c:c+"_t" for c in E}),on=["lat","lon","time"],how="left")
    r=r.join(era.rename({"time":"t_known",**{c:c+"_k" for c in E}}),on=["lat","lon","t_known"],how="left")
    for c in [x for x in E5DIFF+E5PROF if x in E]: r=r.with_columns((pl.col(c+"_t")-pl.col(c+"_k")).alias(c+"_d"))
    # accumulated water balance over (t_known, t]  (the months TWS was unobserved, plus t itself)
    w=era.select(["lat","lon","time","e5P","e5E","e5R","e5PER"]).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).unique().join(w,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg(pl.col("e5PER").sum().alias("e5PER_acc"),pl.col("e5P").sum().alias("e5P_acc"),pl.col("e5E").sum().alias("e5E_acc"),pl.col("e5R").sum().alias("e5R_acc"))
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")
    # anomaly of accumulated balance vs the cell's climatological balance for those months (from era table itself)
    return r
ERA5F=[c+"_t" for c in E5]+[c+"_k" for c in E5]+["e5SW_d","e5SWE_d","e5T2M_d","e5PER_acc","e5P_acc","e5E_acc","e5R_acc"]
