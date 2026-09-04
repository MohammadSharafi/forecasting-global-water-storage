"""ERA5 water-balance features. Loads external/era5/*.nc -> long table (lat, lon, time, vars),
then per row: values at t, at t_known, and accumulated P-E-R over (t_known, t].
Units: tp/e/ro are monthly-mean m/day in monthly-means product -> multiply by days in month
for monthly totals (mm equivalent x1000). swvl are m3/m3; sd is m of water equivalent."""
import polars as pl, numpy as np, glob, xarray as xr
from features import mdiff
VARS={"tp":"tp","e":"e","ro":"ro","sd":"sd","swvl1":"swvl1","swvl2":"swvl2","swvl3":"swvl3","swvl4":"swvl4","t2m":"t2m"}
def load_era5(pattern="external/era5/era5_monthly_1deg_*.nc"):
    ds=xr.open_mfdataset(sorted(glob.glob(pattern)),combine="by_coords")
    tcoord="valid_time" if "valid_time" in ds.coords else "time"
    df=ds.to_dataframe().reset_index().rename(columns={tcoord:"time","latitude":"lat","longitude":"lon"})
    df=pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))
    df=df.with_columns(pl.col("time").dt.month_end().dt.day().alias("dim"))
    # monthly totals in mm; soil water column (m) = sum(layer thickness * volumetric)
    df=df.with_columns([(pl.col("tp")*1000*pl.col("dim")).alias("P"),(-pl.col("e")*1000*pl.col("dim")).alias("E"),(pl.col("ro")*1000*pl.col("dim")).alias("R"),
                        (0.07*pl.col("swvl1")+0.21*pl.col("swvl2")+0.72*pl.col("swvl3")+1.89*pl.col("swvl4")).alias("SW"), (pl.col("sd")*1000).alias("SWE"), pl.col("t2m").alias("T2M")])
    df=df.with_columns((pl.col("P")-pl.col("E")-pl.col("R")).alias("PER"))
    return df.select(["lat","lon","time","P","E","R","PER","SW","SWE","T2M"])
E5=["P","E","R","PER","SW","SWE","T2M"]
def add_era5(r, era):
    """r must have lat, lon, time, t_known, horizon."""
    r=r.join(era.rename({c:c+"_t" for c in E5}),on=["lat","lon","time"],how="left")
    r=r.join(era.rename({"time":"t_known",**{c:c+"_k" for c in E5}}),on=["lat","lon","t_known"],how="left")
    for c in ("SW","SWE","T2M"): r=r.with_columns((pl.col(c+"_t")-pl.col(c+"_k")).alias(c+"_d"))
    # accumulated water balance over (t_known, t]  (the months TWS was unobserved, plus t itself)
    w=era.select(["lat","lon","time","P","E","R","PER"]).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).unique().join(w,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg(pl.col("PER").sum().alias("PER_acc"),pl.col("P").sum().alias("P_acc"),pl.col("E").sum().alias("E_acc"),pl.col("R").sum().alias("R_acc"))
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")
    # anomaly of accumulated balance vs the cell's climatological balance for those months (from era table itself)
    return r
ERA5F=[c+"_t" for c in E5]+[c+"_k" for c in E5]+["SW_d","SWE_d","T2M_d","PER_acc","P_acc","E_acc","R_acc"]
