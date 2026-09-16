import xarray as xr, numpy as np, polars as pl, glob, os
FILES={"P":"prate.sfc.mon.mean.nc","LH":"lhtfl.sfc.mon.mean.nc","R":"runof.sfc.mon.mean.nc","SWE":"weasd.sfc.mon.mean.nc",
       "SW1":"soilw.0-10cm.mon.mean.nc","SW2":"soilw.10-200cm.mon.mean.nc"}
def load_ncep(lats, lons, start="2001-01-01", end="2019-06-01"):
    lat_c=np.array(sorted(set(lats))); lon_c=np.array(sorted(set(lons)))
    out=None
    for name,f in FILES.items():
        p=f"external/ncep/{f}"
        if not (os.path.exists(p) and os.path.getsize(p)>1e5): print("skip",name); continue
        ds=xr.open_dataset(p); v=[k for k in ds.data_vars if k!="time_bnds"][0]
        da=ds[v].sel(time=slice(start,end))

        da=da.assign_coords(lon=(((da.lon+180)%360)-180)).sortby("lon").sortby("lat")
        da=da.interp(lat=lat_c, lon=lon_c, method="linear", kwargs={"fill_value":None})
        df=da.to_dataframe(name=name).reset_index()[["time","lat","lon",name]]
        d=pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))
        out=d if out is None else out.join(d,on=["time","lat","lon"],how="outer_coalesce" if False else "full",coalesce=True)
    out=out.with_columns(pl.col("time").dt.month_end().dt.day().alias("dim"))
    cols=[]
    if "P" in out.columns: out=out.with_columns((pl.col("P")*86400*pl.col("dim")).alias("P")); cols.append("P")
    if "LH" in out.columns: out=out.with_columns((pl.col("LH")/2.5e6*86400*pl.col("dim")).alias("E")).drop("LH"); cols.append("E")
    if "R" in out.columns: cols.append("R")
    if "SWE" in out.columns: cols.append("SWE")
    if "SW1" in out.columns and "SW2" in out.columns: out=out.with_columns((pl.col("SW1")*100+pl.col("SW2")*1900).alias("SW")).drop("SW1","SW2"); cols.append("SW")
    if "P" in cols and "E" in cols: out=out.with_columns((pl.col("P")-pl.col("E")-(pl.col("R") if "R" in cols else 0)).alias("PER")); cols.append("PER")
    return out.select(["lat","lon","time"]+cols).drop("dim",strict=False), cols
def add_ncep(r, nc, cols):
    r=r.join(nc.rename({c:c+"_t" for c in cols}),on=["lat","lon","time"],how="left")
    r=r.join(nc.rename({"time":"t_known",**{c:c+"_k" for c in cols}}),on=["lat","lon","t_known"],how="left")
    for c in [c for c in ("SW","SWE") if c in cols]: r=r.with_columns((pl.col(c+"_t")-pl.col(c+"_k")).alias(c+"_d"))
    acc_cols=[c for c in ("P","E","R","PER") if c in cols]
    w=nc.select(["lat","lon","time"]+acc_cols).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).unique().join(w,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg([pl.col(c).sum().alias(c+"_acc") for c in acc_cols])
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")
    feats=[c+"_t" for c in cols]+[c+"_k" for c in cols]+[c+"_d" for c in ("SW","SWE") if c in cols]+[c+"_acc" for c in acc_cols]
    return r, feats
