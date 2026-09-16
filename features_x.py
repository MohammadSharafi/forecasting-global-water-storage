import xarray as xr, numpy as np, polars as pl, os
from features import mdiff, COV

def _open(p, lat_c, lon_c, start, end):
    ds=xr.open_dataset(p); v=[k for k in ds.data_vars if not k.endswith("_bnds")][0]
    da=ds[v]
    if "level" in da.dims: da=da.isel(level=0)
    da=da.sel(time=slice(start,end))
    da=da.assign_coords(lon=(((da.lon+180)%360)-180)).sortby("lon").sortby("lat")
    da=da.interp(lat=lat_c, lon=lon_c, method="linear", kwargs={"fill_value":None})
    df=da.to_dataframe(name="v").reset_index()[["time","lat","lon","v"]]
    return pl.from_pandas(df).with_columns(pl.col("time").cast(pl.Date).dt.truncate("1mo"))

def load_ncep2(lats, lons, d="external/ncep2", start="2001-01-01", end="2019-06-01"):
    lat_c=np.array(sorted(set(lats))); lon_c=np.array(sorted(set(lons)))
    FILES={"P":"prate.sfc.mon.mean.nc","LH":"lhtfl.sfc.mon.mean.nc","R":"runof.sfc.mon.mean.nc","SWE":"weasd.sfc.mon.mean.nc",
           "SW1":"soilw.0-10cm.mon.mean.nc","SW2":"soilw.10-200cm.mon.mean.nc"}
    out=None
    for name,f in FILES.items():
        d_=_open(f"{d}/{f}",lat_c,lon_c,start,end).rename({"v":name})
        out=d_ if out is None else out.join(d_,on=["time","lat","lon"],how="full",coalesce=True)
    out=out.with_columns(pl.col("time").dt.month_end().dt.day().alias("dim"))
    out=out.with_columns((pl.col("P")*86400*pl.col("dim")).alias("P"),(pl.col("LH")/2.5e6*86400*pl.col("dim")).alias("E"),
                         (pl.col("SW1")*100+pl.col("SW2")*1900).alias("SW"))
    out=out.with_columns((pl.col("P")-pl.col("E")-pl.col("R")).alias("PER"))
    cols=["P","E","R","SWE","SW","PER"]
    return out.select(["lat","lon","time"]+cols).rename({c:"r2"+c for c in cols}), ["r2"+c for c in cols]

def load_cpc(lats, lons, p="external/cpc/soilw.mon.mean.nc", start="2001-01-01", end="2019-06-01"):
    lat_c=np.array(sorted(set(lats))); lon_c=np.array(sorted(set(lons)))
    return _open(p,lat_c,lon_c,start,end).rename({"v":"cpcSW"}), ["cpcSW"]

def add_ext(r, tab, cols, acc_cols=()):
    r=r.join(tab.rename({c:c+"_t" for c in cols}),on=["lat","lon","time"],how="left")
    r=r.join(tab.rename({"time":"t_known",**{c:c+"_k" for c in cols}}),on=["lat","lon","t_known"],how="left")
    for c in cols: r=r.with_columns((pl.col(c+"_t")-pl.col(c+"_k")).alias(c+"_d"))
    feats=[c+s for c in cols for s in ("_t","_k","_d")]
    if acc_cols:
        w=tab.select(["lat","lon","time"]+list(acc_cols)).rename({"time":"tw"})
        j=r.select(["lat","lon","time","t_known"]).unique().join(w,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
        acc=j.group_by(["lat","lon","time","t_known"]).agg([pl.col(c).sum().alias(c+"_acc") for c in acc_cols])
        r=r.join(acc,on=["lat","lon","time","t_known"],how="left"); feats+=[c+"_acc" for c in acc_cols]
    return r, feats

WKEYS=["anom_known","dev24","d2","SPEI_06_t_d","SPEI_12_t_d","spei1_acc","SOIL_MOISTURE_t_d","trend24","ranom_known"]
def add_wide4(r, radius=4):
    from scipy.ndimage import uniform_filter
    pairs=r.select(["time","t_known"]).unique().sort(["time","t_known"]).with_row_index("pid")
    rr=r.join(pairs,on=["time","t_known"],how="left")
    pid=rr["pid"].to_numpy(); li=(rr["lat"].to_numpy()+89.5).round().astype(int); lo=(rr["lon"].to_numpy()+179.5).round().astype(int)
    n=len(pairs); size=(1,2*radius+1,2*radius+1); k2=size[1]*size[2]
    cnt=np.zeros((n,180,360),np.float32); cnt[pid,li,lo]=1.0
    C=uniform_filter(cnt,size=size,mode=("constant","constant","wrap"))*k2-cnt
    out={}
    for k in WKEYS:
        v=rr[k].to_numpy().astype(np.float32); ok=~np.isnan(v)
        arr=np.zeros((n,180,360),np.float32); arr[pid[ok],li[ok],lo[ok]]=v[ok]
        c=np.zeros((n,180,360),np.float32); c[pid[ok],li[ok],lo[ok]]=1.0
        S=uniform_filter(arr,size=size,mode=("constant","constant","wrap"))*k2-arr
        Cc=uniform_filter(c,size=size,mode=("constant","constant","wrap"))*k2-c
        m=np.where(Cc>0.5,S/np.maximum(Cc,1e-6),np.nan)
        out["w4_"+k]=m[pid,li,lo]
    out["w4_n"]=C[pid,li,lo]
    return r.with_columns([pl.Series(k,v) for k,v in out.items()])
WIDE4=["w4_"+k for k in WKEYS]+["w4_n"]

ANWKEYS = ["an_e5PERz_acc", "an_e5Pz_acc", "an_e5Ez_acc", "an_e5PERz_w3", "an_e5PERz_w6",
           "an_e5SWz_d", "an_e5MTWSz_d", "an_PERz_acc", "an_SWz_d", "an_MTWSz_d",
           "an_SOIL_MOISTURE_tz_d"]

def add_anwide_multi(r, radii=(2, 4, 8), keys=None):
    F = []
    for rad in radii:
        r, f = add_anwide(r, radius=rad, keys=keys, suffix=f"_r{rad}")
        F += f
    return r, F

def add_anwide(r, radius=4, keys=None, suffix=""):
    from scipy.ndimage import uniform_filter
    keys = [k for k in (keys or ANWKEYS) if k in r.columns]
    if not keys:
        return r, []
    pairs=r.select(["time","t_known"]).unique().sort(["time","t_known"]).with_row_index("pid")
    rr=r.join(pairs,on=["time","t_known"],how="left")
    pid=rr["pid"].to_numpy(); li=(rr["lat"].to_numpy()+89.5).round().astype(int); lo=(rr["lon"].to_numpy()+179.5).round().astype(int)
    n=len(pairs); size=(1,2*radius+1,2*radius+1); k2=size[1]*size[2]
    out={}
    for k in keys:
        v=rr[k].to_numpy().astype(np.float32); ok=~np.isnan(v)
        arr=np.zeros((n,180,360),np.float32); arr[pid[ok],li[ok],lo[ok]]=v[ok]
        c=np.zeros((n,180,360),np.float32); c[pid[ok],li[ok],lo[ok]]=1.0
        S=uniform_filter(arr,size=size,mode=("constant","constant","wrap"))*k2
        Cc=uniform_filter(c,size=size,mode=("constant","constant","wrap"))*k2
        m=np.where(Cc>0.5,S/np.maximum(Cc,1e-6),np.nan)
        out["aw_"+k+suffix]=m[pid,li,lo]
    return r.with_columns([pl.Series(k,v) for k,v in out.items()]), list(out)

def add_covwin(r, cov_all):
    win=cov_all.select(["lat","lon","time","SPEI_01_t","SPEI_03_t","SPEI_06_t","SOIL_MOISTURE_t"]).rename({"time":"tw"})
    j=r.select(["lat","lon","time","t_known"]).unique().join(win,on=["lat","lon"],how="inner").filter((pl.col("tw")>pl.col("t_known"))&(pl.col("tw")<=pl.col("time")))
    acc=j.group_by(["lat","lon","time","t_known"]).agg(pl.col("SOIL_MOISTURE_t").max().alias("sm_wmax"),pl.col("SOIL_MOISTURE_t").min().alias("sm_wmin"),
        pl.col("SPEI_01_t").min().alias("spei1_wmin"),pl.col("SPEI_01_t").max().alias("spei1_wmax"),pl.col("SPEI_06_t").mean().alias("spei6_win"))
    r=r.join(acc,on=["lat","lon","time","t_known"],how="left")
    for L in (1,2,3):
        r=r.with_columns(pl.col("time").dt.offset_by(f"-{L}mo").alias("tl"))
        r=r.join(cov_all.select(["lat","lon","time","SPEI_01_t","SOIL_MOISTURE_t"]).rename({"time":"tl","SPEI_01_t":f"spei1_l{L}","SOIL_MOISTURE_t":f"sm_l{L}"}),on=["lat","lon","tl"],how="left").drop("tl")
    return r
COVWIN=["sm_wmax","sm_wmin","spei1_wmin","spei1_wmax","spei6_win"]+[f"spei1_l{L}" for L in (1,2,3)]+[f"sm_l{L}" for L in (1,2,3)]

def cell_response(hist):
    h=hist.sort(["lat","lon","time"]).with_columns((pl.col("target")-pl.col("TWS_t")).alias("dy"),
        (pl.col("SOIL_MOISTURE_t")-pl.col("SOIL_MOISTURE_t").shift(1).over(["lat","lon"])).alias("dsm"),
        pl.col("time").shift(1).over(["lat","lon"]).alias("tp"))
    h=h.with_columns(pl.when(mdiff("time","tp")==1).then(pl.col("dsm")).otherwise(None).alias("dsm"))
    g=h.group_by(["lat","lon"]).agg((pl.cov("dy","SPEI_01_t")/pl.col("SPEI_01_t").var()).alias("b_spei1"),
                                    (pl.cov("dy","dsm")/pl.col("dsm").var()).alias("b_dsm"),
                                    pl.corr("dy","SPEI_01_t").alias("rr_spei1"),pl.corr("dy","dsm").alias("rr_dsm"),pl.col("dy").std().alias("dy_sd"))
    return g
def add_response(r, resp):
    r=r.join(resp,on=["lat","lon"],how="left")
    return r.with_columns((pl.col("b_spei1")*pl.col("spei1_acc")).alias("resp_spei"),(pl.col("b_dsm")*pl.col("SOIL_MOISTURE_t_d")).alias("resp_sm"))
RESP=["b_spei1","b_dsm","rr_spei1","rr_dsm","dy_sd","resp_spei","resp_sm"]
