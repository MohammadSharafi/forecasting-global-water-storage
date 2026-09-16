import cdsapi, os, sys, time
import polars as pl
DEST=os.environ.get("ERA5_DIR","external/era5")
os.makedirs(DEST,exist_ok=True)
_c=pl.read_csv("Train.csv",columns=["lat","lon"])
N=min(90.0,_c["lat"].max()+5); S=max(-90.0,_c["lat"].min()-5)
W=max(-180.0,_c["lon"].min()-5); E=min(180.0,_c["lon"].max()+5)
print(f"area N{N} W{W} S{S} E{E}",flush=True)
c=cdsapi.Client(quiet=True)
VARS=["total_precipitation","evaporation","runoff","snow_depth","volumetric_soil_water_layer_1",
      "volumetric_soil_water_layer_2","volumetric_soil_water_layer_3","volumetric_soil_water_layer_4","2m_temperature"]
years=[int(a) for a in sys.argv[1:]] or list(range(2001,2020))
for year in years:
    out=f"{DEST}/era5_monthly_1deg_{year}.nc"
    if os.path.exists(out) and os.path.getsize(out)>1e5: print("have",out,flush=True); continue
    t0=time.time()
    c.retrieve("reanalysis-era5-single-levels-monthly-means",
        {"product_type":["monthly_averaged_reanalysis"],"variable":VARS,"year":[str(year)],
         "month":[f"{m:02d}" for m in range(1,13)],"time":["00:00"],"grid":[1.0,1.0],"area":[N,W,S,E],
         "data_format":"netcdf","download_format":"unarchived"}, out)
    print(f"downloaded {out} {os.path.getsize(out)/1e6:.1f} MB ({time.time()-t0:.0f}s)",flush=True)
print("all done",flush=True)
