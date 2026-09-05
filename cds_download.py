"""Download ERA5 monthly means (single levels) on a .5-centred 1-degree grid, 2001-01..2019-06.
Water-balance and storage fields physically tied to TWS: tp, e, ro, sd, swvl1-4, t2m.
Needs ~/.cdsapirc (free CDS account). Public, operationally available (ERA5T ~5 days after month end)."""
import cdsapi, os, sys, time
os.makedirs("external/era5",exist_ok=True)
c=cdsapi.Client(quiet=True)
VARS=["total_precipitation","evaporation","runoff","snow_depth","volumetric_soil_water_layer_1",
      "volumetric_soil_water_layer_2","volumetric_soil_water_layer_3","volumetric_soil_water_layer_4","2m_temperature"]
years=[int(a) for a in sys.argv[1:]] or list(range(2001,2020))
for year in years:
    out=f"external/era5/era5_monthly_1deg_{year}.nc"
    if os.path.exists(out) and os.path.getsize(out)>1e5: print("have",out,flush=True); continue
    t0=time.time()
    c.retrieve("reanalysis-era5-single-levels-monthly-means",
        {"product_type":["monthly_averaged_reanalysis"],"variable":VARS,"year":[str(year)],
         "month":[f"{m:02d}" for m in range(1,13)],"time":["00:00"],"grid":[1.0,1.0],"area":[89.5,-179.5,-89.5,179.5],
         "data_format":"netcdf","download_format":"unarchived"}, out)
    print(f"downloaded {out} {os.path.getsize(out)/1e6:.1f} MB ({time.time()-t0:.0f}s)",flush=True)
print("all done",flush=True)
