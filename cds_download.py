"""Download ERA5 monthly means (single levels) regridded to 1 degree, 2002-01..2019-01.
Water-balance and storage fields that are physically tied to Total Water Storage:
  tp  total precipitation        e   evaporation           ro  runoff
  sd  snow depth (water equiv.)  swvl1..4 volumetric soil water (0-7,7-28,28-100,100-289 cm)
  t2m 2m temperature
Needs a free CDS account: https://cds.climate.copernicus.eu  -> profile -> API token,
saved as ~/.cdsapirc (two lines: url + key). Public, operationally available (ERA5T ~5 days).
"""
import cdsapi, os
os.makedirs("external/era5",exist_ok=True)
c=cdsapi.Client()
VARS=["total_precipitation","evaporation","runoff","snow_depth","volumetric_soil_water_layer_1",
      "volumetric_soil_water_layer_2","volumetric_soil_water_layer_3","volumetric_soil_water_layer_4","2m_temperature"]
for year in range(2002,2020):
    out=f"external/era5/era5_monthly_1deg_{year}.nc"
    if os.path.exists(out) and os.path.getsize(out)>1e5: print("have",out); continue
    c.retrieve("reanalysis-era5-single-levels-monthly-means",
        {"product_type":"monthly_averaged_reanalysis","variable":VARS,"year":str(year),
         "month":[f"{m:02d}" for m in range(1,13)],"time":"00:00","grid":[1.0,1.0],"area":[89.5,-179.5,-89.5,179.5],"format":"netcdf"}  # .5-centred 1-deg grid = challenge cells,out)
    print("downloaded",out,flush=True)
