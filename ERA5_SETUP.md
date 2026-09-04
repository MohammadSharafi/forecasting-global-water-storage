# Getting the ERA5 covariates (5 minutes of your time, then one command)

1. Register (free): https://cds.climate.copernicus.eu  → log in → click your name → "API token".
2. Accept the ERA5 licence once: open
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download
   and tick "accept terms" at the bottom.
3. Create the file `~/.cdsapirc` with exactly two lines (paste your own token):

       url: https://cds.climate.copernicus.eu/api
       key: <your-token>

4. Run, from the project folder:

       ./.venv/bin/python cds_download.py

   18 small files (~1° global monthly, ~1 MB each). CDS queues requests; usually minutes,
   sometimes longer. It resumes if re-run.

Why: precipitation − evaporation − runoff accumulated over the months where TWS is hidden
IS the change in stored water; total-column soil water and snow water equivalent are the
two largest TWS components. SPEI and near-surface soil moisture are weak proxies for both.
ERA5T is released ~5 days after each month, so this is operationally available and
compliant with the rules; it will be documented in the report.
