# Forecasting Global Water Storage — ITU / Zindi

Code for the "A Step Ahead of Drought: Forecasting Global Water Storage" challenge.
The task is to predict the GRACE terrestrial water storage anomaly for a 1-degree land
cell one month ahead, scored by RMSE over 280,961 test rows.

Final placing: **20th of 1,374 entrants** on the private leaderboard.

## Approach

An ensemble of four model classes over ~370 engineered features, blended with a
post-hoc anchor correction.

| component | notes |
|---|---|
| LightGBM | 255 leaves, 900 rounds, 6 seeds |
| XGBoost | depth 9, 4 seeds |
| MLP | three shapes (256 / 512 / 128 units), PyTorch |
| GRU | per-cell recurrent model over 6/18/36-month histories |
| AR anchor rescaling | per-cell, per-horizon, shrunk toward a pooled slope |
| spatial smoothing | Gaussian over neighbouring cells, residual only |

The models are trained on a residual target measured against the last observed storage
value, so every arm predicts a change rather than a level.

The single largest ensemble gain came from the GRU. It is the weakest model in the stack
on its own, and the most valuable addition, because it correlates ~0.78 with the tree
blend where the tree families correlate ~0.997 with each other. Model diversity, not model
accuracy, is what the blend pays for. CatBoost was tested for the same reason and rejected:
a third tree family still lands at 0.99 and adds nothing.

## Layout

    build_*.py, features*.py   feature construction
    run_models.py              trains one model family on cached matrices
    gru_model.py               the recurrent model
    ar_model.py, ar_blend.py   the per-cell anchor correction
    final_assemble.py          combines families into a submission
    *_probe.py, *_test.py      measurement scripts for individual hypotheses
    submissions/               the two files selected for private scoring

## Data

Competition data is not redistributed here; download `Train.csv` and `Test.csv` from the
competition page. External inputs used, all with a source date at or before the prediction
month: NCEP-R2 and NCEP/NCAR reanalysis, ERA5-Land, CPC soil moisture, GPCC gauge
precipitation, WaterGAP 2.2e modelled storage, GDO drought indices. No GRACE-derived
product is used as an input, and no land-surface model that assimilates GRACE.

## Running

    pip install -r requirements.txt
    python build_mats.py <LAYOUT>
    python run_models.py <LAYOUT> <lgb|lgbd|xgb|cat|mlp> v5x_noll_sa [rounds]
    python final_assemble.py <name> lgbd_v5x_noll:0.65 xgb_v5x_noll:0.35
