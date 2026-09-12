# Last-day investigation

## What the leaderboard actually shows
Read from the entrant's screenshot, 12 Sep:

    1  MOHAR           0.55958798    39 subs
    2  Shankar         0.589305525   25
    3  lode4           0.623134877  127
    4  OverfitStorage  0.62338361   157
    5  Jisoo           0.631662555   61
    6  Gliding Moran   0.647079224    8
    7  Emo HedgeHog    0.647353577   74
    8  Ahsan_496       0.649536249   92
    9  MosCraciunXXX   0.651862897  121
    10 awxlong         0.653388234    5
    11 Ramjas          0.655934963    8

The decisive detail is the submission counts, not the scores. awxlong reached 0.6534 in FIVE
submissions and Ramjas 0.6559 in eight. Five submissions cannot be leaderboard-fitted. So a
straightforward method lands at 0.65, while this project sits at 0.6792 after roughly thirty.

That refutes the hypothesis this project carried for several sessions -- that sub-0.65 implied an
external gap-free TWS product. It does not. The honest reading is that the approach here is
structurally behind by about 3.8% relative to persistence (0.766 of the board's 0.8864 persistence
baseline against 0.738 for rank 10), and the earlier "information ceiling" claims were measurements
of THIS pipeline mistaken for measurements of the problem.

## What the competition rules actually permit
From the challenge page (zindi.world, fetched 12 Sep):
  * metric RMSE, leaderboard is 50% of the final score; Trustworthiness 30%, Innovation 20%
  * public leaderboard is ~30% of the test rows, private is the other 70%
  * allowed: challenge data, publicly available open-source packages, freely and operationally
    available datasets such as satellite data, Copernicus resources, openly available pretrained
    models
  * prohibited: AutoML tools, and leakage of future GRACE/TWS information
So GPCC (rain gauges) and WaterGAP (a forward hydrological model, no GRACE assimilation) are both
squarely allowed. There is no rule that closes off what the leaders are doing.

## What the literature does for this exact problem
Searches on GRACE TWS anomaly forecasting return a consistent picture: LSTM and BiLSTM recurrent
networks are the standard tool for TWSA reconstruction and short-horizon prediction, usually at
0.5 degrees, often combined with EOF decomposition, and compared against XGBoost, ELM and NARX.
The published framing is a per-cell SEQUENCE problem over the cell's own history plus climatic
drivers.

This project never built that. It built a per-row gradient-boosted model with engineered features,
and tried a U-Net over global maps once (rejected: 191 training images, data-starved). A per-cell
sequence model is a different thing from both: 15,715 cells x ~160 months is ~2.5M cell-months of
training signal, which is not data-starved at all. That is the most plausible candidate for what a
five-submission entrant did, and it is the largest untested structural idea remaining.

It cannot be built and validated in the hours left on the last day, so it is recorded here as the
first thing to try with more time rather than attempted and half-finished now.

## What is being done instead, with the time available
The two levers that are measurable today, both of which add information to the model rather than
fitting the board:
  * GPCC gauge precipitation: measured -0.0020 on five layouts, already adopted and shipped.
  * WaterGAP 2.2e modelled storage, on top of GPCC: -0.0025 (Cvn2), -0.0052 (D), -0.0042 (E) so far.
  * an ablation of this pipeline's accumulated training structure (horizon-mix reweighting, recency
    ramp), on the suspicion that structure validated on our layouts is costing score on the real
    test era -- the pattern the five-submission entrants suggest.
