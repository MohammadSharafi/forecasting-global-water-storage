# Overnight run report

Generated 2026-09-08 07:39.
173 steps completed, 0 did not.

## Submission files

- `out/sub_q_alt.csv` (8.2 MB)
- `out/sub_q_base.csv` (8.2 MB)
- `out/sub_q_main.csv` (8.2 MB)
- `out/sub_q_main_nocal.csv` (8.2 MB)
- `out/sub_q_main_nosm.csv` (8.2 MB)

## Configuration chosen

```sh
FINAL_DROPF='scale,bigsa'
FINAL_MODEL=lgb
FINAL_WEIGHTS=ramp
FINAL_HMIX='test'
FINAL_WLGB=0.50
FINAL_STACK=lgb_v5x_noll:0.500,xgb_v5x_noll:0.500,lgbs_v5x_noll:0.000,lgbm_v5x_noll:0.000,cat_v5x_noll:0.000
FINAL_SMOOTH_W1=0.7
FINAL_SMOOTH_W7=0.7
FINAL_SMOOTH_R=1
FINAL_SMOOTH_IT=1
FINAL_SMOOTH_WRAP=0
FINAL_H1BETA=0.00
FINAL_CALIB=1.1300,1.1050,1.0293,0.9719,0.9875,0.9817,0.9998
FINAL_CALIB_B=
```

## Why that configuration: the experiment grid

```

  experiment                      testmix A   testmix B   testmix C   
  e1 base (pre-session-9)         0.6558     0.5577     0.7900
  e2 +covariate anomalies         0.6402     0.5376        -  
  e3 +zonal scale                 0.6551     0.5568        -  
  e4 +anomalies +zonal            0.6397     0.5377     0.7853
  e8 +1500/2500 km anchors        0.6397     0.5376     0.7779
  e5 all, 63 leaves               0.6402     0.5373        -  
  e6 all, 31 leaves               0.6407     0.5384        -  
  e7 all, uniform weights         0.6404     0.5402        -  
  e9 all, test horizon mix        0.6389     0.5344        -  

  features: e2 (+covariate anomalies) -> DROPF='scale,bigsa'
  capacity: lgb
  weights : ramp
  hmix    : test
  (each decision requires a win on ALL of A+B+C; incumbent kept otherwise)
```

## Grid scored under the real test horizon mix -- layout A

```
layout A: 281003 rows
  horizon share   h1  h2  h3  h4  h5  h6  h7
    this layout   0.278  0.277  0.221  0.055  0.055  0.056  0.055
    real test     0.333  0.222  0.167  0.111  0.056  0.056  0.056
    difference    -0.056  +0.054  +0.054  -0.056  -0.000  -0.000  -0.000

  set                       plain  testmix   h1      h2      h3      h4      h5      h6      h7    
  persistence              0.7569   0.7621   0.7078  0.6796  0.7383  0.8561  0.8832  1.0898  0.7302
  e1_base                  0.6447   0.6558   0.6415  0.5550  0.5717  0.7438  0.8198  1.0039  0.5221  (+0.0000)
  e2_anom                  0.6283   0.6402   0.6250  0.5371  0.5511  0.7395  0.8002  0.9889  0.5064  (-0.0155)
  e3_zonal                 0.6442   0.6551   0.6406  0.5548  0.5710  0.7410  0.8201  1.0051  0.5213  (-0.0007)
  e4_anom_zonal            0.6277   0.6397   0.6233  0.5369  0.5534  0.7447  0.7943  0.9866  0.5008  (-0.0160)
  e8_bigsa                 0.6277   0.6397   0.6234  0.5367  0.5516  0.7422  0.7981  0.9882  0.5030  (-0.0161)
  e5_lgbm63                0.6285   0.6402   0.6244  0.5391  0.5516  0.7387  0.8003  0.9869  0.5071  (-0.0156)
  e6_lgbs31                0.6291   0.6407   0.6250  0.5398  0.5525  0.7400  0.7998  0.9862  0.5078  (-0.0150)
  e7_uniform               0.6287   0.6404   0.6205  0.5395  0.5541  0.7442  0.7979  0.9876  0.5167  (-0.0153)
  e9_hmix                  0.6272   0.6389   0.6245  0.5360  0.5502  0.7361  0.7984  0.9873  0.5047  (-0.0169)

  per-horizon difference vs the first set (negative = better):
  e2_anom                -0.0164  -0.0179  -0.0206  -0.0043  -0.0195  -0.0150  -0.0158
  e3_zonal               -0.0009  -0.0002  -0.0007  -0.0027  +0.0004  +0.0012  -0.0009
  e4_anom_zonal          -0.0182  -0.0181  -0.0183  +0.0009  -0.0255  -0.0174  -0.0213
  e8_bigsa               -0.0181  -0.0184  -0.0201  -0.0016  -0.0217  -0.0157  -0.0191
  e5_lgbm63              -0.0170  -0.0160  -0.0201  -0.0050  -0.0195  -0.0170  -0.0150
  e6_lgbs31              -0.0165  -0.0153  -0.0193  -0.0037  -0.0200  -0.0177  -0.0143
  e7_uniform             -0.0209  -0.0156  -0.0176  +0.0005  -0.0218  -0.0163  -0.0055
  e9_hmix                -0.0170  -0.0190  -0.0215  -0.0077  -0.0214  -0.0166  -0.0174
```

## Grid scored under the real test horizon mix -- layout B

```
layout B: 312102 rows
  horizon share   h1  h2  h3  h4  h5  h6  h7
    this layout   0.300  0.249  0.199  0.099  0.050  0.050  0.050
    real test     0.333  0.222  0.167  0.111  0.056  0.056  0.056
    difference    -0.033  +0.027  +0.032  -0.012  -0.006  -0.006  -0.006

  set                       plain  testmix   h1      h2      h3      h4      h5      h6      h7    
  persistence              0.6869   0.6828   0.5693  0.6786  0.7245  0.7018  0.8231  0.8303  0.8240
  e1_base                  0.5617   0.5577   0.5175  0.5710  0.5865  0.5509  0.5832  0.6192  0.5693  (+0.0000)
  e2_anom                  0.5415   0.5376   0.4990  0.5511  0.5576  0.5382  0.5711  0.5985  0.5453  (-0.0200)
  e3_zonal                 0.5608   0.5568   0.5173  0.5705  0.5843  0.5520  0.5752  0.6197  0.5697  (-0.0009)
  e4_anom_zonal            0.5416   0.5377   0.4990  0.5515  0.5574  0.5381  0.5696  0.6002  0.5458  (-0.0199)
  e8_bigsa                 0.5415   0.5376   0.5002  0.5518  0.5571  0.5365  0.5703  0.5963  0.5436  (-0.0201)
  e5_lgbm63                0.5412   0.5373   0.5007  0.5510  0.5566  0.5343  0.5726  0.5969  0.5413  (-0.0204)
  e6_lgbs31                0.5423   0.5384   0.5004  0.5516  0.5579  0.5368  0.5743  0.6005  0.5448  (-0.0193)
  e7_uniform               0.5442   0.5402   0.5011  0.5546  0.5611  0.5420  0.5694  0.5991  0.5483  (-0.0174)
  e9_hmix                  0.5383   0.5344   0.4970  0.5481  0.5533  0.5358  0.5592  0.5987  0.5421  (-0.0232)

  per-horizon difference vs the first set (negative = better):
  e2_anom                -0.0185  -0.0199  -0.0289  -0.0127  -0.0122  -0.0207  -0.0240
  e3_zonal               -0.0002  -0.0005  -0.0022  +0.0010  -0.0081  +0.0005  +0.0004
  e4_anom_zonal          -0.0186  -0.0194  -0.0290  -0.0128  -0.0137  -0.0190  -0.0235
  e8_bigsa               -0.0173  -0.0192  -0.0293  -0.0144  -0.0129  -0.0229  -0.0256
  e5_lgbm63              -0.0168  -0.0200  -0.0299  -0.0166  -0.0107  -0.0223  -0.0279
  e6_lgbs31              -0.0171  -0.0194  -0.0285  -0.0142  -0.0089  -0.0187  -0.0245
  e7_uniform             -0.0164  -0.0164  -0.0254  -0.0089  -0.0138  -0.0201  -0.0210
  e9_hmix                -0.0205  -0.0228  -0.0332  -0.0151  -0.0240  -0.0205  -0.0272
```

## Grid scored under the real test horizon mix -- layout C

```
layout C: 140491 rows
  horizon share   h1  h2  h3  h4  h5  h6  h7
    this layout   0.445  0.110  0.000  0.111  0.000  0.000  0.000
    real test     0.333  0.222  0.167  0.111  0.056  0.056  0.056
    difference    +0.111  -0.112  -0.167  -0.000  -0.056  -0.055  -0.056

  set                       plain  testmix   h1      h2      h3      h4      h5      h6      h7    
  persistence              0.8772   1.0947   0.7743  1.1186  nan  0.6311  nan  2.4860  nan
  e1_base                  0.6536   0.7900   0.6379  0.7982  nan  0.5022  nan  1.6184  nan  (+0.0000)
  e4_anom_zonal            0.6302   0.7853   0.6189  0.7736  nan  0.4746  nan  1.6950  nan  (-0.0047)
  e8_bigsa                 0.6268   0.7779   0.6150  0.7723  nan  0.4669  nan  1.6657  nan  (-0.0121)

  per-horizon difference vs the first set (negative = better):
  e4_anom_zonal          -0.0189  -0.0245  +nan  -0.0276  +nan  +0.0766  +nan
  e8_bigsa               -0.0228  -0.0258  +nan  -0.0352  +nan  +0.0473  +nan
```

## Ensemble weight (lgb vs xgb)

```

  w_lgb   A   B   C
   0.00   0.6387   0.5362   0.7841
   0.05   0.6386   0.5360   0.7838
   0.10   0.6384   0.5359   0.7835
   0.15   0.6383   0.5357   0.7832
   0.20   0.6382   0.5356   0.7830
   0.25   0.6381   0.5355   0.7828
   0.30   0.6380   0.5354   0.7826
   0.35   0.6379   0.5353   0.7824
   0.40   0.6379   0.5353   0.7822
   0.45   0.6378   0.5352   0.7821
   0.50   0.6378   0.5352   0.7820
   0.55   0.6378   0.5352   0.7819
   0.60   0.6378   0.5351   0.7818
   0.65   0.6378   0.5351   0.7817
   0.70   0.6379   0.5352   0.7817
   0.75   0.6379   0.5352   0.7817
   0.80   0.6380   0.5352   0.7817
   0.85   0.6380   0.5353   0.7818
   0.90   0.6381   0.5354   0.7818
   0.95   0.6382   0.5355   0.7819
   1.00   0.6383   0.5356   0.7820

  best per layout: {'A': 0.55, 'B': 0.6, 'C': 0.75}
  leave-one-layout-out choice:
  chosen on B+C -> 0.7; on held-out A: +0.00003
  chosen on A+C -> 0.7; on held-out B: -0.00001
  chosen on A+B -> 0.6; on held-out C: -0.00017
  best overall 0.65: on each layout ['+0.00000', '-0.00003', '-0.00022']
  does not clear 0.0002 out of sample everywhere -- keeping the incumbent 0.5
```

## Ensemble weights over every family

```
  families on every layout: ['lgb_v5x_noll', 'xgb_v5x_noll', 'lgbs_v5x_noll', 'lgbm_v5x_noll', 'cat_v5x_noll']

                                               weights   A  B  C
  incumbent lgb:0.50 xgb:0.50 lgbs:0.00 lgbm:0.00 cat:0.00   0.6378  0.5352  0.7820
  fitted    lgb:0.38 xgb:0.47 lgbs:0.00 lgbm:0.00 cat:0.14   0.6378  0.5366  0.7784
  fitted on B+C, scored on A: +0.00000
  fitted on A+C, scored on B: +0.00232
  fitted on A+B, scored on C: +0.00063
  held-out gain does not clear 0.0003 everywhere -- keeping the incumbent blend
```

## Spatial smoothing

```
... (1 earlier lines omitted)
  layout A: incumbent (0.7, r1, it1, no wrap) = 0.6347
     w1    w7  r  it  wrap   testmix     vs incumbent
   0.50  0.50  1   2 False   0.6346     -0.00007
   0.60  0.60  1   2 False   0.6347     -0.00002
   0.70  0.80  1   1 False   0.6347     -0.00002
   0.70  0.90  1   1 False   0.6347     -0.00001
   0.70  0.80  1   1  True   0.6347     -0.00001
   0.70  0.90  1   1  True   0.6347     -0.00001
   0.80  0.80  1   1 False   0.6347     -0.00000
   0.70  0.70  1   1 False   0.6347     +0.00000

  layout B: incumbent (0.7, r1, it1, no wrap) = 0.5310
     w1    w7  r  it  wrap   testmix     vs incumbent
   0.60  0.60  1   2 False   0.5308     -0.00023
   0.50  0.50  1   2 False   0.5308     -0.00020
   0.70  0.70  1   2 False   0.5309     -0.00015
   0.80  0.80  1   1 False   0.5310     -0.00006
   0.80  0.80  1   1  True   0.5310     -0.00006
   0.80  0.90  1   1 False   0.5310     -0.00004
   0.80  0.90  1   1  True   0.5310     -0.00004
   0.80  0.70  1   1 False   0.5310     -0.00004

  layout C: incumbent (0.7, r1, it1, no wrap) = 0.8655
     w1    w7  r  it  wrap   testmix     vs incumbent
   0.00  0.00  1   1 False   0.7820     -0.08350
   0.00  0.00  1   1  True   0.7820     -0.08350
   0.00  0.00  2   1 False   0.7820     -0.08350
   0.00  0.00  2   1  True   0.7820     -0.08350
   0.00  0.00  1   2 False   0.7820     -0.08350
   0.00  0.00  2   2 False   0.7820     -0.08350
   0.20  0.00  1   1  True   0.7845     -0.08099
   0.20  0.00  1   1 False   0.7845     -0.08099

  leave-one-layout-out choice:
  chosen on B+C -> (0.0, 0.0, 2, 1, False); on held-out A: +0.00312
  chosen on A+C -> (0.0, 0.0, 2, 1, False); on held-out B: +0.00415
  chosen on A+B -> (0.5, 0.5, 1, 2, False); on held-out C: +0.00564
  best overall (0.0, 0.0, 2, 1, False): on each layout ['+0.00312', '+0.00415', '-0.08350']
  does not clear 0.0003 out of sample everywhere -- keeping the incumbent (0.7, 0.7, 1, 1, False)
  chosen: w1=0.7 w7=0.7 radius=1 iters=1 wrap=False
```

## Horizon-1 specialist

```

  layout A: RMSE at h=1 -- general 0.6220, specialist 0.6220
    beta   0.00   0.10   0.20   0.30   0.40   0.50   0.60   0.70   0.80   0.90   1.00
    mix  +0.0000 -0.0002 -0.0003 -0.0004 -0.0004 -0.0005 -0.0004 -0.0004 -0.0003 -0.0002 -0.0000

  layout B: RMSE at h=1 -- general 0.4966, specialist 0.4949
    beta   0.00   0.10   0.20   0.30   0.40   0.50   0.60   0.70   0.80   0.90   1.00
    mix  +0.0000 -0.0002 -0.0003 -0.0004 -0.0005 -0.0006 -0.0006 -0.0006 -0.0006 -0.0006 -0.0005

  layout C: RMSE at h=1 -- general 0.6193, specialist 0.6387
    beta   0.00   0.10   0.20   0.30   0.40   0.50   0.60   0.70   0.80   0.90   1.00
    mix  +0.0000 +0.0004 +0.0009 +0.0015 +0.0021 +0.0028 +0.0035 +0.0044 +0.0052 +0.0062 +0.0072

  leave-one-layout-out choice:
  chosen on B+C -> 0.0; on held-out A: +0.00000
  chosen on A+C -> 0.0; on held-out B: +0.00000
  chosen on A+B -> 0.6; on held-out C: +0.00354
  best overall 0.0: on each layout ['+0.00000', '+0.00000', '+0.00000']
  does not clear 0.0003 out of sample everywhere -- keeping the incumbent 0.0
  not adopted: beta=0.00
```

## Per-horizon calibration

```

  layout A: mean prediction - truth = +0.0160
    h1 h2 h3 h4 h5 h6 h7
    scale  (scale) 1.155 1.065 1.050 1.017 0.928 0.956 0.993
    scale  (affine) 1.177 1.066 1.054 0.998 0.919 0.950 0.991
    offset (affine) -0.049 -0.003 +0.041 -0.039 -0.014 -0.011 -0.006

  layout B: mean prediction - truth = +0.0582
    h1 h2 h3 h4 h5 h6 h7
    scale  (scale) 1.066 1.050 1.038 0.951 1.034 0.989 1.006
    scale  (affine) 1.073 1.052 1.036 0.950 1.037 0.984 0.996
    offset (affine) -0.029 -0.032 -0.034 -0.038 +0.073 -0.037 -0.071

  layout C: mean prediction - truth = -0.0794
    h1 h2 h3 h4 h5 h6 h7
    scale  (scale) 1.170 1.200 1.000 0.948 1.000 1.000 1.000
    scale  (affine) 1.169 1.200 1.000 0.943 1.000 1.000 1.000
    offset (affine) +0.005 +0.100 +0.000 +0.029 +0.000 +0.000 +0.000

  scale: fitted on the other layouts, scored held out: A -0.00150  B -0.00066  C -0.00391

  affine: fitted on the other layouts, scored held out: A -0.00137  B +0.00089  C -0.00041

  adopted the scale form
    scale  1.130 1.105 1.029 0.972 0.988 0.982 1.000
    on each layout: ['-0.00177', '-0.00084', '-0.00681']
```

## What is in the data

```

==============================================================================
COVERAGE
==============================================================================
  Train:  2154021 rows   15715 cells  138 months  2002-05-01 .. 2015-08-01
  Test:   280961 rows   15715 cells   18 months  2015-09-01 .. 2018-12-01
  Train spans 160 months and carries 138; 22 are absent (GRACE gaps)
    2002-06-01 x3, 2003-06-01 x2, 2011-01-01 x2, 2011-06-01 x2, 2012-05-01 x2, 2012-10-01 x2, 2013-03-01 x2, 2013-08-01 x3, 2014-02-01 x2, 2014-07-01 x2
  cells per month in train: min 15510, median 15615, max 15681

==============================================================================
TEST GEOMETRY -- read off Test.csv, not assumed
==============================================================================
  6 blocks: 2015-09-01 x1 | 2016-01-01 x3 | 2016-06-01 x4 | 2016-12-01 x7 | 2018-07-01 x1 | 2018-11-01 x2
  block lengths: [1, 3, 4, 7, 1, 2]
  months skipped between blocks: [3, 2, 2, 12, 3]
  months carrying an observed TWS: 18 -- ['2015-09-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-06-01', '2016-07-01', '2016-08-01', '2016-09-01', '2016-12-01', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01', '2018-07-01', '2018-11-01', '2018-12-01']
  are those exactly the block-first months? False

  eval_mix.TEST_BLOCKS = (1, 3, 4, 7, 1, 2)   file says (1, 3, 4, 7, 1, 2)   MATCH
  horizon mix implied by the file: h1 0.333  h2 0.222  h3 0.167  h4 0.111  h5 0.056  h6 0.056  h7 0.056
  horizon per test MONTH (each covers the whole globe): h1 x18

==============================================================================
REGIME -- did the distribution move between the eras?
==============================================================================
  era                                  n     mean      sd     p05     p95
  Train, all months              2154021   0.1170  0.9130  -1.400   1.547
  Train 2002-2007                 983447   0.2246  0.9910  -1.439   1.654
  Train 2008-2011                 686575   0.0886  0.8243  -1.318   1.375
  Train 2012-2015                 483999  -0.0614  0.8332  -1.434   1.349
  Test, observed months            94048  -0.0510  0.9573  -1.521   1.670
  A shift in the mean here is inherited by every prediction: the models are anchored on
  tws_known, so a global level change is absorbed, but a change in SPREAD is not.

  global mean TWS, last 18 train months (the run-up to the test era):
    2013-11-01  mean -0.0155  sd 0.7142
    2013-12-01  mean -0.0098  sd 0.6880
    2014-01-01  mean -0.0053  sd 0.7462
    2014-04-01  mean +0.0841  sd 0.7997
    2014-05-01  mean -0.0493  sd 0.7632
    2014-06-01  mean +0.0636  sd 0.7579
    2014-09-01  mean +0.0162  sd 0.7058
    2014-10-01  mean -0.1138  sd 0.7648
    2014-11-01  mean +0.0848  sd 0.7088
    2014-12-01  mean +0.0835  sd 0.7023
    2015-01-01  mean -0.1503  sd 0.8176
    2015-02-01  mean -0.1215  sd 0.9588
    2015-03-01  mean -0.1129  sd 1.1356
    2015-04-01  mean -0.1063  sd 0.7555
    2015-05-01  mean -0.1274  sd 0.8107
    2015-06-01  mean -0.1061  sd 0.8794
    2015-07-01  mean -0.3769  sd 1.0423
    2015-08-01  mean -0.0636  sd 0.8700
  sd of the global monthly mean over train: 0.1530   month-to-month change sd: 0.1106
  That second number bounds what a perfect global-offset correction could ever buy.

==============================================================================
SEASONALITY AND PERSISTENCE, by latitude band
==============================================================================
  band            seas amp  seasonal share  lag-1 corr    |d1|
  -60..-30           0.418           0.018       0.785   0.443
  -30..+0            0.427           0.020       0.815   0.405
  +0..+30            0.393           0.023       0.778   0.425
  +30..+60           0.367           0.018       0.823   0.388
  +60..+90           0.400           0.022       0.762   0.408
  seasonal share = fraction of a cell's variance explained by its own monthly
  climatology; |d1| = mean absolute one-month change, the bar for any h=1 model.

==============================================================================
COVARIATE COMPLETENESS
==============================================================================
  Train null rate: SPEI_01_t 0.0%  SPEI_03_t 0.0%  SPEI_06_t 0.0%  SPEI_12_t 0.0%  SOIL_MOISTURE_t 0.0%
  Test null rate: SPEI_01_t 0.0%  SPEI_03_t 0.0%  SPEI_06_t 0.0%  SPEI_12_t 0.0%  SOIL_MOISTURE_t 0.0%
```

## Where the error is -- layout A

```
layout A, predictions lgb_v5x_noll:_bw,xgb_v5x_noll:_bw: 281003 rows
  RMSE 0.6260   persistence 0.7569   gain -0.1309   bias +0.0159

  HORIZON  (share is of the test-weighted MSE: what a fix at that horizon is worth)
    h        n     RMSE     bias  persist     gain  test wt  wtd share
    1    77979   0.6220  +0.0917   0.7078  -0.0858    0.333      31.7%
    2    77721   0.5345  +0.0044   0.6796  -0.1451    0.222      15.6%
    3    61974   0.5491  -0.0804   0.7383  -0.1892    0.167      12.4%
    4    15501   0.7379  +0.0778   0.8561  -0.1182    0.111      14.9%
    5    15546   0.7957  +0.0194   0.8832  -0.0875    0.056       8.6%
    6    15608   0.9895  +0.0156   1.0898  -0.1004    0.056      13.4%
    7    15535   0.5030  +0.0107   0.7302  -0.2273    0.056       3.5%

  VS PERSISTENCE
    h=1: model is worse than the last observation on  44.5% of rows
    h=2: model is worse than the last observation on  39.5% of rows
    h=3: model is worse than the last observation on  38.0% of rows
    h=4: model is worse than the last observation on  43.0% of rows
    h=5: model is worse than the last observation on  44.9% of rows
    h=6: model is worse than the last observation on  43.4% of rows
    h=7: model is worse than the last observation on  38.7% of rows

        LAT BAND        n     RMSE     bias  persist     gain  MSE share
         +0..+30    55522   0.6870  -0.0722   0.7826  -0.0956      23.8%
        +30..+60   100228   0.5849  +0.0083   0.7213  -0.1364      31.1%
        +60..+90    69967   0.5990  +0.0564   0.7384  -0.1393      22.8%
         -30..+0    45051   0.6680  +0.1106   0.8328  -0.1647      18.3%
        -60..-30    10235   0.6564  -0.1268   0.7299  -0.0735       4.0%

  CELL SD DECILE        n     RMSE     bias  persist     gain  MSE share
               0    28114   0.2305  +0.0428   0.2301  +0.0004       1.4%
               1    28104   0.3778  +0.0347   0.4494  -0.0716       3.6%
               2    28100   0.5019  +0.0210   0.6077  -0.1059       6.4%
               3    28091   0.5627  +0.0303   0.6780  -0.1154       8.1%
               4    28104   0.6056  +0.0181   0.7250  -0.1194       9.4%
               5    28106   0.6435  +0.0177   0.7860  -0.1425      10.6%
               6    28094   0.6907  +0.0244   0.8381  -0.1475      12.2%
               7    28093   0.7286  +0.0305   0.8921  -0.1636      13.5%
               8    28109   0.7832  +0.0027   0.9651  -0.1818      15.7%
               9    28088   0.8675  -0.0634   1.0357  -0.1682      19.2%
    (decile 0 = the steadiest cells, 9 = the most variable)

    TARGET MONTH        n     RMSE     bias  persist     gain  MSE share
               1    46949   0.6161  -0.0081   0.7198  -0.1037      16.2%
               2    31335   0.7947  +0.0600   0.8918  -0.0971      18.0%
               3    15670   0.5070  +0.0139   0.7312  -0.2242       3.7%
               6    15581   1.0678  +0.2912   1.1816  -0.1138      16.1%
               7    31178   0.5692  -0.0263   0.7290  -0.1598       9.2%
               8    31067   0.5692  -0.0041   0.7708  -0.2016       9.1%
               9    31099   0.4700  +0.0585   0.6421  -0.1720       6.2%
              10    15603   0.4906  -0.1262   0.5582  -0.0675       3.4%
              11    31248   0.4722  -0.0774   0.5723  -0.1001       6.3%
              12    31273   0.6434  +0.0549   0.7727  -0.1293      11.8%

  SPATIAL STRUCTURE OF THE RESIDUAL (correlation with the neighbour at grid lag d)
    lag 1 cells east: corr +0.967   (n=265007)
    lag 2 cells east: corr +0.899   (n=253617)
    lag 3 cells east: corr +0.809   (n=244374)
    lag 5 cells east: corr +0.621   (n=230030)
    lag 8 cells east: corr +0.447   (n=212526)
    Residual correlation still present at lag 2-3 means spatial smoothing has more to
    take; correlation gone by lag 1 means it is finished.

  WHERE THE MSE IS  (band x horizon, % of total)
      band         h1   h2   h3   h4   h5   h6   h7
    -60..-30    0.7  0.9  0.7  0.2  0.4  0.9  0.2
     -30..+0    5.1  3.4  2.7  1.6  2.2  2.8  0.6
     +0..+30    5.4  4.5  3.5  2.3  1.6  4.4  0.9
    +30..+60    8.1  6.3  5.3  2.2  3.6  4.5  1.1
    +60..+90    8.2  5.1  4.8  1.4  1.2  1.4  0.7

  20 worst cells carry 0.67% of total MSE:
    (22,16), (22,58), (44,96), (22,18), (42,98), (42,96), (36,82), (20,58), (34,82), (68,14), (20,16), (44,96), (34,82), (22,24), (34,104), (22,16), (32,104), (36,80), (20,16), (20,18)
    25% of the error comes from 1370 cells (8.7% of the grid)
    50% of the error comes from 3605 cells (22.9% of the grid)
    80% of the error comes from 7948 cells (50.6% of the grid)
```

## Where the error is -- layout B

```
layout B, predictions lgb_v5x_noll:_bw,xgb_v5x_noll:_bw: 312102 rows
  RMSE 0.5390   persistence 0.6869   gain -0.1480   bias +0.0581

  HORIZON  (share is of the test-weighted MSE: what a fix at that horizon is worth)
    h        n     RMSE     bias  persist     gain  test wt  wtd share
    1    93609   0.4966  +0.0577   0.5693  -0.0727    0.333      28.7%
    2    77708   0.5477  +0.0629   0.6786  -0.1309    0.222      23.3%
    3    62022   0.5542  +0.0684   0.7245  -0.1703    0.167      17.9%
    4    30958   0.5391  +0.0770   0.7018  -0.1628    0.111      11.3%
    5    15483   0.5663  -0.1466   0.8231  -0.2569    0.056       6.2%
    6    15515   0.5997  +0.0734   0.8303  -0.2306    0.056       7.0%
    7    15587   0.5413  +0.1406   0.8240  -0.2827    0.056       5.7%

  VS PERSISTENCE
    h=1: model is worse than the last observation on  44.8% of rows
    h=2: model is worse than the last observation on  40.7% of rows
    h=3: model is worse than the last observation on  39.2% of rows
    h=4: model is worse than the last observation on  41.2% of rows
    h=5: model is worse than the last observation on  34.6% of rows
    h=6: model is worse than the last observation on  39.5% of rows
    h=7: model is worse than the last observation on  33.4% of rows

        LAT BAND        n     RMSE     bias  persist     gain  MSE share
         +0..+30    61558   0.5477  +0.0022   0.6839  -0.1361      20.4%
        +30..+60   111318   0.5123  +0.0918   0.6972  -0.1849      32.2%
        +60..+90    77752   0.5540  +0.1598   0.6937  -0.1397      26.3%
         -30..+0    50085   0.5496  -0.0759   0.6500  -0.1004      16.7%
        -60..-30    11389   0.5916  -0.0737   0.7142  -0.1226       4.4%

  CELL SD DECILE        n     RMSE     bias  persist     gain  MSE share
               0    31225   0.2372  +0.1598   0.2149  +0.0223       1.9%
               1    31205   0.3224  +0.1117   0.3803  -0.0578       3.6%
               2    31211   0.4196  +0.0566   0.5343  -0.1147       6.1%
               3    31214   0.4954  +0.0732   0.6400  -0.1446       8.5%
               4    31204   0.5392  +0.0643   0.7084  -0.1692      10.0%
               5    31213   0.5763  +0.0590   0.7469  -0.1706      11.4%
               6    31214   0.5997  +0.0685   0.7718  -0.1722      12.4%
               7    31210   0.6311  +0.0307   0.8079  -0.1767      13.7%
               8    31204   0.6688  +0.0145   0.8523  -0.1835      15.4%
               9    31202   0.7037  -0.0570   0.8938  -0.1902      17.0%
    (decile 0 = the steadiest cells, 9 = the most variable)

    TARGET MONTH        n     RMSE     bias  persist     gain  MSE share
               1    15648   0.6009  +0.0723   0.8305  -0.2296       6.2%
               2    31324   0.5142  +0.1474   0.7023  -0.1881       9.1%
               3    31312   0.5581  +0.1330   0.6620  -0.1039      10.8%
               4    31189   0.5465  +0.0721   0.7035  -0.1570      10.3%
               5    15543   0.5684  -0.0356   0.7336  -0.1652       5.5%
               7    15586   0.5135  -0.0341   0.6190  -0.1055       4.5%
               8    46610   0.5021  +0.0232   0.5928  -0.0907      13.0%
               9    31115   0.5415  +0.0539   0.6728  -0.1314      10.1%
              10    46861   0.5305  +0.0870   0.6643  -0.1338      14.5%
              11    31280   0.5497  +0.0809   0.7104  -0.1607      10.4%
              12    15634   0.5665  -0.1474   0.8238  -0.2573       5.5%

  SPATIAL STRUCTURE OF THE RESIDUAL (correlation with the neighbour at grid lag d)
    lag 1 cells east: corr +0.969   (n=294326)
    lag 2 cells east: corr +0.915   (n=281670)
    lag 3 cells east: corr +0.845   (n=271414)
    lag 5 cells east: corr +0.688   (n=255494)
    lag 8 cells east: corr +0.476   (n=236076)
    Residual correlation still present at lag 2-3 means spatial smoothing has more to
    take; correlation gone by lag 1 means it is finished.

  WHERE THE MSE IS  (band x horizon, % of total)
      band         h1   h2   h3   h4   h5   h6   h7
    -60..-30    1.1  1.4  0.7  0.4  0.3  0.3  0.2
     -30..+0    3.3  4.0  3.9  1.8  1.5  1.5  0.6
     +0..+30    5.4  5.4  3.6  2.0  0.9  1.4  0.7
    +30..+60    8.2  7.5  6.6  3.7  1.5  2.3  2.3
    +60..+90    7.5  7.4  6.2  2.0  1.3  0.7  1.2

  20 worst cells carry 0.76% of total MSE:
    (62,-68), (38,138), (62,-66), (38,140), (62,-68), (38,140), (36,138), (40,142), (40,140), (40,142), (62,-64), (36,136), (40,140), (42,140), (46,100), (44,98), (44,98), (38,142), (40,92), (36,138)
    25% of the error comes from 1397 cells (8.9% of the grid)
    50% of the error comes from 3699 cells (23.5% of the grid)
    80% of the error comes from 8106 cells (51.6% of the grid)
```

## How much the covariates can explain (before the rebuild)

```
/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU/.venv/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py:3045: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU/.venv/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py:3046: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
layout A: 281003 rows, dy sd = 0.7563

  feature                 global r  within-cell r  r at h=1   what it is
  ------------------------------------------------------------------------------
  raw   e5PER_acc            0.004          0.028       nan   ERA5 P-E-R accumulated over the gap
  anom  an_e5PERz_acc        0.177          0.189       nan   
  raw   PER_acc             -0.051         -0.012       nan   NCEP P-E-R accumulated over the gap
  anom  an_PERz_acc          0.115          0.157       nan   
  raw   e5SW_d              -0.002          0.027       nan   ERA5 soil-water change over the gap
  anom  an_e5SWz_d           0.062          0.050       nan   
  raw   e5SWE_d              0.034          0.025       nan   ERA5 snow change over the gap
  anom  an_e5SWEz_d          0.068          0.068       nan   
  raw   e5P_acc             -0.032          0.019       nan   ERA5 precipitation accumulated
  anom  an_e5Pz_acc          0.235          0.259       nan   

  ridge on the 34 an_ features (fit on A_tr, scored on A_va):
    R^2 on the validation change = 0.0392   (corr 0.265)
    that is a LINEAR bound; the trees should exceed it
      h=1: R^2 +0.0473  (n=77979)
      h=2: R^2 +0.0562  (n=77721)
      h=3: R^2 +0.1072  (n=61974)
      h=4: R^2 -0.0079  (n=15501)
      h=5: R^2 -0.1288  (n=15546)
      h=6: R^2 -0.0342  (n=15608)
      h=7: R^2 +0.0145  (n=15535)
```

## How much the covariates can explain (after the rebuild)

```
/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU/.venv/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py:3045: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/moe/Programming/Forecasting Global Water Storage Challenge by ITU/.venv/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py:3046: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
layout A: 281003 rows, dy sd = 0.7563

  feature                 global r  within-cell r  r at h=1   what it is
  ------------------------------------------------------------------------------
  raw   e5PER_acc            0.004          0.028       nan   ERA5 P-E-R accumulated over the gap
  anom  an_e5PERz_acc        0.177          0.189       nan   
  raw   PER_acc             -0.051         -0.012       nan   NCEP P-E-R accumulated over the gap
  anom  an_PERz_acc          0.115          0.157       nan   
  raw   e5SW_d              -0.002          0.027       nan   ERA5 soil-water change over the gap
  anom  an_e5SWz_d           0.062          0.050       nan   
  raw   e5SWE_d              0.034          0.025       nan   ERA5 snow change over the gap
  anom  an_e5SWEz_d          0.068          0.068       nan   
  raw   e5P_acc             -0.032          0.019       nan   ERA5 precipitation accumulated
  anom  an_e5Pz_acc          0.235          0.259       nan   
  anom  an_e5MTWSz_d         0.074          0.061       nan   modelled total water storage change
  anom  an_MTWSz_d           0.069          0.087       nan   modelled total water storage change

  ridge on the 88 an_ features (fit on A_tr, scored on A_va):
    R^2 on the validation change = 0.0503   (corr 0.292)
    that is a LINEAR bound; the trees should exceed it
      h=1: R^2 +0.0762  (n=77979)
      h=2: R^2 +0.1018  (n=77721)
      h=3: R^2 +0.1326  (n=61974)
      h=4: R^2 -0.0207  (n=15501)
      h=5: R^2 -0.3144  (n=15546)
      h=6: R^2 -0.0455  (n=15608)
      h=7: R^2 +0.1183  (n=15535)
```

## The global month-to-month shift

```
layout A, predictions lgb_v5x_noll: 281003 rows, RMSE 0.6307

  per target month, mean change over all cells:
  month            true  predicted   missed
  2012-07-01     0.0901     0.0852   0.0050
  2012-08-01     0.0199     0.0861  -0.0662
  2012-09-01     0.0459     0.0653  -0.0194
  2013-01-01     0.0050     0.0561  -0.0511
  2013-02-01    -0.0946     0.0118  -0.1064
  2013-11-01     0.0033    -0.0031   0.0065
  2013-12-01     0.0070     0.0104  -0.0034
  2014-01-01     0.0963    -0.0269   0.1232
  2014-09-01    -0.1260    -0.0284  -0.0976
  2014-10-01     0.0699    -0.0645   0.1344
  2014-11-01     0.0706    -0.0892   0.1598
  2014-12-01    -0.1644    -0.0836  -0.0808
  2015-01-01    -0.1347    -0.1042  -0.0305
  2015-02-01    -0.1272    -0.0989  -0.0283
  2015-03-01    -0.1199    -0.1114  -0.0085
  2015-06-01    -0.2726     0.0224  -0.2950
  2015-07-01     0.0429    -0.0245   0.0675
  2015-08-01     0.0268    -0.0519   0.0787

  sd of the true monthly global shift : 0.1027
  sd of the predicted one             : 0.0637   (ratio 0.62)
  corr(pred, true) = +0.306    regression slope = +0.494   

  share of total MSE that is pure global-offset error: 2.7%
  ORACLE, perfect global correction : RMSE 0.6221   (from 0.6307, i.e. -0.0086)
    persistence of the previous month's offset, a=0.25: RMSE 0.6316
    persistence of the previous month's offset, a=0.50: RMSE 0.6335
    persistence of the previous month's offset, a=0.75: RMSE 0.6365
    persistence of the previous month's offset, a=1.00: RMSE 0.6405
    corr(previous offset, this offset) = -0.121   <- if this is near zero the offset is temporally white and NOT correctable

  same decomposition per 30-degree latitude band:
    share of total MSE that is band-offset error: 12.9%
    ORACLE, perfect band correction: RMSE 0.5886   (-0.0421)

  Read it this way: the oracle rows bound the prize. If they are small, no global or
  zonal correction can help and this avenue is closed. If they are large, the
  persistence row says whether any of it is reachable with information at <= t.
```

## The global month-to-month shift (after the rebuild)

```
layout A, predictions lgb_v5x_noll_e8: 281003 rows, RMSE 0.6277

  per target month, mean change over all cells:
  month            true  predicted   missed
  2012-07-01     0.0901     0.1003  -0.0102
  2012-08-01     0.0199     0.0908  -0.0709
  2012-09-01     0.0459     0.0693  -0.0234
  2013-01-01     0.0050     0.0793  -0.0743
  2013-02-01    -0.0946     0.0258  -0.1203
  2013-11-01     0.0033     0.0047  -0.0014
  2013-12-01     0.0070     0.0188  -0.0117
  2014-01-01     0.0963    -0.0149   0.1112
  2014-09-01    -0.1260    -0.0321  -0.0939
  2014-10-01     0.0699    -0.0676   0.1375
  2014-11-01     0.0706    -0.0915   0.1621
  2014-12-01    -0.1644    -0.0900  -0.0744
  2015-01-01    -0.1347    -0.1042  -0.0304
  2015-02-01    -0.1272    -0.1074  -0.0199
  2015-03-01    -0.1199    -0.1087  -0.0112
  2015-06-01    -0.2726     0.0344  -0.3070
  2015-07-01     0.0429    -0.0198   0.0627
  2015-08-01     0.0268    -0.0478   0.0746

  sd of the true monthly global shift : 0.1027
  sd of the predicted one             : 0.0695   (ratio 0.68)
  corr(pred, true) = +0.307    regression slope = +0.453   

  share of total MSE that is pure global-offset error: 2.9%
  ORACLE, perfect global correction : RMSE 0.6187   (from 0.6277, i.e. -0.0090)
    persistence of the previous month's offset, a=0.25: RMSE 0.6284
    persistence of the previous month's offset, a=0.50: RMSE 0.6303
    persistence of the previous month's offset, a=0.75: RMSE 0.6332
    persistence of the previous month's offset, a=1.00: RMSE 0.6372
    corr(previous offset, this offset) = -0.081   <- if this is near zero the offset is temporally white and NOT correctable

  same decomposition per 30-degree latitude band:
    share of total MSE that is band-offset error: 13.0%
    ORACLE, perfect band correction: RMSE 0.5854   (-0.0423)

  Read it this way: the oracle rows bound the prize. If they are small, no global or
  zonal correction can help and this avenue is closed. If they are large, the
  persistence row says whether any of it is reachable with information at <= t.
```

## Compliance audit

```

==============================================================================
R1  FEATURES ACTUALLY USED  (layout FINAL)
==============================================================================
  [PASS] used_FINAL_lgb_v5x_noll_f1.json: 254 features, no lat/lon   model=lgb featset=v5x_noll dropf=['bigsa', 'scale'] anchor=tws weights=ramp hmix=test hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7  covariate anomaly 88
  [PASS] used_FINAL_lgb_v5x_noll_f3.json: 166 features, no lat/lon   model=lgb featset=v5x_noll dropf=['anom', 'bigsa', 'scale'] anchor=tws weights=ramp hmix=- hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7
  [PASS] used_FINAL_lgbs_v5x_noll_f2.json: 254 features, no lat/lon   model=lgbs featset=v5x_noll dropf=['bigsa', 'scale'] anchor=tws weights=ramp hmix=test hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7  covariate anomaly 88
  [PASS] used_FINAL_xgb_v5x_noll_f1.json: 254 features, no lat/lon   model=xgb featset=v5x_noll dropf=['bigsa', 'scale'] anchor=tws weights=ramp hmix=test hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7  covariate anomaly 88
  [PASS] used_FINAL_xgb_v5x_noll_f2.json: 254 features, no lat/lon   model=xgb featset=v5x_noll dropf=['bigsa', 'scale'] anchor=tws weights=ramp hmix=test hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7  covariate anomaly 88
  [PASS] used_FINAL_xgb_v5x_noll_f3.json: 166 features, no lat/lon   model=xgb featset=v5x_noll dropf=['anom', 'bigsa', 'scale'] anchor=tws weights=ramp hmix=- hfilt=-
         CPC 3  ERA5 21  NCEP 18  NCEP2 21  TWS/competition 96  anchor 7

==============================================================================
R2  ROW GEOMETRY AND ANCHORING
==============================================================================
  [PASS] t_known <= t for every row   min gap 0 months, max 6
  [PASS] horizon == months(t_known -> t) + 1 for every row
  [PASS] tws_known equals the observed TWS at t_known, reconstructed independently   max |difference| 1.19e-07 over 280961 rows
  [PASS] clim_next reproduces a HISTORY-ONLY climatology exactly   max |difference| 2.98e-08 over 280326 rows

==============================================================================
R4/R5  EXTERNAL DATA INVENTORY
==============================================================================
          0.0 MB  external/.DS_Store
          0.0 MB  external/oni.data
          0.0 MB  external/oni.parquet
         23.3 MB  external/ncep2/lhtfl.sfc.mon.mean.nc
         25.1 MB  external/ncep2/prate.sfc.mon.mean.nc
          6.6 MB  external/ncep2/runof.sfc.mon.mean.nc
         12.2 MB  external/ncep2/soilw.0-10cm.mon.mean.nc
         12.2 MB  external/ncep2/soilw.10-200cm.mon.mean.nc
          5.9 MB  external/ncep2/weasd.sfc.mon.mean.nc
         52.3 MB  external/ncep/lhtfl.sfc.mon.mean.nc
         51.6 MB  external/ncep/prate.sfc.mon.mean.nc
         13.5 MB  external/ncep/runof.sfc.mon.mean.nc
         19.0 MB  external/ncep/soilw.0-10cm.mon.mean.nc
         18.2 MB  external/ncep/soilw.10-200cm.mon.mean.nc
          5.1 MB  external/ncep/weasd.sfc.mon.mean.nc
        255.3 MB  external/cpc/soilw.mon.mean.nc
          7.6 MB  external/era5/era5_monthly_1deg_2001.nc
          7.6 MB  external/era5/era5_monthly_1deg_2002.nc
          7.6 MB  external/era5/era5_monthly_1deg_2003.nc
          7.6 MB  external/era5/era5_monthly_1deg_2004.nc
          7.6 MB  external/era5/era5_monthly_1deg_2005.nc
          7.6 MB  external/era5/era5_monthly_1deg_2006.nc
          7.7 MB  external/era5/era5_monthly_1deg_2007.nc
          7.7 MB  external/era5/era5_monthly_1deg_2008.nc
          7.6 MB  external/era5/era5_monthly_1deg_2009.nc
          7.5 MB  external/era5/era5_monthly_1deg_2010.nc
          7.6 MB  external/era5/era5_monthly_1deg_2011.nc
          7.6 MB  external/era5/era5_monthly_1deg_2012.nc
          7.6 MB  external/era5/era5_monthly_1deg_2013.nc
          7.6 MB  external/era5/era5_monthly_1deg_2014.nc
          7.5 MB  external/era5/era5_monthly_1deg_2015.nc
          7.5 MB  external/era5/era5_monthly_1deg_2016.nc
          7.6 MB  external/era5/era5_monthly_1deg_2017.nc
          7.6 MB  external/era5/era5_monthly_1deg_2018.nc
          7.6 MB  external/era5/era5_monthly_1deg_2019.nc
          3.2 MB  external/era5/era5_monthly_1deg_2015_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2015_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2011_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2011_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2001_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2001_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2005_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2005_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2016_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2016_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2018_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2018_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2012_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2012_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2008_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2008_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2002_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2002_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2006_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2006_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2017_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2017_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2013_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2013_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2019_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2019_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2003_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2003_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2009_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2009_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2007_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2007_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2014_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2014_x/data_stream-moda_stepType-avgua.nc
          3.1 MB  external/era5/era5_monthly_1deg_2010_x/data_stream-moda_stepType-avgad.nc
          4.4 MB  external/era5/era5_monthly_1deg_2010_x/data_stream-moda_stepType-avgua.nc
          3.2 MB  external/era5/era5_monthly_1deg_2004_x/data_stream-moda_stepType-avgad.nc
          4.5 MB  external/era5/era5_monthly_1deg_2004_x/data_stream-moda_stepType-avgua.nc
  73 external files. Every one must be a non-GRACE covariate whose source date is <= t.
  ERA5 / NCEP / CPC are reanalysis and observation products, not GRACE derivatives.

==============================================================================
CODE PATTERNS THAT HAVE CAUSED PROBLEMS BEFORE
==============================================================================
  traj_smooth: 5 reference(s) outside its own module  (trajectory smoothing uses the next horizon's prediction)
      ./compliance.py:140:    for pat, why in (("traj_smooth", "trajectory smoothing uses the next horizon's predict
      ./blend_eval.py:3:from smooth import smooth, traj_smooth
      ./blend_eval.py:10:for n,p in P.items(): print(f"  {n:28} {r(p):.4f}  +smooth {r(smooth(va,p,0.7)):.4f}  +traj
      ./blend_eval.py:16:    print("  NNLS weights:",dict(zip(names,np.round(wts,3))),f"-> {r(pb):.4f}  +smooth+traj
      ./blend_eval.py:17:    pe=M.mean(1); print(f"  equal mean -> {r(pe):.4f}  +smooth+traj {r(traj_smooth(va,smoot
  TWS_t_masked: 15 reference(s) outside its own module  (reads of the masked flag -- must only ever EXCLUDE rows)
      ./build_mats.py:36:    obs_hist=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_hist, te.filte
      ./data_report.py:79:    mask_col = "TWS_t_masked" if "TWS_t_masked" in te.columns else None
      ./predict_test.py:12:obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","tim
      ./build_final_fallback.py:16:obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, 
      ./compliance.py:103:        if "TWS_t_masked" in te.columns:
      ./compliance.py:104:            obs = pl.concat([obs, te.filter(~pl.col("TWS_t_masked"))

==============================================================================
RESULT
==============================================================================
  every check passed
```

## Feature completeness of the FINAL matrix

```
FINAL carries all 266 features
```
