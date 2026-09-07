"""Append great-circle smoothed-anchor features to the cached matrices, in the same row order.
Training rows use history-only observations; evaluation rows use the observations available at
prediction time (history + test rows whose TWS_t is not masked), exactly as build_mats.py does."""
import polars as pl, numpy as np, sys, time, os
from anchor import anchor_fields, sample
L=sys.argv[1]; t0=time.time()
# 1500/2500 km give the CONTINENTAL scale. Session 3 measured that this test period's
# unpredictable component is regional and spatially coherent, and nothing else in the feature
# set aggregates above ~800 km. A great-circle disc is also more hydrologically coherent than a
# latitude ring, which at 45N mixes Oregon, Iowa, France, Kazakhstan and Mongolia.
RAD=tuple(int(x) for x in os.environ.get("ANCHOR_RADII","300,500,800,1500,2500").split(","))
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
if L=="FINAL":
    te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    obs_hist=tr.select(["lat","lon","time","TWS_t"])
    obs_eval=pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
else:
    sfx={"A":"","B":"_B","C":"_C"}[L]; tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"])
    obs_eval=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
for part,obs in (("tr",obs_hist),("va",obs_eval)):
    m=pl.read_parquet(f"out/mats/{L}_{part}.parquet",columns=["lat","lon","t_known","tws_known"])
    F,ti=anchor_fields(m["t_known"].unique().to_list(),obs,RAD); S=sample(F,ti,m)
    k=m["tws_known"].to_numpy(); cols={}
    for km,v in S.items():
        v=np.where(np.isnan(v),k,v).astype(np.float32)
        cols[f"sa{km}"]=v; cols[f"dsa{km}"]=(k-v).astype(np.float32)
    if 300 in RAD and 800 in RAD:  cols["sa_grad"] =(cols["sa300"]-cols["sa800"]).astype(np.float32)
    if 800 in RAD and 2500 in RAD: cols["sa_grad2"]=(cols["sa800"]-cols["sa2500"]).astype(np.float32)
    pl.DataFrame(cols).write_parquet(f"out/mats/{L}_{part}_anchor.parquet")
    print(f"{L} {part}: {len(m)} rows, cols {list(cols)} ({time.time()-t0:.0f}s)",flush=True)
