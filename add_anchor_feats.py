import polars as pl, numpy as np, sys, time, os
from anchor import anchor_fields, sample
from build_mats import base_of
L=sys.argv[1]; t0=time.time()

RAD=tuple(int(x) for x in os.environ.get("ANCHOR_RADII","300,500,800,1500,2500").split(","))
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
if base_of(L)=="FINAL":
    te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    obs_hist=tr.select(["lat","lon","time","TWS_t"])
    obs_eval=pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
else:
    _b=base_of(L); sfx="" if _b=="A" else f"_{_b}"
    tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
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
