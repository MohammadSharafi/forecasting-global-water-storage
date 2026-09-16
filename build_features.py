import polars as pl, numpy as np, gc, sys, time
from features import COV, training_rows, cell_stats
from features2 import clim_sums, assemble2, FEATS2
from features4 import add_ar, AR
F=FEATS2+AR; SEED=int(sys.argv[1]) if len(sys.argv)>1 else 2
t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date()); te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
cov_all=pl.concat([tr.select(["lat","lon","time"]+COV), te.select(["lat","lon","time"]+COV)])
obs_train=tr.select(["lat","lon","time","TWS_t"]); obs_all=pl.concat([obs_train, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
sums=clim_sums(tr); _,cell=cell_stats(tr)

known=(te.select(["lat","lon","time"]).join(obs_all.rename({"time":"t_obs"}).select(["lat","lon","t_obs"]),on=["lat","lon"],how="inner")
        .filter(pl.col("t_obs")<=pl.col("time")).group_by(["lat","lon","time"]).agg(pl.col("t_obs").max().alias("t_known")))
rows_te=te.select(["ID","lat","lon","time"]).join(known,on=["lat","lon","time"],how="left"); assert rows_te["t_known"].null_count()==0
Xte=add_ar(assemble2(rows_te,cov_all,obs_all,sums,cell,loyo=False),obs_all)
Xte.select(list(dict.fromkeys(["ID","lat","lon","time","t_known","horizon","tws_known"]+F))).with_columns([pl.col(c).cast(pl.Float32) for c in F if c not in ("lat","lon")]).write_parquet("out/Xte_v4.parquet")
print(f"test matrix {Xte.shape} written ({time.time()-t0:.0f}s)",flush=True); del Xte, te, obs_all, known, rows_te; gc.collect()

Xtr=add_ar(assemble2(training_rows(tr,np.random.default_rng(SEED),per_row=3),cov_all,obs_train,sums,cell,loyo=True),obs_train)
Xtr.select(list(dict.fromkeys(["target","tws_known"]+F))).with_columns([pl.col(c).cast(pl.Float32) for c in dict.fromkeys(["target","tws_known"]+F)]).write_parquet(f"out/Xtr_v4_s{SEED}.parquet")
print(f"train matrix {Xtr.shape} written ({time.time()-t0:.0f}s)")
