import polars as pl, numpy as np
from smooth import smooth
for sfx in ["","_B"]:
    tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs=pl.concat([hist.select(["lat","lon","time","TWS_t"]), tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
    tp=tp.join(obs.rename({"time":"t_known","TWS_t":"tws_known"}),on=["lat","lon","t_known"],how="left")
    if "horizon" not in tp.columns: tp=tp.with_columns(((pl.col("time").dt.year()-pl.col("t_known").dt.year())*12+(pl.col("time").dt.month()-pl.col("t_known").dt.month())+1).alias("horizon"))
    y=tp["target"].to_numpy(); k=tp["tws_known"].to_numpy(); h=tp["horizon"].to_numpy()
    r=lambda a: float(np.sqrt(np.mean((y-a)**2)))
    pn=np.load(f"out/valN{sfx}_v6w_ncep.npy"); p5=np.load(f"out/val5{'_B' if sfx else ''}_pred.npy") if sfx=="" else np.load("out/val4_B_pred.npy")
    print(f"== layout{sfx or ' A'}: ncep={r(pn):.4f} v5/v4={r(p5):.4f}")
    for w in (0.3,0.5,0.7): print(f"  blend w_ncep={w}: {r(w*pn+(1-w)*p5):.4f}")
    pb=0.5*pn+0.5*p5
    ps=smooth(tp,pb,w=0.7,radius=1,iters=1); print(f"  smoothed blend: {r(ps):.4f}")

    for a in (0.9,1.0,1.1,1.2):
        print(f"  change x{a}: {r(k+a*(ps-k)):.4f}", "  by h:", " ".join(f"{hh}:{np.sqrt(np.mean(((y-(k+a*(ps-k)))**2)[h==hh])):.3f}" for hh in range(1,8)))

    d=tp.select(["lat","lon","t_known","horizon"]).with_columns(pl.Series("p",ps),pl.Series("k",k)).with_columns((pl.col("p")-pl.col("k")).alias("res"))
    d=d.sort(["lat","lon","t_known","horizon"])
    for w in (0.2,0.35,0.5):
        e=d.with_columns(pl.col("res").shift(1).over(["lat","lon","t_known"]).alias("rm"),pl.col("res").shift(-1).over(["lat","lon","t_known"]).alias("rp"),
                         pl.col("horizon").shift(1).over(["lat","lon","t_known"]).alias("hm"),pl.col("horizon").shift(-1).over(["lat","lon","t_known"]).alias("hp"))
        e=e.with_columns(pl.when(pl.col("hm")==pl.col("horizon")-1).then(pl.col("rm")).otherwise(None).alias("rm"),pl.when(pl.col("hp")==pl.col("horizon")+1).then(pl.col("rp")).otherwise(None).alias("rp"))
        e=e.with_columns(pl.mean_horizontal(["rm","rp"]).alias("nb")).with_columns(pl.when(pl.col("nb").is_null()).then(pl.col("res")).otherwise((1-w)*pl.col("res")+w*pl.col("nb")).alias("res2"))
        e=e.join(tp.select(["lat","lon","t_known","horizon"]).with_row_index("i"),on=["lat","lon","t_known","horizon"]).sort("i")
        q=(e["k"]+e["res2"]).to_numpy(); print(f"  traj smooth w={w}: {r(q):.4f}")
