import polars as pl, numpy as np, sys, time
from anchor import anchor_fields, sample
L=sys.argv[1]; RAD=(300,500); t0=time.time()
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date())
if L=="FINAL":
    te=pl.read_csv("Test.csv").with_columns(pl.col("time").str.to_date())
    hist=tr; obs_hist=tr.select(["lat","lon","time","TWS_t"])
    obs_eval=pl.concat([obs_hist, te.filter(~pl.col("TWS_t_masked")).select(["lat","lon","time","TWS_t"])])
else:
    sfx={"A":"","B":"_B"}[L]; tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
    obs_hist=hist.select(["lat","lon","time","TWS_t"])
    obs_eval=pl.concat([obs_hist, tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])

fit=hist.select(["lat","lon","time","TWS_t","target"]).rename({"time":"t_known"})
F,ti=anchor_fields(sorted(set(fit["t_known"].to_list())),obs_hist,RAD); S=sample(F,ti,fit)
yr=fit["t_known"].dt.year().to_numpy(); k=fit["TWS_t"].to_numpy(); tgt=fit["target"].to_numpy()
cells=fit.select(["lat","lon"]); acc={}
for km in RAD:
    sa=S[km]; ok=~np.isnan(sa)
    d=pl.DataFrame({"lat":cells["lat"],"lon":cells["lon"],"yr":yr,
                    "x":np.where(ok,k-sa,np.nan),"z":np.where(ok,tgt-sa,np.nan)}).drop_nulls()
    g=d.group_by(["lat","lon","yr"]).agg(pl.col("x").sum().alias("Sx"),pl.col("z").sum().alias("Sz"),
        (pl.col("x")*pl.col("x")).sum().alias("Sxx"),(pl.col("x")*pl.col("z")).sum().alias("Sxz"),pl.len().alias("N"))
    tot=g.group_by(["lat","lon"]).agg([pl.col(c).sum().alias(c+"_t") for c in ("Sx","Sz","Sxx","Sxz","N")])
    acc[km]=(g.join(tot,on=["lat","lon"]),tot)
def coefs(Sx,Sz,Sxx,Sxz,N):
    N=np.maximum(N,1); vx=Sxx-Sx*Sx/N; cxz=Sxz-Sx*Sz/N
    b=np.where((N>=8)&(vx>1e-6),cxz/np.maximum(vx,1e-9),np.nan); b=np.clip(b,-0.5,1.5)
    a=np.where(np.isnan(b),np.nan,Sz/N-b*(Sx/N)); return a,b,N
for part,obs in (("tr",obs_hist),("va",obs_eval)):
    m=pl.read_parquet(f"out/mats/{L}_{part}.parquet",columns=["lat","lon","t_known","tws_known","horizon"]).with_row_index("i")
    Fm,tim=anchor_fields(sorted(set(m["t_known"].to_list())),obs,RAD); Sm=sample(Fm,tim,m)
    my=m["t_known"].dt.year().to_numpy(); mk=m["tws_known"].to_numpy(); cols={}
    for km in RAD:
        gy,tot=acc[km]
        if part=="tr":
            j=m.with_columns(pl.Series("yr",my)).join(gy,on=["lat","lon","yr"],how="left").join(tot,on=["lat","lon"],how="left").sort("i")
            arr=lambda c: np.nan_to_num(j[c].to_numpy()) if c in j.columns else 0
            a,b,N=coefs(arr("Sx_t")-arr("Sx"),arr("Sz_t")-arr("Sz"),arr("Sxx_t")-arr("Sxx"),arr("Sxz_t")-arr("Sxz"),arr("N_t")-arr("N"))
        else:
            j=m.join(tot,on=["lat","lon"],how="left").sort("i")
            a,b,N=coefs(*[np.nan_to_num(j[c+"_t"].to_numpy()) for c in ("Sx","Sz","Sxx","Sxz","N")])
        sa=np.where(np.isnan(Sm[km]),mk,Sm[km]); dev=mk-sa
        bb=np.where(np.isnan(b),0.5,b); aa=np.where(np.isnan(a),0.0,a)
        cols[f"lb{km}"]=bb.astype(np.float32); cols[f"la{km}"]=aa.astype(np.float32); cols[f"ln{km}"]=N.astype(np.float32)
        cols[f"lp{km}"]=(sa+aa+bb*dev).astype(np.float32)
        cols[f"lpd{km}"]=(sa+aa+bb*dev-mk).astype(np.float32)
    pl.DataFrame(cols).write_parquet(f"out/mats/{L}_{part}_anchor3.parquet")
    print(f"{L} {part}: {len(m)} rows, {len(cols)} cols, beta mean {np.nanmean(cols['lb300']):.3f} ({time.time()-t0:.0f}s)",flush=True)
