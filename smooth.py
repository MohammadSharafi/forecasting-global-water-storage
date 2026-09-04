"""Spatial smoothing of predicted residuals (pred - last known) over neighbours at the same month."""
import polars as pl, numpy as np
def smooth(df, pred, w, radius=1, iters=1):
    """df: lat, lon, time, tws_known.  Returns smoothed predictions."""
    d=df.select(["lat","lon","time","tws_known"]).with_columns(pl.Series("res",pred-df["tws_known"].to_numpy()))
    for _ in range(iters):
        nb=[]
        for dl in range(-radius,radius+1):
            for dn in range(-radius,radius+1):
                if dl or dn: nb.append(d.select(["lat","lon","time","res"]).with_columns((pl.col("lat")+dl).alias("lat"),(pl.col("lon")+dn).alias("lon")))
        nb=pl.concat(nb).group_by(["lat","lon","time"]).agg(pl.col("res").mean().alias("nb"))
        d=d.join(nb,on=["lat","lon","time"],how="left").with_columns(((1-w)*pl.col("res")+w*pl.col("nb").fill_null(pl.col("res"))).alias("res")).drop("nb")
    return (d["tws_known"]+d["res"]).to_numpy()
if __name__=="__main__":
    for sfx in ["","_B"]:
        tp=pl.read_parquet(f"out/pseudo_test{sfx}.parquet"); hist=pl.read_parquet(f"out/pseudo_hist{sfx}.parquet")
        obs=pl.concat([hist.select(["lat","lon","time","TWS_t"]), tp.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])])
        tp=tp.join(obs.rename({"time":"t_known","TWS_t":"tws_known"}),on=["lat","lon","t_known"],how="left")
        p=np.load(f"out/val4{sfx}_pred.npy"); y=tp["target"].to_numpy(); r=lambda a: float(np.sqrt(np.mean((y-a)**2)))
        print(f"layout{sfx or ' A'}: raw={r(p):.4f}")
        for radius,iters,w in [(1,1,0.5),(1,1,0.7),(1,1,1.0),(1,2,0.5),(1,3,0.5),(2,1,0.5),(2,1,0.7),(2,2,0.5)]:
            print(f"   radius={radius} iters={iters} w={w}: {r(smooth(tp,p,w,radius,iters)):.4f}")
