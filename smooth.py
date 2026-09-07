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

def traj_smooth(df, pred, w=0.35, causal=True):
    """Smooth the predicted change along the horizon within a block: for the same (cell, t_known),
    blend each residual with the mean of its horizon neighbours (h-1, h+1). df: lat, lon, t_known, horizon, tws_known."""
    d=df.select(["lat","lon","t_known","horizon","tws_known"]).with_row_index("i").with_columns(pl.Series("res",pred-df["tws_known"].to_numpy())).sort(["lat","lon","t_known","horizon"])
    g=["lat","lon","t_known"]
    e=d.with_columns(pl.col("res").shift(1).over(g).alias("rm"),pl.col("res").shift(-1).over(g).alias("rp"),pl.col("horizon").shift(1).over(g).alias("hm"),pl.col("horizon").shift(-1).over(g).alias("hp"))
    e=e.with_columns(pl.when(pl.col("hm")==pl.col("horizon")-1).then(pl.col("rm")).otherwise(None).alias("rm"),pl.when(pl.col("hp")==pl.col("horizon")+1).then(pl.col("rp")).otherwise(None).alias("rp"))
    if causal: e=e.with_columns(pl.lit(None,dtype=pl.Float64).alias("rp"))   # only the previous horizon (information <= t) is used
    e=e.with_columns(pl.mean_horizontal(["rm","rp"]).alias("nb")).with_columns(pl.when(pl.col("nb").is_null()).then(pl.col("res")).otherwise((1-w)*pl.col("res")+w*pl.col("nb")).alias("res2")).sort("i")
    return (e["tws_known"]+e["res2"]).to_numpy()


# ---------------------------------------------------------------------------------------------
# session 9m: the same neighbourhood blend, but with a per-row weight, an explicit row order and
# optional longitude wrap-around. `smooth` above hardcodes one scalar w for every row and drops
# the dateline neighbours; both are now tunable (smooth_scan.py) rather than assumed.
def nb_mean(df, res, radius=1, wrap=False):
    """Mean of `res` over the 8 (or 24) grid neighbours at the same month, in row order.

    Pulled out of smooth_field because one blending pass is LINEAR in the neighbour mean:
    res' = res + w * (nb - res).  So a whole grid of blend weights -- including a different
    weight per horizon -- can be scored from a single neighbour pass instead of one pass each,
    which is what makes smooth_scan.py cheap enough to run on every configuration."""
    d = (df.select(["lat", "lon", "time"]).with_columns(pl.Series("res", np.asarray(res, float)))
           .with_row_index("_i"))
    lo = int(df["lon"].min()); span = int(df["lon"].max()) - lo + 1
    wrap = wrap and span >= 350
    nb = []
    for dl in range(-radius, radius + 1):
        for dn in range(-radius, radius + 1):
            if not (dl or dn):
                continue
            nb.append(d.select(["lat", "lon", "time", "res"]).with_columns(
                (pl.col("lat") + dl).alias("lat"),
                (((pl.col("lon") + dn - lo) % span + lo) if wrap
                 else (pl.col("lon") + dn)).alias("lon")))
    nb = pl.concat(nb).group_by(["lat", "lon", "time"]).agg(pl.col("res").mean().alias("nb"))
    d = d.join(nb, on=["lat", "lon", "time"], how="left").sort("_i")
    return d["nb"].fill_null(d["res"]).to_numpy()


def smooth_field(df, pred, w, radius=1, iters=1, wrap=False):
    """df: lat, lon, time, tws_known.   w: scalar, or one blend weight per row.

    Per-row w exists because the right amount of smoothing is not the same at every horizon:
    at h=1 the prediction is dominated by the cell's own observed state and blurring it in
    costs signal, while at h=7 almost nothing is left but the regional mean, which is exactly
    what the neighbourhood estimates.

    wrap=True joins the cells either side of the dateline. On a global 1-degree grid the two
    edge columns otherwise smooth against half a neighbourhood."""
    k = df["tws_known"].to_numpy()
    wv = np.full(len(df), float(w)) if np.isscalar(w) else np.asarray(w, dtype=float)
    assert len(wv) == len(df)
    res = np.asarray(pred, float) - k
    for _ in range(iters):
        res = res + wv * (nb_mean(df, res, radius, wrap) - res)
    return k + res


def horizon_w(h, w1, w7):
    """Linear ramp of the smoothing weight from horizon 1 to horizon 7."""
    return np.clip(w1 + (w7 - w1) * (np.asarray(h, float) - 1.0) / 6.0, 0.0, 1.0)
