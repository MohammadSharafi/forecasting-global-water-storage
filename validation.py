"""Test-mimicking validation. Real test = 6 blocks of consecutive months; TWS_t is
observed only in a block's first month, hidden afterwards (horizon 1..7). We carve the
same structure out of the END of train and evaluate simple baselines with only
information available at or before t."""
import polars as pl, numpy as np
tr=pl.read_csv("Train.csv").with_columns(pl.col("time").str.to_date(), pl.col("time").str.to_date().dt.month().alias("m"))
months=sorted(tr["time"].unique().to_list())
# pseudo-test blocks from the last ~3 years, mimicking real block lengths (1,3,4,7 months etc.)
BLOCKS=[["2012-07","2012-08","2012-09"],["2013-01","2013-02"],["2013-11","2013-12","2014-01"],
        ["2014-09","2014-10","2014-11","2014-12","2015-01","2015-02","2015-03"],["2015-06","2015-07","2015-08"]]
blocks=[[pl.date(*map(int,(m+"-01").split("-")[:2]),1) for m in b] for b in BLOCKS]
import datetime as dt
blocks=[[dt.date(int(m[:4]),int(m[5:7]),1) for m in b] for b in BLOCKS]
test_months=[m for b in blocks for m in b]; first={b[0] for b in blocks}
cut=min(test_months)
hist=tr.filter(pl.col("time")<cut)              # "train" part: everything before first pseudo-test month
te=tr.filter(pl.col("time").is_in(test_months)).with_columns(pl.col("time").is_in(list(first)).not_().alias("masked"))
# information available at t: TWS_t only in first month of each block. Last-known TWS per cell:
# from block-first months (observed) or from hist's last month before the block.
obs=pl.concat([hist.select(["lat","lon","time","TWS_t"]),
               te.filter(~pl.col("masked")).select(["lat","lon","time","TWS_t"])]).sort("time")
rows=[]
for b in blocks:
    for m in b:
        sub=te.filter(pl.col("time")==m)
        known=obs.filter(pl.col("time")<=m).group_by(["lat","lon"]).agg(pl.col("time").max().alias("t_known"), pl.col("TWS_t").sort_by("time").last().alias("tws_known"))
        sub=sub.join(known,on=["lat","lon"],how="left")
        rows.append(sub)
te=pl.concat(rows)
te=te.with_columns(((pl.col("time").dt.year()-pl.col("t_known").dt.year())*12+(pl.col("time").dt.month()-pl.col("t_known").dt.month())+1).alias("horizon"))
# climatology from hist only (leak-free), of TWS by calendar month
clim=hist.group_by(["lat","lon","m"]).agg(pl.col("TWS_t").mean().alias("clim_t"), pl.col("TWS_t").std().alias("sd_t"))
tm=te.with_columns(((pl.col("m")%12)+1).alias("m_next"), pl.col("t_known").dt.month().alias("m_known"))
tm=tm.join(clim.rename({"m":"m_next","clim_t":"clim_next"}).select(["lat","lon","m_next","clim_next"]),on=["lat","lon","m_next"],how="left")
tm=tm.join(clim.rename({"m":"m_known","clim_t":"clim_known"}).select(["lat","lon","m_known","clim_known"]),on=["lat","lon","m_known"],how="left")
tm=tm.with_columns((pl.col("clim_next")+(pl.col("tws_known")-pl.col("clim_known"))).alias("anom_persist"))
y=tm["target"].to_numpy()
def rmse(p): p=np.asarray(p,float); ok=~np.isnan(p); return np.sqrt(np.mean((y[ok]-p[ok])**2)), ok.mean()
print(f"pseudo-test rows={len(tm)}  masked share={tm['masked'].mean():.3f}  horizons: {tm.group_by('horizon').len().sort('horizon').to_dict(as_series=False)}")
print(f"{'baseline':46} {'RMSE':>8}  coverage")
for name,p in [("last-known TWS (persistence)",tm["tws_known"]),("per-cell climatology of target month",tm["clim_next"]),
               ("anomaly persistence: clim_next + (known-clim_known)",tm["anom_persist"]),
               ("0.5*persist + 0.5*clim",0.5*tm["tws_known"]+0.5*tm["clim_next"]),
               ("0.5*anom_persist + 0.5*clim",0.5*tm["anom_persist"]+0.5*tm["clim_next"])]:
    r,c=rmse(p); print(f"{name:46} {r:8.4f}  {c:.3f}")
print("\nby horizon (anomaly persistence vs persistence vs clim):")
for h,g in tm.group_by("horizon",maintain_order=True):
    g=g.sort("horizon"); yy=g["target"].to_numpy()
    f=lambda c: np.sqrt(np.nanmean((yy-g[c].to_numpy())**2))
    print(f"  h={h[0]:>2} n={len(g):>6}  persist={f('tws_known'):.4f}  anom_persist={f('anom_persist'):.4f}  clim={f('clim_next'):.4f}")
tm.write_parquet("out/pseudo_test.parquet"); hist.write_parquet("out/pseudo_hist.parquet")
