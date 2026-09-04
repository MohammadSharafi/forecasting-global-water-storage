"""v5: wider (5x5) neighbourhood means of the key dynamic features at the same (t, t_known)."""
import polars as pl
KEYS=["anom_known","dev24","d2","d6","SPEI_06_t_d","SPEI_12_t_d","spei1_acc","SOIL_MOISTURE_t_d","slope","trend24"]
def add_wide(r, radius=2):
    base=r.select(["lat","lon","time","t_known"]+KEYS)
    nb=[]
    for dl in range(-radius,radius+1):
        for dn in range(-radius,radius+1):
            if dl or dn: nb.append(base.with_columns((pl.col("lat")+dl).alias("lat"),(pl.col("lon")+dn).alias("lon")))
    nb=pl.concat(nb).group_by(["lat","lon","time","t_known"]).agg([pl.col(k).mean().alias("w_"+k) for k in KEYS]+[pl.len().alias("w_n")])
    return r.join(nb,on=["lat","lon","time","t_known"],how="left")
WIDE=["w_"+k for k in KEYS]+["w_n"]
