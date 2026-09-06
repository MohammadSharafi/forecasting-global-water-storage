"""Great-circle neighbourhood mean of the last-observed TWS field ("smoothed anchor").
Permitted per the 24 Aug organiser ruling: neighbouring cells' TWS at months <= t, including
spatial filtering and aggregation. Coordinates are used only to define the neighbourhood.
Longitude window scales with 1/cos(lat) so the neighbourhood is a physical radius, not a box."""
import numpy as np, polars as pl
from scipy.ndimage import uniform_filter

def anchor_fields(anchor_months, obs, radii_km=(300,500,800)):
    """obs: lat, lon, time, TWS_t (observed only). Returns {km: (T,180,360) array}, index map."""
    tks=sorted(anchor_months); ti={t:i for i,t in enumerate(tks)}
    A=np.full((len(tks),180,360),np.nan,np.float32)
    of=obs.filter(pl.col("time").is_in(tks))
    A[[ti[t] for t in of["time"].to_list()],(of["lat"].to_numpy()+89.5).round().astype(int),(of["lon"].to_numpy()+179.5).round().astype(int)]=of["TWS_t"].to_numpy()
    M=~np.isnan(A); A0=np.where(M,A,0).astype(np.float32); Mf=M.astype(np.float32)
    lat_c=np.arange(180)-89.5; out={}
    for km in radii_km:
        S=np.zeros_like(A0); C=np.zeros_like(A0)
        for i in range(180):
            dlat=max(1,int(round(km/111.0))); dlon=max(1,int(round(km/(111.0*max(np.cos(np.radians(lat_c[i])),0.1)))))
            i0,i1=max(0,i-dlat),min(180,i+dlat+1); n=i1-i0; w=2*dlon+1
            sb=uniform_filter(A0[:,i0:i1,:],size=(1,n,w),mode=("constant","constant","wrap"))*n*w
            cb=uniform_filter(Mf[:,i0:i1,:],size=(1,n,w),mode=("constant","constant","wrap"))*n*w
            S[:,i,:]=sb[:,n//2,:]; C[:,i,:]=cb[:,n//2,:]
        out[km]=np.where(C>0.5,S/np.maximum(C,1e-6),np.nan)
    return out, ti

def sample(fields, ti, rows):
    """rows: polars frame with lat, lon, t_known. Returns {km: values aligned to rows}."""
    i=(rows["lat"].to_numpy()+89.5).round().astype(int); j=(rows["lon"].to_numpy()+179.5).round().astype(int)
    p=np.array([ti[t] for t in rows["t_known"].to_list()])
    return {km:F[p,i,j] for km,F in fields.items()}
