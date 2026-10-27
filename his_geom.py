import sys, numpy as np, xarray as xr
ds = xr.open_dataset(sys.argv[1])
x = ds["cross_section_geom_node_coordx"].values; y = ds["cross_section_geom_node_coordy"].values
cnt = ds["cross_section_geom_node_count"].values.astype(int)
names = ds["cross_section_name"].values if "cross_section_name" in ds else [None]*len(cnt)
s = 0
for i, c in enumerate(cnt):
    xs, ys = x[s:s+c], y[s:s+c]; s += c
    km = xs.mean()/1000
    if 19 <= km <= 46:
        n = names[i].decode().strip() if isinstance(names[i], bytes) else str(names[i]).strip()
        print(f"{i:3d} {n:20s} n={c:3d} x {xs.min():9.1f}-{xs.max():9.1f}  y {ys.min():8.1f}-{ys.max():8.1f}")
for v in ds.data_vars:
    if v.startswith("cross_section") and "time" in ds[v].dims: print(v, ds[v].attrs.get("units"), ds[v].attrs.get("long_name"))
