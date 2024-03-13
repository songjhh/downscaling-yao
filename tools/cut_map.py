import xarray
import geopandas as gpd
from shapely.geometry import mapping
import os
import matplotlib.pyplot as plt
import rioxarray

root_dir = os.path.abspath("../data/2020-month/")
changjiang_shp_path = os.path.abspath("../data/shp/boundary/changjiang.shp")
result_dir = os.path.abspath("../data/2020-changjiang-month/")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        xds = xarray.open_dataset(os.path.join(root, file))
        xds = xds[["precipitation"]].transpose("time", "lat", "lon")
        xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        xds.rio.write_crs("EPSG:4326", inplace=True)

        geodf = gpd.read_file(r"%s" % changjiang_shp_path)
        clipped = xds.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
        clipped.to_netcdf(result_dir + "/" + file, mode="w", format="NETCDF4")
        clipped_ds = xarray.open_dataset(result_dir + "/" + file, decode_times=False)
        clipped_ds.attrs = xds.attrs
        clipped_ds.close()
