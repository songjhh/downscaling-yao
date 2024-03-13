import xarray as xr
import geopandas as gpd
import numpy as np
import os

root_dir = os.path.abspath("../data/2020-changjiang-month/")
station_locations = gpd.read_file(
    os.path.abspath("../data/shp/station/lat_lon_changjiang.shp")
)
save_dir = os.path.abspath("../data/2020-changjiang-month-by-station/")
for root, dirs, files in os.walk(root_dir):
    for file in files:
        nc_data = xr.open_dataset(os.path.join(root, file))
        precipitation_data = nc_data["precipitation"]

        station_precipitation = {}
        for index, station in station_locations.iterrows():
            lat = station["LATITUDE"]
            lon = station["LONGITUDE"]
            lat_index = np.abs(nc_data.lat - lat).argmin()
            lon_index = np.abs(nc_data.lon - lon).argmin()
            precipitation_value = precipitation_data[:, lat_index, lon_index].values
            station_precipitation[station["STATION_CO"]] = precipitation_value[0]
            sorted_items = sorted(station_precipitation.items())
            sorted_dict = {k: v for k, v in sorted_items}

        values_array = np.array(list(sorted_dict.values()))
        np.save(
            save_dir + "/" + file[:-4] + ".npy",
            values_array,
        )
