import xarray as xr
import geopandas as gpd
import numpy as np
import os
import pandas as pd

root_dir = os.path.abspath("../data/2020-changjiang-month/")
station_locations = gpd.read_file(
    os.path.abspath("../data/shp/station/lat_lon_changjiang.shp")
)
save_dir = os.path.abspath("../data/2020-changjiang-month-by-station/")
station_dir = os.path.abspath("../data/2020-station-month/")

station_data = pd.read_csv(
    os.path.abspath("../data/station-data/2020china_2400stations_precipitation.csv")
)

for root, dirs, files in os.walk(root_dir):
    for file in files:
        nc_data = xr.open_dataset(os.path.join(root, file))
        precipitation_data = nc_data["precipitation"]

        station_precipitation = {}
        for index, station in station_locations.iterrows():
            if station["STATION_CO"] not in station_data["station_code"].values:
                continue
            lat = station["LATITUDE"]
            lon = station["LONGITUDE"]
            lat_index = np.abs(nc_data.lat - lat).argmin()
            lon_index = np.abs(nc_data.lon - lon).argmin()
            precipitation_value = precipitation_data[:, lat_index, lon_index].values
            station_precipitation[station["STATION_CO"]] = precipitation_value[0]
            sorted_items = sorted(station_precipitation.items())
            sorted_dict = {k: v for k, v in sorted_items}

        values_array = np.array(list(sorted_dict.values()))
        print(len(values_array))
        np.save(
            save_dir + "/" + file[:-4] + ".npy",
            values_array,
        )
