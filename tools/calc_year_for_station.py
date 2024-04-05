import pandas as pd
import os
import geopandas as gpd
import numpy as np

result_dir = os.path.abspath("../data/2020-station-month/")
df = pd.read_csv(
    os.path.abspath("../data/station-data/2020china_2400stations_precipitation.csv")
)
station_locations = gpd.read_file(
    os.path.abspath("../data/shp/station/lat_lon_changjiang.shp")
)

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
arr1 = []
for month in months:
    if month < 10:
        monthStr = "0" + str(month)
    else:
        monthStr = str(month)
    data = np.load(result_dir + "/2020" + monthStr + ".npy")
    if (len(arr1) == 0):
        arr1 = data
    else:
        arr1 = arr1 + data
print(len(arr1))
np.save(
    result_dir + "/2020.npy",
    arr1,
)