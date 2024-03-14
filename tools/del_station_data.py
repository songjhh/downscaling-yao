import pandas as pd
import os
import geopandas as gpd
import numpy as np

station_file = os.path.abspath(
    "../data/station-data/2020china_2400stations_precipitation.csv"
)
station_data = pd.read_csv(station_file)
result = station_data.groupby(["station_code"])["value"].count().reset_index()
result = result[result["value"] != 366]
print(len(result))

# print(len(station_data))
# filtered_station_data = station_data[
#     (~station_data["station_code"].isin(result["station_code"]))
#     & (station_data["value"] >= 0)
#     & (station_data["station_code"] != 58255)
# ]

# print(len(filtered_station_data))

# filtered_station_data = station_data[
#     (station_data["station_code"] != 1) & (station_data["value"] >= 0)
# ]
# filtered_station_data.to_csv(station_file, index=False)
