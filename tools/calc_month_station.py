import pandas as pd
import os
import geopandas as gpd
import numpy as np

# 读取原始 CSV 文件
station_file = os.path.abspath(
    "../data/station-data/2020china_2400stations_precipitation.csv"
)
result_dir = os.path.abspath("../data/2020-statin-month/")
df = pd.read_csv(station_file)
station_locations = gpd.read_file(
    os.path.abspath("../data/shp/station/lat_lon_changjiang.shp")
)

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


for month in months:
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    filtered_df = df[df["date"].dt.month == 10]

    ids = set(station_locations["STATION_CO"])
    filtered_df = filtered_df[filtered_df["station_code"].isin(ids)]

    result = (
        filtered_df.groupby(["station_code", filtered_df["date"].dt.month])["value"]
        .sum()
        .reset_index()
    )
    result_sorted = result.sort_values(by='station_code')
    values_array = result_sorted['value'].to_numpy()

    if month < 10:
        monthStr = "0" + str(month)
    else:
        monthStr = str(month)
    np.save(
        result_dir + "/2020" + monthStr + ".npy",
        values_array,
    )
