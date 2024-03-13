import pandas as pd
import os

# 读取原始 CSV 文件
station_file = os.path.abspath(
    "../data/station-data/2020china_2400stations_precipitation.csv"
)
result_file = os.path.abspath("../data/station-data/2020-station-month.csv")
df = pd.read_csv(station_file)

# 将 year、month 和 day 合并为一个日期列
df["date"] = pd.to_datetime(df[["year", "month", "day"]])

# 按站点代码和月份分组，计算每个组的 value 总和
result = (
    df.groupby(["station_code", "longitude", "latitude", df["date"].dt.month])["value"]
    .sum()
    .reset_index()
)

# 保存结果到新的 CSV 文件
result.to_csv(result_file, index=False)
