import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取GPM降尺度后的降水数据
gpm_data = xr.open_dataset(
    "/Users/jianghouhong/code/songjhh/depth-learning/downscaling-yao/data/2020-changjiang-month/202001.nc4"
)

# 读取站点观测降水数据
observed_data = pd.read_csv(
    "/Users/jianghouhong/code/songjhh/depth-learning/downscaling-yao/data/station-data/2020-station-month.csv"
)

# 对数据进行匹配和预处理
# 确保数据按时间和空间匹配

# 假设gpm_data和observed_data的时间和空间均匹配

# 从.nc4文件中提取经纬度范围内的降水数据
# 假设您的.nc4文件中有一个名为'precipitation'的变量
gpm_precipitation = gpm_data["precipitation"]

# 从站点观测数据中提取观测值
observed_values = observed_data["value"]
print(observed_values)

# 提取与站点观测数据对应的GPM降水数据
# 这取决于您的站点数据如何与GPM数据相匹配，可以使用最近邻插值等方法
# 假设您已经有了一个函数get_nearest_gpm_data，它接受站点的经纬度，返回最近的GPM降水数据
# model_values = [
#     get_nearest_gpm_data(latitude, longitude)
#     for latitude, longitude in zip(
#         observed_data["latitude"], observed_data["longitude"]
#     )
# ]

# # 计算相关系数
# correlation_coefficient, _ = pearsonr(model_values, observed_values)

# # 计算均方根误差
# rmse = np.sqrt(mean_squared_error(model_values, observed_values))

# # 计算平均绝对误差
# mae = mean_absolute_error(model_values, observed_values)

# print("Correlation Coefficient:", correlation_coefficient)
# print("Root Mean Square Error (RMSE):", rmse)
# print("Mean Absolute Error (MAE):", mae)
