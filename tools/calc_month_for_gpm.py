import xarray as xr
import dask.array as da
import os
from netCDF4 import Dataset as ncdataset

monthday = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

root_dir = os.path.abspath("../data/original-data/")
result_dir = os.path.abspath("../data/2020-month/")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        ds = xr.open_dataset(os.path.join(root, file), group="Grid")
        ds_subset = ds["precipitation"]
        # 获取文件名
        new_file_name = "/2020" + str(file)[-12:-10] + ".nc4"
        # 计算每个月的总小时数
        total_hours = monthday[int(file[-12:-10]) - 1] * 24
        # 求平均降水
        # ds_subset_avg = ds_subset.resample(time="1M").mean()
        # 插值到新的经纬度范围
        ds_resampled_x1 = ds_subset.interp(
            lat=da.arange(24, 36, 0.1), lon=da.arange(90, 122, 0.1)
        )
        # 将小时转换为月
        ds_resampled_x1 *= total_hours
        ds_resampled_x1.attrs["units"] = "mm/month"
        ds_resampled_x1.attrs["Units"] = "mm/month"

        ds_resampled_x1.to_netcdf(result_dir + new_file_name)
