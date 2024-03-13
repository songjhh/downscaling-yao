import os
from netCDF4 import Dataset as ncdataset
import numpy as np

root_dir = os.path.abspath("../data/2020-month/")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        dst = ncdataset(os.path.join(root, file))
        precipitation_var = dst.variables["precipitation"]
        print(precipitation_var)
        print(precipitation_var[0, 0, 0])
        print(precipitation_var[0, 0, 1])
        print(precipitation_var[0, 1, 0])
        print(precipitation_var[0, 1, 1])
        # print(dst.variables["precipitation"])
        # print(np.max(dst.variables["precipitation"]))
        # print(dst.variables["precipitation"].isel(time=0))
        break
