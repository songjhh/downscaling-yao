import os
from netCDF4 import Dataset as ncdataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray

root_dir = os.path.abspath("../data/2020-changjiang-month/")
# root_dir = os.path.abspath("../data/2020-month/")
pic_dir = os.path.abspath("../data/temp/")

for root, dirs, files in os.walk(root_dir):
    for file in files:

        plt.close("all")
        xds = xarray.open_dataset(os.path.join(root, file))
        target = xds["precipitation"]
        plt.figure(figsize=(32, 12))
        target.transpose("time", "lat", "lon").plot()
        # plt.show()
        plt.savefig(pic_dir + "/" + file[:-4] + ".png")

        # dst = ncdataset(os.path.join(root, file))
        # target = dst.variables["precipitation"]
        # target = np.squeeze(target)
        # target = np.flipud(target)
        # sns.set()
        # plt.close("all")
        # plt.figure(figsize=(32, 12))
        # ax = sns.heatmap(
        #     target, vmin=0, yticklabels=False, xticklabels=False, vmax=np.max(target)
        # )
        # plt.show()
        # plt.savefig(pic_dir + "/" + file[:-4] + ".png")
