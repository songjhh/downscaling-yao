import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

observed_dir = os.path.abspath("../data/2020-station-month/")
predicted_dir = os.path.abspath("../data/2020-changjiang-month-by-station/")
# pic_dir = os.path.abspath("../analyse/regression-pic/")
pic_dir = os.path.abspath("../data/temp/pic/")

for root, dirs, files in os.walk(observed_dir):
    for file in files:
        # 从npy文件中加载数据
        predicted_precipitation = np.load(os.path.join(predicted_dir, file))
        observed_precipitation = np.load(os.path.join(observed_dir, file))

        # 绘制散点图
        plt.close("all")
        plt.scatter(observed_precipitation, predicted_precipitation, label="Data")

        # 计算回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            observed_precipitation, predicted_precipitation
        )
        line = slope * observed_precipitation + intercept

        # 绘制回归线
        plt.plot(observed_precipitation, line, color="red", label="Regression line")

        # 添加标签和标题
        plt.xlabel("Observed precipitation (mm)")
        plt.ylabel("Predicted precipitation (mm)")
        plt.title("Observed vs Predicted Precipitation")
        plt.legend()

        plt.savefig(pic_dir + "/" + file[:-4] + ".png")
        # plt.show()
