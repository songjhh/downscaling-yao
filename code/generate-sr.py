import torch
import functions
import size
import generate
import argparse
import config
import torch.nn as nn
from pathlib import Path
from netCDF4 import Dataset as ncdataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_nc4(save_path, save_name, extracted_data, time):
    new_lat_range = (24, 36)
    new_lon_range = (90, 122)
    lon_shape = extracted_data.shape[0]
    lat_shape = extracted_data.shape[1]

    # 创建新的NetCDF文件
    new_dataset = ncdataset(
        os.path.join(save_path, save_name + ".nc4"), "w", format="NETCDF4"
    )

    # 创建经纬度维度
    new_dataset.createDimension("lat", lat_shape)
    new_dataset.createDimension("lon", lon_shape)

    # 创建经纬度变量
    latitudes = new_dataset.createVariable("lat", np.float32, ("lat",))
    longitudes = new_dataset.createVariable("lon", np.float32, ("lon",))

    # 创建目标变量
    new_variable = new_dataset.createVariable(
        "precipitation",
        np.float32,
        (
            "lon",
            "lat",
        ),
    )

    # 写入经纬度数据
    latitudes[:] = np.linspace(new_lat_range[0], new_lat_range[1], lat_shape)
    longitudes[:] = np.linspace(new_lon_range[0], new_lon_range[1], lon_shape)

    # 写入目标数据
    new_variable[:] = extracted_data

    # 关闭文件
    new_dataset.close()


if __name__ == "__main__":
    # torch.nn.Module.dump_patches = True
    # 处理配置
    parser = config.get_arguments()
    parser.add_argument("--model_dir", default="")
    parser.add_argument(
        "--sr_factor", help="super resolution factor", type=float, default=10
    )
    parser.add_argument("--mode", help="task to be done", default="SR")
    parser.set_defaults(nc_im=1)
    parser.set_defaults(nc_z=1)
    parser.set_defaults(alpha=30)

    opt = parser.parse_args()
    if opt.not_cuda:
        Gs = torch.load(opt.model_dir + "/Gs.pth", map_location=torch.device("cpu"))
        Zs = torch.load(opt.model_dir + "/Zs.pth", map_location=torch.device("cpu"))
        reals = torch.load(
            opt.model_dir + "/reals.pth", map_location=torch.device("cpu")
        )
        NoiseAmp = torch.load(
            opt.model_dir + "/NoiseAmp.pth", map_location=torch.device("cpu")
        )
    else:
        Gs = torch.load(opt.model_dir + "/Gs.pth")
        Zs = torch.load(opt.model_dir + "/Zs.pth")
        reals = torch.load(opt.model_dir + "/reals.pth")
        NoiseAmp = torch.load(opt.model_dir + "/NoiseAmp.pth")

    root_dir = os.path.abspath("../data/2020-month/")
    result_save_path = os.path.abspath("../data/result/singan-202001-x10/result/")
    original_save_path = os.path.abspath("../data/result/singan-202001-x10/original/")
    for testing_file in Path(root_dir).rglob("*.nc4"):
        save_name = str(testing_file)[-10:-4]
        parser.set_defaults(input_name="%s" % (save_name))
        opt = parser.parse_args()
        opt = config.post_config(opt)
        Zs_sr = []
        reals_sr = []
        NoiseAmp_sr = []
        Gs_sr = []
        reals = []

        dst = ncdataset(testing_file)
        target = dst.variables["precipitation"]
        target = np.squeeze(target)
        maxsd = [np.max(target)]

        target_torch = torch.from_numpy(target)
        target_torch = target_torch / np.max(target)
        target_torch = functions.norm(target_torch)
        target_torch = target_torch[None, None, :, :]

        ud = size.mimresize(target_torch, 1, maxsd, opt)

        real = size.adjust_scales2image_SR(ud, maxsd, opt)
        real_ = real
        reals = size.creat_reals_pyramid(real_, reals, maxsd, opt)
        real = reals[-1]
        real_ = real

        in_scale, iter_num = functions.calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale

        for j in range(1, iter_num + 1, 1):
            real_ = size.mimresize(real_, pow(1 / opt.scale_factor, 1), maxsd, opt)
            reals_sr.append(real_)
            Gs_sr.append(Gs[-1])
            NoiseAmp_sr.append(NoiseAmp[-1])
            z_opt = torch.full(real_.shape, 0, device=opt.device)
            m = nn.ZeroPad2d(5)
            z_opt = m(z_opt)
            Zs_sr.append(z_opt)

        opt.num_samples = 1
        out = generate.SinGAN_generate(
            Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, maxsd, in_s=reals_sr[0]
        )
        out = out[
            :,
            :,
            0 : int(opt.sr_factor * reals[-1].shape[2]),
            0 : int(opt.sr_factor * reals[-1].shape[3]),
        ]

        outt = size.denorm(out)
        inp = outt[-1, -1, :, :].to(torch.device("cpu"))
        inp = inp.numpy()
        inpp = inp * maxsd

        create_nc4(
            original_save_path,
            save_name,
            target,
            dst.variables["time"],
        )
        create_nc4(result_save_path, save_name, inpp, dst.variables["time"])
