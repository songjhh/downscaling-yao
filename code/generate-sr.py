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


if __name__ == "__main__":
    # 处理配置
    parser = config.get_arguments()
    parser.add_argument("--model_dir", default="")
    parser.add_argument(
        "--sr_factor", help="super resolution factor", type=float, default=4
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

    # path = Path("/content/testing-set")
    path = Path(
        "/Users/jianghouhong/code/songjhh/depth-learning/downscaling-yao/data/2020-month"
    )
    # saveData = "/content/drive/MyDrive/code/SinGAN-songjhh"
    saveData = (
        "/Users/jianghouhong/code/songjhh/depth-learning/downscaling-yao/data/result/singan-0101-x100"
    )
    # savePic = "/content"
    savePic = (
        "/Users/jianghouhong/code/songjhh/depth-learning/downscaling-yao/data/result/singan-0101-x100"
    )
    # compareSet = "/Users/jianghouhong/code/songjhh/depth-learning/SinGAN-songjhh/data/compare-set"
    for testing_file in path.rglob("*.nc4"):
        # save_name = str(testing_file)[-12:-4]
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

        # compare = ncdataset(compareSet + "/%s.nc4" % (save_name)).variables["precipitation"]
        # compare = np.squeeze(compare)

        target_torch = torch.from_numpy(target)
        target_torch = target_torch / np.max(target)
        target_torch = functions.norm(target_torch)
        target_torch = target_torch[None, None, :, :]

        ud = size.mimresize(target_torch, 1, maxsd, opt)

        # resampled = size.mimresize_in(target, scale_factor=1 / 4)
        # sns.set()
        # plt.figure(figsize=(32, 12))
        # ax = sns.heatmap(
        #     resampled,
        #     vmin=0,
        #     yticklabels=False,
        #     xticklabels=False,
        #     vmax=np.max(resampled),
        # )
        # plt.title("resampled data")
        # plt.savefig(savePic + "/pic/resampled-" + save_name + ".png")
        # # print(len(resampled))
        # # print(len(resampled[0]))
        # np.savetxt(
        #     saveData + "/Resampled/%s.txt" % save_name,
        #     resampled,
        # )

        # reals = []
        real = size.adjust_scales2image_SR(ud, maxsd, opt)
        real_ = real
        reals = size.creat_reals_pyramid(real_, reals, maxsd, opt)
        real = reals[-1]  # read_image(opt)
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

        dir2save = functions.generate_dir2save(opt)
        # plt.imsave(
        #     "%s/%s_HR.png" % (dir2save, opt.input_name),
        #     functions.convert_image_np(out.detach()),
        #     vmin=0,
        #     vmax=1,
        # )

        outt = size.denorm(out)
        inp = outt[-1, -1, :, :].to(torch.device("cpu"))
        inp = inp.numpy()
        inpp = inp * maxsd

        # np.savetxt(
        #     "/content/drive/MyDrive/code/SinGAN-songjhh/Results/%s.txt" % save_name,
        #     inpp,
        # )
        np.savetxt(
            saveData + "/Results/%s.txt" % save_name,
            inpp,
        )
        np.savetxt(
            saveData + "/Original/%s.txt" % save_name,
            target,
        )
        # np.savetxt(
        #     saveData + "/Compare/%s.txt" % save_name,
        #     compare,
        # )

        sns.set()
        plt.close("all")
        plt.figure(figsize=(32, 12))
        ax = sns.heatmap(
            inpp[::-1], vmin=0, yticklabels=False, xticklabels=False, vmax=np.max(inpp)
        )
        inpp.transpose("time", "lat", "lon").plot()
        print(len(inpp))
        print(len(inpp[0]))
        plt.title("Result from model")
        plt.savefig(savePic + "/pic/result-" + save_name + ".png")

        sns.set()
        plt.figure(figsize=(32, 12))
        ax = sns.heatmap(
            target[::-1], vmin=0, yticklabels=False, xticklabels=False, vmax=np.max(target)
        )
        # print(len(target))
        # print(len(target[0]))
        plt.title("Original data")
        plt.savefig(savePic + "/pic/original-" + save_name + ".png")
        break
        # sns.set()
        # plt.figure(figsize=(14, 12))
        # ax = sns.heatmap(
        #     compare, vmin=0, yticklabels=False, xticklabels=False, vmax=np.max(compare)
        # )
        # print(len(compare))
        # print(len(compare[0]))
        # plt.title("Compare data")
        # plt.savefig(savePic + "/pic/compare-" + save_name + ".png")
