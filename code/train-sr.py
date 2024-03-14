import config
import generate
import functions
import training
import size
from pathlib import Path
import os
from netCDF4 import Dataset as ncdataset
import numpy as np
import torch

if __name__ == "__main__":
    # 处理配置
    parser = config.get_arguments()
    parser.add_argument(
        "--sr_factor", help="super resolution factor", type=float, default=4
    )
    parser.add_argument("--mode", help="task to be done", default="SR")
    parser.set_defaults(nc_im=1)
    parser.set_defaults(nc_z=1)
    parser.set_defaults(alpha=30)

    # 读取文件
    path = Path("/content/training-set")
    for training_file in path.rglob("*.nc4"):
        save_name = str(training_file)[-12:-4]
        parser.set_defaults(input_name="%s" % (save_name))
        opt = parser.parse_args()
        opt = config.post_config(opt)

        Gs = []
        Zs = []
        reals = []
        NoiseAmp = []

        dir2save = generate.generate_dir2save(opt)
        if dir2save is None:
            print("task does not exist")
        else:
            try:
                os.makedirs(dir2save)
            except OSError:
                pass

        mode = opt.mode
        in_scale, iter_num = functions.calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        opt.mode = "train"

        dir2trained_model = generate.generate_dir2save(opt)
        if os.path.exists(dir2trained_model):
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = mode
        else:
            print("%f" % pow(in_scale, iter_num))
            print("*** Train SinGAN for SR from " + str(training_file) + "***")

            dst = ncdataset(training_file)
            target = dst.variables["precipitation"]
            target = np.squeeze(target)
            maxsd = [np.max(target)]
            target_torch = torch.from_numpy(target)
            target_torch = target_torch / np.max(target)
            target_torch = functions.norm(target_torch)
            target_torch = target_torch[None, None, :, :]
            ud = size.mimresize(
                target_torch, 1, maxsd, opt
            )  # 1/4 of original as input
            opt.min_size = 10
            opt.max_size = 700
            real = size.adjust_scales2image_SR(ud, maxsd, opt)
            real_ = real
            reals = size.creat_reals_pyramid(real_, reals, maxsd, opt)
            training.train(opt, Gs, Zs, reals, NoiseAmp, maxsd)
            opt.mode = mode
