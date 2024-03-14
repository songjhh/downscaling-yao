import argparse
import random
import torch

# 获取配置
def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', help='task to be done', default='train')
    # workspace:
    parser.add_argument(
        "--not_cuda", action="store_true", help="disables cuda", default=0
    )

    # load, input, save configurations:
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument("--manualSeed", type=int, help="manual seed")
    parser.add_argument("--nc_z", type=int, help="noise # channels", default=3)
    parser.add_argument("--nc_im", type=int, help="image # channels", default=3)
    parser.add_argument("--out", help="output folder", default="Output")

    # networks hyper parameters:
    parser.add_argument("--nfc", type=int, default=32)
    parser.add_argument("--min_nfc", type=int, default=32)
    parser.add_argument("--ker_size", type=int, help="kernel size", default=3)
    parser.add_argument("--num_layer", type=int, help="number of layers", default=5)
    parser.add_argument("--stride", help="stride", default=1)
    parser.add_argument(
        "--padd_size", type=int, help="net pad size", default=0
    )  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument(
        "--scale_factor", type=float, help="pyramid scale factor", default=0.75
    )  # pow(0.5,1/6))
    parser.add_argument(
        "--noise_amp", type=float, help="addative noise cont weight", default=0.1
    )
    parser.add_argument(
        "--min_size",
        type=int,
        help="image minimal size at the coarser scale",
        default=25,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        help="image minimal size at the coarser scale",
        default=250,
    )

    # optimization hyper parameters:
    parser.add_argument(
        "--niter", type=int, default=2000, help="number of epochs to train per scale"
    )
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.1)
    parser.add_argument(
        "--lr_g", type=float, default=0.0005, help="learning rate, default=0.0005"
    )
    parser.add_argument(
        "--lr_d", type=float, default=0.0005, help="learning rate, default=0.0005"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--Gsteps", type=int, help="Generator inner steps", default=3)
    parser.add_argument(
        "--Dsteps", type=int, help="Discriminator inner steps", default=3
    )
    parser.add_argument(
        "--lambda_grad", type=float, help="gradient penelty weight", default=0.1
    )
    parser.add_argument(
        "--alpha", type=float, help="reconstruction loss weight", default=10
    )

    return parser

# 初始化固定参数
def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = "TrainedModels/%s/scale_factor=%f/" % (
        opt.input_name,
        opt.scale_factor,
    )
    if opt.mode == "SR":
        opt.alpha = 100
    elif opt.mode == "animation":
        opt.num_samples = 1
    else:
        opt.num_samples = opt.num_samples

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # if torch.cuda.is_available() and opt.not_cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt
