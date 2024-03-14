import math
import torch
import os
import numpy as np
import torch.nn as nn


# 计算初始尺度
def calc_init_scale(opt):
    in_scale = math.pow(1 / 2, 1 / 3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale, iter_num


# 加载已训练模型
def load_trained_pyramid(opt):
    # dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = "train"
    if (mode == "animation_train") | (mode == "SR_train") | (mode == "paint_train"):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if os.path.exists(dir):
        Gs = torch.load("%s/Gs.pth" % dir)
        Zs = torch.load("%s/Zs.pth" % dir)
        reals = torch.load("%s/reals.pth" % dir)
        NoiseAmp = torch.load("%s/NoiseAmp.pth" % dir)
    else:
        print("no appropriate trained model is exist, please train first")
    opt.mode = mode
    return Gs, Zs, reals, NoiseAmp


# 反归一化
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# 归一化
def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


# 转换图像
def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp, 0, 1)
    return inp


# 生成噪声
def generate_noise(size, num_samp=1, device="cuda", type="gaussian", scale=1):
    if type == "gaussian":
        noise = torch.randn(
            num_samp,
            size[0],
            round(size[1] / scale),
            round(size[2] / scale),
            device=device,
        )
        noise = upsampling(noise, size[1], size[2])
    if type == "gaussian_mixture":
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == "uniform":
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


# 上采样
def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode="bilinear", align_corners=True)
    return m(im)


# 计算梯度
def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(
            device
        ),  # .cuda(), #if use_cuda else torch.ones(
        # disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# 保存网络
def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), "%s/netG.pth" % (opt.outf))
    torch.save(netD.state_dict(), "%s/netD.pth" % (opt.outf))
    torch.save(z, "%s/z_opt.pth" % (opt.outf))


# 重置梯度
def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


# 从cpu加载
def move_to_cpu(t):
    t = t.to(torch.device("cpu"))
    return t


def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == "train") | (opt.mode == "SR_train"):
        dir2save = "TrainedModels/%s/scale_factor=%f,alpha=%d" % (
            opt.input_name[:-4],
            opt.scale_factor_init,
            opt.alpha,
        )
    elif opt.mode == "animation_train":
        dir2save = "TrainedModels/%s/scale_factor=%f_noise_padding" % (
            opt.input_name[:-4],
            opt.scale_factor_init,
        )
    elif opt.mode == "paint_train":
        dir2save = "TrainedModels/%s/scale_factor=%f_paint/start_scale=%d" % (
            opt.input_name[:-4],
            opt.scale_factor_init,
            opt.paint_start_scale,
        )
    elif opt.mode == "random_samples":
        dir2save = "%s/RandomSamples/%s/gen_start_scale=%d" % (
            opt.out,
            opt.input_name[:-4],
            opt.gen_start_scale,
        )
    elif opt.mode == "random_samples_arbitrary_sizes":
        dir2save = "%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f" % (
            opt.out,
            opt.input_name[:-4],
            opt.scale_v,
            opt.scale_h,
        )
    elif opt.mode == "animation":
        dir2save = "%s/Animation/%s" % (opt.out, opt.input_name[:-4])
    elif opt.mode == "SR":
        dir2save = "%s/SR/%s" % (opt.out, opt.sr_factor)
    elif opt.mode == "harmonization":
        dir2save = "%s/Harmonization/%s/%s_out" % (
            opt.out,
            opt.input_name[:-4],
            opt.ref_name[:-4],
        )
    elif opt.mode == "editing":
        dir2save = "%s/Editing/%s/%s_out" % (
            opt.out,
            opt.input_name[:-4],
            opt.ref_name[:-4],
        )
    elif opt.mode == "paint2image":
        dir2save = "%s/Paint2image/%s/%s_out" % (
            opt.out,
            opt.input_name[:-4],
            opt.ref_name[:-4],
        )
        if opt.quantization_flag:
            dir2save = "%s_quantized" % dir2save
    return dir2save
