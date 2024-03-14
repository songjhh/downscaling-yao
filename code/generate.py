import torch
import torch.nn as nn
import functions
import size
import os
import matplotlib.pyplot as plt


def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == "train") | (opt.mode == "SR_train"):
        dir2save = "TrainedModels/%s/scale_factor=%f,alpha=%d" % (
            opt.input_name,
            opt.scale_factor_init,
            opt.alpha,
        )
    elif opt.mode == "SR":
        dir2save = "%s/SR/%s" % (opt.out, opt.sr_factor)

    return dir2save


def SinGAN_generate(
    Gs,
    Zs,
    reals,
    NoiseAmp,
    opt,
    maxsd,
    in_s=None,
    scale_v=1,
    scale_h=1,
    n=0,
    gen_start_scale=0,
):
    #####z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
    # if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0, opt.num_samples, 1):
            if n == 0:
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, 1, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise(
                    [opt.nc_z, nzx, nzy], device=opt.device
                )
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                # I_prev = m(I_prev)
                # I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                # I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = size.mimresize(I_prev, 1 / opt.scale_factor, maxsd, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[
                        :,
                        :,
                        0 : round(scale_v * reals[n].shape[2]),
                        0 : round(scale_h * reals[n].shape[3]),
                    ]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:, :, 0 : z_curr.shape[2], 0 : z_curr.shape[3]]
                    I_prev = functions.upsampling(
                        I_prev, z_curr.shape[2], z_curr.shape[3]
                    )
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp * (z_curr) + I_prev
            I_curr = G(z_in.detach(), I_prev)

            if n == len(reals) - 1:
                if opt.mode == "train":
                    dir2save = "%s/RandomSamples/%s/gen_start_scale=%d" % (
                        opt.out,
                        opt.input_name,
                        gen_start_scale,
                    )
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (
                    (opt.mode != "harmonization")
                    & (opt.mode != "editing")
                    & (opt.mode != "SR")
                    & (opt.mode != "paint2image")
                ):
                    plt.imsave(
                        "%s/%d.png" % (dir2save, i),
                        functions.convert_image_np(I_curr.detach()),
                        vmin=0,
                        vmax=1,
                    )
                    # plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    # plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n += 1
    return I_curr.detach()
