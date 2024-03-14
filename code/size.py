######original SinGAN from https://github.com/tamarott/SinGAN
import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from skimage import color
from math import pi
import torch
import math


# 调整图像尺度以适应超分辨率任务
def adjust_scales2image_SR(real_, maxs, opt):
    opt.min_size = 18
    opt.num_scales = (
        int(
            (
                math.log(
                    opt.min_size / min(real_.shape[2], real_.shape[3]),
                    opt.scale_factor_init,
                )
            )
        )
        + 1
    )
    scale2stop = int(
        math.log(
            min(opt.max_size, max(real_.shape[2], real_.shape[3]))
            / max(real_.shape[0], real_.shape[3]),
            opt.scale_factor_init,
        )
    )
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(
        opt.max_size / max([real_.shape[2], real_.shape[3]]), 1
    )  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = mimresize(real_, opt.scale1, maxs, opt)
    # opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(
        opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale)
    )
    scale2stop = int(
        math.log(
            min(opt.max_size, max(real_.shape[2], real_.shape[3]))
            / max(real_.shape[0], real_.shape[3]),
            opt.scale_factor_init,
        )
    )
    opt.stop_scale = opt.num_scales - scale2stop
    return real


def mimresize(im, scale, maxs, opt):
    # s = im.shape
    im = mtorch2uint8(im, maxs)
    im = mimresize_in(im, scale_factor=scale)
    im = mnp2torch(im, maxs, opt)
    return im


def mnp2torch(x, maxs, opt):  ###input is (120, 80, 2)
    aaa = []
    if len(maxs) == 1:
        aa = x[:, :, 0] / maxs[0]
        x = aa[None, None, :, :]
    if len(maxs) == 2:
        for i in range(len(maxs)):
            aa = x[:, :, i] / maxs[i]
            aa = aa[None, None, :, :]
            aaa.append(aa)
        x = np.concatenate((aaa[0], aaa[1]), 1)
    if len(maxs) == 3:
        for i in range(len(maxs)):
            aa = x[:, :, i] / maxs[i]
            aa = aa[None, None, :, :]
            aaa.append(aa)
        x = np.concatenate((aaa[0], aaa[1], aaa[2]), 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = (
        x.type(torch.cuda.FloatTensor)
        if not (opt.not_cuda)
        else x.type(torch.FloatTensor)
    )
    x = norm(x)
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def move_to_gpu(t):
    if torch.cuda.is_available():
        t = t.to(torch.device("cuda"))
    return t


def mtorch2uint8(x, maxs):
    aaa = []
    x = denorm(x)
    if len(maxs) == 1:
        x = x[:, 0, :, :] * maxs[0]
    if len(maxs) == 2:
        for i in range(len(maxs)):
            aa = x[:, i, :, :] * maxs[i]
            aaa.append(aa)
        x = torch.cat((aaa[0], aaa[1]), 0)
    if len(maxs) == 3:
        for i in range(len(maxs)):
            aa = x[:, i, :, :] * maxs[i]
            aaa.append(aa)
        x = torch.cat((aaa[0], aaa[1], aaa[2]), 0)
    x = x.permute((1, 2, 0))
    x = x.cpu().numpy()
    # x = x.astype(np.uint8)
    return x


def mimresize_in(
    im,
    scale_factor=None,
    output_shape=None,
    kernel=None,
    antialiasing=True,
    kernel_shift_flag=False,
):
    scale_factor, output_shape = fix_scale_and_size(
        im.shape, output_shape, scale_factor
    )

    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0),
    }.get(kernel)
    antialiasing *= scale_factor[0] < 1
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()
    out_im = np.copy(im)
    for dim in sorted_dims:
        if scale_factor[dim] == 1.0:
            continue
        weights, field_of_view = contributions(
            im.shape[dim],
            output_shape[dim],
            scale_factor[dim],
            method,
            kernel_width,
            antialiasing,
        )
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    if scale_factor is not None:
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(
            input_shape[len(output_shape) :]
        )
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length + 1)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)
    left_boundary = np.floor(match_coordinates - kernel_width / 2)
    expanded_kernel_width = np.ceil(kernel_width) + 2
    field_of_view = np.squeeze(
        np.uint(
            np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1
        )
    )
    weights = fixed_kernel(
        1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1
    )
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)
    mirror = np.uint(
        np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1)))
    )
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])
    return weights, field_of_view


def resize_along_dim(im, dim, weights, field_of_view):
    tmp_im = np.swapaxes(im, dim, 0)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)
    return np.swapaxes(tmp_out_im, dim, 0)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
    return out_im[
        np.round(
            np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])
        ).astype(int)[:, None],
        np.round(
            np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])
        ).astype(int),
        :,
    ]


def kernel_shift(kernel, sf):
    current_center_of_mass = measurements.center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (
        sf - (kernel.shape[0] % 2)
    )
    shift_vec = wanted_center_of_mass - current_center_of_mass
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, "constant")
    return interpolation.shift(kernel, shift_vec)


def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((1 < absx) & (absx <= 2))


def lanczos2(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps)
    ) * (abs(x) < 2)


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps)
    ) * (abs(x) < 3)


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


# 创建图像金字塔
def creat_reals_pyramid(real, reals, maxs, opt):
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = mimresize(real, scale, maxs, opt)
        reals.append(curr_real)
    return reals
