#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :-1] - disp[:, :, 1:])
    grad_disp_y = torch.abs(disp[:, :-1, :] - disp[:, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :-1, :] - img[:, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def regularizer_loss(base):
    white = (base[..., 0:1] + base[..., 1:2] + base[..., 2:3]) / 3.0
    return torch.mean(torch.abs(base - white))

def get_mask(gt_image):
    if gt_image.shape[0]==3:
        return gt_image, None
    mask = gt_image[3:4,...]
    return gt_image[0:3,...], mask

def tv_loss(image):
    """Computes the TV loss for a luma image
    """
    h_tv = torch.abs(image[ :, :-1, :] - image[:, 1:, :])
    w_tv = torch.abs(image[:, :, :-1] - image[:, :, 1:])
    return h_tv.mean() + w_tv.mean()


# =============================================================================
# Robust (Huber) losses and ground-truth albedo prior losses
# =============================================================================

def huber_elementwise(pred, gt, delta=0.1):
    """Element-wise Huber / smooth-L1 (beta = delta).

    Quadratic for |pred-gt| < delta, linear beyond. Normalised so the linear
    region has slope 1, i.e. it matches the magnitude of an L1 loss for large
    residuals (drop-in replacement for l1_loss that is less sensitive to
    outliers near zero).
    """
    diff = pred - gt
    absd = torch.abs(diff)
    return torch.where(absd < delta, 0.5 * diff * diff / delta, absd - 0.5 * delta)


def huber_loss(pred, gt, delta=0.1, mask=None):
    """Mean Huber loss, optionally restricted to a (broadcastable) mask."""
    h = huber_elementwise(pred, gt, delta)
    if mask is None:
        return h.mean()
    m = mask.expand_as(h)
    return (h * m).sum() / m.sum().clamp_min(1.0)


def _foreground_mask(gt, eps=1e-6):
    """1xHxW mask of pixels where the GT buffer is non-zero (object, not bg)."""
    return (gt.sum(dim=0, keepdim=True) > eps).to(gt.dtype)


def albedo_prior_loss(rendered, gt, mode="direct", lambda_dssim=0.4, delta=0.1):
    """GT-supervised albedo loss with selectable invariance to the
    albedo<->light intensity/colour ambiguity.

    Modes:
      * "direct"     : Huber + DSSIM on raw values (pins absolute albedo).
      * "lstsq"      : per-channel least-squares gain aligns the rendered albedo
                       to GT before comparing, so only spatial structure and
                       relative colour are supervised (the global per-channel
                       scale is free, absorbing the envmap intensity ambiguity).
      * "log_chroma" : supervises chromaticity (intensity-invariant colour) plus
                       a scale-invariant log-intensity term (per-channel mean
                       shift removed), the intrinsic-image style decomposition.
    """
    if mode == "lstsq":
        mask = _foreground_mask(gt)
        # Closed-form per-channel gain s_c = <r,g> / <r,r> over foreground.
        num = (rendered * gt * mask).sum(dim=(1, 2))
        den = (rendered * rendered * mask).sum(dim=(1, 2)).clamp_min(1e-8)
        s = (num / den).detach().clamp_min(0.0).view(-1, 1, 1)
        aligned = rendered * s
        data = huber_loss(aligned, gt, delta)
        struct = 1.0 - ssim(aligned, gt)
        return (1.0 - lambda_dssim) * data + lambda_dssim * struct

    if mode == "log_chroma":
        eps = 1e-3
        mask = _foreground_mask(gt)
        # Chromaticity: intensity-invariant colour (per-pixel channel ratios).
        chroma_r = rendered / (rendered.sum(dim=0, keepdim=True) + eps)
        chroma_g = gt / (gt.sum(dim=0, keepdim=True) + eps)
        loss_chroma = huber_loss(chroma_r, chroma_g, delta, mask=mask)
        # Scale-invariant log intensity: remove per-channel mean of the log
        # difference (a global multiplicative gain becomes an additive shift).
        log_diff = (torch.log(rendered.clamp_min(0.0) + eps)
                    - torch.log(gt.clamp_min(0.0) + eps))
        n = mask.sum().clamp_min(1.0)
        mean_shift = (log_diff * mask).sum(dim=(1, 2), keepdim=True) / n
        centered = (log_diff - mean_shift)
        loss_log = huber_loss(centered, torch.zeros_like(centered), delta, mask=mask)
        return loss_chroma + loss_log

    # Default: "direct"
    data = huber_loss(rendered, gt, delta)
    struct = 1.0 - ssim(rendered, gt)
    return (1.0 - lambda_dssim) * data + lambda_dssim * struct

    