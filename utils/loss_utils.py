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


def _ssim_structure_loss(img1, img2, window_size=11, lum_weight=0.1):
    """Structure/contrast-focused SSIM loss.

    The standard SSIM index factorises into luminance (l), contrast (c) and
    structure (s) terms; the usual implementation groups contrast*structure
    into a single `cs` map. Here we keep the full contrast*structure term but
    raise the luminance term to a small exponent `lum_weight` (< 1) so that
    differences in absolute brightness barely affect the loss, while shape /
    texture (structure) and local contrast are matched at full strength.

    Returns 1 - mean(l**lum_weight * cs), so 0 is a perfect structural match.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

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

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)          # in (0, 1]
    contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    ssim_struct = luminance.clamp_min(1e-6).pow(lum_weight) * contrast_structure
    return 1.0 - ssim_struct.mean()


def albedo_prior_loss(rendered, gt, mode="direct", lambda_dssim=0.4, delta=0.1):
    """GT-supervised albedo loss with selectable invariance to the
    albedo<->light intensity/colour ambiguity.

    Modes:
      * "direct"      : Huber + DSSIM on raw values (pins absolute albedo).
      * "lstsq"       : per-channel least-squares gain aligns the rendered albedo
                        to GT before comparing, so only spatial structure and
                        relative colour are supervised (the global per-channel
                        scale is free, absorbing the envmap intensity ambiguity).
      * "log_chroma"  : supervises chromaticity (intensity-invariant colour) plus
                        a scale-invariant log-intensity term (per-channel mean
                        shift removed), the intrinsic-image style decomposition.
      * "gradient"    : spatial-gradient (edge) loss. Matches the rendered albedo
                        gradient to the GT albedo gradient along X and Y with an
                        L1 penalty,  L = |dx_pred - dx_gt| + |dy_pred - dy_gt|.
                        Ignores any global brightness/colour shift between the two
                        BRDF implementations and only forces texture boundaries /
                        edges to land in the same places as the GT.
      * "zncc"        : scale-and-shift-invariant loss (ZNCC / Pearson). Each of
                        the rendered and GT albedo is standardised per channel
                        over the foreground (subtract mean, divide by std) before
                        comparison, so the rendered albedo may be proportionally
                        brighter / darker or have different contrast as long as
                        the relative distribution of values matches.
      * "zncc_local"  : locally-normalised cross-correlation. Like "zncc" but the
                        per-channel mean/std are computed inside a sliding
                        Gaussian window, so the loss is invariant to a
                        SPATIALLY-VARYING gain. Smooth baked shading (soft
                        shadows / brightness gradients) is ignored while local
                        texture must still agree — better albedo<->light
                        decoupling for relighting.
      * "zncc_grad"   : ZNCC computed in the spatial-gradient domain. The albedo
                        is differentiated first (removing low-frequency baked
                        shading), then the X / Y gradient channels are
                        standardised over the foreground and their global
                        correlation is maximised, supervising high-frequency
                        texture edges only.
      * "ssim_struct" : SSIM on the albedo with the luminance term heavily
                        down-weighted, so shapes / textures must match the GT but
                        absolute brightness / contrast may drift.
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

    if mode == "gradient":
        # Spatial-gradient (edge) loss: L1 between the rendered and GT albedo
        # gradients along X and Y. Penalises the *rate of change* between
        # neighbouring pixels, not the absolute colour, so a global brightness
        # or energy-conservation shift in the GT is ignored while texture
        # boundaries / edges are forced to land in the same places.
        mask = _foreground_mask(gt)
        # Finite-difference image gradients (signed).
        r_dx = rendered[:, :, 1:] - rendered[:, :, :-1]
        g_dx = gt[:, :, 1:] - gt[:, :, :-1]
        r_dy = rendered[:, 1:, :] - rendered[:, :-1, :]
        g_dy = gt[:, 1:, :] - gt[:, :-1, :]
        # Valid only where BOTH neighbouring pixels are foreground (so the
        # object<->background silhouette step is not counted as a GT edge).
        mask_dx = mask[:, :, 1:] * mask[:, :, :-1]
        mask_dy = mask[:, 1:, :] * mask[:, :-1, :]
        # L1 on the gradient difference (Charbonnier-smoothed via Huber).
        loss_dx = huber_loss(r_dx, g_dx, delta, mask=mask_dx)
        loss_dy = huber_loss(r_dy, g_dy, delta, mask=mask_dy)
        return loss_dx + loss_dy

    if mode == "zncc":
        # Scale-and-shift-invariant (ZNCC / Pearson) loss. Standardise each of
        # the rendered and GT albedo per channel over the foreground, then
        # compare. Invariant to a per-channel linear transform (gain + bias).
        eps = 1e-4
        mask = _foreground_mask(gt)
        n = mask.sum().clamp_min(1.0)

        def _standardize(x):
            mean = (x * mask).sum(dim=(1, 2), keepdim=True) / n
            xc = (x - mean) * mask
            var = (xc * xc).sum(dim=(1, 2), keepdim=True) / n
            return xc / torch.sqrt(var + eps)

        rp = _standardize(rendered)
        gp = _standardize(gt)
        return huber_loss(rp, gp, delta, mask=mask)

    if mode == "zncc_local":
        # Locally normalised cross-correlation. Standardise the rendered and GT
        # albedo inside a sliding Gaussian window (per channel), then maximise
        # the local correlation. Invariant to a SPATIALLY-VARYING per-channel
        # gain, so smooth baked shading (soft shadows / brightness gradients) is
        # ignored while local texture must agree.
        eps = 1e-4
        window_size = 11
        channel = rendered.size(-3)
        window = create_window(window_size, channel)
        if rendered.is_cuda:
            window = window.cuda(rendered.get_device())
        window = window.type_as(rendered)

        mask = _foreground_mask(gt)
        r = rendered * mask
        g = gt * mask

        def _conv(x):
            return F.conv2d(x, window, padding=window_size // 2, groups=channel)

        mu_r = _conv(r)
        mu_g = _conv(g)
        var_r = (_conv(r * r) - mu_r * mu_r).clamp_min(0.0)
        var_g = (_conv(g * g) - mu_g * mu_g).clamp_min(0.0)
        cov = _conv(r * g) - mu_r * mu_g
        # Local Pearson correlation in [-1, 1]; 1 - corr is the loss.
        corr = cov / torch.sqrt(var_r * var_g + eps)
        loss_map = (1.0 - corr) * mask
        return loss_map.sum() / mask.sum().clamp_min(1.0)

    if mode == "zncc_grad":
        # ZNCC in the spatial-gradient domain. Differentiate first (kills the
        # low-frequency baked shading), then standardise each gradient channel
        # over the foreground and maximise the global correlation.
        eps = 1e-4
        mask = _foreground_mask(gt)
        r_dx = rendered[:, :, 1:] - rendered[:, :, :-1]
        g_dx = gt[:, :, 1:] - gt[:, :, :-1]
        r_dy = rendered[:, 1:, :] - rendered[:, :-1, :]
        g_dy = gt[:, 1:, :] - gt[:, :-1, :]
        mask_dx = mask[:, :, 1:] * mask[:, :, :-1]
        mask_dy = mask[:, 1:, :] * mask[:, :-1, :]

        def _corr_loss(rp, gp, m):
            n = m.sum().clamp_min(1.0)
            mean_r = (rp * m).sum(dim=(1, 2), keepdim=True) / n
            mean_g = (gp * m).sum(dim=(1, 2), keepdim=True) / n
            rc = (rp - mean_r) * m
            gc = (gp - mean_g) * m
            std_r = torch.sqrt((rc * rc).sum(dim=(1, 2), keepdim=True) / n + eps)
            std_g = torch.sqrt((gc * gc).sum(dim=(1, 2), keepdim=True) / n + eps)
            corr = ((rc / std_r) * (gc / std_g) * m).sum(dim=(1, 2)) / n
            return (1.0 - corr).mean()

        return _corr_loss(r_dx, g_dx, mask_dx) + _corr_loss(r_dy, g_dy, mask_dy)

    if mode == "ssim_struct":
        # Structure-focused SSIM: match shapes / textures of the GT albedo while
        # down-weighting absolute brightness (luminance) and contrast drift.
        return _ssim_structure_loss(rendered, gt, lum_weight=0.1)

    # Default: "direct"
    data = huber_loss(rendered, gt, delta)
    struct = 1.0 - ssim(rendered, gt)
    return (1.0 - lambda_dssim) * data + lambda_dssim * struct


def decode_normal_to_world(encoded, camera_space=False, R_cam2world=None,
                           convention="opengl"):
    """Decode an encoded normal map [3,H,W] in [0,1] to a unit normal in [-1,1].

    GIR renders surface normals in WORLD space, so the GT normal prior must be
    expressed in world space too before it can be compared (cosine loss).

      * Synthetic / Blender priors are already stored in world space, so
        `camera_space=False` simply decodes (x*2-1) and normalises.
      * Real-world (COLMAP) priors from the diffusion models are stored in the
        CAMERA / view space of each frame. With `camera_space=True` the decoded
        normal is first converted from the model's view-space axis convention to
        the COLMAP camera convention (X right, Y down, Z forward) and then
        rotated into world space by the camera-to-world rotation `R_cam2world`
        (n_world = R_cam2world @ n_cam).

    `convention`:
      * "opengl" : model normals use OpenGL view axes (X right, Y up, Z toward
                   the viewer); Y and Z are flipped to reach COLMAP camera axes.
      * "opencv" / "colmap" : model normals already use COLMAP camera axes.
    """
    n = encoded * 2.0 - 1.0  # [-1, 1], shape [3, H, W]
    if camera_space and R_cam2world is not None:
        if convention == "opengl":
            flip = torch.tensor([1.0, -1.0, -1.0], device=n.device, dtype=n.dtype).view(3, 1, 1)
            n = n * flip
        C, H, W = n.shape
        R = R_cam2world.to(device=n.device, dtype=n.dtype)
        n = (R @ n.reshape(3, -1)).reshape(3, H, W)
    return F.normalize(n, p=2, dim=0)
