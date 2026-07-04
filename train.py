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

import os
import json
import math
import time
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, smooth_loss, regularizer_loss, get_mask, tv_loss, huber_loss, albedo_prior_loss, decode_normal_to_world
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, mse
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torchvision
from envlight.utils import cubemap_to_latlong
from lpipsPyTorch import lpips as compute_lpips
from lpipsPyTorch.modules.lpips import LPIPS

def load_relighted_gt(viewpoint, hdri_name, white_background):
    import os
    import numpy as np
    from PIL import Image
    import torch
    from utils.general_utils import PILtoTorch

    image_path = getattr(viewpoint, "image_path", None)
    if image_path is None:
        return None

    dir_name = os.path.dirname(image_path)
    parent_dir = os.path.dirname(dir_name)
    basename = os.path.basename(image_path)
    stem, ext = os.path.splitext(basename)
    frame_idx_str = stem.split("_")[-1]

    relighted_dir = os.path.join(parent_dir, f"rgba_{hdri_name}")
    relighted_file = f"rgba_{hdri_name}_{frame_idx_str}{ext}"
    rel_path = os.path.join(relighted_dir, relighted_file)

    if not os.path.exists(rel_path):
        return None

    # Load and preprocess
    img = Image.open(rel_path)
    im_data = np.array(img.convert("RGBA"))
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    norm_data = im_data / 255.0
    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    arr = np.concatenate((arr, norm_data[..., 3:4]), -1)
    img_rgba = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGBA")

    # Resize to the same resolution as viewpoint.original_image
    resolution = (viewpoint.image_width, viewpoint.image_height)
    resized_tensor = PILtoTorch(img_rgba, resolution)

    # The RGB channels are already composited over `bg` above (matching the
    # background the model renders against), so we return them directly. We must
    # NOT re-multiply by the alpha mask here: that would force the background to
    # black and create a mismatch against the rendered image whenever
    # white_background is True (penalizing relight PSNR/SSIM on white-bg scenes).
    gt_image = resized_tensor[:3, ...].clamp(0.0, 1.0).to(viewpoint.data_device)

    return gt_image

def scale_aligned_psnr(img, gt, return_gain=False):
    """PSNR after fitting a single global gain g* = <img,gt>/<img,img> that
    minimises ||g*img - gt||^2. This removes a pure exposure/scale mismatch
    (common after relighting / envmap intensity ambiguity) so the number
    reflects decomposition quality rather than overall brightness.

    With return_gain=True also returns g* itself: at relight g* > 1 means the
    render is systematically too dark vs the GT (the light-transport energy
    deficit), g* < 1 too bright."""
    num = (img * gt).sum()
    den = (img * img).sum().clamp_min(1e-8)
    gain = (num / den).clamp_min(0.0)
    aligned = torch.clamp(img * gain, 0.0, 1.0)
    value = psnr(aligned, gt).mean().item()
    if return_gain:
        return value, gain.item()
    return value

def envmap_recovery_metrics(envlight_base, gt_hdr_path, hdr_rotation=False):
    """Recovery error between the learned base environment map and a GT HDRI.

    The model's cubemap is converted to a lat-long image and compared to the GT
    HDRI in log space after a single global gain alignment (the recovered
    environment has an arbitrary absolute intensity). Returns
    (log_psnr, rel_l1) or None when the GT file is missing/unreadable.

    When training WITHOUT --hdr_rotation on Blender-rendered data, the model
    stores radiance indexed by raw (Z-up) world directions, while the GT HDRI
    lat-long uses the Y-up convention: L_world(d) = tex_gt(R d) with GIR's
    rotation R(x,y,z) = (-y, z, -x). In that case the GT is resampled at R v so
    both maps live in the same frame (verified: identity anti-correlates,
    R gives the best log-correlation). With --hdr_rotation the model's map is
    already in the GT convention and is compared directly.
    """
    if not gt_hdr_path or not os.path.exists(gt_hdr_path):
        return None
    try:
        import imageio
        import numpy as np
        gt_np = imageio.imread(gt_hdr_path).astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Could not read envmap GT {gt_hdr_path}: {e}")
        return None

    gt = torch.from_numpy(gt_np[..., :3]).cuda().clamp_min(0.0)
    H, W = gt.shape[0], gt.shape[1]
    # Cap resolution to keep the periodic metric cheap.
    target_h, target_w = min(H, 512), min(W, 1024)
    if (target_h, target_w) != (H, W):
        gt = F.interpolate(gt.permute(2, 0, 1).unsqueeze(0), size=(target_h, target_w),
                           mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()

    if not hdr_rotation:
        # Resample the GT at R v (world-frame convention of the learned map).
        import numpy as np
        import nvdiffrast.torch as dr
        from envlight.utils import latlong_to_cubemap
        gt_cube = latlong_to_cubemap(gt, [512, 512])
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / target_h, 1.0 - 1.0 / target_h, target_h, device="cuda"),
            torch.linspace(-1.0 + 1.0 / target_w, 1.0 - 1.0 / target_w, target_w, device="cuda"),
            indexing="ij")
        sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
        sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
        v = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
        rv = torch.stack((-v[..., 1], v[..., 2], -v[..., 0]), dim=-1)
        gt = dr.texture(gt_cube[None, ...], rv[None, ...].contiguous(),
                        filter_mode="linear", boundary_mode="cube")[0].clamp_min(0.0)

    model_ll = cubemap_to_latlong(envlight_base.detach(), [target_h, target_w]).clamp_min(0.0)

    gain = ((model_ll * gt).sum() / (model_ll * model_ll).sum().clamp_min(1e-8)).clamp_min(0.0)
    aligned = model_ll * gain

    log_a = torch.log1p(aligned)
    log_g = torch.log1p(gt)
    mse = ((log_a - log_g) ** 2).mean().clamp_min(1e-12)
    peak = log_g.max().clamp_min(1e-6)
    log_psnr = (20.0 * torch.log10(peak / torch.sqrt(mse))).item()
    rel_l1 = ((aligned - gt).abs().sum() / gt.abs().sum().clamp_min(1e-6)).item()
    # Absolute brightness calibration of the learned light: > 1 means the
    # envmap trains brighter than the GT HDRI (it absorbs the transport
    # deficit), and that factor is what the relit renders are missing when the
    # native-intensity GT HDRI is swapped in (try_7: mean_ratio ~ relight gain).
    mean_ratio = (model_ll.mean() / gt.mean().clamp_min(1e-8)).item()
    return log_psnr, rel_l1, mean_ratio

def gt_normal_world(viewpoint, gt_normal):
    """Decode a viewpoint's GT normal prior into a unit WORLD-space normal.

    For synthetic priors this is just (x*2-1) normalised; for real-world COLMAP
    priors (flagged `normal_in_camera_space`) the camera->world rotation stored
    on the camera is applied so the prior lines up with GIR's world normals.
    """
    return decode_normal_to_world(
        gt_normal,
        camera_space=getattr(viewpoint, "normal_in_camera_space", False),
        R_cam2world=getattr(viewpoint, "R_cam2world", None),
        convention=getattr(viewpoint, "normal_camera_convention", "opengl"))

def periodic_evaluation(iteration, scene, gaussians, pipe, background, first_stage_step, second_stage_step, remove_noise, hdr_rotation, ema_loss, lpips_model, save_visuals=False, run_relighting_eval=False, eval_relight_hdris=[], envmap_gt_path="", albedo_geometry_warmup=False, light_linear_indirect=False, relight_max_views=0):
    """Evaluate metrics on test and train cameras at the current iteration."""
    torch.cuda.empty_cache()
    lpips_model.to("cuda")
    results = {"iteration": iteration, "train_loss": ema_loss, "num_gaussians": gaussians.get_xyz.shape[0]}

    eval_configs = [
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), max(1, len(scene.getTrainCameras()) // 5))][:5]},
    ]

    for config in eval_configs:
        if not config["cameras"] or len(config["cameras"]) == 0:
            continue

        psnr_vals, ssim_vals, lpips_vals, l1_vals, mse_vals = [], [], [], [], []
        psnr_aligned_vals = []
        gain_vals = []
        albedo_psnr_vals, albedo_ssim_vals, albedo_l1_vals = [], [], []
        albedo_psnr_aligned_vals = []
        normal_angular_error_vals = []
        metallic_mae_vals, roughness_mae_vals = [], []
        visual_pairs = []  # (render, gt) for visual comparison

        for idx, viewpoint in enumerate(config["cameras"]):
            render_pkg = render(viewpoint, gaussians, pipe, background,
                                iteration=iteration, is_train=False,
                                first_stage_step=first_stage_step,
                                second_stage_step=second_stage_step,
                                remove_noise=remove_noise,
                                hdr_rotation=hdr_rotation,
                                albedo_geometry_warmup=albedo_geometry_warmup,
                                light_linear_indirect=light_linear_indirect)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_image, _ = get_mask(gt_image)

            psnr_vals.append(psnr(image, gt_image).mean().item())
            pa_val, g_val = scale_aligned_psnr(image, gt_image, return_gain=True)
            psnr_aligned_vals.append(pa_val)
            gain_vals.append(g_val)
            ssim_vals.append(ssim(image, gt_image).item())
            l1_vals.append(l1_loss(image, gt_image).item())
            mse_vals.append(mse(image, gt_image).mean().item())
            gt_albedo = getattr(viewpoint, 'albedo_gt', None)
            if gt_albedo is not None:
                rendered_albedo = render_pkg.get("rendered_albedo", None)
                if rendered_albedo is not None:
                    rendered_albedo_clamped = torch.clamp(rendered_albedo, 0.0, 1.0)
                    albedo_psnr_vals.append(psnr(rendered_albedo_clamped, gt_albedo).mean().item())
                    # Scale-aligned albedo PSNR: removes a single global gain
                    # before comparing, so albedo modes that only recover
                    # relative structure / colour (log_chroma, gradient) are
                    # judged on decomposition quality rather than absolute scale.
                    albedo_psnr_aligned_vals.append(scale_aligned_psnr(rendered_albedo_clamped, gt_albedo))
                    albedo_ssim_vals.append(ssim(rendered_albedo_clamped, gt_albedo).item())
                    albedo_l1_vals.append(l1_loss(rendered_albedo_clamped, gt_albedo).item())

            gt_normal = getattr(viewpoint, 'normal_gt', None)
            if gt_normal is not None:
                rendered_normal = render_pkg.get("rendered_normal", None)
                if rendered_normal is not None:
                    pred_n = rendered_normal * 2.0 - 1.0
                    pred_n = F.normalize(pred_n, p=2, dim=0)
                    gt_n = gt_normal_world(viewpoint, gt_normal)
                    cos_sim = torch.clamp(torch.sum(pred_n * gt_n, dim=0), -1.0, 1.0)
                    ang_error = torch.acos(cos_sim) * 180.0 / 3.141592653589793
                    # Restrict to the object silhouette. Background pixels carry
                    # no meaningful normal (rendered normal is ~0 there -> 90 deg
                    # error), which would otherwise dominate and inflate the mean.
                    alpha_map = render_pkg.get("alpha", None)
                    if alpha_map is not None:
                        fg = (alpha_map.squeeze(0) > 0.5).to(ang_error.dtype)
                        denom = fg.sum().clamp_min(1.0)
                        normal_angular_error_vals.append(((ang_error * fg).sum() / denom).item())
                    else:
                        normal_angular_error_vals.append(ang_error.mean().item())

            # Per-channel material errors (metallic / roughness), when GT exists.
            rendered_metallic = render_pkg.get("rendered_metallic", None)
            gt_metallic_val = getattr(viewpoint, 'metallic_gt', None)
            if rendered_metallic is not None and gt_metallic_val is not None:
                if isinstance(gt_metallic_val, torch.Tensor):
                    gt_metallic = gt_metallic_val.to(rendered_metallic.device)
                else:
                    gt_metallic = torch.full_like(rendered_metallic, float(gt_metallic_val))
                metallic_mae_vals.append(l1_loss(torch.clamp(rendered_metallic, 0.0, 1.0), gt_metallic).item())

            rendered_roughness = render_pkg.get("rendered_roughness", None)
            gt_roughness_val = getattr(viewpoint, 'roughness_gt', None)
            if rendered_roughness is not None and gt_roughness_val is not None:
                if isinstance(gt_roughness_val, torch.Tensor):
                    gt_roughness = gt_roughness_val.to(rendered_roughness.device)
                else:
                    gt_roughness = torch.full_like(rendered_roughness, float(gt_roughness_val))
                roughness_mae_vals.append(l1_loss(torch.clamp(rendered_roughness, 0.0, 1.0), gt_roughness).item())

            # LPIPS with cached model (already on GPU)
            with torch.no_grad():
                lpips_val = lpips_model(image.unsqueeze(0) if image.dim() == 3 else image,
                                       gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
                lpips_vals.append(lpips_val.item())

            if save_visuals and idx < 3:  # Save first 3 views
                visual_pairs.append((image.detach().cpu(), gt_image.detach().cpu()))

        prefix = config["name"]
        results[f"{prefix}_psnr"] = sum(psnr_vals) / len(psnr_vals)
        results[f"{prefix}_psnr_aligned"] = sum(psnr_aligned_vals) / len(psnr_aligned_vals)
        # Fitted gain under the TRAINING light. Sanity reference for the
        # relight_*_gain metrics: the photometric loss pins this near 1.0
        # (try_7: ~1.09), so any relight gain above it is a relight-specific
        # energy deficit, not a metric artifact.
        results[f"{prefix}_gain"] = sum(gain_vals) / len(gain_vals)
        results[f"{prefix}_ssim"] = sum(ssim_vals) / len(ssim_vals)
        results[f"{prefix}_lpips"] = sum(lpips_vals) / len(lpips_vals)
        results[f"{prefix}_l1"] = sum(l1_vals) / len(l1_vals)
        results[f"{prefix}_mse"] = sum(mse_vals) / len(mse_vals)

        if albedo_psnr_vals:
            results[f"{prefix}_albedo_psnr"] = sum(albedo_psnr_vals) / len(albedo_psnr_vals)
            results[f"{prefix}_albedo_psnr_aligned"] = sum(albedo_psnr_aligned_vals) / len(albedo_psnr_aligned_vals)
            results[f"{prefix}_albedo_ssim"] = sum(albedo_ssim_vals) / len(albedo_ssim_vals)
            results[f"{prefix}_albedo_l1"] = sum(albedo_l1_vals) / len(albedo_l1_vals)
        if normal_angular_error_vals:
            results[f"{prefix}_normal_ang_err"] = sum(normal_angular_error_vals) / len(normal_angular_error_vals)
        if metallic_mae_vals:
            results[f"{prefix}_metallic_mae"] = sum(metallic_mae_vals) / len(metallic_mae_vals)
        if roughness_mae_vals:
            results[f"{prefix}_roughness_mae"] = sum(roughness_mae_vals) / len(roughness_mae_vals)

        # Save visual comparison grids
        if save_visuals and visual_pairs:
            vis_path = os.path.join(scene.model_path, "eval_visuals", prefix)
            os.makedirs(vis_path, exist_ok=True)
            for vi, (rend, gt) in enumerate(visual_pairs):
                grid = torchvision.utils.make_grid([rend, gt], nrow=2, padding=4, pad_value=1.0)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"iter{iteration:06d}_view{vi}.png"))

        # Relighting evaluation (only on test cameras and at run_relighting_eval steps)
        if config["name"] == "test" and run_relighting_eval:
            hdris_dir = os.path.join(scene.source_path, "hdris")
            if os.path.exists(hdris_dir):
                hdr_files = sorted([f for f in os.listdir(hdris_dir) if f.endswith(".hdr")])
                hdr_stems = [os.path.splitext(f)[0] for f in hdr_files]

                selected_hdris = []
                for t in eval_relight_hdris:
                    if t in hdr_stems:
                        dir_exists = False
                        for split in ["train", "val", "test"]:
                            if os.path.isdir(os.path.join(scene.source_path, split, f"rgba_{t}")):
                                dir_exists = True
                                break
                        if dir_exists:
                            selected_hdris.append(t)

                for h in hdr_stems:
                    if len(selected_hdris) >= len(eval_relight_hdris) or len(selected_hdris) >= len(hdr_stems):
                        break
                    if h not in selected_hdris:
                        dir_exists = False
                        for split in ["train", "val", "test"]:
                            if os.path.isdir(os.path.join(scene.source_path, split, f"rgba_{h}")):
                                dir_exists = True
                                break
                        if dir_exists:
                            selected_hdris.append(h)

                if selected_hdris:
                    print(f"\nEvaluating relighting under HDRIs: {selected_hdris}")
                    sys.stdout.flush()

                    white_background = (background[0] > 0.5).item()

                    # Relighting is by far the most expensive part of the eval
                    # (n_hdris x n_test_cams renders + LPIPS). Cap it to an
                    # evenly spaced, DETERMINISTIC subset of the test cameras:
                    # the same views are used at every eval and in every run,
                    # so the numbers stay comparable across runs while the cost
                    # drops by len(cams)/relight_max_views.
                    relight_cams = config["cameras"]
                    if relight_max_views and len(relight_cams) > relight_max_views:
                        step = (len(relight_cams) - 1) / (relight_max_views - 1)
                        idxs = sorted({int(round(i * step)) for i in range(relight_max_views)})
                        relight_cams = [relight_cams[i] for i in idxs]
                        print(f"  (relight eval capped at {len(relight_cams)} of "
                              f"{len(config['cameras'])} test views)")

                    for hdri_name in selected_hdris:
                        hdri_path = os.path.join(hdris_dir, f"{hdri_name}.hdr")
                        
                        # Backup envlight state
                        envlight_obj = gaussians.envlight
                        orig_image = envlight_obj.image.clone() if hasattr(envlight_obj, "image") and envlight_obj.image is not None else None
                        orig_base = envlight_obj.base.clone()
                        orig_specular = [m.clone() for m in envlight_obj.specular] if hasattr(envlight_obj, "specular") and envlight_obj.specular is not None else None
                        orig_diffuse = envlight_obj.diffuse.clone() if hasattr(envlight_obj, "diffuse") and envlight_obj.diffuse is not None else None

                        relight_psnr_vals = []
                        relight_psnr_aligned_vals = []
                        relight_gain_vals = []
                        relight_ssim_vals = []
                        relight_lpips_vals = []
                        relight_save_pairs = []

                        try:
                            # Load new HDRI and build MIPs
                            envlight_obj.load(hdri_path)
                            envlight_obj.build_mips()

                            # Evaluate over the (possibly capped) test cameras
                            for idx, viewpoint in enumerate(relight_cams):
                                gt_relight = load_relighted_gt(viewpoint, hdri_name, white_background)
                                if gt_relight is None:
                                    continue

                                render_pkg = render(viewpoint, gaussians, pipe, background,
                                                    iteration=iteration, is_train=False,
                                                    first_stage_step=first_stage_step,
                                                    second_stage_step=second_stage_step,
                                                    remove_noise=remove_noise,
                                                    hdr_rotation=hdr_rotation,
                                                    albedo_geometry_warmup=albedo_geometry_warmup,
                                                    light_linear_indirect=light_linear_indirect)
                                image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                                relight_psnr_vals.append(psnr(image, gt_relight).mean().item())
                                pa, g = scale_aligned_psnr(image, gt_relight, return_gain=True)
                                relight_psnr_aligned_vals.append(pa)
                                relight_gain_vals.append(g)
                                relight_ssim_vals.append(ssim(image, gt_relight).item())
                                with torch.no_grad():
                                    relight_lpips_vals.append(lpips_model(
                                        image.unsqueeze(0) if image.dim() == 3 else image,
                                        gt_relight.unsqueeze(0) if gt_relight.dim() == 3 else gt_relight).item())

                                if save_visuals and idx < 3:
                                    relight_save_pairs.append((viewpoint.image_name, image.detach().cpu(), gt_relight.detach().cpu()))

                        finally:
                            # Restore envlight state
                            if orig_image is not None:
                                envlight_obj.image = orig_image
                            envlight_obj.base.data = orig_base
                            if orig_specular is not None:
                                envlight_obj.specular = orig_specular
                            if orig_diffuse is not None:
                                envlight_obj.diffuse = orig_diffuse

                        # Store and report results
                        if relight_psnr_vals:
                            avg_psnr = sum(relight_psnr_vals) / len(relight_psnr_vals)
                            avg_psnr_aligned = sum(relight_psnr_aligned_vals) / len(relight_psnr_aligned_vals)
                            avg_ssim = sum(relight_ssim_vals) / len(relight_ssim_vals)
                            results[f"relight_{hdri_name}_psnr"] = avg_psnr
                            results[f"relight_{hdri_name}_psnr_aligned"] = avg_psnr_aligned
                            results[f"relight_{hdri_name}_ssim"] = avg_ssim
                            if relight_gain_vals:
                                # Fitted global gain render->GT: > 1 means the
                                # relit render is too dark (energy deficit).
                                results[f"relight_{hdri_name}_gain"] = sum(relight_gain_vals) / len(relight_gain_vals)
                            if relight_lpips_vals:
                                avg_lpips = sum(relight_lpips_vals) / len(relight_lpips_vals)
                                results[f"relight_{hdri_name}_lpips"] = avg_lpips

                            print(f"  HDRI {hdri_name:20s} — PSNR: {avg_psnr:.2f} (aligned {avg_psnr_aligned:.2f}), SSIM: {avg_ssim:.4f}")
                            sys.stdout.flush()

                        # Save visuals as separate images
                        if save_visuals and relight_save_pairs:
                            vis_path = os.path.join(scene.model_path, "eval_visuals", f"relight_{hdri_name}")
                            os.makedirs(vis_path, exist_ok=True)
                            for view_name, rend, gt in relight_save_pairs:
                                torchvision.utils.save_image(rend, os.path.join(vis_path, f"iter{iteration:06d}_{view_name}_render.png"))
                                torchvision.utils.save_image(gt, os.path.join(vis_path, f"iter{iteration:06d}_{view_name}_gt.png"))

    albedo_info = ""
    if "test_albedo_psnr" in results:
        albedo_info = f", Alb PSNR: {results['test_albedo_psnr']:.2f}, Norm Err: {results['test_normal_ang_err']:.2f} deg"

    # --- Environment map recovery vs. GT base-light HDRI ---
    env_metrics = envmap_recovery_metrics(gaussians.envlight.base, envmap_gt_path, hdr_rotation=hdr_rotation)
    if env_metrics is not None:
        results["envmap_log_psnr"], results["envmap_rel_l1"], results["envmap_mean_ratio"] = env_metrics
        albedo_info += (f", EnvMap logPSNR: {results['envmap_log_psnr']:.2f}, relL1: {results['envmap_rel_l1']:.3f}, "
                        f"meanRatio: {results['envmap_mean_ratio']:.2f}")

    print(f"\n[ITER {iteration}] Eval — "
          f"Test PSNR: {results.get('test_psnr', 0):.2f}, "
          f"SSIM: {results.get('test_ssim', 0):.4f}, "
          f"LPIPS: {results.get('test_lpips', 0):.4f}, "
          f"L1: {results.get('test_l1', 0):.6f}, "
          f"MSE: {results.get('test_mse', 0):.6f}{albedo_info}")
    sys.stdout.flush()
    lpips_model.to("cpu")
    torch.cuda.empty_cache()
    return results


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, first_stage_step, second_stage_step, remove_noise, hdr_rotation, reg_hdr_weight=0.001, reg_material_weight=0.1, eval_interval=2000, visual_interval=10000, lambda_albedo_gt=0.1, lambda_normal_gt=0.1, lambda_metallic_gt=0.05, lambda_roughness_gt=0.05, use_prior_weight_scheduler=False, prior_weight_scheduler_ratio=0.15, prior_weight_final_ratio=1.0, albedo_prior_mode="direct", huber_delta=0.1, exclude_prior_loss=False, eval_relight_hdris=['snowy_forest', 'moonless_night', 'fireplace'], envmap_gt_path="", tv_reduction_factor=1.0, reduce_geo_lr_third_stage=1.0, geo_lr_final_iter=0, disable_reset_third_stage=False, albedo_geometry_warmup=False, warmup_albedo_prior_mode="", prior_geom_grad_scale=1.0, albedo_anchor_weight=0.0, lambda_normal_third_stage_scale=1.0, light_linear_indirect=False, relight_max_views=0):
    # Respect user-specified intervals if they differ from the default values of 2000 / 10000.
    # Otherwise, use more reasonable dynamic defaults to avoid slowing down training.
    user_eval_set = (eval_interval != 2000)
    user_visual_set = (visual_interval != 10000)

    if not user_eval_set:
        if opt.iterations < 10000:
            eval_interval = max(1, opt.iterations // 10)  # Target ~10 evaluations
        elif opt.iterations < 30000:
            eval_interval = max(1, opt.iterations // 30)  # Target ~30 evaluations
        else:
            eval_interval = 2000
            
    if not user_visual_set:
        if opt.iterations < 10000:
            visual_interval = max(1, opt.iterations // 3)   # Target ~3 visual saves
        elif opt.iterations < 30000:
            visual_interval = max(1, opt.iterations // 10)  # Target ~10 visual saves
        else:
            visual_interval = 10000


    # A property's smoothness (TV / edge-aware) regularizer is meant to fill in
    # information we DON'T have. When a GT prior is supervising that property we
    # no longer need to artificially enforce smoothness, so its TV weight is
    # scaled by `tv_reduction_factor` (0.0 = off, 1.0 = unchanged). Properties
    # without a GT prior keep their full TV weight.
    has_albedo_gt = bool(getattr(dataset, "albedo_gt_dir", ""))
    has_normal_gt = bool(getattr(dataset, "normal_gt_dir", ""))
    has_metallic_gt = bool(getattr(dataset, "metallic_gt_dir", ""))
    has_roughness_gt = bool(getattr(dataset, "roughness_gt_dir", ""))
    albedo_tv_scale = tv_reduction_factor if has_albedo_gt else 1.0
    normal_tv_scale = tv_reduction_factor if has_normal_gt else 1.0
    metallic_tv_scale = tv_reduction_factor if has_metallic_gt else 1.0
    roughness_tv_scale = tv_reduction_factor if has_roughness_gt else 1.0

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # Optional third-stage geometry-LR reduction (xyz / scaling / rotation). The
    # final iteration defaults to the end of training when not set (<=0). Inert
    # when reduce_geo_lr_third_stage == 1.0, so the baseline is unaffected.
    _geo_lr_final_iter = geo_lr_final_iter if geo_lr_final_iter and geo_lr_final_iter > 0 else opt.iterations
    gaussians.set_geo_lr_schedule(second_stage_step, _geo_lr_final_iter, reduce_geo_lr_third_stage)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # --- Periodic evaluation setup ---
    metrics_log_path = os.path.join(scene.model_path, "metrics_log.json")
    if checkpoint and os.path.exists(metrics_log_path):
        with open(metrics_log_path, "r") as f:
            metrics_log = json.load(f)
        # Remove entries at or after the resumed iteration to avoid duplicates
        metrics_log = [m for m in metrics_log if m["iteration"] < first_iter]
        print(f"Loaded {len(metrics_log)} existing metric entries from {metrics_log_path}")
    else:
        metrics_log = []

    # --- Periodic loss components setup ---
    loss_log_path = os.path.join(scene.model_path, "train_process", "loss_components.json")
    if checkpoint and os.path.exists(loss_log_path):
        with open(loss_log_path, "r") as f:
            try:
                loss_logs = json.load(f)
                # Remove entries at or after the resumed iteration to avoid duplicates
                loss_logs = [l for l in loss_logs if l["iteration"] < first_iter]
                print(f"Loaded {len(loss_logs)} existing loss component entries from {loss_log_path}")
            except Exception:
                loss_logs = []
    else:
        loss_logs = []
        # If not resuming, remove any stale loss_components.json from previous runs
        if os.path.exists(loss_log_path):
            try:
                os.remove(loss_log_path)
            except OSError:
                pass

    # Cache the LPIPS model on CPU to save GPU memory; moved to GPU only during eval
    lpips_model = LPIPS("vgg", "0.1").cpu()
    lpips_model.eval()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    eval_count = 0
    # Cross-view state for stateful albedo prior modes (si_ema keeps its
    # EMA-shared per-channel log-gain here).
    albedo_prior_state = {}
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, iteration=iteration, is_train=False, first_stage_step=first_stage_step, second_stage_step=second_stage_step, remove_noise=remove_noise, hdr_rotation=hdr_rotation, albedo_geometry_warmup=albedo_geometry_warmup, light_linear_indirect=light_linear_indirect)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration >= second_stage_step:
            # Refresh the occlusion voxel grid every 1000 iters, and ALWAYS at
            # the stage boundary itself: the PBR stage needs the grid on its
            # first iteration, and second_stage_step may not be a multiple of 1000.
            if iteration == second_stage_step or iteration % 1000 == 0:
                gaussians.get_diffuse_occ()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration>second_stage_step and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, random_bg_color=bg, iteration=iteration, is_train=True, first_stage_step=first_stage_step, second_stage_step=second_stage_step, remove_noise=remove_noise, hdr_rotation=hdr_rotation, albedo_geometry_warmup=albedo_geometry_warmup, prior_geom_grad_scale=prior_geom_grad_scale, light_linear_indirect=light_linear_indirect)
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        rendered_normal = render_pkg["rendered_normal"]
        rendered_albedo = render_pkg["rendered_albedo"]
        rendered_metallic = render_pkg["rendered_metallic"]
        rendered_roughness = render_pkg["rendered_roughness"]
        rendered_diffuse_color = render_pkg["rendered_diffuse_color"]
        rendered_specular_color = render_pkg["rendered_specular_color"]
        rendered_diffuse_light = render_pkg["rendered_diffuse_light"]
        rendered_specular_light = render_pkg["rendered_specular_light"]
        rendered_diffuse_albedo = render_pkg["rendered_diffuse_albedo"]
        rendered_specular_albedo = render_pkg["rendered_specular_albedo"]
        rendered_specular_indirect_light = render_pkg["rendered_specular_indirect_light"]
        rendered_specular_direct_light = render_pkg["rendered_specular_direct_light"]
        rendered_specular_indirect_color = render_pkg["rendered_specular_indirect_color"]
        rendered_specular_direct_color = render_pkg["rendered_specular_direct_color"]
        rendered_occ = render_pkg["rendered_occ"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"] 
        # Loss
        multiplier = 0.0
        loss_albedo_gt_val = torch.tensor(0.0).cuda()
        loss_normal_gt_val = torch.tensor(0.0).cuda()
        loss_metallic_gt_val = torch.tensor(0.0).cuda()
        loss_roughness_gt_val = torch.tensor(0.0).cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        gt_image, gt_mask = get_mask(gt_image)
        if gt_mask is None:
            # 3-channel datasets (plain COLMAP) carry no alpha: treat everything
            # as foreground so the compositing / prior masking below still work.
            gt_mask = torch.ones_like(gt_image[:1])
        gt_image = gt_image * gt_mask + bg.unsqueeze(-1).unsqueeze(-1).repeat(1, gt_image.shape[1], gt_image.shape[2]) * (1-gt_mask)
        Ll1 = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_image
        # ── Albedo geometry warm-up (Phases 1 & 2) ──────────────────
        # PBR albedo is view-independent, so during the geometry / normal-
        # alignment stages we fit geometry to the GT albedo prior directly
        # instead of to the RGB images via the photometric loss. The rendered
        # `image` here IS the flat per-gaussian albedo (see the renderer's
        # warm-up path); the photometric / PBR loss is withheld until the PBR
        # stage (iter > second_stage_step). The Phase-2 normal prior below still
        # adds on top of this.
        if albedo_geometry_warmup and not exclude_prior_loss and iteration <= second_stage_step:
            gt_albedo_warmup = getattr(viewpoint_cam, 'albedo_gt', None)
            if gt_albedo_warmup is not None:
                # The warm-up (stages 1 & 2) may use a DIFFERENT albedo prior
                # formulation than the PBR stage (stage 3): `warmup_albedo_prior_mode`
                # selects it, falling back to `albedo_prior_mode` when left empty.
                warmup_mode = warmup_albedo_prior_mode or albedo_prior_mode
                # Composite the GT albedo over the same background as the
                # rendered image so empty pixels match exactly and do not skew
                # the (scale-sensitive) albedo prior.
                gt_albedo_warmup = gt_albedo_warmup * gt_mask + bg.unsqueeze(-1).unsqueeze(-1).repeat(1, gt_albedo_warmup.shape[1], gt_albedo_warmup.shape[2]) * (1 - gt_mask)
                loss = lambda_albedo_gt * albedo_prior_loss(image, gt_albedo_warmup,
                                                            mode=warmup_mode,
                                                            lambda_dssim=opt.lambda_dssim,
                                                            delta=huber_delta,
                                                            alpha_mask=gt_mask,
                                                            state=albedo_prior_state)
                loss_albedo_gt_val = l1_loss(image, gt_albedo_warmup).detach()
        if iteration > second_stage_step:
            multiplier = 1.0
            if use_prior_weight_scheduler:
                # Up-then-hold/(in/de)crease schedule for the prior / envmap-reg
                # weights:
                #   * linearly ramp 0 -> 1 over the first `prior_weight_scheduler_ratio`
                #     fraction of the PBR stage (let the decomposition settle first),
                #   * then linearly interpolate 1 -> `prior_weight_final_ratio` over the
                #     rest. final == 1.0 HOLDS the priors at full strength (recommended
                #     for relighting, so the GT prior keeps constraining the late phase
                #     where the model otherwise overfits the single training light);
                #     final > 1.0 INCREASES the pull; final < 1.0 reproduces the old
                #     decay-toward-a-floor behavior (e.g. 0.5).
                total_second_stage_steps = opt.iterations - second_stage_step
                warmup_steps = int(prior_weight_scheduler_ratio * total_second_stage_steps)
                progress = iteration - second_stage_step
                if warmup_steps > 0 and progress < warmup_steps:
                    multiplier = float(progress) / float(warmup_steps)
                else:
                    decay_steps = total_second_stage_steps - warmup_steps
                    if decay_steps > 0:
                        t = min(max(float(progress - warmup_steps) / float(decay_steps), 0.0), 1.0)
                        final = prior_weight_final_ratio
                        multiplier = 1.0 + (final - 1.0) * t
                    else:
                        multiplier = 1.0

            loss_albedo = tv_loss(rendered_albedo) * 0.1 * albedo_tv_scale
            loss_normal = smooth_loss(rendered_normal, gt_image) * 0.01 * normal_tv_scale # 1
            loss_regularizer = regularizer_loss(gaussians.envlight.base) * reg_hdr_weight * multiplier
            loss_metallic = tv_loss(rendered_metallic) * reg_material_weight * metallic_tv_scale
            loss_roughness = tv_loss(rendered_roughness) * reg_material_weight * roughness_tv_scale
            loss = loss + loss_albedo + loss_normal + loss_metallic + loss_roughness + loss_regularizer #+ Ll1_alpha

            # GT priors — each property is supervised independently when its GT
            # is present. A prior whose folder was set to "" is simply absent
            # from the camera here, so it contributes nothing (weight is 0).
            gt_albedo = getattr(viewpoint_cam, 'albedo_gt', None)
            gt_normal = getattr(viewpoint_cam, 'normal_gt', None)
            gt_metallic = getattr(viewpoint_cam, 'metallic_gt', None)
            gt_roughness = getattr(viewpoint_cam, 'roughness_gt', None)

            if exclude_prior_loss:
                # Compute raw L1 / (1 - cos) errors without gradients — logging
                # only, so the numbers stay comparable across runs regardless of
                # the optimized prior formulation.
                with torch.no_grad():
                    if gt_albedo is not None:
                        loss_albedo_gt_val = l1_loss(rendered_albedo, gt_albedo)
                    if gt_normal is not None:
                        pred_n = rendered_normal * 2.0 - 1.0
                        gt_n = gt_normal_world(viewpoint_cam, gt_normal)
                        cos_sim = F.cosine_similarity(pred_n, gt_n, dim=0)
                        loss_normal_gt_val = (1.0 - cos_sim).mean()
                    if gt_metallic is not None:
                        loss_metallic_gt_val = l1_loss(rendered_metallic, gt_metallic)
                    if gt_roughness is not None:
                        loss_roughness_gt_val = l1_loss(rendered_roughness, gt_roughness)
            else:
                # Optimized prior loss. The albedo term uses the selected
                # invariance mode; normal/material terms use robust losses.
                loss_prior = torch.zeros((), device="cuda")

                if gt_albedo is not None:
                    prior_albedo = albedo_prior_loss(rendered_albedo, gt_albedo,
                                                     mode=albedo_prior_mode,
                                                     lambda_dssim=opt.lambda_dssim,
                                                     delta=huber_delta,
                                                     alpha_mask=gt_mask,
                                                     state=albedo_prior_state)
                    with torch.no_grad():
                        loss_albedo_gt_val = l1_loss(rendered_albedo, gt_albedo)
                    loss_prior = loss_prior + lambda_albedo_gt * prior_albedo
                    # Weak ABSOLUTE anchor on top of the (gain-invariant) albedo
                    # mode: prevents the global gain drift observed with pure
                    # invariant losses (albedo scale wandering off while the
                    # envmap compensates), while staying too weak to fight local
                    # shading disagreements.
                    if albedo_anchor_weight > 0.0:
                        anchor = huber_loss(rendered_albedo, gt_albedo, huber_delta, mask=gt_mask)
                        loss_prior = loss_prior + albedo_anchor_weight * anchor

                if gt_normal is not None:
                    pred_n = rendered_normal * 2.0 - 1.0
                    gt_n = gt_normal_world(viewpoint_cam, gt_normal)
                    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=0)
                    # Restrict to the object silhouette: background pixels carry
                    # no meaningful GT normal (and the rendered normal there is
                    # just the background colour), so they must not contribute
                    # gradients — critical for diffusion priors, whose bg pixels
                    # contain hallucinated normals.
                    m = gt_mask.squeeze(0)
                    prior_normal = ((1.0 - cos_sim) * m).sum() / m.sum().clamp_min(1.0)
                    loss_normal_gt_val = prior_normal.detach()
                    # The stage-3 normal weight can be scaled down independently
                    # of the Phase-2 weight: noisy (diffusion) normal priors are
                    # valuable while geometry forms in Phase 2 but degrade the
                    # settled geometry when kept at full strength against the
                    # photometric loss in the PBR stage.
                    loss_prior = loss_prior + lambda_normal_gt * lambda_normal_third_stage_scale * prior_normal

                if gt_metallic is not None:
                    prior_metallic = huber_loss(rendered_metallic, gt_metallic, huber_delta, mask=gt_mask)
                    with torch.no_grad():
                        loss_metallic_gt_val = l1_loss(rendered_metallic, gt_metallic)
                    loss_prior = loss_prior + lambda_metallic_gt * prior_metallic

                if gt_roughness is not None:
                    prior_roughness = huber_loss(rendered_roughness, gt_roughness, huber_delta, mask=gt_mask)
                    with torch.no_grad():
                        loss_roughness_gt_val = l1_loss(rendered_roughness, gt_roughness)
                    loss_prior = loss_prior + lambda_roughness_gt * prior_roughness

                loss = loss + loss_prior * multiplier
        elif iteration > first_stage_step and not exclude_prior_loss:
            # ── Phase 2 — normal alignment ──────────────────────────────────
            # Between the radiance warm-up (Phase 1) and the full PBR stage
            # (Phase 3) we lock in geometry by supervising ONLY the shading
            # normal with its GT prior (back-props into gaussian rotation /
            # scaling). Albedo / material priors stay off until Phase 3. A
            # reduced edge-aware smoothness keeps the normals stable.
            #
            # This phase is SKIPPED for the GIR baseline (exclude_prior_loss):
            # the baseline must reproduce the original GIR pipeline exactly,
            # which adds no loss between the two stage boundaries (only the RGB
            # photometric loss above). Engaging the normal supervision here
            # would back-prop into the gaussian geometry and diverge from — and
            # in practice destabilise — the untouched baseline.
            if rendered_normal is not None:
                loss_normal = smooth_loss(rendered_normal, gt_image) * 0.01 * normal_tv_scale
                loss = loss + loss_normal

                gt_normal = getattr(viewpoint_cam, 'normal_gt', None)
                if gt_normal is not None:
                    pred_n = rendered_normal * 2.0 - 1.0
                    gt_n = gt_normal_world(viewpoint_cam, gt_normal)
                    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=0)
                    # Foreground-only, same as the Phase-3 normal prior.
                    m = gt_mask.squeeze(0)
                    prior_normal = ((1.0 - cos_sim) * m).sum() / m.sum().clamp_min(1.0)
                    loss_normal_gt_val = prior_normal.detach()
                    if not exclude_prior_loss:
                        # Flat weight in Phase 2 (the up-then-down scheduler only
                        # governs the Phase-3 priors).
                        loss = loss + lambda_normal_gt * prior_normal
        loss.backward()

        iter_end.record()

        if iteration % 500 == 0:
            render_path = os.path.join(scene.model_path, "train_process", "renders")
            os.makedirs(render_path, exist_ok=True)
            torchvision.utils.save_image(image.clamp(0.0, 1.0).detach().cpu(), os.path.join(render_path, '{0:05d}'.format(iteration) + ".png"))

            gts_path = os.path.join(scene.model_path, "train_process", "gt")
            os.makedirs(gts_path, exist_ok=True)
            torchvision.utils.save_image(gt_image.detach().cpu(), os.path.join(gts_path, '{0:05d}'.format(iteration) + ".png"))
            
            depth_path = os.path.join(scene.model_path, "train_process", "depth")
            os.makedirs(depth_path, exist_ok=True)
            torchvision.utils.save_image(depth.detach().cpu()/10.0, os.path.join(depth_path, '{0:05d}'.format(iteration) + ".png"))
            
            alpha_path = os.path.join(scene.model_path, "train_process", "alpha")
            os.makedirs(alpha_path, exist_ok=True)
            torchvision.utils.save_image(alpha.detach().cpu(), os.path.join(alpha_path, '{0:05d}'.format(iteration) + ".png"))

            if rendered_diffuse_color is not None:
                normal_path = os.path.join(scene.model_path, "train_process", "normal")
                os.makedirs(normal_path, exist_ok=True)
                torchvision.utils.save_image(rendered_normal.detach().cpu(), os.path.join(normal_path, '{0:05d}'.format(iteration) + ".png"))
                
                albedo_path = os.path.join(scene.model_path, "train_process", "albedo")
                os.makedirs(albedo_path, exist_ok=True)
                torchvision.utils.save_image(rendered_albedo.detach().cpu(), os.path.join(albedo_path, '{0:05d}'.format(iteration) + ".png"))

                metallic_path = os.path.join(scene.model_path, "train_process", "metallic")
                os.makedirs(metallic_path, exist_ok=True)
                torchvision.utils.save_image(rendered_metallic.detach().cpu(), os.path.join(metallic_path, '{0:05d}'.format(iteration) + ".png"))

                roughness_path = os.path.join(scene.model_path, "train_process", "roughness")
                os.makedirs(roughness_path, exist_ok=True)
                torchvision.utils.save_image(rendered_roughness.detach().cpu(), os.path.join(roughness_path, '{0:05d}'.format(iteration) + ".png"))

                hdr_path = os.path.join(scene.model_path, "train_process", "hdr")
                os.makedirs(hdr_path, exist_ok=True)
                hdr_image = cubemap_to_latlong(gaussians.envlight.base.detach(), [1024, 2048]).permute(2,0,1).contiguous()
                torchvision.utils.save_image(hdr_image.detach().cpu(), os.path.join(hdr_path, '{0:05d}'.format(iteration) + ".png"))

                diffuse_color_path = os.path.join(scene.model_path, "train_process", "diffuse_color")
                os.makedirs(diffuse_color_path, exist_ok=True)
                torchvision.utils.save_image(rendered_diffuse_color.detach().cpu(), os.path.join(diffuse_color_path, '{0:05d}'.format(iteration) + ".png"))

                specular_color_path = os.path.join(scene.model_path, "train_process", "specular_color")
                os.makedirs(specular_color_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_color.detach().cpu(), os.path.join(specular_color_path, '{0:05d}'.format(iteration) + ".png"))

                diffuse_albedo_path = os.path.join(scene.model_path, "train_process", "diffuse_albedo")
                os.makedirs(diffuse_albedo_path, exist_ok=True)
                torchvision.utils.save_image(rendered_diffuse_albedo.detach().cpu(), os.path.join(diffuse_albedo_path, '{0:05d}'.format(iteration) + ".png"))

                specular_albedo_path = os.path.join(scene.model_path, "train_process", "specular_albedo")
                os.makedirs(specular_albedo_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_albedo.detach().cpu(), os.path.join(specular_albedo_path, '{0:05d}'.format(iteration) + ".png"))

                diffuse_light_path = os.path.join(scene.model_path, "train_process", "diffuse_light")
                os.makedirs(diffuse_light_path, exist_ok=True)
                torchvision.utils.save_image(rendered_diffuse_light.detach().cpu(), os.path.join(diffuse_light_path, '{0:05d}'.format(iteration) + ".png"))

                specular_light_path = os.path.join(scene.model_path, "train_process", "specular_light")
                os.makedirs(specular_light_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_light.detach().cpu(), os.path.join(specular_light_path, '{0:05d}'.format(iteration) + ".png"))

            if rendered_specular_indirect_light is not None:
                specular_indirect_light_path = os.path.join(scene.model_path, "train_process", "specular_indirect_light")
                os.makedirs(specular_indirect_light_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_indirect_light.detach().cpu(), os.path.join(specular_indirect_light_path, '{0:05d}'.format(iteration) + ".png"))

                specular_direct_light_path = os.path.join(scene.model_path, "train_process", "specular_direct_light")
                os.makedirs(specular_direct_light_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_direct_light.detach().cpu(), os.path.join(specular_direct_light_path, '{0:05d}'.format(iteration) + ".png"))

                specular_indirect_color_path = os.path.join(scene.model_path, "train_process", "specular_indirect_color")
                os.makedirs(specular_indirect_color_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_indirect_color.detach().cpu(), os.path.join(specular_indirect_color_path, '{0:05d}'.format(iteration) + ".png"))

                specular_direct_color_path = os.path.join(scene.model_path, "train_process", "specular_direct_color")
                os.makedirs(specular_direct_color_path, exist_ok=True)
                torchvision.utils.save_image(rendered_specular_direct_color.detach().cpu(), os.path.join(specular_direct_color_path, '{0:05d}'.format(iteration) + ".png"))
            
                occ_path = os.path.join(scene.model_path, "train_process", "occ")
                os.makedirs(occ_path, exist_ok=True)
                torchvision.utils.save_image(rendered_occ.detach().cpu(), os.path.join(occ_path, '{0:05d}'.format(iteration) + ".png"))
            
            if hasattr(viewpoint_cam, 'albedo_gt') and viewpoint_cam.albedo_gt is not None and rendered_albedo is not None and rendered_normal is not None:
                albedo_comp_path = os.path.join(scene.model_path, "train_process", "albedo_gt_comparison")
                os.makedirs(albedo_comp_path, exist_ok=True)
                albedo_grid = torchvision.utils.make_grid([rendered_albedo.detach().cpu(), viewpoint_cam.albedo_gt.detach().cpu()], nrow=2, padding=4, pad_value=1.0)
                torchvision.utils.save_image(albedo_grid, os.path.join(albedo_comp_path, '{0:05d}'.format(iteration) + ".png"))

                normal_comp_path = os.path.join(scene.model_path, "train_process", "normal_gt_comparison")
                os.makedirs(normal_comp_path, exist_ok=True)
                normal_grid = torchvision.utils.make_grid([rendered_normal.detach().cpu(), viewpoint_cam.normal_gt.detach().cpu()], nrow=2, padding=4, pad_value=1.0)
                torchvision.utils.save_image(normal_grid, os.path.join(normal_comp_path, '{0:05d}'.format(iteration) + ".png"))

                albedo_err_path = os.path.join(scene.model_path, "train_process", "albedo_error")
                os.makedirs(albedo_err_path, exist_ok=True)
                albedo_err = torch.abs(rendered_albedo.detach().cpu() - viewpoint_cam.albedo_gt.detach().cpu()).mean(dim=0, keepdim=True).repeat(3, 1, 1)
                torchvision.utils.save_image(albedo_err, os.path.join(albedo_err_path, '{0:05d}'.format(iteration) + ".png"))

                normal_err_path = os.path.join(scene.model_path, "train_process", "normal_error")
                os.makedirs(normal_err_path, exist_ok=True)
                normal_err = torch.abs(rendered_normal.detach().cpu() - viewpoint_cam.normal_gt.detach().cpu()).mean(dim=0, keepdim=True).repeat(3, 1, 1)
                torchvision.utils.save_image(normal_err, os.path.join(normal_err_path, '{0:05d}'.format(iteration) + ".png"))

                if hasattr(gaussians, 'envlight') and gaussians.envlight is not None:
                    hdr_base = gaussians.envlight.base.detach()
                    print(f"\n[ITER {iteration}] EnvMap Stats — Min: {hdr_base.min().item():.4f}, Max: {hdr_base.max().item():.4f}, Mean: {hdr_base.mean().item():.4f}")
                
                print(f"[ITER {iteration}] Extended Loss — albedo_gt: {loss_albedo_gt_val.item():.4f}, normal_gt: {loss_normal_gt_val.item():.4f}, metallic_gt: {loss_metallic_gt_val.item():.4f}, roughness_gt: {loss_roughness_gt_val.item():.4f}")
                loss_entry = {
                    "iteration": iteration,
                    "albedo_gt": loss_albedo_gt_val.item(),
                    "normal_gt": loss_normal_gt_val.item(),
                    "metallic_gt": loss_metallic_gt_val.item(),
                    "roughness_gt": loss_roughness_gt_val.item(),
                    "lambda_albedo": lambda_albedo_gt * multiplier,
                    "lambda_normal": lambda_normal_gt * multiplier,
                    "lambda_metallic": lambda_metallic_gt * multiplier,
                    "lambda_roughness": lambda_roughness_gt * multiplier,
                    "lambda_reg_hdr": reg_hdr_weight * multiplier,
                    "prior_multiplier": multiplier,
                    "albedo_prior_mode": albedo_prior_mode,
                }
                loss_logs.append(loss_entry)
                os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
                with open(loss_log_path, "w") as f:
                    json.dump(loss_logs, f, indent=2)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, first_stage_step, second_stage_step, remove_noise, hdr_rotation, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (do not densify or reset opacity on the very last iteration)
            if iteration < opt.densify_until_iter and iteration < opt.iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    alpha_ = 0.2
                    # Optional hard cap: stop *growing* the gaussian count once the cap is
                    # reached (pruning still runs to keep memory in check). 0 = unlimited.
                    below_cap = (opt.max_gaussians <= 0) or (gaussians.get_xyz.shape[0] < opt.max_gaussians)
                    if below_cap:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent * alpha_, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    # Optionally skip opacity resets in the third (PBR) stage: by
                    # then the geometry is settled, and a hard reset can disturb
                    # the decomposition. A/B-controlled via disable_reset_third_stage.
                    if not (disable_reset_third_stage and iteration > second_stage_step):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # --- Periodic evaluation ---
            if iteration % eval_interval == 0 or iteration == opt.iterations:
                eval_count += 1
                run_relighting_eval = (eval_count % 2 == 0) or (iteration == opt.iterations)
                save_visuals = (iteration % visual_interval == 0) or (iteration == opt.iterations)
                eval_results = periodic_evaluation(
                    iteration, scene, gaussians, pipe, background,
                    first_stage_step, second_stage_step, remove_noise, hdr_rotation,
                    ema_loss_for_log, lpips_model, save_visuals=save_visuals,
                    run_relighting_eval=run_relighting_eval,
                    eval_relight_hdris=eval_relight_hdris,
                    envmap_gt_path=envmap_gt_path,
                    albedo_geometry_warmup=albedo_geometry_warmup,
                    light_linear_indirect=light_linear_indirect,
                    relight_max_views=relight_max_views
                )
                metrics_log.append(eval_results)
                with open(metrics_log_path, "w") as f:
                    json.dump(metrics_log, f, indent=2)

    # --- Generate PDF report after training ---
    print("\nGenerating training report PDF...")
    try:
        from generate_report import generate_report
        report_path = generate_report(scene.model_path)
        print(f"Training report saved to: {report_path}")
    except Exception as e:
        print(f"Warning: Could not generate PDF report: {e}")
        print("You can generate it manually with: python generate_report.py --model_path " + scene.model_path)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    args.model_path = os.path.abspath(args.model_path)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, first_stage_step, second_stage_step, remove_noise, hdr_rotation, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, iteration=iteration, is_train=False, first_stage_step=first_stage_step, second_stage_step=second_stage_step, remove_noise=remove_noise, hdr_rotation=hdr_rotation, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image, _ = get_mask(gt_image)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                sys.stdout.flush()
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--first_stage_step", type=int, default = 5_000)
    parser.add_argument("--second_stage_step", type=int, default = 30_000)
    parser.add_argument("--remove_noise", action="store_true", default=False)
    parser.add_argument("--hdr_rotation", action="store_true", default=False)
    parser.add_argument("--reg_hdr_weight", type=float, default=0.001)
    parser.add_argument("--reg_material_weight", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluate metrics every N iterations")
    parser.add_argument("--relight_max_views", type=int, default=0, help="Cap the relighting eval to this many evenly spaced test cameras (0 = use all test cameras); cuts the relight-eval cost by n_test/N while staying deterministic and comparable across runs")
    parser.add_argument("--visual_interval", type=int, default=10000, help="Save visual comparisons every N iterations")
    parser.add_argument("--lambda_albedo_gt", type=float, default=0.1, help="Weight for albedo GT loss")
    parser.add_argument("--lambda_normal_gt", type=float, default=0.1, help="Weight for normal GT loss")
    parser.add_argument("--lambda_metallic_gt", type=float, default=0.05, help="Weight for metallic GT loss")
    parser.add_argument("--lambda_roughness_gt", type=float, default=0.05, help="Weight for roughness GT loss")
    parser.add_argument("--use_prior_weight_scheduler", action="store_true", default=False, help="Ramp prior/envmap-reg weights up then partially back down (off = constant weights)")
    parser.add_argument("--prior_weight_scheduler_ratio", type=float, default=0.15, help="Warm-up ratio (fraction of PBR-stage steps spent ramping 0->1)")
    parser.add_argument("--prior_weight_final_ratio", type=float, default=1.0, help="Value the prior/envmap weight schedule ramps to after warm-up (fraction of max weight). 1.0 = hold at full strength, >1 = increase, <1 = decay toward a floor (e.g. 0.5 = old behavior)")
    parser.add_argument("--albedo_prior_mode", type=str, default="direct", choices=["direct", "lstsq", "log_chroma", "gradient", "zncc", "zncc_local", "zncc_grad", "ssim_struct", "si_ema"], help="Albedo prior formulation (PBR stage 3): direct | lstsq (per-channel gain align) | log_chroma (log+chromaticity) | gradient (spatial-gradient/edge) | zncc (scale-&-shift-invariant) | zncc_local (locally-normalised correlation) | zncc_grad (gradient-domain correlation) | ssim_struct (structure-focused SSIM) | si_ema (scale-invariant log loss with EMA-shared global gain + log-gradient term)")
    parser.add_argument("--warmup_albedo_prior_mode", type=str, default="", choices=["", "direct", "lstsq", "log_chroma", "gradient", "zncc", "zncc_local", "zncc_grad", "ssim_struct", "si_ema"], help="Albedo prior formulation used during the geometry/normal WARM-UP stages (1 & 2) when --albedo_geometry_warmup is set. Empty = use the same mode as --albedo_prior_mode. Lets you e.g. warm up with 'direct' and decompose with 'zncc' in stage 3.")
    parser.add_argument("--huber_delta", type=float, default=0.1, help="Delta for the robust Huber prior losses")
    parser.add_argument("--exclude_prior_loss", action="store_true", default=False, help="Exclude GT priors loss from optimization, but keep calculating it for debug/logging purposes")
    parser.add_argument("--eval_relight_hdris", nargs="+", type=str, default=['snowy_forest', 'moonless_night', 'fireplace'], help="List of HDRI names to evaluate relighting on")
    parser.add_argument("--envmap_gt_path", type=str, default="", help="Path to the GT base-light HDRI for the environment-map recovery metric (optional)")
    parser.add_argument("--tv_reduction_factor", type=float, default=1.0, help="Scale (0..1) applied to a property's TV/smoothness regularizer when that property has a GT prior (0 = off, 1 = unchanged)")
    parser.add_argument("--reduce_geo_lr_third_stage", type=float, default=1.0, help="Final multiplier on geometry LRs (xyz/scaling/rotation), cosine-annealed across the third stage. 1.0 = off (no change), e.g. 0.05 = reduce to 5%%")
    parser.add_argument("--geo_lr_final_iter", type=int, default=0, help="Iteration at which reduce_geo_lr_third_stage is fully reached (<=0 means end of training)")
    parser.add_argument("--disable_reset_third_stage", action="store_true", default=False, help="Skip opacity resets once iteration > second_stage_step (third / PBR stage)")
    parser.add_argument("--albedo_geometry_warmup", action="store_true", default=False, help="Supervise the flat rendered albedo against the GT albedo prior during the geometry/normal stages (iter <= second_stage_step) instead of the RGB photometric loss; the photometric/PBR loss is reintroduced only in the PBR stage. The warmed-up albedo is carried over (not reset) into the PBR stage. Combine with reduce_geo_lr_third_stage to freeze geometry once PBR begins.")
    parser.add_argument("--prior_geom_grad_scale", type=float, default=1.0, help="Scale on the geometry gradients of the stage-3 albedo/material prior rasters: 1.0 = unchanged, 0.0 = fully blocked (priors teach only material values; geometry stays owned by the photometric + normal losses), intermediate values let priors nudge geometry gently")
    parser.add_argument("--albedo_anchor_weight", type=float, default=0.0, help="Weight of a weak ABSOLUTE (direct Huber) albedo anchor added on top of the gain-invariant albedo prior mode in stage 3; prevents global albedo gain drift (0 = off)")
    parser.add_argument("--light_linear_indirect", action="store_true", help="Express the baked indirect terms (specular SH indirect + a new diffuse bounce for occluded sample directions) as reflectance x mean radiance of the CURRENT envmap, so bounce energy rescales with the light at relight instead of staying frozen at the training light's level")
    parser.add_argument("--lambda_normal_third_stage_scale", type=float, default=1.0, help="Extra multiplier on lambda_normal_gt applied ONLY in the PBR stage (Phase 2 keeps the full weight); lower it for noisy diffusion normal priors")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # A GT prior whose folder is unset ("") is disabled: force its loss weight
    # to 0 so it is neither optimized nor (under exclude_prior_loss) logged.
    if not args.albedo_gt_dir:
        args.lambda_albedo_gt = 0.0
    if not args.normal_gt_dir:
        args.lambda_normal_gt = 0.0
    if not args.metallic_gt_dir:
        args.lambda_metallic_gt = 0.0
    if not args.roughness_gt_dir:
        args.lambda_roughness_gt = 0.0

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.first_stage_step, args.second_stage_step, args.remove_noise, args.hdr_rotation, args.reg_hdr_weight, args.reg_material_weight, args.eval_interval, args.visual_interval, args.lambda_albedo_gt, args.lambda_normal_gt, args.lambda_metallic_gt, args.lambda_roughness_gt, args.use_prior_weight_scheduler, args.prior_weight_scheduler_ratio, args.prior_weight_final_ratio, args.albedo_prior_mode, args.huber_delta, args.exclude_prior_loss, args.eval_relight_hdris, args.envmap_gt_path, args.tv_reduction_factor, args.reduce_geo_lr_third_stage, args.geo_lr_final_iter, args.disable_reset_third_stage, args.albedo_geometry_warmup, args.warmup_albedo_prior_mode, args.prior_geom_grad_scale, args.albedo_anchor_weight, args.lambda_normal_third_stage_scale, args.light_linear_indirect, args.relight_max_views)

    # All done
    print("\nTraining complete.")
