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
import time
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, smooth_loss, regularizer_loss, get_mask, tv_loss
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

def periodic_evaluation(iteration, scene, gaussians, pipe, background, first_stage_step, second_stage_step, remove_noise, hdr_rotation, ema_loss, lpips_model, save_visuals=False):
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
        albedo_psnr_vals, albedo_ssim_vals, albedo_l1_vals = [], [], []
        normal_angular_error_vals = []
        visual_pairs = []  # (render, gt) for visual comparison

        for idx, viewpoint in enumerate(config["cameras"]):
            render_pkg = render(viewpoint, gaussians, pipe, background,
                                iteration=iteration, is_train=False,
                                first_stage_step=first_stage_step,
                                second_stage_step=second_stage_step,
                                remove_noise=remove_noise,
                                hdr_rotation=hdr_rotation)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_image, _ = get_mask(gt_image)

            psnr_vals.append(psnr(image, gt_image).mean().item())
            ssim_vals.append(ssim(image, gt_image).item())
            l1_vals.append(l1_loss(image, gt_image).item())
            mse_vals.append(mse(image, gt_image).mean().item())
            if hasattr(viewpoint, 'albedo_gt') and viewpoint.albedo_gt is not None:
                gt_albedo = viewpoint.albedo_gt
                rendered_albedo = render_pkg.get("rendered_albedo", None)
                if rendered_albedo is not None:
                    rendered_albedo_clamped = torch.clamp(rendered_albedo, 0.0, 1.0)
                    albedo_psnr_vals.append(psnr(rendered_albedo_clamped, gt_albedo).mean().item())
                    albedo_ssim_vals.append(ssim(rendered_albedo_clamped, gt_albedo).item())
                    albedo_l1_vals.append(l1_loss(rendered_albedo_clamped, gt_albedo).item())
                
                gt_normal = viewpoint.normal_gt
                rendered_normal = render_pkg.get("rendered_normal", None)
                if rendered_normal is not None:
                    pred_n = rendered_normal * 2.0 - 1.0
                    gt_n = gt_normal * 2.0 - 1.0
                    pred_n = F.normalize(pred_n, p=2, dim=0)
                    gt_n = F.normalize(gt_n, p=2, dim=0)
                    cos_sim = torch.clamp(torch.sum(pred_n * gt_n, dim=0), -1.0, 1.0)
                    ang_error = torch.acos(cos_sim) * 180.0 / 3.141592653589793
                    normal_angular_error_vals.append(ang_error.mean().item())

            # LPIPS with cached model (already on GPU)
            with torch.no_grad():
                lpips_val = lpips_model(image.unsqueeze(0) if image.dim() == 3 else image,
                                       gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
                lpips_vals.append(lpips_val.item())

            if save_visuals and idx < 3:  # Save first 3 views
                visual_pairs.append((image.detach().cpu(), gt_image.detach().cpu()))

        prefix = config["name"]
        results[f"{prefix}_psnr"] = sum(psnr_vals) / len(psnr_vals)
        results[f"{prefix}_ssim"] = sum(ssim_vals) / len(ssim_vals)
        results[f"{prefix}_lpips"] = sum(lpips_vals) / len(lpips_vals)
        results[f"{prefix}_l1"] = sum(l1_vals) / len(l1_vals)
        results[f"{prefix}_mse"] = sum(mse_vals) / len(mse_vals)

        if albedo_psnr_vals:
            results[f"{prefix}_albedo_psnr"] = sum(albedo_psnr_vals) / len(albedo_psnr_vals)
            results[f"{prefix}_albedo_ssim"] = sum(albedo_ssim_vals) / len(albedo_ssim_vals)
            results[f"{prefix}_albedo_l1"] = sum(albedo_l1_vals) / len(albedo_l1_vals)
        if normal_angular_error_vals:
            results[f"{prefix}_normal_ang_err"] = sum(normal_angular_error_vals) / len(normal_angular_error_vals)

        # Save visual comparison grids
        if save_visuals and visual_pairs:
            vis_path = os.path.join(scene.model_path, "eval_visuals", prefix)
            os.makedirs(vis_path, exist_ok=True)
            for vi, (rend, gt) in enumerate(visual_pairs):
                grid = torchvision.utils.make_grid([rend, gt], nrow=2, padding=4, pad_value=1.0)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"iter{iteration:06d}_view{vi}.png"))

    albedo_info = ""
    if "test_albedo_psnr" in results:
        albedo_info = f", Alb PSNR: {results['test_albedo_psnr']:.2f}, Norm Err: {results['test_normal_ang_err']:.2f} deg"

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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, first_stage_step, second_stage_step, remove_noise, hdr_rotation, reg_hdr_weight=0.001, reg_material_weight=0.1, eval_interval=2000, visual_interval=10000, lambda_albedo_gt=0.5, lambda_normal_gt=0.1, lambda_metallic_gt=0.05, exclude_prior_loss=False, freeze_uncertainty_weights=False):
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


    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
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
                    net_image = render(viewpoint_cam, gaussians, pipe, background, random_bg_color=bg, iteration=iteration, is_train=True, first_stage_step=first_stage_step, second_stage_step=second_stage_step, remove_noise=remove_noise, hdr_rotation=hdr_rotation)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration >= second_stage_step:
            if iteration % 1000==0:
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

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, random_bg_color=bg, iteration=iteration, is_train=True, first_stage_step=first_stage_step, second_stage_step=second_stage_step, remove_noise=remove_noise, hdr_rotation=hdr_rotation)
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
        loss_albedo_gt_val = torch.tensor(0.0).cuda()
        loss_normal_gt_val = torch.tensor(0.0).cuda()
        loss_metallic_gt_val = torch.tensor(0.0).cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        gt_image, gt_mask = get_mask(gt_image)        
        gt_image = gt_image * gt_mask + bg.unsqueeze(-1).unsqueeze(-1).repeat(1, gt_image.shape[1], gt_image.shape[2]) * (1-gt_mask)
        Ll1 = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_image
        if iteration > second_stage_step:
            loss_albedo = tv_loss(rendered_albedo) * 0.1
            loss_normal = smooth_loss(rendered_normal, gt_image) * 0.01 # 1
            loss_regularizer = regularizer_loss(gaussians.envlight.base) * reg_hdr_weight
            loss_metallic = tv_loss(rendered_metallic) * reg_material_weight
            loss_roughness = tv_loss(rendered_roughness) * reg_material_weight
            loss = loss + loss_albedo + loss_normal + loss_metallic + loss_roughness + loss_regularizer #+ Ll1_alpha

            # --- Extended GT-supervised PBR losses (Kendall et al. uncertainty weighting) ---
            if hasattr(viewpoint_cam, 'albedo_gt') and viewpoint_cam.albedo_gt is not None:
                gt_albedo = viewpoint_cam.albedo_gt
                gt_normal = viewpoint_cam.normal_gt
                gt_metallic_val = viewpoint_cam.metallic_gt

                if isinstance(gt_metallic_val, torch.Tensor):
                    gt_metallic = gt_metallic_val
                else:
                    gt_metallic = torch.full_like(rendered_metallic, gt_metallic_val)

                if exclude_prior_loss:
                    # Compute base losses without gradients — logging only
                    with torch.no_grad():
                        loss_albedo_gt_val = (1.0 - opt.lambda_dssim) * l1_loss(rendered_albedo, gt_albedo) \
                                             + opt.lambda_dssim * (1.0 - ssim(rendered_albedo, gt_albedo))
                        pred_n = rendered_normal * 2.0 - 1.0
                        gt_n = gt_normal * 2.0 - 1.0
                        cos_sim = F.cosine_similarity(pred_n, gt_n, dim=0)
                        loss_normal_gt_val = (1.0 - cos_sim).mean()
                        loss_metallic_gt_val = l1_loss(rendered_metallic, gt_metallic)
                else:
                    # Compute base (unweighted) losses for each G-buffer channel
                    base_loss_albedo = (1.0 - opt.lambda_dssim) * l1_loss(rendered_albedo, gt_albedo) \
                                       + opt.lambda_dssim * (1.0 - ssim(rendered_albedo, gt_albedo))
                    pred_n = rendered_normal * 2.0 - 1.0
                    gt_n = gt_normal * 2.0 - 1.0
                    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=0)
                    base_loss_normal = (1.0 - cos_sim).mean()
                    base_loss_metallic = l1_loss(rendered_metallic, gt_metallic)

                    # Store for logging (detached from graph)
                    loss_albedo_gt_val = base_loss_albedo.detach()
                    loss_normal_gt_val = base_loss_normal.detach()
                    loss_metallic_gt_val = base_loss_metallic.detach()

                    # Kendall et al. multi-task uncertainty weighting:
                    #   L_prior = sum_i [ L_i * exp(-w_i) + w_i ]
                    # where w_i is a learnable log-variance scalar.
                    #
                    # CRITICAL STABILITY POLISH: To prevent exponential overflow (exp(-w) -> infinity)
                    # and numerical NaN crashes when base losses become extremely close to zero,
                    # we clamp the log-variance parameters w to a safe operational range of [-5.0, 10.0].
                    # This allows effective loss weights to scale up to exp(5.0) ≈ 148.4x down to exp(-10.0) ≈ 4.5e-5x.
                    w_a = torch.clamp(gaussians._w_albedo, min=-5.0, max=10.0)
                    w_m = torch.clamp(gaussians._w_metallic, min=-5.0, max=10.0)
                    w_n = torch.clamp(gaussians._w_normal, min=-5.0, max=10.0)

                    loss_prior = (base_loss_albedo * torch.exp(-w_a) + w_a) \
                               + (base_loss_metallic * torch.exp(-w_m) + w_m) \
                               + (base_loss_normal * torch.exp(-w_n) + w_n)

                    loss = loss + loss_prior
        loss.backward()

        # When frozen, zero w_ gradients so they stay at w=0 (unit weighting)
        if freeze_uncertainty_weights:
            for w_param in (gaussians._w_albedo, gaussians._w_metallic, gaussians._w_normal):
                if w_param.grad is not None:
                    w_param.grad.zero_()

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
                
                print(f"[ITER {iteration}] Extended Loss — albedo_gt: {loss_albedo_gt_val.item():.4f}, normal_gt: {loss_normal_gt_val.item():.4f}, metallic_gt: {loss_metallic_gt_val.item():.4f}")
                w_a, w_m, w_n = gaussians._w_albedo.item(), gaussians._w_metallic.item(), gaussians._w_normal.item()
                print(f"[ITER {iteration}] Uncertainty w — albedo: {w_a:.4f} (eff: {torch.exp(-gaussians._w_albedo).item():.4f}), "
                      f"metallic: {w_m:.4f} (eff: {torch.exp(-gaussians._w_metallic).item():.4f}), "
                      f"normal: {w_n:.4f} (eff: {torch.exp(-gaussians._w_normal).item():.4f})")
                
                loss_entry = {
                    "iteration": iteration,
                    "albedo_gt": loss_albedo_gt_val.item(),
                    "normal_gt": loss_normal_gt_val.item(),
                    "metallic_gt": loss_metallic_gt_val.item(),
                    "w_albedo": gaussians._w_albedo.item(),
                    "w_metallic": gaussians._w_metallic.item(),
                    "w_normal": gaussians._w_normal.item(),
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent * alpha_, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
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
                save_visuals = (iteration % visual_interval == 0) or (iteration == opt.iterations)
                eval_results = periodic_evaluation(
                    iteration, scene, gaussians, pipe, background,
                    first_stage_step, second_stage_step, remove_noise, hdr_rotation,
                    ema_loss_for_log, lpips_model, save_visuals=save_visuals
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
    parser.add_argument("--visual_interval", type=int, default=10000, help="Save visual comparisons every N iterations")
    parser.add_argument("--lambda_albedo_gt", type=float, default=0.5, help="Weight for albedo GT loss")
    parser.add_argument("--lambda_normal_gt", type=float, default=0.1, help="Weight for normal GT loss")
    parser.add_argument("--lambda_metallic_gt", type=float, default=0.05, help="Weight for metallic GT loss")
    parser.add_argument("--exclude_prior_loss", action="store_true", default=False, help="Exclude GT priors loss from optimization, but keep calculating it for debug/logging purposes")
    parser.add_argument("--freeze_uncertainty_weights", action="store_true", default=False, help="Keep Kendall uncertainty weights frozen at w=0 (unit weighting) for A/B comparison")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.first_stage_step, args.second_stage_step, args.remove_noise, args.hdr_rotation, args.reg_hdr_weight, args.reg_material_weight, args.eval_interval, args.visual_interval, args.lambda_albedo_gt, args.lambda_normal_gt, args.lambda_metallic_gt, args.exclude_prior_loss, args.freeze_uncertainty_weights)

    # All done
    print("\nTraining complete.")
