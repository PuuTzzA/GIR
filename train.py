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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, smooth_loss, regularizer_loss, get_mask, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torchvision
from envlight.utils import cubemap_to_latlong

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, first_stage_step, second_stage_step, remove_noise, hdr_rotation, reg_hdr_weight=0.001, reg_material_weight=0.1, lambda_albedo=0.0, lambda_normal=0.0, lambda_metallic=0.0, gt_loss_type="l1"):
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
        del render_pkg  # Free the dictionary reference to allow GC of unused buffers
        # Loss
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
        # ========== GT Material Supervision Losses (masked to foreground only) ==========
        # Entirely skipped when all lambdas are 0 (baseline mode) to save GPU memory.
        if lambda_albedo > 0 or lambda_normal > 0 or lambda_metallic > 0:
            def _masked_gt_loss(pred, target, mask, n_fg_pixels):
                diff = pred - target
                if gt_loss_type == "huber":
                    per_pixel = torch.nn.functional.huber_loss(pred, target, reduction='none', delta=0.1)
                else:  # "l1"
                    per_pixel = torch.abs(diff)
                return (per_pixel * mask).sum() / (n_fg_pixels * pred.shape[0])
            fg_mask = viewpoint_cam.gt_alpha_mask.cuda()  # (1, H, W), 1=foreground
            n_fg = fg_mask.sum().clamp(min=1.0)  # avoid div-by-zero
            if lambda_albedo > 0 and rendered_albedo is not None and viewpoint_cam.albedo_gt is not None:
                gt_albedo = viewpoint_cam.albedo_gt.cuda()
                loss = loss + lambda_albedo * _masked_gt_loss(rendered_albedo, gt_albedo, fg_mask, n_fg)
            if lambda_normal > 0 and rendered_normal is not None and viewpoint_cam.normal_gt is not None:
                gt_normal = viewpoint_cam.normal_gt.cuda()
                loss = loss + lambda_normal * _masked_gt_loss(rendered_normal, gt_normal, fg_mask, n_fg)
            if lambda_metallic > 0 and rendered_metallic is not None:
                zero_metallic = torch.zeros_like(rendered_metallic)
                loss = loss + lambda_metallic * _masked_gt_loss(rendered_metallic, zero_metallic, fg_mask, n_fg)
            del fg_mask
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
                torchvision.utils.save_image(rendered_metallic.expand(3,-1,-1).detach().cpu(), os.path.join(metallic_path, '{0:05d}'.format(iteration) + ".png"))

                roughness_path = os.path.join(scene.model_path, "train_process", "roughness")
                os.makedirs(roughness_path, exist_ok=True)
                torchvision.utils.save_image(rendered_roughness.expand(3,-1,-1).detach().cpu(), os.path.join(roughness_path, '{0:05d}'.format(iteration) + ".png"))

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
            

        # === Save intermediate renders + point cloud every 5000 iterations ===
        if iteration % 5000 == 0 and iteration > 0:
            with torch.no_grad():
                intermediate_dir = os.path.join(scene.model_path, "intermediate_renders", f"iter_{iteration}")
                os.makedirs(intermediate_dir, exist_ok=True)
                # Save RGB from the training render (already computed above)
                torchvision.utils.save_image(image.clamp(0.0, 1.0).detach().cpu(), os.path.join(intermediate_dir, "rgb.png"))
                # Save G-buffers from the training render (None during first stage, available after)
                if rendered_albedo is not None:
                    torchvision.utils.save_image(rendered_albedo.clamp(0.0, 1.0).detach().cpu(), os.path.join(intermediate_dir, "albedo.png"))
                if rendered_normal is not None:
                    torchvision.utils.save_image(rendered_normal.clamp(0.0, 1.0).detach().cpu(), os.path.join(intermediate_dir, "normal.png"))
                if rendered_metallic is not None:
                    torchvision.utils.save_image(rendered_metallic.expand(3,-1,-1).clamp(0.0, 1.0).detach().cpu(), os.path.join(intermediate_dir, "metallic.png"))
                if rendered_roughness is not None:
                    torchvision.utils.save_image(rendered_roughness.expand(3,-1,-1).clamp(0.0, 1.0).detach().cpu(), os.path.join(intermediate_dir, "roughness.png"))
                # Save point cloud (for re-rendering later)
                gaussians.save_ply(os.path.join(intermediate_dir, "point_cloud.ply"))
                # Save full checkpoint (Gaussian params + envlight state for re-rendering)
                torch.save((gaussians.capture(), iteration), os.path.join(intermediate_dir, "checkpoint.pth"))
                print(f"\n[ITER {iteration}] Saved renders + point cloud + checkpoint to {intermediate_dir}")
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

            # Densification
            if iteration < opt.densify_until_iter:
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

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.first_stage_step, args.second_stage_step, args.remove_noise, args.hdr_rotation, args.reg_hdr_weight, args.reg_material_weight, args.lambda_albedo, args.lambda_normal, args.lambda_metallic, args.gt_loss_type)

    # All done
    print("\nTraining complete.")
