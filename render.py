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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from envlight.utils import cubemap_to_latlong

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_religt=False, second_stage_step = 30000, hdr_rotation = False, light_linear_indirect=False):
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, iteration=iteration, is_train= not render_religt, second_stage_step=second_stage_step, hdr_rotation=hdr_rotation, light_linear_indirect=light_linear_indirect)
        image = render_pkg["render"]
        
        gt = view.original_image[0:3, :, :]

        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        makedirs(render_path, exist_ok=True)
        torchvision.utils.save_image(image.clamp(0.0, 1.0), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        if not render_religt:
            gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
            makedirs(gts_path, exist_ok=True)
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, save_name : str, render_religt : bool, second_stage_step : int, hdr_rotation : bool, environment_texture : str, environment_scale : float = 1.0, light_linear_indirect : bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, environment_texture=environment_texture, environment_scale=environment_scale)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.get_diffuse_occ()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train_"+save_name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_religt, second_stage_step, hdr_rotation, light_linear_indirect)

        if not skip_test:
             render_set(dataset.model_path, "test_"+save_name, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_religt, second_stage_step, hdr_rotation, light_linear_indirect)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--render_relight", action="store_true")
    parser.add_argument("--second_stage_step", default=30000, type=int)
    parser.add_argument("--hdr_rotation", action="store_true")
    parser.add_argument("--environment_texture", type=str, default="hdri/flower_road_no_sun_2k.hdr")
    parser.add_argument("--light_linear_indirect", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.save_name, args.render_relight, args.second_stage_step, args.hdr_rotation, args.environment_texture, light_linear_indirect=args.light_linear_indirect)

    