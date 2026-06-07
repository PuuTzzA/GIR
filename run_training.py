#
# Central training launcher for the (modified) GIR engine.
#
# This file collects ALL parameters that can be used to train GIR in a single
# place. Edit the CONFIG dictionary below to configure a run, then launch with:
#
#     python run_training.py
#
# Any value in CONFIG may also be overridden from the command line, e.g.:
#
#     python run_training.py --source_path /data/lego --model_path outputs/lego \
#         --eval --lambda_albedo_gt 0.5 --lambda_normal_gt 0.1
#
# The parameters are grouped exactly like the engine groups them:
#   - ModelParams        (scene / data loading)
#   - PipelineParams     (rasterizer pipeline)
#   - OptimizationParams (learning rates, densification, ...)
#   - Engine / staging   (two-stage PBR training control)
#   - Priors             (albedo / normal / metallic GT-prior supervision that
#                         was ADDED on top of the original GIR engine)
#   - Logging / eval     (evaluation + visual logging + relighting eval)
#
# For inquiries about the original GIR engine contact george.drettakis@inria.fr
#

import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from gaussian_renderer import network_gui
from train import training


# =============================================================================
# CENTRAL CONFIGURATION
# -----------------------------------------------------------------------------
# Every parameter the GIR engine understands lives here. Values set to None are
# left at the engine default (defined in arguments/__init__.py or train.py).
# =============================================================================
CONFIG = {
    # ----------------------------------------------------------------------
    # ModelParams - scene / dataset loading
    # ----------------------------------------------------------------------
    "sh_degree": 3,                 # default: 3      | order of spherical harmonics
    "source_path": "",              # default: ""     | -s : path to the COLMAP / dataset folder (REQUIRED)
    "model_path": "",               # default: ""     | output folder for the trained model
    "images": "images",             # default: images | sub-folder containing input images
    "resolution": -1,               # default: -1     | -1 = keep native resolution
    "white_background": False,       # default: False  | -w : use a white background
    "data_device": "cuda",          # default: cuda   | device the dataset tensors live on
    "eval": False,                   # default: False  | hold out test cameras for evaluation

    # ----------------------------------------------------------------------
    # PipelineParams - rasterizer / rendering pipeline
    # ----------------------------------------------------------------------
    "convert_SHs_python": True,      # default: True   | compute SH -> RGB in python
    "compute_cov3D_python": False,   # default: False  | compute 3D covariance in python
    "debug": False,                  # default: False  | rasterizer debug mode

    # ----------------------------------------------------------------------
    # OptimizationParams - schedule
    # ----------------------------------------------------------------------
    "iterations": 60_000,            # default: 60_000 | total training iterations

    # Gaussian position learning rate schedule
    "position_lr_init": 0.00016,     # default: 0.00016
    "position_lr_final": 0.0000016,  # default: 0.0000016
    "position_lr_delay_mult": 0.01,  # default: 0.01
    "position_lr_max_steps": 60_000, # default: 60_000

    # Per-attribute learning rates
    "feature_lr": 0.0025,            # default: 0.0025 | SH features
    "opacity_lr": 0.05,              # default: 0.05
    "scaling_lr": 0.005,             # default: 0.005
    "rotation_lr": 0.001,            # default: 0.001

    # HDR environment map learning rate schedule
    "hdr_lr_init": 0.00025,          # default: 0.00025
    "hdr_lr_final": 0.00005,         # default: 0.00005
    "hdr_lr_delay_mult": 0.01,       # default: 0.01
    "hdr_lr_max_steps": 55_000,      # default: 55_000

    # HDR base (diffuse) learning rate schedule
    "hdr_base_lr_init": 0.0025,      # default: 0.0025
    "hdr_base_lr_final": 0.0005,     # default: 0.0005
    "hdr_base_lr_delay_mult": 0.01,  # default: 0.01
    "hdr_base_lr_max_steps": 55_000, # default: 55_000

    # Albedo learning rate schedule
    "albedo_lr_init": 0.0025,        # default: 0.0025
    "albedo_lr_final": 0.0005,       # default: 0.0005
    "albedo_lr_delay_mult": 0.01,    # default: 0.01
    "albedo_lr_max_steps": 55_000,   # default: 55_000

    # Material (roughness / metallic) learning rate schedule
    "material_lr_init": 0.0025,      # default: 0.0025
    "material_lr_final": 0.0005,     # default: 0.0005
    "material_lr_delay_mult": 0.01,  # default: 0.01
    "material_lr_max_steps": 55_000, # default: 55_000

    # Densification / pruning
    "percent_dense": 0.01,           # default: 0.01
    "lambda_dssim": 0.4,             # default: 0.4    | weight of the D-SSIM term in RGB loss
    "densification_interval": 100,   # default: 100
    "opacity_reset_interval": 3000,  # default: 3000
    "densify_from_iter": 500,        # default: 500
    "densify_until_iter": 45_000,    # default: 45_000
    "densify_grad_threshold": 0.0002,# default: 0.0002
    "max_gaussians": 0,              # default: 0      | hard cap on #gaussians (0 = unlimited)
    "random_background": False,      # default: False

    # ----------------------------------------------------------------------
    # Engine / staging - two-stage PBR training control
    # ----------------------------------------------------------------------
    "first_stage_step": 5_000,       # default: 5_000  | end of stage 1 (radiance warm-up)
    "second_stage_step": 30_000,     # default: 30_000 | end of stage 2 (start of PBR decomposition)
    "remove_noise": False,           # default: False  | remove floaters / noisy gaussians
    "hdr_rotation": False,           # default: False  | optimize HDR rotation
    "reg_hdr_weight": 0.001,         # default: 0.001  | HDR smoothness regularization weight
    "reg_material_weight": 0.1,      # default: 0.1    | material smoothness regularization weight

    # ----------------------------------------------------------------------
    # Priors - GT prior supervision ADDED to the engine
    # (albedo / normal / metallic priors extracted by the prior_extractors)
    # ----------------------------------------------------------------------
    "lambda_albedo_gt": 0.1,         # default: 0.1    | fixed weight for albedo GT-prior loss
    "lambda_normal_gt": 0.1,         # default: 0.1    | fixed weight for normal GT-prior loss
    "lambda_metallic_gt": 0.05,      # default: 0.05   | fixed weight for metallic GT-prior loss
    "lambda_roughness_gt": 0.05,     # default: 0.05   | fixed weight for roughness GT-prior loss
    "use_prior_weight_scheduler": True, # default: True  | ramp up prior and envmap weights
    "prior_weight_scheduler_ratio": 0.15, # default: 0.15  | fraction of post-second-stage steps for warmup
    "exclude_prior_loss": False,     # default: False  | compute prior losses for logging only,
                                     #                  | but do NOT backprop them (no-prior ablation)

    # ----------------------------------------------------------------------
    # Logging / evaluation
    # ----------------------------------------------------------------------
    "eval_interval": 2_000,          # default: 2_000  | evaluate metrics every N iterations
    "visual_interval": 10_000,       # default: 10_000 | save visual comparisons every N iters
    "eval_relight_hdris": ["snowy_forest", "moonless_night", "fireplace"],
                                     # default: ['snowy_forest', 'moonless_night', 'fireplace']

    # Iterations at which to test / save / checkpoint
    # default: [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000]
    "test_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
    "save_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
    "checkpoint_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
    "start_checkpoint": None,        # default: None   | path to a checkpoint to resume from

    # ----------------------------------------------------------------------
    # Runtime / misc
    # ----------------------------------------------------------------------
    "ip": "127.0.0.1",               # default: 127.0.0.1 | network GUI host
    "port": 6000,                    # default: 6000   | network GUI port
    "debug_from": -1,                # default: -1     | start rasterizer debug at this iteration
    "detect_anomaly": False,         # default: False  | torch autograd anomaly detection
    "quiet": False,                  # default: False  | suppress console output
}


def build_args():
    """Build the engine argument namespace from CONFIG + optional CLI overrides.

    Defaults are taken from the engine's own parameter groups so this launcher
    stays in sync with arguments/__init__.py and train.py. CONFIG values then
    override those defaults, and finally any command-line flag overrides CONFIG.
    """
    parser = ArgumentParser(description="Central GIR training launcher")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # The extra (non param-group) arguments defined in train.py.
    parser.add_argument("--ip", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--debug_from", type=int)
    parser.add_argument("--detect_anomaly", action="store_true", default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int)
    parser.add_argument("--save_iterations", nargs="+", type=int)
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int)
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--first_stage_step", type=int)
    parser.add_argument("--second_stage_step", type=int)
    parser.add_argument("--remove_noise", action="store_true", default=None)
    parser.add_argument("--hdr_rotation", action="store_true", default=None)
    parser.add_argument("--reg_hdr_weight", type=float)
    parser.add_argument("--reg_material_weight", type=float)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--visual_interval", type=int)
    parser.add_argument("--lambda_albedo_gt", type=float)
    parser.add_argument("--lambda_normal_gt", type=float)
    parser.add_argument("--lambda_metallic_gt", type=float)
    parser.add_argument("--lambda_roughness_gt", type=float)
    parser.add_argument("--use_prior_weight_scheduler", action="store_true", default=None)
    parser.add_argument("--prior_weight_scheduler_ratio", type=float)
    parser.add_argument("--exclude_prior_loss", action="store_true", default=None)
    parser.add_argument("--eval_relight_hdris", nargs="+", type=str)

    # 1. Start from engine defaults.
    args = parser.parse_args([])

    # 2. Apply CONFIG overrides.
    for key, value in CONFIG.items():
        if not hasattr(args, key):
            raise KeyError(f"Unknown parameter in CONFIG: '{key}'")
        setattr(args, key, value)

    # 3. Apply command-line overrides (anything explicitly passed != None).
    cli_args = parser.parse_args(sys.argv[1:])
    for key, value in vars(cli_args).items():
        if value is not None:
            setattr(args, key, value)

    return args, lp, op, pp


def main():
    args, lp, op, pp = build_args()

    if not args.source_path:
        raise ValueError(
            "CONFIG['source_path'] (or --source_path) must be set to the dataset path."
        )

    # Always evaluate at the final iteration too.
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.first_stage_step,
        args.second_stage_step,
        args.remove_noise,
        args.hdr_rotation,
        args.reg_hdr_weight,
        args.reg_material_weight,
        args.eval_interval,
        args.visual_interval,
        args.lambda_albedo_gt,
        args.lambda_normal_gt,
        args.lambda_metallic_gt,
        args.lambda_roughness_gt,
        args.use_prior_weight_scheduler,
        args.prior_weight_scheduler_ratio,
        args.exclude_prior_loss,
        args.eval_relight_hdris,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
