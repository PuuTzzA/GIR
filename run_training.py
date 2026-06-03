"""
Simple training launcher for PBR-3DGS / GIR.
Edit the variables below, then run:  python run_training.py
"""
import os
import sys
# Memory optimisation: expandable CUDA memory segments reduce fragmentation.
# Must be set BEFORE any CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ============================================================
# EDIT THESE TWO PATHS
# ============================================================
INPUT_DATASET = "../data/sphere"          # Path to dataset folder (contains transforms_train.json)
OUTPUT_MODEL  = "../outputs/outputs_sphere_with_losses_075"  # Path where model + intermediate renders will be saved
# ============================================================
# Optional settings (change as needed)
RESOLUTION             = 2          # Image resolution divisor (1=full, 2=half, 4=quarter)
DENSIFY_GRAD_THRESHOLD = 0.0004     #
EVAL                   = True       # Hold out test set for evaluation
PORT                   = 6008       # Network GUI port
ITERATIONS             = 60000      # Total training iterations (default GIR: 60k)
# ============================================================
# BASELINE MODE — run plain GIR with NO custom additions
# When True: disables GT supervision, saves checkpoints every 10k iterations.
# Use this if you're running out of CUDA memory.
# ============================================================
BASELINE_MODE = False
# ============================================================
# GT MATERIAL LOSS WEIGHTS
# Only used when BASELINE_MODE = False AND DISABLE_GT_SUPERVISION = False
# Set > 0.0 to enable. These are multiplied with the computed
# masked L1 loss for each material property.
# ============================================================
DISABLE_GT_SUPERVISION = False   # Set True to completely disable GT supervision regardless of lambdas
LAMBDA_ALBEDO   = 0.75       # Weight for albedo GT supervision loss
LAMBDA_NORMAL   = 0.75       # Weight for normal GT supervision loss
LAMBDA_METALLIC = 0.05      # Weight for metallic loss (assumes zero/non-metallic scene)
GT_LOSS_TYPE    = "l1"      # Loss function for GT supervision: "l1" or "huber"
# ============================================================
# Apply baseline overrides
# ============================================================
if BASELINE_MODE or DISABLE_GT_SUPERVISION:
    LAMBDA_ALBEDO = 0.0
    LAMBDA_NORMAL = 0.0
    LAMBDA_METALLIC = 0.0
# Checkpoint interval: every 10k in baseline mode, every 5k otherwise
CHECKPOINT_INTERVAL = 10_000 if BASELINE_MODE else 5_000
SAVE_ITERS = list(range(CHECKPOINT_INTERVAL, ITERATIONS + 1, CHECKPOINT_INTERVAL))
# ============================================================
# Build argv and run training
# ============================================================
sys.argv = [
    "train.py",
    "-s", INPUT_DATASET,
    "-m", OUTPUT_MODEL,
    "--port", str(PORT),
    "--iterations", str(ITERATIONS),
    "-r", str(RESOLUTION),
    "--densify_grad_threshold", str(DENSIFY_GRAD_THRESHOLD),
    "--lambda_albedo", str(LAMBDA_ALBEDO),
    "--lambda_normal", str(LAMBDA_NORMAL),
    "--lambda_metallic", str(LAMBDA_METALLIC),
    "--gt_loss_type", GT_LOSS_TYPE,
]
if EVAL:
    sys.argv.append("--eval")
# Import and run
from train import training
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from utils.general_utils import safe_state
from gaussian_renderer import network_gui
import torch
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6000)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=SAVE_ITERS)
parser.add_argument("--save_iterations", nargs="+", type=int, default=SAVE_ITERS)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=SAVE_ITERS)
parser.add_argument("--start_checkpoint", type=str, default=None)
parser.add_argument("--first_stage_step", type=int, default=5_000)
parser.add_argument("--second_stage_step", type=int, default=30_000)
parser.add_argument("--remove_noise", action="store_true", default=False)
parser.add_argument("--hdr_rotation", action="store_true", default=False)
parser.add_argument("--reg_hdr_weight", type=float, default=0.001)
parser.add_argument("--reg_material_weight", type=float, default=0.1)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)
print(f"{'='*60}")
print(f"  BASELINE MODE:          {BASELINE_MODE}")
print(f"  GT SUPERVISION OFF:     {DISABLE_GT_SUPERVISION}")
print(f"  Checkpoint interval:    every {CHECKPOINT_INTERVAL} iters")
print(f"{'='*60}")
print(f"Input dataset:   {os.path.abspath(INPUT_DATASET)}")
print(f"Output model:    {os.path.abspath(OUTPUT_MODEL)}")
print(f"Resolution:      1/{RESOLUTION}")
print(f"Densify grad threshold: {DENSIFY_GRAD_THRESHOLD}")
print(f"Lambda albedo:   {args.lambda_albedo}")
print(f"Lambda normal:   {args.lambda_normal}")
print(f"Lambda metallic: {args.lambda_metallic}")
print(f"GT loss type:    {args.gt_loss_type}")
print(f"Optimizing {args.model_path}")
safe_state(args.quiet)
network_gui.init(args.ip, args.port)
torch.autograd.set_detect_anomaly(args.detect_anomaly)
training(
    lp.extract(args), op.extract(args), pp.extract(args),
    args.test_iterations, args.save_iterations, args.checkpoint_iterations,
    args.start_checkpoint, args.debug_from,
    args.first_stage_step, args.second_stage_step,
    args.remove_noise, args.hdr_rotation,
    args.reg_hdr_weight, args.reg_material_weight,
    args.lambda_albedo, args.lambda_normal, args.lambda_metallic, args.gt_loss_type,
)
print("\nTraining complete.")
