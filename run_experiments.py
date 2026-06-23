#!/usr/bin/env python3
"""
Experiment runner + multi-run comparison report for the (modified) GIR engine.

This script launches several training runs that differ ONLY in how the
albedo / normal / metallic GT-priors are used, then builds a single comparison
PDF + CSV that overlays their:

    * Novel-view performance        -> test_psnr / test_ssim / test_lpips
    * Novel-view + relighting        -> relight_<hdri>_psnr / _ssim (test cams
                                        rendered under unseen HDRIs)
    * Prior loss curves              -> albedo_gt / normal_gt / metallic_gt
    * Novel-view performance        -> test_psnr / test_ssim / test_lpips
    * Novel-view + relighting        -> relight_<hdri>_psnr / _ssim (test cams
                                        rendered under unseen HDRIs)
    * Prior loss curves              -> albedo_gt / normal_gt / metallic_gt
    * Training loss / #gaussians

------------------------------------------------------------------------------
THREE-PHASE TRAINING PIPELINE
------------------------------------------------------------------------------
The GIR engine is driven by two stage boundaries, `first_stage_step` and
`second_stage_step`, which this batch uses to realise an explicit 3-phase
schedule (priors are introduced gradually, geometry first):

  * Phase 1 — GEOMETRY  (iter <= first_stage_step)
        GIR's radiance warm-up. Gaussian positions / scales / rotations /
        opacity and a plain view-dependent colour are optimised against the RGB
        images so densification places gaussians where the geometry is. No PBR
        decomposition, no GT priors, no material TV losses yet.

  * Phase 2 — NORMAL ALIGNMENT  (first_stage_step < iter <= second_stage_step)
        The geometry-derived shading normal becomes differentiable every
        iteration and is supervised by the GT normal prior (this back-props into
        gaussian rotation / scaling, locking in surface orientation before any
        material is decomposed). Albedo / metallic / roughness priors stay OFF.
        Only a (reduced) edge-aware normal smoothness is applied.

  * Phase 3 — FULL PBR  (iter > second_stage_step)
        The full GIR PBR decomposition runs. All available GT priors are applied
        (albedo via the selected mode, plus normal / metallic / roughness) under
        the up-then-down weight scheduler, together with the TV / smoothness
        regularizers — each TV term scaled by `tv_reduction_factor` for any
        property that has a GT prior (the GT already constrains it, so artificial
        smoothness is unnecessary / harmful).

------------------------------------------------------------------------------
THE FOUR RUNS (lego, resolution 2, 60k iterations)
------------------------------------------------------------------------------
This batch keeps the priors SMALL (only albedo + normal GT; metallic / roughness
priors stay OFF) and isolates the albedo prior formulation. The normal GT prior
is handled exactly as before (cosine loss).

    1. baseline_no_prior : GIR baseline. Priors computed for logging only
                           (NOT optimized), full TV losses, envmap regularizer
                           ON, no prior scheduler.
    2. albedo_gradient   : NEW gradient-domain albedo prior. Matches the spatial
                           gradient of the rendered albedo to the gradient of the
                           GT albedo, so flat GT regions push the rendered albedo
                           gradient to zero (and GT edges are reproduced).
    3. albedo_direct     : "direct" albedo loss (Huber + DSSIM on raw values).
    4. albedo_log_chroma : log-intensity + chromaticity (intrinsic-image) albedo
                           loss.

   The per-channel least-squares albedo mode ("lstsq") is kept in the engine but
   dropped from this batch.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    # Run all experiments, then build the comparison report:
    python run_experiments.py

    # Only (re)build the comparison report from existing run folders:
    python run_experiments.py --report-only

    # Run a subset by name:
    python run_experiments.py --only no_prior fixed_lambda

Edit the COMMON dict and the EXPERIMENTS list below to configure runs.
"""

import os
import sys
import json
import glob
import shlex
import argparse
import subprocess
from datetime import datetime

# Directory of this file = the GIR engine root (where train.py lives).
GIR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(GIR_DIR)


# =============================================================================
# CONFIGURATION  -- edit here to add / change runs
# =============================================================================

# Parameters shared by every run in this experiment batch. Tuned for an
# extensive-but-affordable lego comparison at resolution 8 with 12k iterations.
# The stage boundaries realise the 3-phase pipeline described in the module
# docstring; densification / opacity-reset are arranged to FINISH well before
# the end so nothing disturbs the final gaussians.
LEGO_DIR = os.path.join(REPO_DIR, "data", "datasets_with_priors", "lego")
REAL_LIFE_DIR = os.path.join(REPO_DIR, "data", "datasets_with_priors", "bicycle")

LEGO_DIR_RELIGHT_HDRs = ["fireplace", "night", "snow"]  # HDRIs for blender datasets (lego, armadillo)
REAL_LIFE_DIR_RELIGHT_HDRs = []                          # real photos: no relight GT, so empty

COMMON = {
    "eval": True,                 # hold out the test cameras for novel-view eval
    "random_background": False,   # default: False

    "iterations": 45_000,          # total iterations (good-GPU run)

    # ── 3-phase schedule ────────────────────────────────────────────────
    #   Phase 1 GEOMETRY        : iter 0    .. 2000   (radiance warm-up)
    #   Phase 2 NORMAL ALIGN    : iter 2000 .. 5000   (GT normal prior only)
    #   Phase 3 FULL PBR        : iter 5000 .. 12000  (all priors + materials)
    "first_stage_step": 5_000,      # end of Phase 1 (radiance warm-up)
    "second_stage_step": 30_000,     # end of Phase 2 / start of Phase 3 (PBR)

    # Densification / pruning. densify_until_iter (7000) < iterations (12000),
    # and opacity_reset_interval (3000) only fires at 3000 & 6000 (both inside
    # the densify window), so the last 5000 iters settle cleanly with no
    # densification / opacity reset disturbing the final result.
    "percent_dense": 0.01,           # default: 0.01
    "lambda_dssim": 0.4,             # default: 0.4    | weight of the D-SSIM term in RGB loss
    "densification_interval": 100,   # default: 100
    "opacity_reset_interval": 3000,  # resets only at 3000 & 6000 (< densify_until_iter)
    "densify_from_iter": 500,        # default: 500
    "densify_until_iter": 38_000,    # default: 45_000
    "densify_grad_threshold": 0.0002,# default: 0.0002
    "max_gaussians": 300_000,        # hard cap on #gaussians (0 = unlimited)

    "eval_interval": 2_000,          # evaluate metrics every N iters
    "visual_interval": 5_000,        # save visual comparisons every N iters

    # Keep disk usage small: only save/checkpoint at the very end.
    # Denser cadence around 40-50k so the best relight checkpoint (which for the
    # baseline peaks ~44k before the post-densify overfit) is captured.
    "save_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 42_000, 44_000, 46_000, 48_000, 50_000, 60_000],
    "test_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 42_000, 44_000, 46_000, 48_000, 50_000, 60_000],
    "checkpoint_iterations": [5_000, 10_000, 20_000, 30_000, 40_000, 42_000, 44_000, 46_000, 48_000, 50_000, 60_000],

    # Prior loss weights (used by every run that optimizes the priors).
    # A weight is auto-forced to 0 when its *_gt_dir (set per-dataset) is "".
    "lambda_albedo_gt": 0.25,
    "lambda_normal_gt": 0.8,
    "lambda_metallic_gt": 0.05,
    "lambda_roughness_gt": 0.05,

    # Prior weight schedule (Phase 3 only): warm up over the first 15% of the
    # PBR stage, then linearly interpolate from the full weight (1.0) to
    # `prior_weight_final_ratio` over the rest. 1.0 = warm up then HOLD at full
    # strength (no down-ramp); the old up-then-down decay used 0.5.
    "prior_weight_scheduler_ratio": 0.15,
    "prior_weight_final_ratio": 1.0,

    # Robust Huber delta for the prior losses.
    "huber_delta": 0.2,

    # Default albedo prior formulation; overridden per albedo variant below.
    "albedo_prior_mode": "direct",

    # TV / smoothness reduction for properties that have a GT prior. 1.0 keeps
    # the regularizer at full strength (baseline); per-variant overrides lower
    # or remove it. Properties WITHOUT a GT prior always keep full TV.
    "tv_reduction_factor": 1.0,

    # Third-stage geometry-LR reduction (xyz / scaling / rotation), cosine from
    # second_stage_step to geo_lr_final_iter. 1.0 = off (baseline / reference);
    # per-experiment overrides anneal it down to curb late light-baking.
    "reduce_geo_lr_third_stage": 1.0,
    "geo_lr_final_iter": 45_000,

    # Skip opacity resets once iter > second_stage_step. Off by default; one
    # experiment toggles it for the reset A/B.
    "disable_reset_third_stage": False,

    # Envmap-neutrality regularizer. Kept small for prior runs (GT albedo helps
    # resolve the albedo/light-colour ambiguity); baseline raises it slightly.
    "reg_hdr_weight": 0.0001,
}

# =============================================================================
# DATASETS  -- per-dataset overrides applied on top of COMMON
# =============================================================================
# Each dataset supplies its own source path, resolution, prior-folder names,
# relight HDRIs and (for synthetic data) the GT base-light envmap. The same
# four albedo variants below are run on every dataset.
#
# lego    : Blender synthetic-with-priors. Real GT albedo/normal in WORLD space
#           ("albedo_gt"/"normal_gt"), relighting GT available, sunset base HDRI.
# bicycle : real-world COLMAP-with-priors. NO real GT (estimated priors only) so
#           albedo="albedo", normal="normal"; the normals are in CAMERA space and
#           are rotated to world space at train time (normal_camera_convention).
#           No relight GT and no GT base HDRI, so those are left empty.
DATASETS = [
    {
        "name": "lego",
        "args": {
            "source_path": LEGO_DIR,
            "white_background": False,         # lego is a Blender-synthetic scene
            "resolution": 4,                   # -1 = keep native resolution
            "albedo_gt_dir": "albedo_gt",      # WORLD-space GT albedo
            "normal_gt_dir": "normal_gt",      # WORLD-space GT normal
            "metallic_gt_dir": "",
            "roughness_gt_dir": "",
            "eval_relight_hdris": LEGO_DIR_RELIGHT_HDRs,
            "envmap_gt_path": os.path.join(LEGO_DIR, "hdris", "sunset.hdr"),
        },
    },
    #{
    #    "name": "bicycle",
    #    "args": {
    #        "source_path": REAL_LIFE_DIR,
    #        "white_background": False,
    #        "resolution": 4,
    #        # Real photos: no ground truth, use the ESTIMATED priors. Change
    #        # these to "*_video" to use the video-consistent variants instead.
    #        "albedo_gt_dir": "albedo",
    #        "normal_gt_dir": "normal",
    #        "metallic_gt_dir": "",
    #        "roughness_gt_dir": "",
    #        # The COLMAP normal priors are in camera/view space; this controls
    #        # how they are mapped to camera axes before being rotated to world
    #        # space ("opengl" = flip Y,Z; "opencv"/"colmap" = no flip).
    #        "normal_camera_convention": "opengl",
    #        "eval_relight_hdris": REAL_LIFE_DIR_RELIGHT_HDRs,  # empty -> no relight eval
    #        "envmap_gt_path": "",                              # no GT base light
    #     },
    #},
]

# =============================================================================
# ALBEDO VARIANTS  -- the baseline + three albedo-prior formulations
# =============================================================================
# Each variant overrides only the albedo handling; all other settings come from
# COMMON / the dataset. The three new formulations isolate WHAT the albedo prior
# supervises (edges / relative distribution / structure) while staying invariant
# to the global brightness / colour shift between Cycles and the GIR BRDF.
ALBEDO_VARIANTS = [
    {
        # GIR baseline: priors computed for logging only (NOT optimized), full TV
        # losses, envmap regularizer at its paper value, no prior scheduler.
        "name": "baseline_no_prior",
        "args": {"reg_hdr_weight": 0.001, "tv_reduction_factor": 1.0},
        "flags": {"exclude_prior_loss": True},
    },
    #{
    #    # (1) Spatial-gradient / edge loss: match the rendered albedo gradient to
    #    # the GT albedo gradient (L1). Ignores any global brightness/colour shift
    #    # and only forces texture boundaries / edges into the right places.
    #    "name": "albedo_gradient",
    #    "args": {"albedo_prior_mode": "gradient", "tv_reduction_factor": 0.75},
    #    "flags": {"use_prior_weight_scheduler": True},
    #},
    {
        # (2) Scale-and-shift-invariant loss (ZNCC / Pearson): standardise both
        # albedos per channel (mean 0, std 1) before comparing, so the GIR albedo
        # may be proportionally brighter / darker / different contrast.
        "name": "albedo_zncc",
        "args": {"albedo_prior_mode": "zncc", "tv_reduction_factor": 0.75, "reg_hdr_weight": 0.001},
        "flags": {"use_prior_weight_scheduler": True},
    },
    {
        # (2) Scale-and-shift-invariant loss (ZNCC / Pearson): standardise both
        # albedos per channel (mean 0, std 1) before comparing, so the GIR albedo
        # may be proportionally brighter / darker / different contrast.
        "name": "albedo_zncc_reg_hdr",
        "args": {"albedo_prior_mode": "zncc", "tv_reduction_factor": 0.75, "reg_hdr_weight": 0.001},
        "flags": {"use_prior_weight_scheduler": True},
    },
    #{
    #    # (3) Structure-focused SSIM: SSIM on the albedo with the luminance term
    #    # heavily down-weighted, so shapes / textures must match the GT but the
    #    # overall brightness / contrast may drift.
    #    "name": "albedo_ssim_struct",
    #    "args": {"albedo_prior_mode": "ssim_struct", "tv_reduction_factor": 0.75},
    #    "flags": {"use_prior_weight_scheduler": True},
    #},
    #{
    #    # (4) Locally-normalised cross-correlation (relight-focused refinement of
    #    # zncc): standardise both albedos inside a sliding Gaussian window so the
    #    # loss is invariant to SPATIALLY-VARYING gain. Removes smooth baked
    #    # shading the global zncc leaves behind, keeping illumination out of the
    #    # recovered albedo for cleaner relighting.
    #    "name": "albedo_zncc_local",
    #    "args": {"albedo_prior_mode": "zncc_local", "tv_reduction_factor": 0.75, "reg_hdr_weight": 0.0007},
    #    "flags": {"use_prior_weight_scheduler": True},
    #},
    {
        # (5) Gradient-domain ZNCC (relight-focused refinement of zncc): match
        # the scale-&-shift-invariant correlation of the albedo SPATIAL GRADIENTS
        # so low-frequency baked shading is differentiated away and only
        # high-frequency texture edges are supervised.
        "name": "albedo_zncc_grad",
        "args": {"albedo_prior_mode": "zncc_grad", "tv_reduction_factor": 0.75, "reg_hdr_weight": 0.0007},
        "flags": {"use_prior_weight_scheduler": True},
    },
]

# Where all run folders for this batch live.
EXPERIMENT_ROOT = os.path.join(REPO_DIR, "outputs", "new_experiments_try_1")

# Build the 8 experiments = every albedo variant on every dataset. Each
# experiment merges dataset args first, then the variant args (variant wins).
EXPERIMENTS = []
for _dataset in DATASETS:
    for _variant in ALBEDO_VARIANTS:
        _args = dict(_dataset["args"])
        _args.update(_variant.get("args", {}))
        EXPERIMENTS.append({
            "name": f"{_dataset['name']}_{_variant['name']}",
            "args": _args,
            "flags": dict(_variant.get("flags", {})),
        })

# =============================================================================
# RUN LAUNCHING
# =============================================================================

def _fmt_value(v):
    """Format a config value as command-line argument token(s)."""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def build_command(exp, model_path):
    """Build the `python train.py ...` command for one experiment."""
    cfg = dict(COMMON)
    cfg.update(exp.get("args", {}))

    # Boolean store_true flags handled separately (engine + run via train.py).
    bool_flags = {
        "eval", "white_background", "exclude_prior_loss", "use_prior_weight_scheduler",
        "remove_noise", "hdr_rotation", "random_background", "quiet",
        "disable_reset_third_stage",
    }

    flags = dict(exp.get("flags", {}))
    # Promote any bool entries living in cfg into the flags set.
    for key in list(cfg.keys()):
        if key in bool_flags and isinstance(cfg[key], bool):
            flags.setdefault(key, cfg.pop(key))

    cmd = [sys.executable, "train.py"]
    cmd += ["--source_path", cfg.pop("source_path")]
    cmd += ["--model_path", model_path]

    for key, value in cfg.items():
        # An empty list (e.g. eval_relight_hdris=[] for real-world data) would
        # produce a "--flag" with no values, which argparse nargs="+" rejects.
        # Skip it so the engine falls back to its default / no-op behaviour.
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        cmd.append(f"--{key}")
        cmd += _fmt_value(value)

    for key, enabled in flags.items():
        if enabled:
            cmd.append(f"--{key}")

    return cmd


def run_experiment(exp):
    """Launch a single experiment as a subprocess. Returns the model_path."""
    model_path = os.path.join(EXPERIMENT_ROOT, exp["name"])
    os.makedirs(model_path, exist_ok=True)

    cmd = build_command(exp, model_path)
    print("\n" + "=" * 78)
    print(f"RUN: {exp['name']}")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print("=" * 78, flush=True)

    log_path = os.path.join(model_path, "train_stdout.log")
    # Stream the child's stdout/stderr live to this console AND tee it to a log
    # file. We read in small chunks (not full lines) so the tqdm progress bar,
    # which uses carriage returns, updates in real time.
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=GIR_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for chunk in iter(lambda: proc.stdout.read(1), ""):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                logf.write(chunk)
        finally:
            proc.stdout.close()
            returncode = proc.wait()

    if returncode != 0:
        print(f"\n[ERROR] Run '{exp['name']}' exited with code {returncode}. "
              f"See {log_path}", flush=True)
    return model_path, returncode


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def _load_metrics(model_path):
    p = os.path.join(model_path, "metrics_log.json")
    if not os.path.exists(p):
        return []
    with open(p, "r") as f:
        return json.load(f)


def _load_loss_components(model_path):
    p = os.path.join(model_path, "train_process", "loss_components.json")
    if not os.path.exists(p):
        return []
    with open(p, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def _series(entries, key):
    xs, ys = [], []
    for e in entries:
        if key in e and e[key] is not None:
            xs.append(e["iteration"])
            ys.append(e[key])
    return xs, ys


def _relight_keys(entries, suffix):
    keys = set()
    for e in entries:
        for k in e:
            if k.startswith("relight_") and k.endswith(suffix):
                keys.add(k)
    return sorted(keys)


def _avg_relight_series(entries, suffix):
    """Mean over all relight_* HDRIs per evaluation point."""
    keys = _relight_keys(entries, suffix)
    xs, ys = [], []
    for e in entries:
        vals = [e[k] for k in keys if k in e and e[k] is not None]
        if vals:
            xs.append(e["iteration"])
            ys.append(sum(vals) / len(vals))
    return xs, ys


def generate_comparison(runs, out_dir):
    """runs: list of (name, model_path). Builds comparison PDF + CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    palette = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    data = []
    for i, (name, mp) in enumerate(runs):
        data.append({
            "name": name,
            "color": palette[i % len(palette)],
            "metrics": _load_metrics(mp),
            "loss": _load_loss_components(mp),
        })

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "comparison_report.pdf")
    csv_path = os.path.join(out_dir, "comparison_summary.csv")

    def plot_metric(ax, key, ylabel, title):
        any_data = False
        for d in data:
            xs, ys = _series(d["metrics"], key)
            if xs:
                ax.plot(xs, ys, "o-", color=d["color"], markersize=3,
                        linewidth=1.5, label=d["name"])
                any_data = True
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
        if any_data:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    def plot_avg_relight(ax, suffix, ylabel, title):
        any_data = False
        for d in data:
            xs, ys = _avg_relight_series(d["metrics"], suffix)
            if xs:
                ax.plot(xs, ys, "o-", color=d["color"], markersize=3,
                        linewidth=1.5, label=d["name"])
                any_data = True
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
        if any_data:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No relight data", ha="center", va="center", transform=ax.transAxes)

    def plot_loss(ax, key, ylabel, title):
        any_data = False
        for d in data:
            xs, ys = _series(d["loss"], key)
            if xs:
                ax.plot(xs, ys, "o-", color=d["color"], markersize=3,
                        linewidth=1.5, label=d["name"])
                any_data = True
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
        if any_data:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Title + final summary table ──────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.95, "GIR Prior-Ablation Comparison", fontsize=24,
                fontweight="bold", ha="center", va="top", color="#1a1a2e")
        ax.text(0.5, 0.88, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                fontsize=10, ha="center", va="top", color="#888")

        # Build summary table from each run's final metrics entry.
        header = ["Run", "PSNR", "SSIM", "LPIPS", "Relight PSNR", "Relight SSIM"]
        rows = []
        for d in data:
            m = d["metrics"][-1] if d["metrics"] else {}
            _, rp = _avg_relight_series(d["metrics"], "_psnr")
            _, rs = _avg_relight_series(d["metrics"], "_ssim")
            rows.append([
                d["name"],
                f"{m.get('test_psnr', float('nan')):.3f}",
                f"{m.get('test_ssim', float('nan')):.4f}",
                f"{m.get('test_lpips', float('nan')):.4f}",
                f"{rp[-1]:.3f}" if rp else "N/A",
                f"{rs[-1]:.4f}" if rs else "N/A",
            ])
        if rows:
            table = ax.table(cellText=rows, colLabels=header, loc="center",
                             cellLoc="center", bbox=[0.05, 0.35, 0.9, 0.4])
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
            for j in range(len(header)):
                table[0, j].set_facecolor("#1a1a2e")
                table[0, j].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 2: Novel-view performance ───────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Novel-View Performance (held-out test cameras)",
                     fontsize=14, fontweight="bold")
        plot_metric(axes[0, 0], "test_psnr", "PSNR (dB)", "Test PSNR \u2191")
        plot_metric(axes[0, 1], "test_ssim", "SSIM", "Test SSIM \u2191")
        plot_metric(axes[1, 0], "test_lpips", "LPIPS", "Test LPIPS \u2193")
        plot_metric(axes[1, 1], "train_loss", "Loss", "Training Loss (EMA) \u2193")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 3: Novel-view + relighting ──────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Novel-View + Relighting (test cameras under unseen HDRIs)",
                     fontsize=14, fontweight="bold")
        plot_avg_relight(axes[0, 0], "_psnr", "PSNR (dB)", "Mean Relight PSNR \u2191")
        plot_avg_relight(axes[0, 1], "_ssim", "SSIM", "Mean Relight SSIM \u2191")
        # Per-HDRI final PSNR bars
        ax_bar = axes[1, 0]
        all_hdris = sorted({k.replace("relight_", "").replace("_psnr", "")
                            for d in data for k in _relight_keys(d["metrics"], "_psnr")})
        if all_hdris:
            import numpy as np
            x = np.arange(len(all_hdris)); w = 0.8 / max(1, len(data))
            for i, d in enumerate(data):
                m = d["metrics"][-1] if d["metrics"] else {}
                vals = [m.get(f"relight_{h}_psnr", float("nan")) for h in all_hdris]
                ax_bar.bar(x + i * w, vals, w, color=d["color"], label=d["name"])
            ax_bar.set_xticks(x + (len(data) - 1) * w / 2)
            ax_bar.set_xticklabels(all_hdris, rotation=20, ha="right", fontsize=8)
            ax_bar.set_ylabel("PSNR (dB)"); ax_bar.set_title("Final per-HDRI Relight PSNR \u2191")
            ax_bar.legend(loc="best", fontsize=8)
        else:
            ax_bar.text(0.5, 0.5, "No relight data", ha="center", va="center",
                        transform=ax_bar.transAxes)
        # Scale-aligned relight PSNR (single gain fit -> decomposition quality).
        plot_avg_relight(axes[1, 1], "_psnr_aligned", "PSNR (dB)",
                         "Mean Relight PSNR, scale-aligned \u2191")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4: Prior losses ───────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Prior Losses",
                     fontsize=14, fontweight="bold")
        plot_loss(axes[0, 0], "albedo_gt", "Loss", "Albedo prior loss \u2193")
        plot_loss(axes[0, 1], "normal_gt", "Loss", "Normal prior loss \u2193")
        plot_loss(axes[1, 0], "metallic_gt", "Loss", "Metallic prior loss \u2193")
        plot_loss(axes[1, 1], "roughness_gt", "Loss", "Roughness prior loss \u2193")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 5: Decomposition quality + environment-map recovery ──────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Decomposition Quality & Envmap Recovery",
                     fontsize=14, fontweight="bold")
        plot_metric(axes[0, 0], "test_albedo_psnr", "PSNR (dB)", "Albedo PSNR \u2191")
        plot_metric(axes[0, 1], "test_normal_ang_err", "Degrees", "Normal angular error \u2193")
        plot_metric(axes[1, 0], "test_metallic_mae", "MAE", "Metallic MAE \u2193")
        plot_metric(axes[1, 1], "envmap_log_psnr", "log-PSNR (dB)", "EnvMap recovery \u2191")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── CSV summary of final metrics ─────────────────────────────────────
    with open(csv_path, "w") as f:
        cols = ["run", "test_psnr", "test_psnr_aligned", "test_ssim", "test_lpips",
                "mean_relight_psnr", "mean_relight_psnr_aligned", "mean_relight_ssim",
                "test_albedo_psnr", "test_normal_ang_err", "test_metallic_mae", "test_roughness_mae",
                "envmap_log_psnr", "envmap_rel_l1",
                "albedo_gt", "normal_gt", "metallic_gt", "roughness_gt", "num_gaussians"]
        f.write(",".join(cols) + "\n")
        for d in data:
            m = d["metrics"][-1] if d["metrics"] else {}
            lc = d["loss"][-1] if d["loss"] else {}
            _, rp = _avg_relight_series(d["metrics"], "_psnr")
            _, rpa = _avg_relight_series(d["metrics"], "_psnr_aligned")
            _, rs = _avg_relight_series(d["metrics"], "_ssim")
            row = [
                d["name"],
                m.get("test_psnr", ""), m.get("test_psnr_aligned", ""), m.get("test_ssim", ""), m.get("test_lpips", ""),
                rp[-1] if rp else "", rpa[-1] if rpa else "", rs[-1] if rs else "",
                m.get("test_albedo_psnr", ""), m.get("test_normal_ang_err", ""),
                m.get("test_metallic_mae", ""), m.get("test_roughness_mae", ""),
                m.get("envmap_log_psnr", ""), m.get("envmap_rel_l1", ""),
                lc.get("albedo_gt", ""), lc.get("normal_gt", ""), lc.get("metallic_gt", ""), lc.get("roughness_gt", ""),
                m.get("num_gaussians", ""),
            ]
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nComparison report : {pdf_path}")
    print(f"Comparison summary: {csv_path}")
    return pdf_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="GIR experiment runner + comparison report")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip training; only (re)build the comparison report.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the named experiments (default: all).")
    args = ap.parse_args()

    selected = EXPERIMENTS
    if args.only:
        selected = [e for e in EXPERIMENTS if e["name"] in args.only]
        if not selected:
            print(f"No experiments match {args.only}. Available: "
                  f"{[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    runs = []
    if args.report_only:
        for e in selected:
            runs.append((e["name"], os.path.join(EXPERIMENT_ROOT, e["name"])))
    else:
        os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
        for e in selected:
            model_path, rc = run_experiment(e)
            runs.append((e["name"], model_path))

    generate_comparison(runs, EXPERIMENT_ROOT)
    print("\nAll done.")


if __name__ == "__main__":
    main()
