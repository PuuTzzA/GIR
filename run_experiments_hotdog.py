#!/usr/bin/env python3
"""
HOTDOG HOTDOG
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
THE RUNS (lego, resolution 2, 60k iterations) — try_10: PAPER SCALE
------------------------------------------------------------------------------
try_9 findings this batch is built on:
  * The try_9 diagnostic ANSWERED: with the albedo truly pinned
    (zncc+anchor: albedo 29.2 dB, gain 0.92) the envmap ratio did NOT drop —
    it ROSE through phase 3 (1.66 -> 1.75; no-LLI anchor run: 2.28) while the
    envmap logPSNR stayed healthy (30.2). The leftover relight gain is
    GENUINE missing transport energy (no interreflection in the renderer),
    which the training envmap absorbs because it is the only remaining free
    knob — and envmap-side compensation does NOT transfer to a swapped HDRI.
  * diff_zncc_grad_lli2 PROVED the transport energy can live on the
    reflectance side instead: with loose priors (no anchor, weak metallic
    0.05) the optimizer made ~26% of the (opacity-weighted) gaussians
    metallic > 0.5 (opa-wtd mean 0.217 vs 0.008-0.02 in every GT run) with
    roughness 0.83 — a rough-specular pseudo-bounce whose direct part samples
    the swapped envmap and whose indirect part is the light-linear SH term.
    Result: envmap ratio 0.87, relight gains 0.85-1.04 (calibrated!), best
    raw relight 26.49 and best relight SSIM 0.906 — but the worst
    decomposition (albedo 23.7 dB, normals 16.8 deg, metallic MAE 0.058).
  * So raw-vs-aligned is a calibration-vs-structure split: GT anchor runs own
    the structure (aligned 27.6-27.7), the diffusion run owns the calibration
    (raw ~= aligned). The anchor is NOT the problem — it is doing its job;
    the renderer needs a legitimate light-tracking home for the bounce energy
    (engine lever queued for try_11: bounce-energy encouragement /
    irradiance-normalized bounce / fractional occlusion).

This batch VALIDATES the three headline configs at the PAPER's operating
point before any new engine lever: resolution -r 2 (400x400) and the paper's
60k schedule (stages 5k/30k, densify to 45k) instead of the exploration
setting (-r 4, 37.5k, stages 5k/25k, densify to 30k).

    1. baseline_no_prior      : GIR exactly as in the paper / reference repo
                                (GIR_Reference): train.py -s <lego> --eval
                                --random_background --hdr_rotation
                                --reg_hdr_weight 0.1 --reg_material_weight
                                0.05, everything else at engine defaults
                                (60k, stages 5k/30k, densify 500..45k,
                                lambda_dssim 0.4, no priors, no LLI, no
                                geo-LR anneal, TV at full strength) — except
                                the batch-wide 1.5M gaussian cap (above).
                                Verified against GIR_Reference/GIR: with
                                these values every engine addition is inert
                                (see FINDINGS.md "Baseline delta" section).
    2. gt_zncc_grad_anchor_lli2: best GT raw-relight config (try_8/9 lineage)
                                — zncc_grad + anchor 0.05 + detached LLI.
                                zncc_grad is the mode that must survive for
                                the diffusion-prior end goal.
    3. gt_zncc_anchor_lli2_envpen : CHALLENGER — try_9's best decomposition
                                (zncc + anchor 0.05 + LLI) plus the NEW
                                envmap-mean penalty (--reg_env_mean_weight
                                0.005): downward pressure on the envmap's
                                mean radiance re-homes the transport deficit
                                into the light-linear bounce, which transfers
                                at relight. Success = envmap ratio ~1, gain
                                < 1.22, raw relight > 25.4, albedo pinned.
    4. diff_zncc_grad_lli2    : the diffusion-prior arm (DiffusionRenderer
                                albedo/normal/metallic/roughness), no anchor.
    5. gt_zncc_grad_anchor15_lli2 : REPAIR-BATCH ADDITION (post-try_10,
                                hyperparameter-only): identical to run 2
                                except albedo_anchor_weight 0.05 -> 0.15.

try_10 RESULTS (details in FINDINGS.md):
  * Runs 1+3 completed to 60k; runs 2+4 OOMed at ~47k/48k — AFTER their 45k
    checkpoints were saved (allocator fragmentation in get_diffuse_occ:
    1.8 GiB alloc failed with ~7 GiB reserved-but-unallocated; the launcher
    now sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which fixes
    exactly this). Densification ends at 45k, so resuming from 45k replays
    the identical remaining schedule. Their CSV metrics are 45k values.
  * Run 2 (gt_zncc_grad_anchor_lli2) is the BEST RUN TO DATE at 45k: raw
    relight 27.62 dB (baseline 25.79), relight gain 1.002 — the try_7
    success criterion (gain -> 1.0 at aligned-level raw PSNR) met at paper
    scale; envmap ratio 1.14 and falling. Sole flaw: albedo gain 0.815
    (zncc_grad channel drift; anchor 0.05 too weak) -> run 5 fixes that.
  * Run 3 (envpen) FAILED its success criteria: envmap ratio rose 1.49->2.00
    through phase 3 DESPITE the mean penalty, envmap logPSNR 26.7 (run 2:
    28.9), raw relight 26.89 < 27.62. Lever retired — do NOT re-run. (Its
    zncc albedo is the best of the batch, 27.8/33.7 dB, so the run stays in
    the report as the zncc reference.)
  * Run 4 (diffusion) at 45k: aligned relight 28.62 TIES run 2 (structure is
    fine!) but the envmap trained 1.4x too DARK (ratio 0.71, relight gain
    0.80 = renders too bright): the r4 metallic escape hatch OVERSHOOTS at
    r2. Raw relight 24.58 < baseline. Resume to 60k before judging.

Cost note: NOTHING here can reuse the r4 warm-up buffer (resolution,
iterations and stage boundaries are all in the fingerprint) — runs 2+3 share
ONE fresh warm-up via the buffer, baseline and diffusion each run full.
Expect this batch to take DAYS on a 16 GB GPU. ALL runs (baseline included)
carry a max_gaussians=1.5M cap — OOM insurance at r2 that keeps the four
runs comparable; it deviates from the paper's unlimited densification but
should only bind if r2 densification runs away (r4: baseline 140k, prior
runs ~1.2M).
EXTERNAL_RUNS is EMPTY on purpose: every earlier batch ran at -r 4 / 37.5k,
whose numbers are not comparable at this resolution.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    # Run all experiments, then build the comparison report:
    python run_experiments.py

    # Only (re)build the comparison report from existing run folders:
    python run_experiments.py --report-only

    # Run a subset by name:
    python run_experiments.py --only no_prior fixed_lambda

    # Restart a crashed run from the latest checkpoint in its OWN model_path
    # (bypasses the warm-up buffer, so config edits that change the warm-up
    # fingerprint — e.g. a lower max_gaussians after an OOM — are fine):
    python run_experiments.py --only lego_gt_zncc_grad_anchor_lli2 --resume

    # Same, but from a specific checkpoint and with a lower gaussian cap
    # (the cap only acts while densification runs, so resume from a
    # checkpoint BEFORE densify_until_iter for it to change anything):
    python run_experiments.py --only lego_gt_zncc_grad_anchor_lli2 \\
        --resume-iter 30000 --set max_gaussians=1100000

    # ---- try_10 REPAIR SEQUENCE (run one at a time, in this order) ----
    # 1) finish the headline run 45k -> 60k (~6 h):
    python run_experiments.py --only lego_gt_zncc_grad_anchor_lli2 --resume-iter 45000
    # 2) finish the diffusion arm 45k -> 60k (~5 h):
    python run_experiments.py --only lego_diff_zncc_grad_lli2 --resume-iter 45000
    # 3) the new anchor-0.15 variant (warm-up buffer HIT -> Phase 3 only, ~15 h):
    python run_experiments.py --only lego_gt_zncc_grad_anchor15_lli2
    # Any run that already owns a chkpnt*.pth is auto-skipped unless --resume
    # is given, so a bare `python run_experiments.py` cannot clobber existing
    # results (it would only launch the new anchor15 run).

Edit the COMMON dict and the EXPERIMENTS list below to configure runs.
"""

import os
import re
import sys
import json
import glob
import shlex
import shutil
import hashlib
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
HOTDOG_DIR = os.path.join(REPO_DIR, "data", "datasets_with_priors", "hotdog")
REAL_LIFE_DIR = os.path.join(REPO_DIR, "data", "datasets_with_priors", "bicycle")

HOTDOG_DIR_RELIGHT_HDRs = ["fireplace", "bridge", "night", "snow", "city", "courtyard", "forest"]  # HDRIs for blender datasets (lego, armadillo)
REAL_LIFE_DIR_RELIGHT_HDRs = []                          # real photos: no relight GT, so empty

COMMON = {
    "eval": True,                 # hold out the test cameras for novel-view eval
    # Reference launch scripts train with --random_background; it also prevents
    # background-colour baking, so every run uses it.
    "random_background": True,

    # Blender-rendered data is Z-up while the envlight lat-long convention is
    # Y-up: the reference trains Blender/TensoIR scenes with --hdr_rotation.
    # Without it every relight eval samples the loaded HDRI sideways (verified:
    # the learned envmap matches the GT sunset only under GIR's rotation map,
    # log-corr +0.40 vs -0.21 for identity). REQUIRED on our lego dataset.
    "hdr_rotation": True,

    # try_10 runs the PAPER's schedule (reference train.py defaults): 60k
    # iterations, stages 5k/30k, densification 500..45k. This also makes the
    # LR schedules line up exactly with the reference (position_lr_max_steps
    # 60k, hdr/albedo/material LR max steps 55k are engine defaults tuned for
    # a 60k run).
    "iterations": 60_000,

    # ── 3-phase schedule ────────────────────────────────────────────────
    "first_stage_step": 5_000,      # end of Phase 1 (radiance warm-up)
    "second_stage_step": 30_000,    # end of Phase 2 / start of Phase 3 (PBR, paper value)

    "percent_dense": 0.01,           # default: 0.01
    "lambda_dssim": 0.4,             # default: 0.4    | weight of the D-SSIM term in RGB loss
    "densification_interval": 100,   # default: 100
    "opacity_reset_interval": 3000,
    "densify_from_iter": 500,        # default: 500
    "densify_until_iter": 45_000,    # paper default (was 30k in the 37.5k batches)
    "densify_grad_threshold": 0.0002,# default: 0.0002
    # Hard cap for EVERY run in this batch (user decision, try_10): 1.5M
    # gaussians. Deviates from the reference's unlimited densification, but at
    # r4 the baseline stayed at 140k and prior runs at ~1.2M, so the cap
    # should only bind if r2 densification runs away — it is OOM insurance
    # for the 16 GB GPU that keeps all four runs comparable.
    "max_gaussians": 1_500_000,

    # TV weight on metallic/roughness: the reference launch scripts use 0.05
    # (engine default is 0.1).
    "reg_material_weight": 0.05,

    "eval_interval": 5_000,          # evaluate metrics every N iters
    "visual_interval": 5_000,        # save visual comparisons every N iters

    # 60k-run milestones: stage-2 boundary (30k, also the warm-up-buffer
    # checkpoint), densify end (45k) and final. Checkpoints at r2 are large
    # (~1.5 GB at 1.2M gaussians), so keep the list short.
    "save_iterations": [30_000, 45_000, 60_000],
    "test_iterations": [30_000, 45_000, 60_000],
    "checkpoint_iterations": [30_000, 45_000, 60_000],

    # Prior loss weights (used by every run that optimizes the priors).
    # A weight is auto-forced to 0 when its *_gt_dir (set per-dataset) is "".
    "lambda_albedo_gt": 0.25,
    "lambda_normal_gt": 0.8,
    "lambda_metallic_gt": 0.15,
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
    # In albedo geometry warm-up runs this is the STAGE-3 (PBR) formulation;
    # `warmup_albedo_prior_mode` selects a (possibly different) formulation for
    # the warm-up stages 1 & 2. Empty string = reuse `albedo_prior_mode`.
    "albedo_prior_mode": "direct",
    "warmup_albedo_prior_mode": "direct",

    # TV / smoothness reduction for properties that have a GT prior. 1.0 keeps
    # the regularizer at full strength (baseline); per-variant overrides lower
    # or remove it. Properties WITHOUT a GT prior always keep full TV.
    "tv_reduction_factor": 1.0,

    # Third-stage geometry-LR reduction (xyz / scaling / rotation), cosine from
    # second_stage_step to geo_lr_final_iter. 1.0 = off (baseline / reference);
    # per-experiment overrides anneal it down to curb late light-baking.
    "reduce_geo_lr_third_stage": 1.0,
    "geo_lr_final_iter": 66_000,

    # Skip opacity resets once iter > second_stage_step. Off by default; one
    # experiment toggles it for the reset A/B.
    "disable_reset_third_stage": False,

    # Envmap-neutrality regularizer. Kept small for prior runs (GT albedo helps
    # resolve the albedo/light-colour ambiguity); baseline raises it slightly.
    "reg_hdr_weight": 0.0001,

    # Relight eval on 24 of the 200 test cameras (deterministic subset): the
    # full 6-HDRI x 200-view relight eval took ~1 h per eval point in try_7.
    "relight_max_views": 24,
}

# reduce_geo_lr_third_stage — optimizer-level, time-scheduled, loss-agnostic.
# It scales the learning rates of the geometry parameters (xyz, scaling, rotation), cosine-annealed 
# from 1.0 down to the given factor between second_stage_step and geo_lr_final_iter, then held. 
# It throttles how fast geometry can move regardless of which loss produced the gradient — the photometric loss, 
# the normal prior, and the albedo prior are all slowed equally. It answers: "how plastic is geometry over time?"
# 
# prior_geom_grad_scale — gradient-path-level, constant, loss-selective.
# It scales (0.0 = blocks) the geometry gradients flowing through one specific path: 
# the stage-3 albedo/material prior rasters. The rendered maps are unchanged in the 
# forward pass; only the backward contribution of those rasters to means/scales/rotations/opacity is damped. 
# The photometric loss and the normal prior keep full-strength geometry gradients. 
# It answers: "which losses are allowed to shape geometry at all?" This targets the 
# try_5 failure mode directly — per-view-inconsistent diffusion albedo restructuring 
# gaussians through the raster (zncc → 63° normals vs zncc_grad → 36°).
# 
# The practical difference in your batch: the LR reduction is a blunt instrument 
# — it's likely a big part of why try_5 prior runs lost ~4 dB test PSNR vs baseline 
# (geometry at 5% LR couldn't recover from opacity resets and fresh densification clones).
# The grad-scale is surgical — it removes only the harmful channel while leaving photometric 
# geometry refinement at full speed. The _plus runs currently use both (LR→0.05 and grad-scale 0.0), 
# which is somewhat redundant; if the batch confirms the grad-scale works, a natural follow-up is 
# relaxing reduce_geo_lr_third_stage toward 0.3–1.0 to win back the photometric quality — 
# that's the A/B I'd queue next.


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
        "name": "hotdog",
        "args": {
            "source_path": HOTDOG_DIR,
            "white_background": False,         # hotdog is a Blender-synthetic scene
            "resolution": 2,                   # 800x800 native -> 400x400 (-r 2)
            "albedo_gt_dir": "albedo_gt",      # WORLD-space GT albedo
            "normal_gt_dir": "normal_gt",      # WORLD-space GT normal
            "metallic_gt_dir": "metallic_simulated_zero",
            "roughness_gt_dir": "",
            "eval_relight_hdris": HOTDOG_DIR_RELIGHT_HDRs,
            "envmap_gt_path": os.path.join(HOTDOG_DIR, "hdris", "sunset.hdr"),
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
# ALBEDO VARIANTS  -- try_7: light-linear indirect + anchor/reg_hdr unbundling
# =============================================================================
# Every GT variant below copies the try_6 gt_zncc_grad_ctrl stage-1/2-relevant
# parameters EXACTLY (tv_reduction_factor 0.75, lambda_normal_gt 0.8 from
# COMMON, albedo warm-up on) so the warm-up-buffer fingerprint matches the
# buffered try_6 checkpoints and only Phase 3 is recomputed. Same for the
# diffusion variant vs the try_6 diff runs. All new knobs (anchor, LLI,
# reg_hdr, roughness prior, detach) are Phase-3-only.
# geo_lr_final_iter 66k keeps the r4 batches' RELATIVE anneal shape: there the
# window was second_stage + 1.2x the phase-3 length (25k + 1.2*12.5k = 40k on
# a 37.5k run), so at 60k it is 30k + 1.2*30k = 66k — the factor reached at
# the final iteration (~0.11) matches the earlier batches.
_GT_BASE_ARGS = {
    "albedo_prior_mode": "zncc_grad", "tv_reduction_factor": 0.75,
    "reg_hdr_weight": 0.001,
    "reduce_geo_lr_third_stage": 0.05, "geo_lr_final_iter": 66_000,
}
_PRIOR_FLAGS = {
    "use_prior_weight_scheduler": True,
    "albedo_geometry_warmup": True,
}

ALBEDO_VARIANTS = [
    {
        # GIR baseline exactly as in the paper / reference repo: the
        # train_tensoir.sh launch line is
        #   train.py -s <scene> --eval --random_background --hdr_rotation \
        #            --reg_hdr_weight 0.1 --reg_material_weight 0.05
        # with every other value at the engine defaults, which COMMON now
        # mirrors (60k, stages 5k/30k, densify 500..45k, lambda_dssim 0.4,
        # percent_dense 0.01, opacity resets every 3k). Sole deviation: the
        # batch-wide max_gaussians 1.5M cap (COMMON) — kept on the baseline
        # too so all four runs are comparable; it only binds on runaway
        # densification at r2.
        # exclude_prior_loss keeps the added prior losses OUT of the
        # optimization (they are still computed for logging); all other new
        # engine knobs are verified inert at their defaults (tv_reduction 1.0,
        # reduce_geo_lr 1.0, grad-scale 1.0, no warm-up, no anchor, no LLI).
        "name": "baseline_no_prior",
        "args": {"reg_hdr_weight": 0.1, "tv_reduction_factor": 1.0},
        "flags": {"exclude_prior_loss": True},
    },
    {
        "name" : "diff_zncc_zncc",
        "args": {
            **_GT_BASE_ARGS,
            "albedo_prior_mode": "zncc",
            "albedo_gt_dir": "albedo",
            "normal_gt_dir": "normal",
            "metallic_gt_dir": "metallic",
            "roughness_gt_dir": "roughness",
            "normal_camera_convention": "opengl",
            "lambda_metallic_gt": 0.05,
            "lambda_normal_gt": 0.4,
        },
        "flags": {**_PRIOR_FLAGS, "light_linear_indirect": True},
    },
    {
        "name" : "gt_zncc_zncc_neu",
        "args": {
            **_GT_BASE_ARGS,
            "albedo_prior_mode": "zncc",
            "albedo_anchor_weight": 0,
        },
        "flags": {**_PRIOR_FLAGS, "light_linear_indirect": True},
    },
]

# Where all run folders for this batch live.
EXPERIMENT_ROOT = os.path.join(REPO_DIR, "outputs", "new_experiments_try_10_paper_scale_hotdog")

# Finished runs from earlier batches to overlay in the comparison report
# WITHOUT re-running them (name shown in the report, absolute model path).
# NOTE: t6/t7 relight metrics were computed over ALL 200 test views; t8 and
# the new runs use 24 (expect ±0.1-0.2 dB sampling difference — do not
# over-read).
# try_10 runs at -r 2 / 60k: every earlier batch (t6-t9) ran at -r 4 / 37.5k,
# so their metrics are NOT comparable at this operating point and nothing is
# overlaid. (Put (name, model_path) tuples here to overlay future r2 runs.)
EXTERNAL_RUNS = []

# Build the experiments = every albedo variant on every dataset. Each
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
# STAGE-1/2 WARM-UP BUFFER  -- reuse the geometry + normal-alignment stages
# =============================================================================
# Phases 1 (geometry warm-up) and 2 (normal alignment) — everything at
# iter <= second_stage_step — depend only on a subset of the configuration.
# Any two runs that agree on ALL of those parameters produce an identical
# stage-2 checkpoint, so we can compute it ONCE, stash the checkpoint in a
# shared buffer keyed by a fingerprint of those parameters, and let every later
# run that matches resume straight into Phase 3 (full PBR) via train.py's
# `--start_checkpoint`.
#
# Buffer layout:
#   outputs/warmup_buffer/<fingerprint-hash>/
#       metadata.json              # the fingerprint + provenance (human-readable)
#       chkpnt<second_stage_step>.pth
WARMUP_BUFFER_ROOT = os.path.join(REPO_DIR, "outputs", "warmup_buffer")

# Parameters that influence the geometry warm-up + normal-alignment stages.
# (Stage-3-only knobs — reg_*_weight, the prior-weight scheduler, geo-LR
# reduction, the third-stage reset toggle, metallic/roughness priors, eval /
# relight / logging settings — are deliberately EXCLUDED so runs that differ
# only in Phase 3 share the same warm-up.)
WARMUP_FINGERPRINT_KEYS = [
    # scene / data
    "source_path", "resolution", "white_background", "random_background",
    "normal_gt_dir", "normal_camera_convention",
    # stage boundaries / total length
    "first_stage_step", "second_stage_step", "iterations",
    # densification (runs throughout stages 1 & 2)
    "percent_dense", "densification_interval", "opacity_reset_interval",
    "densify_from_iter", "densify_until_iter", "densify_grad_threshold",
    "max_gaussians",
    # losses active in stages 1 & 2
    "lambda_dssim", "lambda_normal_gt", "tv_reduction_factor",
]
# Boolean store_true flags that influence stages 1 & 2.
WARMUP_FINGERPRINT_FLAG_KEYS = [
    "white_background", "random_background",
    "exclude_prior_loss", "albedo_geometry_warmup",
    "hdr_rotation",  # rotates the stage-2 envmap queries
]


def _full_config(exp):
    """Merge COMMON + the experiment's args + flags into one flat dict.

    Mirrors the precedence used by build_command (dataset/variant args win over
    COMMON, flags add the booleans), so the fingerprint sees exactly the values
    that train.py will receive.
    """
    cfg = dict(COMMON)
    cfg.update(exp.get("args", {}))
    cfg.update(exp.get("flags", {}))
    return cfg


# Bump this whenever an ENGINE change alters what stages 1 & 2 compute, so
# buffered checkpoints from before the change can never be reused.
#   v2: envlight cubemap/latlong/FG-LUT sampling fixed (nvdiffrast dr.texture),
#       camera-space diffusion normals, alpha-masked prior losses.
#   v3: distCUDA2 scale init restored (reference behaviour); hdr_rotation now
#       part of the fingerprint (affects the stage-2 light queries).
WARMUP_CODE_VERSION = 3


def warmup_fingerprint(exp):
    """Build the ordered dict of stage-1/2-influencing parameters for `exp`."""
    cfg = _full_config(exp)
    fp = {"warmup_code_version": WARMUP_CODE_VERSION}
    for k in WARMUP_FINGERPRINT_KEYS:
        fp[k] = cfg.get(k, None)
    for k in WARMUP_FINGERPRINT_FLAG_KEYS:
        fp[k] = bool(cfg.get(k, False))
    # The albedo prior only takes part in the warm-up stages when albedo
    # geometry warm-up is enabled; otherwise its settings are irrelevant here
    # (the albedo prior is a Phase-3-only loss), so we leave them out to keep
    # the buffer shared across albedo-only variations.
    if fp.get("albedo_geometry_warmup"):
        fp["albedo_gt_dir"] = cfg.get("albedo_gt_dir", "")
        fp["lambda_albedo_gt"] = cfg.get("lambda_albedo_gt", None)
        fp["huber_delta"] = cfg.get("huber_delta", None)
        # Stages 1 & 2 use warmup_albedo_prior_mode, falling back to
        # albedo_prior_mode when empty (same rule as train.py).
        fp["warmup_albedo_prior_mode"] = (
            cfg.get("warmup_albedo_prior_mode", "") or cfg.get("albedo_prior_mode", "")
        )
    return fp


def _fingerprint_hash(fingerprint):
    blob = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _second_stage_step(exp):
    return int(_full_config(exp).get("second_stage_step"))


def find_buffered_checkpoint(exp):
    """Return the path to a buffered stage-2 checkpoint matching `exp`, or None."""
    fp = warmup_fingerprint(exp)
    sss = _second_stage_step(exp)
    entry_dir = os.path.join(WARMUP_BUFFER_ROOT, _fingerprint_hash(fp))
    meta_path = os.path.join(entry_dir, "metadata.json")
    ckpt_path = os.path.join(entry_dir, f"chkpnt{sss}.pth")
    if not (os.path.exists(meta_path) and os.path.exists(ckpt_path)):
        return None
    # Guard against the (unlikely) hash collision: confirm an exact match.
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None
    if meta.get("fingerprint") == fp:
        return ckpt_path
    return None


def save_to_buffer(exp, model_path):
    """Copy the freshly-computed stage-2 checkpoint of `exp` into the buffer."""
    fp = warmup_fingerprint(exp)
    sss = _second_stage_step(exp)
    src = os.path.join(model_path, f"chkpnt{sss}.pth")
    if not os.path.exists(src):
        print(f"[buffer] WARNING: expected stage-2 checkpoint '{src}' not found; "
              f"nothing buffered for '{exp['name']}'.", flush=True)
        return
    entry_dir = os.path.join(WARMUP_BUFFER_ROOT, _fingerprint_hash(fp))
    os.makedirs(entry_dir, exist_ok=True)
    dst = os.path.join(entry_dir, f"chkpnt{sss}.pth")
    shutil.copy2(src, dst)
    with open(os.path.join(entry_dir, "metadata.json"), "w") as f:
        json.dump({
            "fingerprint": fp,
            "second_stage_step": sss,
            "source_run": os.path.basename(model_path),
            "saved": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    print(f"[buffer] Buffered stage-2 checkpoint -> {dst}", flush=True)


# =============================================================================
# RUN LAUNCHING
# =============================================================================

def _fmt_value(v):
    """Format a config value as command-line argument token(s)."""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def build_command(exp, model_path, start_checkpoint=None, ensure_checkpoint_iter=None):
    """Build the `python train.py ...` command for one experiment.

    start_checkpoint        : if given, resume from this checkpoint (stage-2
                              buffer) so only Phase 3 is computed.
    ensure_checkpoint_iter  : if given, make sure this iteration is in
                              checkpoint_iterations so the stage-2 checkpoint is
                              written (and can later be copied into the buffer).
    """
    cfg = dict(COMMON)
    cfg.update(exp.get("args", {}))

    # Guarantee a checkpoint is written at the stage-2 boundary so it can be
    # buffered afterwards (only relevant for runs that actually compute it).
    if ensure_checkpoint_iter is not None:
        ckpt_iters = list(cfg.get("checkpoint_iterations", []))
        if ensure_checkpoint_iter not in ckpt_iters:
            ckpt_iters.append(ensure_checkpoint_iter)
            ckpt_iters.sort()
        cfg["checkpoint_iterations"] = ckpt_iters

    # Boolean store_true flags handled separately (engine + run via train.py).
    bool_flags = {
        "eval", "white_background", "exclude_prior_loss", "use_prior_weight_scheduler",
        "remove_noise", "hdr_rotation", "random_background", "quiet",
        "disable_reset_third_stage", "albedo_geometry_warmup",
        "light_linear_indirect",
    }

    flags = dict(exp.get("flags", {}))
    # Promote any bool entries living in cfg into the flags set.
    for key in list(cfg.keys()):
        if key in bool_flags and isinstance(cfg[key], bool):
            flags.setdefault(key, cfg.pop(key))

    cmd = [sys.executable, "train.py"]
    cmd += ["--source_path", cfg.pop("source_path")]
    cmd += ["--model_path", model_path]

    if start_checkpoint:
        # train.py loads this via torch.load and resumes from the stored
        # iteration (== second_stage_step), so only Phase 3 runs.
        cmd += ["--start_checkpoint", start_checkpoint]

    for key, value in cfg.items():
        # An empty list (e.g. eval_relight_hdris=[] for real-world data) would
        # produce a "--flag" with no values, which argparse nargs="+" rejects.
        # Skip it so the engine falls back to its default / no-op behaviour.
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        # An empty string scalar (e.g. warmup_albedo_prior_mode="") likewise
        # just means "use the engine default", so skip it too.
        if isinstance(value, str) and value == "":
            continue
        cmd.append(f"--{key}")
        cmd += _fmt_value(value)

    for key, enabled in flags.items():
        if enabled:
            cmd.append(f"--{key}")

    return cmd


def own_checkpoints(model_path):
    """Map iteration -> path for every chkpnt<iter>.pth inside `model_path`."""
    found = {}
    for p in glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        m = re.fullmatch(r"chkpnt(\d+)\.pth", os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    return found


def run_experiment(exp, use_buffer=True, resume=False, resume_iter=None):
    """Launch a single experiment as a subprocess. Returns (model_path, rc).

    When `use_buffer` is on we first look for a buffered stage-2 checkpoint
    whose fingerprint matches this run's Phase-1/2 configuration:
      * HIT  -> resume from it (Phase 3 only).
      * MISS -> run all three phases, then copy the stage-2 checkpoint into the
                buffer for future runs.

    `resume` restarts a crashed/killed run from a checkpoint that already sits
    in its OWN model_path (`resume_iter` picks which; default: the latest).
    This bypasses the warm-up buffer on BOTH ends: no fingerprint lookup — so
    config edits that change the fingerprint (e.g. lowering max_gaussians
    after an OOM) don't force a fresh 3-phase run — and no buffering
    afterwards, because the resumed checkpoint was produced under the OLD
    config and the new fingerprint would mis-describe it.
    """
    model_path = os.path.join(EXPERIMENT_ROOT, exp["name"])
    os.makedirs(model_path, exist_ok=True)

    # Guard: this batch is repaired incrementally (--only / --resume), and a
    # bare `python run_experiments.py` would otherwise re-train every listed
    # run from scratch INTO ITS EXISTING FOLDER, clobbering days of results.
    # Any run that already owns a checkpoint is therefore skipped unless the
    # user explicitly resumes it.
    if not resume:
        existing = own_checkpoints(model_path)
        if existing:
            final_iter = int(_full_config(exp).get("iterations", 0))
            state = ("complete" if final_iter in existing
                     else f"partial (latest chkpnt{max(existing)}.pth)")
            print(f"[skip] '{exp['name']}' already has checkpoints "
                  f"({state}). Use --resume to continue it, or delete "
                  f"{model_path} to re-run from scratch.", flush=True)
            return model_path, 0

    buffered_ckpt = None
    if resume:
        ckpts = own_checkpoints(model_path)
        ckpt_iter = resume_iter if resume_iter is not None else (max(ckpts) if ckpts else None)
        if ckpt_iter not in ckpts:
            wanted = f"chkpnt{resume_iter}.pth" if resume_iter is not None else "chkpnt<iter>.pth"
            print(f"[resume] ERROR: no {wanted} in {model_path}; nothing to "
                  f"resume for '{exp['name']}'. Available: "
                  f"{sorted(ckpts) or 'none'}.", flush=True)
            return model_path, 1
        # Preserve the crashed attempt's artifacts: checkpoints past the
        # resume point would be silently overwritten once training passes
        # those iterations again, and the stdout log is truncated below.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for it in sorted(ckpts):
            if it > ckpt_iter:
                os.rename(ckpts[it], f"{ckpts[it]}.stale-{stamp}")
                print(f"[resume] set aside chkpnt{it}.pth -> .stale-{stamp}", flush=True)
        old_log = os.path.join(model_path, "train_stdout.log")
        if os.path.exists(old_log):
            os.rename(old_log, f"{old_log}.{stamp}")
        print(f"[resume] '{exp['name']}': resuming from its own "
              f"chkpnt{ckpt_iter}.pth (warm-up buffer bypassed).", flush=True)
        cmd = build_command(exp, model_path, start_checkpoint=ckpts[ckpt_iter])
    else:
        buffered_ckpt = find_buffered_checkpoint(exp) if use_buffer else None
        sss = _second_stage_step(exp)
        if buffered_ckpt:
            print(f"[buffer] HIT for '{exp['name']}': reusing stage-2 checkpoint "
                  f"{buffered_ckpt} (running Phase 3 only).", flush=True)
            cmd = build_command(exp, model_path, start_checkpoint=buffered_ckpt)
        else:
            if use_buffer:
                print(f"[buffer] MISS for '{exp['name']}': computing all three phases "
                      f"(stage-2 checkpoint will be buffered at iter {sss}).", flush=True)
            cmd = build_command(exp, model_path,
                                ensure_checkpoint_iter=(sss if use_buffer else None))

    print("\n" + "=" * 78)
    print(f"RUN: {exp['name']}")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print("=" * 78, flush=True)

    log_path = os.path.join(model_path, "train_stdout.log")
    # The try_10 OOM (gt_zncc_grad_anchor_lli2, iter 46990) failed a 1.83 GiB
    # alloc with 6.98 GiB *reserved but unallocated* — caching-allocator
    # fragmentation, not true memory pressure. expandable_segments lets the
    # allocator grow segments instead of hoarding fixed-size blocks; it does
    # not change numerics. A PYTORCH_CUDA_ALLOC_CONF set by the user wins.
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Stream the child's stdout/stderr live to this console AND tee it to a log
    # file. We read in small chunks (not full lines) so the tqdm progress bar,
    # which uses carriage returns, updates in real time.
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=GIR_DIR, env=env,
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
    elif use_buffer and not resume and not buffered_ckpt:
        # Fresh full run succeeded: stash its stage-2 checkpoint for reuse.
        save_to_buffer(exp, model_path)
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

    palette = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
               "#795548", "#607D8B", "#F44336", "#3F51B5", "#8BC34A", "#FFC107"]

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

        # \u2500\u2500 Page 3b: Energy calibration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # relight_<hdri>_gain is the fitted global gain g* (render -> GT):
        # g* > 1 means the relit render is too DARK (the light-transport
        # energy deficit); g* ~ 1 is calibrated. test_gain is the same fit
        # under the TRAINING light (photometric loss pins it near 1; try_7:
        # ~1.09) \u2014 the relight-vs-test gap is the relight-specific deficit.
        # envmap_mean_ratio is learned-envmap brightness / GT-HDRI brightness;
        # in try_7 it tracked the relight gain (the envmap absorbs the deficit).
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Energy Calibration (fitted gain render\u2192GT; 1.0 = calibrated)",
                     fontsize=14, fontweight="bold")
        plot_avg_relight(axes[0, 0], "_gain", "gain g*", "Mean Relight Gain (\u21921.0)")
        axes[0, 0].axhline(1.0, color="#888", linewidth=1, linestyle=":")
        ax_gbar = axes[0, 1]
        gain_hdris = sorted({k.replace("relight_", "").replace("_gain", "")
                             for d in data for k in _relight_keys(d["metrics"], "_gain")})
        if gain_hdris:
            import numpy as np
            x = np.arange(len(gain_hdris)); w = 0.8 / max(1, len(data))
            for i, d in enumerate(data):
                m = d["metrics"][-1] if d["metrics"] else {}
                vals = [m.get(f"relight_{h}_gain", float("nan")) for h in gain_hdris]
                ax_gbar.bar(x + i * w, vals, w, color=d["color"], label=d["name"])
            ax_gbar.set_xticks(x + (len(data) - 1) * w / 2)
            ax_gbar.set_xticklabels(gain_hdris, rotation=20, ha="right", fontsize=8)
            ax_gbar.axhline(1.0, color="#888", linewidth=1, linestyle=":")
            ax_gbar.set_ylabel("gain g*"); ax_gbar.set_title("Final per-HDRI Relight Gain (\u21921.0)")
            ax_gbar.legend(loc="best", fontsize=7)
        else:
            ax_gbar.text(0.5, 0.5, "No gain data (pre-try_7 runs)", ha="center",
                         va="center", transform=ax_gbar.transAxes)
        # Bookkeeping identity (try_8): relight_gain ~ envmap_mean_ratio x
        # albedo_gain \u2014 these two curves decompose the relight gain above.
        # (test_gain, the ~1.09 train-light sanity constant, lives in the CSV.)
        plot_metric(axes[1, 0], "test_albedo_gain", "gain g*",
                    "Albedo Gain (<1 = albedo too bright, \u21921.0)")
        axes[1, 0].axhline(1.0, color="#888", linewidth=1, linestyle=":")
        plot_metric(axes[1, 1], "envmap_mean_ratio", "model / GT",
                    "EnvMap Brightness Ratio (\u21921.0)")
        axes[1, 1].axhline(1.0, color="#888", linewidth=1, linestyle=":")
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
        cols = ["run", "test_psnr", "test_psnr_aligned", "test_gain", "test_ssim", "test_lpips",
                "mean_relight_psnr", "mean_relight_psnr_aligned", "mean_relight_gain", "mean_relight_ssim",
                "test_albedo_psnr", "test_albedo_psnr_aligned", "test_albedo_gain", "test_normal_ang_err", "test_metallic_mae", "test_roughness_mae",
                "envmap_log_psnr", "envmap_rel_l1", "envmap_mean_ratio",
                "albedo_gt", "normal_gt", "metallic_gt", "roughness_gt", "num_gaussians"]
        f.write(",".join(cols) + "\n")
        for d in data:
            m = d["metrics"][-1] if d["metrics"] else {}
            lc = d["loss"][-1] if d["loss"] else {}
            _, rp = _avg_relight_series(d["metrics"], "_psnr")
            _, rpa = _avg_relight_series(d["metrics"], "_psnr_aligned")
            _, rg = _avg_relight_series(d["metrics"], "_gain")
            _, rs = _avg_relight_series(d["metrics"], "_ssim")
            row = [
                d["name"],
                m.get("test_psnr", ""), m.get("test_psnr_aligned", ""), m.get("test_gain", ""), m.get("test_ssim", ""), m.get("test_lpips", ""),
                rp[-1] if rp else "", rpa[-1] if rpa else "", rg[-1] if rg else "", rs[-1] if rs else "",
                m.get("test_albedo_psnr", ""), m.get("test_albedo_psnr_aligned", ""), m.get("test_albedo_gain", ""), m.get("test_normal_ang_err", ""),
                m.get("test_metallic_mae", ""), m.get("test_roughness_mae", ""),
                m.get("envmap_log_psnr", ""), m.get("envmap_rel_l1", ""), m.get("envmap_mean_ratio", ""),
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

def _parse_override(token):
    """Parse one --set 'KEY=VALUE' token; value typed as int, float, bool or str."""
    key, sep, raw = token.partition("=")
    if not sep or not key:
        sys.exit(f"--set expects KEY=VALUE, got '{token}'")
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw


def main():
    ap = argparse.ArgumentParser(description="GIR experiment runner + comparison report")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip training; only (re)build the comparison report.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the named experiments (default: all).")
    ap.add_argument("--no-buffer", action="store_true",
                    help="Disable the stage-1/2 warm-up buffer (always compute "
                         "all three phases and never read/write the buffer).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume each selected experiment from the latest "
                         "chkpnt<iter>.pth already in its model_path (e.g. "
                         "after an OOM). Bypasses the warm-up buffer, so "
                         "config edits that change the warm-up fingerprint "
                         "(max_gaussians, ...) do not force a fresh run.")
    ap.add_argument("--resume-iter", type=int, default=None,
                    help="With --resume: resume from this exact checkpoint "
                         "iteration instead of the latest. Implies --resume.")
    ap.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                    help="Override scalar config values for the SELECTED "
                         "experiments, e.g. --set max_gaussians=1100000. "
                         "Merged into the experiment args (highest "
                         "precedence), so warm-up fingerprints of fresh runs "
                         "stay consistent with what train.py receives.")
    args = ap.parse_args()

    selected = EXPERIMENTS
    if args.only:
        selected = [e for e in EXPERIMENTS if e["name"] in args.only]
        if not selected:
            print(f"No experiments match {args.only}. Available: "
                  f"{[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    if args.set:
        overrides = dict(_parse_override(t) for t in args.set)
        for e in selected:
            e["args"] = {**e.get("args", {}), **overrides}
        print(f"[override] {overrides} -> {[e['name'] for e in selected]}")

    resume = args.resume or (args.resume_iter is not None)

    runs = []
    # Overlay finished runs from earlier batches (never re-run here). Listed
    # first so the report orders them as the reference curves.
    for name, path in EXTERNAL_RUNS:
        if os.path.exists(os.path.join(path, "metrics_log.json")):
            runs.append((name, path))
        else:
            print(f"[external] WARNING: '{path}' has no metrics_log.json; skipped.")

    if args.report_only:
        for e in selected:
            runs.append((e["name"], os.path.join(EXPERIMENT_ROOT, e["name"])))
    else:
        os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
        # Archive this script alongside the results so the batch config is
        # preserved even after the file is edited for the next batch.
        try:
            shutil.copy2(os.path.abspath(__file__), os.path.join(EXPERIMENT_ROOT, "run_experiments.py"))
        except shutil.SameFileError:
            pass
        for e in selected:
            run_experiment(e, use_buffer=not args.no_buffer,
                           resume=resume, resume_iter=args.resume_iter)
        # The report always covers every experiment in the batch that has
        # metrics — a --only / --resume repair run must not shrink the
        # comparison PDF down to the runs it happened to (re)launch.
        for e in EXPERIMENTS:
            mp = os.path.join(EXPERIMENT_ROOT, e["name"])
            if os.path.exists(os.path.join(mp, "metrics_log.json")):
                runs.append((e["name"], mp))

    generate_comparison(runs, EXPERIMENT_ROOT)
    print("\nAll done.")


if __name__ == "__main__":
    main()
