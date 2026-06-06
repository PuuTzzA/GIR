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
    * (Learned) uncertainty weights  -> w_albedo / w_metallic / w_normal
    * Training loss / #gaussians

The three default runs for the `cube_colorful` quick test are:
    1. no_prior      : priors computed for logging only, NOT optimized
    2. fixed_lambda  : priors weighted by fixed lambda_albedo/normal/metallic_gt
    3. uncertainty   : priors weighted by learnable Kendall uncertainty weights

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    # Run all experiments, then build the comparison report:
    python run_experiments.py

    # Only (re)build the comparison report from existing run folders:
    python run_experiments.py --report-only

    # Run a subset by name:
    python run_experiments.py --only no_prior uncertainty

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

# Parameters shared by every run in this experiment batch.
# These are deliberately "quick test" values: few iterations and an EARLY start
# of the PBR (second) stage so the prior losses kick in soon.
COMMON = {
    "source_path": os.path.join(REPO_DIR, "data", "datasets_with_priors", "cube_colorful"),
    "eval": True,                 # hold out the test cameras for novel-view eval
    "white_background": False,    # cube_colorful is a Blender-synthetic scene
    "densify_grad_threshold": 0.0004,# default: 0.0002
    "resolution": 2,               # default: -1     | -1 = keep native resolution
    "max_gaussians": 300000,       # default: 0      | hard cap on #gaussians (0 = unlimited)

    "iterations": 30000,           # total iterations (default engine: 60_000)
    "first_stage_step": 2000,     # end of radiance warm-up (default: 5_000)
    "second_stage_step": 4000,    # START of PBR decomposition EARLY (default: 30_000)

    "eval_interval": 191,         # evaluate metrics every N iters
    "visual_interval": 3500,      # save visual comparisons every N iters

    # HDRIs available for cube_colorful (rgba_<name> folders + hdris/<name>.hdr)
    "eval_relight_hdris": ["snowy_forest", "moonless_night", "gym_entrance"],

    # Keep disk usage small for the quick test: only save/checkpoint at the end.
    "save_iterations": [7000],
    "test_iterations": [7000],
    "checkpoint_iterations": [7000],

    # Fixed-lambda weights (used by the fixed_lambda run).
    "lambda_albedo_gt": 0.5,
    "lambda_normal_gt": 0.1,
    "lambda_metallic_gt": 0.05,
}

# Where all run folders for this batch live.
EXPERIMENT_ROOT = os.path.join(REPO_DIR, "outputs", "experiments_cube_colorful")

# Each experiment = a display name + a dict of args that OVERRIDE / EXTEND COMMON.
# `flags` are boolean store_true switches passed only when True.
EXPERIMENTS = [
    {
        "name": "no_prior",
        "args": {},
        "flags": {"exclude_prior_loss": True},   # compute priors but don't optimize them
    },
    {
        "name": "fixed_lambda",
        "args": {},
        "flags": {"use_uncertainty_weights": False},  # use fixed lambda_*_gt weights
    },
    {
        "name": "uncertainty",
        "args": {},
        "flags": {"use_uncertainty_weights": True},   # learnable Kendall weights
    },
]


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
        "eval", "white_background", "exclude_prior_loss",
        "use_uncertainty_weights", "freeze_uncertainty_weights",
        "remove_noise", "hdr_rotation", "random_background", "quiet",
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
        axes[1, 1].axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4: Prior losses + uncertainty weights ───────────────────
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle("Prior Losses & Uncertainty Weights",
                     fontsize=14, fontweight="bold")
        plot_loss(axes[0, 0], "albedo_gt", "Loss", "Albedo prior loss \u2193")
        plot_loss(axes[0, 1], "normal_gt", "Loss", "Normal prior loss \u2193")
        plot_loss(axes[0, 2], "metallic_gt", "Loss", "Metallic prior loss \u2193")
        plot_loss(axes[1, 0], "w_albedo", "w", "w_albedo (log-var)")
        plot_loss(axes[1, 1], "w_metallic", "w", "w_metallic (log-var)")
        plot_loss(axes[1, 2], "w_normal", "w", "w_normal (log-var)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── CSV summary of final metrics ─────────────────────────────────────
    with open(csv_path, "w") as f:
        cols = ["run", "test_psnr", "test_ssim", "test_lpips",
                "mean_relight_psnr", "mean_relight_ssim",
                "albedo_gt", "normal_gt", "metallic_gt", "num_gaussians"]
        f.write(",".join(cols) + "\n")
        for d in data:
            m = d["metrics"][-1] if d["metrics"] else {}
            lc = d["loss"][-1] if d["loss"] else {}
            _, rp = _avg_relight_series(d["metrics"], "_psnr")
            _, rs = _avg_relight_series(d["metrics"], "_ssim")
            row = [
                d["name"],
                m.get("test_psnr", ""), m.get("test_ssim", ""), m.get("test_lpips", ""),
                rp[-1] if rp else "", rs[-1] if rs else "",
                lc.get("albedo_gt", ""), lc.get("normal_gt", ""), lc.get("metallic_gt", ""),
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
