#!/usr/bin/env python3
"""
Generate a PDF training report from metrics_log.json.

Usage:
    python generate_report.py --model_path ../outputs
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Style configuration
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "test":  "#2196F3",   # Blue
    "train": "#FF9800",   # Orange
    "single": "#4CAF50",  # Green
}


def _extract(metrics_log, key):
    """Extract (iterations, values) for a given key, skipping missing entries."""
    iters, vals = [], []
    for entry in metrics_log:
        if key in entry and entry[key] is not None:
            iters.append(entry["iteration"])
            vals.append(entry[key])
    return iters, vals


def _plot_dual(ax, metrics_log, test_key, train_key, ylabel, title, higher_is_better=True):
    """Plot test and train curves on the same axes."""
    iters_t, vals_t = _extract(metrics_log, test_key)
    iters_tr, vals_tr = _extract(metrics_log, train_key)

    if iters_t:
        ax.plot(iters_t, vals_t, "o-", color=COLORS["test"], markersize=3, linewidth=1.5, label="Test")
    if iters_tr:
        ax.plot(iters_tr, vals_tr, "s--", color=COLORS["train"], markersize=3, linewidth=1.2, label="Train")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")

    # Mark the best value
    if iters_t and vals_t:
        best_fn = max if higher_is_better else min
        best_val = best_fn(vals_t)
        best_iter = iters_t[vals_t.index(best_val)]
        ax.annotate(f"{best_val:.4f}",
                    xy=(best_iter, best_val), xytext=(10, 10 if higher_is_better else -15),
                    textcoords="offset points", fontsize=8, color=COLORS["test"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["test"], lw=0.8))


def _plot_single(ax, metrics_log, key, ylabel, title, color=None):
    """Plot a single-series curve."""
    iters, vals = _extract(metrics_log, key)
    if not iters:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    c = color or COLORS["single"]
    ax.plot(iters, vals, "o-", color=c, markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def generate_report(model_path):
    """Generate a training report PDF and return its path."""
    metrics_path = os.path.join(model_path, "metrics_log.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics_log.json not found in {model_path}")

    with open(metrics_path, "r") as f:
        metrics_log = json.load(f)

    if not metrics_log:
        raise ValueError("metrics_log.json is empty")

    report_path = os.path.join(model_path, "training_report.pdf")

    # Try to read config
    cfg_text = ""
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_text = f.read().strip()

    with PdfPages(report_path) as pdf:
        # ── Page 1: Title ─────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax = fig.add_subplot(111)
        ax.axis("off")

        scene_name = os.path.basename(os.path.normpath(model_path))
        total_iters = metrics_log[-1]["iteration"]
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        ax.text(0.5, 0.72, "PBR-3DGS Training Report", fontsize=28, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes, color="#1a1a2e")
        ax.text(0.5, 0.60, f"Scene: {scene_name}", fontsize=16,
                ha="center", va="center", transform=ax.transAxes, color="#16213e")
        ax.text(0.5, 0.52, f"Total iterations: {total_iters:,}  |  Evaluation points: {len(metrics_log)}",
                fontsize=12, ha="center", va="center", transform=ax.transAxes, color="#555")
        ax.text(0.5, 0.46, f"Generated: {date_str}", fontsize=10,
                ha="center", va="center", transform=ax.transAxes, color="#888")

        # Final metrics summary box
        final = metrics_log[-1]
        summary_lines = []
        for key, label in [("test_psnr", "PSNR"), ("test_ssim", "SSIM"), ("test_lpips", "LPIPS"),
                           ("test_l1", "L1"), ("test_mse", "MSE")]:
            if key in final:
                summary_lines.append(f"{label}: {final[key]:.4f}")
        if summary_lines:
            summary_text = "Final Test Metrics:  " + "  |  ".join(summary_lines)
            ax.text(0.5, 0.34, summary_text, fontsize=10, ha="center", va="center",
                    transform=ax.transAxes, color="#333",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e3f2fd", edgecolor="#90caf9", alpha=0.8))

        if cfg_text:
            # Show a compact config summary
            ax.text(0.5, 0.15, f"Config: {cfg_text[:200]}{'...' if len(cfg_text) > 200 else ''}",
                    fontsize=7, ha="center", va="center", transform=ax.transAxes,
                    color="#666", fontstyle="italic", wrap=True)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Main metrics (PSNR, SSIM, LPIPS) ─────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Image Quality Metrics Over Training", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#FAFAFA")

        _plot_dual(axes[0, 0], metrics_log, "test_psnr", "train_psnr", "PSNR (dB)", "PSNR ↑", higher_is_better=True)
        _plot_dual(axes[0, 1], metrics_log, "test_ssim", "train_ssim", "SSIM", "SSIM ↑", higher_is_better=True)
        _plot_dual(axes[1, 0], metrics_log, "test_lpips", "train_lpips", "LPIPS", "LPIPS ↓", higher_is_better=False)
        _plot_single(axes[1, 1], metrics_log, "train_loss", "Loss", "Training Loss (EMA)", color="#E91E63")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 3: Loss metrics (L1, MSE) & model stats ─────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Loss Metrics & Model Statistics", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#FAFAFA")

        _plot_dual(axes[0, 0], metrics_log, "test_l1", "train_l1", "L1 Loss", "L1 Loss ↓", higher_is_better=False)
        _plot_dual(axes[0, 1], metrics_log, "test_mse", "train_mse", "MSE", "MSE ↓", higher_is_better=False)
        _plot_single(axes[1, 0], metrics_log, "num_gaussians", "Count", "Number of Gaussians", color="#9C27B0")

        # Summary table on bottom-right
        ax_table = axes[1, 1]
        ax_table.axis("off")
        ax_table.set_title("Final Metrics Summary")

        table_data = []
        header = ["Metric", "Test", "Train"]
        for metric_name, test_key, train_key in [
            ("PSNR (dB)", "test_psnr", "train_psnr"),
            ("SSIM", "test_ssim", "train_ssim"),
            ("LPIPS", "test_lpips", "train_lpips"),
            ("L1 Loss", "test_l1", "train_l1"),
            ("MSE", "test_mse", "train_mse"),
        ]:
            test_val = final.get(test_key, None)
            train_val = final.get(train_key, None)
            table_data.append([
                metric_name,
                f"{test_val:.4f}" if test_val is not None else "N/A",
                f"{train_val:.4f}" if train_val is not None else "N/A",
            ])
        if "train_loss" in final:
            table_data.append(["Train Loss", f"{final['train_loss']:.6f}", "—"])
        if "num_gaussians" in final:
            table_data.append(["# Gaussians", f"{final['num_gaussians']:,}", "—"])

        table = ax_table.table(cellText=table_data, colLabels=header,
                               loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        # Style header
        for j in range(len(header)):
            table[0, j].set_facecolor("#1a1a2e")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(header)):
                table[i, j].set_facecolor("#f5f5f5" if i % 2 == 0 else "white")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 4+: Visual comparisons ───────────────────────────────
        vis_dir = os.path.join(model_path, "eval_visuals")
        if os.path.exists(vis_dir):
            for split in ["test", "train"]:
                split_dir = os.path.join(vis_dir, split)
                if not os.path.isdir(split_dir):
                    continue

                image_files = sorted(glob.glob(os.path.join(split_dir, "*.png")))
                if not image_files:
                    continue

                # Group by iteration
                iter_groups = {}
                for img_path in image_files:
                    fname = os.path.basename(img_path)
                    # Format: iter{NNNNNN}_view{N}.png
                    iter_str = fname.split("_")[0].replace("iter", "")
                    try:
                        it = int(iter_str)
                    except ValueError:
                        continue
                    iter_groups.setdefault(it, []).append(img_path)

                for it in sorted(iter_groups.keys()):
                    imgs = iter_groups[it]
                    n_imgs = len(imgs)
                    fig, axes = plt.subplots(1, n_imgs, figsize=(11, 3.5 + 0.5))
                    if n_imgs == 1:
                        axes = [axes]
                    fig.suptitle(f"Visual Comparison — {split.capitalize()} Set — Iteration {it:,}",
                                 fontsize=12, fontweight="bold", y=1.0)
                    fig.patch.set_facecolor("#FAFAFA")

                    for ax, img_path in zip(axes, imgs):
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                        ax.axis("off")
                        view_name = os.path.basename(img_path).replace(".png", "").split("_")[-1]
                        ax.set_title(f"Render | GT  ({view_name})", fontsize=9)

                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training report PDF from metrics log")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model output directory containing metrics_log.json")
    args = parser.parse_args()

    generate_report(args.model_path)
