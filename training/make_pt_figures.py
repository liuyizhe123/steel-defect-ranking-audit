# -*- coding: utf-8 -*-
"""Regenerate manuscript figures from the final PyTorch-only results."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def style_axis(ax):
    ax.grid(True, color="0.78", linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_png(fig, path):
    """Save through the Agg canvas to avoid a Python 3.14 savefig crash."""
    fig.set_dpi(300)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.frombuffer(
        "RGBA",
        (width, height),
        fig.canvas.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1,
    ).convert("RGB")
    image.save(path, dpi=(300, 300))


def annotation_figure():
    jitter = np.array([0, 5, 10, 20])
    jitter50 = np.array([73.33, 71.89, 66.85, 45.87])
    jitter95 = np.array([39.84, 36.81, 30.09, 20.10])
    jitter_repeat50 = 48.03
    jitter_repeat95 = 20.85

    kept = np.array([100, 83.3, 69.6, 56.7])
    omitted = 100 - kept
    drop50 = np.array([73.33, 70.98, 69.84, 65.74])
    drop95 = np.array([39.84, 37.65, 37.03, 34.46])
    drop_repeat50 = 66.27
    drop_repeat95 = 34.96

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax in axes:
        style_axis(ax)
    ax = axes[0]
    ax.plot(jitter, jitter50, marker="o", lw=1.8, ms=7, color="#d9541a", label="mAP@0.5, seed 2026")
    ax.plot(jitter, jitter95, marker="s", lw=1.8, ms=7, color="#0073b8", label="mAP@0.5:0.95, seed 2026")
    ax.errorbar(20, jitter_repeat50, yerr=2.02, fmt="D", ms=8, capsize=4, color="#d9541a", mfc="white", label="mAP@0.5, 3-seed mean")
    ax.errorbar(20, jitter_repeat95, yerr=0.68, fmt="D", ms=8, capsize=4, color="#0073b8", mfc="white", label="mAP@0.5:0.95, 3-seed mean")
    for x, y in zip(jitter, jitter50):
        offset = -2.2 if x == jitter[-1] else 1.5
        va = "top" if offset < 0 else "bottom"
        ax.text(x, y + offset, f"{y:.1f}", ha="center", va=va)
    for x, y in zip(jitter, jitter95):
        ax.text(x, y - 2.3, f"{y:.1f}", ha="center", va="top")
    ax.set_xlabel("Box jitter σ (% of box size)")
    ax.set_ylabel("mAP (%)")
    ax.set_title("(a) Box-location/size noise")
    ax.set_ylim(0, 82)
    ax.legend(loc="lower left", frameon=True)

    ax = axes[1]
    ax.plot(omitted, drop50, marker="o", lw=1.8, ms=7, color="#d9541a", label="mAP@0.5, seed 2026")
    ax.plot(omitted, drop95, marker="s", lw=1.8, ms=7, color="#0073b8", label="mAP@0.5:0.95, seed 2026")
    ax.errorbar(43.3, drop_repeat50, yerr=0.73, fmt="D", ms=8, capsize=4, color="#d9541a", mfc="white", label="mAP@0.5, 3-seed mean")
    ax.errorbar(43.3, drop_repeat95, yerr=0.44, fmt="D", ms=8, capsize=4, color="#0073b8", mfc="white", label="mAP@0.5:0.95, 3-seed mean")
    for x, y in zip(omitted, drop50):
        offset = -2.0 if x == omitted[-1] else 1.3
        va = "top" if offset < 0 else "bottom"
        ax.text(x, y + offset, f"{y:.1f}", ha="center", va=va)
    for x, y in zip(omitted, drop95):
        ax.text(x, y - 2.0, f"{y:.1f}", ha="center", va="top")
    ax.set_xlabel("Target instances omitted (%)")
    ax.set_ylabel("mAP (%)")
    ax.set_title("(b) Missing annotations")
    ax.set_xlim(-2, 46)
    ax.set_ylim(0, 82)
    ax.legend(loc="lower left", frameon=True)

    save_png(fig, os.path.join(OUT, "fig5_annotation.png"))
    plt.close(fig)


def resolution_figure():
    conditions = ["Clean", "σ=10% jitter", "40% instances omitted"]
    m50_640 = np.array([73.33, 66.85, 69.84])
    m50_320 = np.array([71.65, 67.17, 68.40])
    m95_640 = np.array([39.84, 30.09, 37.03])
    m95_320 = np.array([39.08, 31.67, 35.57])
    x = np.arange(len(conditions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, a, b, title, ylabel, ylim in [
        (axes[0], m50_640, m50_320, "(a) mAP@0.5", "mAP@0.5 (%)", (0, 82)),
        (axes[1], m95_640, m95_320, "(b) mAP@0.5:0.95", "mAP@0.5:0.95 (%)", (0, 48)),
    ]:
        style_axis(ax)
        bars1 = ax.bar(x - width / 2, a, width, color="#4c73b0", label="640×640")
        bars2 = ax.bar(x + width / 2, b, width, color="#d99933", label="320×320")
        ax.set_xticks(x, conditions)
        ax.tick_params(axis="x", labelrotation=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(loc="lower left", frameon=True)
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    save_png(fig, os.path.join(OUT, "fig6_resolution.png"))
    plt.close(fig)


def complexity_figure():
    points = [
        (3.012, 73.86, 0.69, "YOLOv8n+CIoU (3 seeds)", "#315ba8", (2.72, 75.2)),
        (3.035, 72.57, 1.58, "ESCM-only+CIoU (3 seeds)", "#2b8c7e", (3.08, 71.3)),
        (3.012, 72.07, 2.72, "YOLOv8n+FWN (3 seeds)", "#77a83c", (2.72, 68.0)),
        (0.978, 71.31, 0.90, "ESCM+SAF+CIoU (3 seeds)", "#8a3f91", (1.22, 72.4)),
        (0.978, 70.10, None, "ESCM+SAF+FWN (1 seed)", "#d9541a", (1.22, 68.7)),
        (0.955, 66.64, None, "SAF only+CIoU (1 seed)", "#6b4b9a", (1.20, 65.8)),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    style_axis(ax)
    for x, y, yerr, label, color, text_xy in points:
        ax.scatter(x, y, s=110, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.2,
                        capsize=4, zorder=2)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=text_xy,
            ha="left" if text_xy[0] > x else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7),
        )
    ax.plot([3.012, 3.012], [73.86, 72.07], "k:", lw=0.9)
    ax.plot([3.012, 3.035], [73.86, 72.57], "k:", lw=0.9)
    ax.plot([0.978, 0.978], [71.31, 70.10], "k:", lw=0.9)
    ax.set_xlim(0.8, 3.25)
    ax.set_ylim(64.5, 76.0)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Complexity versus accuracy (NEU-DET, PyTorch protocol)")
    save_png(fig, os.path.join(OUT, "fig7_complexity.png"))
    plt.close(fig)


def validation_figure():
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), constrained_layout=True)
    for ax in axes.flat:
        style_axis(ax)

    ax = axes[0, 0]
    gc10_labels = ["v8n\nCIoU", "v8n\nFWN", "LSC\nCIoU", "LSC\nFWN"]
    gc10_values = [59.80, 58.55, 58.24, 53.41]
    colors = ["#315ba8", "#77a83c", "#8a3f91", "#d9541a"]
    bars = ax.bar(gc10_labels, gc10_values, color=colors)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 65)
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("(a) GC10-DET selected VOC mirror")

    ax = axes[0, 1]
    conditions = ["Clean", "σ=10% jitter", "40% instances omitted"]
    v8 = np.array([73.33, 66.85, 69.84])
    v10 = np.array([70.36, 59.33, 63.56])
    x = np.arange(3)
    width = 0.34
    b1 = ax.bar(x - width / 2, v8, width, color="#315ba8", label="YOLOv8n")
    b2 = ax.bar(x + width / 2, v10, width, color="#cc6b2c", label="YOLOv10n")
    ax.set_xticks(x, conditions)
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(0, 80)
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("(b) Cross-model training perturbation")
    ax.legend(loc="lower left", frameon=True)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 0]
    jitter = np.array([0, 5, 10, 20])
    y8_jitter = np.array([73.33, 71.06, 63.01, 31.71])
    y10_jitter = np.array([70.36, 68.42, 60.97, 30.36])
    e8_jitter = np.array([0.0, 0.0, 0.76, 1.08])
    e10_jitter = np.array([0.0, 0.0, 0.68, 0.59])
    ax.errorbar(
        jitter, y8_jitter, yerr=e8_jitter, marker="o", lw=1.8,
        capsize=3, color="#315ba8", label="YOLOv8n"
    )
    ax.errorbar(
        jitter, y10_jitter, yerr=e10_jitter, marker="s", lw=1.8,
        capsize=3, color="#cc6b2c", label="YOLOv10n"
    )
    ax.set_xticks(jitter)
    ax.set_xlabel("Test-annotation box jitter σ (%)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_ylim(0, 80)
    ax.set_title("(c) Evaluation-side box jitter")
    ax.legend(loc="lower left", frameon=True)

    ax = axes[1, 1]
    omitted = np.array([0, 20, 40, 60])
    y8_drop = np.array([73.33, 63.67, 56.18, 46.30])
    y10_drop = np.array([70.36, 61.45, 53.40, 44.51])
    e8_drop = np.array([0.0, 0.98, 0.82, 0.0])
    e10_drop = np.array([0.0, 1.53, 0.94, 0.0])
    ax.errorbar(
        omitted, y8_drop, yerr=e8_drop, marker="o", lw=1.8,
        capsize=3, color="#315ba8", label="YOLOv8n"
    )
    ax.errorbar(
        omitted, y10_drop, yerr=e10_drop, marker="s", lw=1.8,
        capsize=3, color="#cc6b2c", label="YOLOv10n"
    )
    ax.set_xticks(omitted)
    ax.set_xlim(-5, 65)
    ax.set_xlabel("Test instances omitted (%)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_ylim(0, 80)
    ax.set_title("(d) Evaluation-side instance omission")
    ax.legend(loc="lower left", frameon=True)

    save_png(fig, os.path.join(OUT, "fig5_validation.png"))
    plt.close(fig)


def standard_split_figure():
    summary_path = os.path.join(BASE, "review_matrix_summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    labels = ["YOLOv8n\nCIoU", "YOLOv8n\nFWN", "ESCM+SAF\nCIoU"]
    keys = {
        "YOLOv8n\nCIoU": "v8n_ciou",
        "YOLOv8n\nFWN": "v8n_fwn",
        "ESCM+SAF\nCIoU": "lsc_ciou",
    }
    values50 = []
    values95 = []
    std50 = []
    std95 = []
    for label in labels:
        model = keys[label]
        rows = [
            summary[f"standard721_{model}_s{seed}"]
            for seed in ("26", "27", "28")
        ]
        m50 = np.array([row["mAP50"] * 100 for row in rows])
        m95 = np.array([row["mAP5095"] * 100 for row in rows])
        values50.append(m50.mean())
        values95.append(m95.mean())
        std50.append(m50.std(ddof=1))
        std95.append(m95.std(ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    colors = ["#315ba8", "#77a83c", "#8a3f91"]
    for ax, values, errors, title, ylabel, ymax in [
        (axes[0], values50, std50, "(a) mAP@0.5", "mAP@0.5 (%)", 82),
        (axes[1], values95, std95, "(b) mAP@0.5:0.95", "mAP@0.5:0.95 (%)", 50),
    ]:
        style_axis(ax)
        bars = ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    save_png(fig, os.path.join(OUT, "fig8_standard_split.png"))
    plt.close(fig)


def architecture_figure():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax in axes:
        ax.set_axis_off()

    def box(ax, x, y, w, h, text, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.03",
            linewidth=1.1, edgecolor="#333333", facecolor=color
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(ax, a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                     linewidth=0.9, color="#333333"))

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) ESCM", fontsize=11, fontweight="bold")
    box(ax, 0.01, 0.42, 0.16, 0.16, "input\n$C$", "#dce8f7")
    box(ax, 0.23, 0.62, 0.17, 0.12, "$x_1$\n$C/4$", "#f5e6cb")
    box(ax, 0.23, 0.28, 0.17, 0.18, "$x_2$\n$3C/4$", "#e6f2df")
    box(ax, 0.45, 0.25, 0.15, 0.16, "1x1\nPW", "#e6f2df")
    box(ax, 0.66, 0.38, 0.14, 0.11, "1x7 DW", "#dce8f7")
    box(ax, 0.66, 0.16, 0.14, 0.11, "7x1 DW", "#dce8f7")
    box(ax, 0.45, 0.60, 0.15, 0.12, "channel\ngate", "#f5d9df")
    box(ax, 0.84, 0.21, 0.14, 0.18, "shuffle\n+ 1x1\n+ concat", "#eadff2")
    arrow(ax, (0.17, 0.50), (0.23, 0.68))
    arrow(ax, (0.17, 0.50), (0.23, 0.37))
    arrow(ax, (0.40, 0.37), (0.45, 0.33))
    arrow(ax, (0.60, 0.33), (0.66, 0.43))
    arrow(ax, (0.60, 0.33), (0.66, 0.22))
    arrow(ax, (0.73, 0.43), (0.84, 0.32))
    arrow(ax, (0.73, 0.22), (0.84, 0.28))
    arrow(ax, (0.60, 0.33), (0.52, 0.60))
    arrow(ax, (0.60, 0.66), (0.84, 0.35))
    ax.text(0.62, 0.78, "$g=\\sigma(GAP(W_g z))$", ha="center", fontsize=7)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) SAF neck", fontsize=11, fontweight="bold")
    box(ax, 0.02, 0.72, 0.17, 0.14, "backbone\nP2", "#dce8f7")
    box(ax, 0.02, 0.49, 0.17, 0.14, "backbone\nP3", "#dce8f7")
    box(ax, 0.02, 0.26, 0.17, 0.14, "backbone\nP4", "#dce8f7")
    box(ax, 0.28, 0.10, 0.18, 0.14, "SPPF on\nP4", "#f5e6cb")
    box(ax, 0.28, 0.69, 0.19, 0.14, "Concat + C2f\nP2/P3", "#e6f2df")
    box(ax, 0.28, 0.44, 0.19, 0.14, "Concat + C2f\nP3/P4", "#e6f2df")
    box(ax, 0.57, 0.10, 0.18, 0.14, "DSConv\nP3/P4", "#eadff2")
    box(ax, 0.80, 0.69, 0.15, 0.12, "P2 head", "#f5d9df")
    box(ax, 0.80, 0.44, 0.15, 0.12, "P3 head", "#f5d9df")
    box(ax, 0.80, 0.19, 0.15, 0.12, "P4 head", "#f5d9df")
    arrow(ax, (0.19, 0.52), (0.28, 0.72))
    arrow(ax, (0.19, 0.77), (0.28, 0.77))
    arrow(ax, (0.19, 0.33), (0.28, 0.49))
    arrow(ax, (0.19, 0.30), (0.28, 0.17))
    arrow(ax, (0.47, 0.76), (0.80, 0.75))
    arrow(ax, (0.47, 0.51), (0.80, 0.50))
    arrow(ax, (0.46, 0.17), (0.80, 0.25))
    ax.text(0.47, 0.91, "P5 removed", ha="center", fontsize=8, color="#b03a2e")
    ax.plot([0.23, 0.39], [0.92, 0.92], color="#b03a2e", lw=1.1, ls="--")

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(c) FWN box loss", fontsize=11, fontweight="bold")
    box(ax, 0.02, 0.58, 0.17, 0.18, "predicted\nbox", "#dce8f7")
    box(ax, 0.02, 0.22, 0.17, 0.18, "target\nbox", "#f5e6cb")
    box(ax, 0.28, 0.55, 0.19, 0.18, "IoU\n$L_{IoU}$\n$L_{Foc}$", "#e6f2df")
    box(ax, 0.28, 0.22, 0.19, 0.18, "NWD\n$L_{NWD}$", "#dce8f7")
    box(ax, 0.55, 0.68, 0.16, 0.13, "scale\nweight $\\lambda$", "#f5d9df")
    box(ax, 0.55, 0.30, 0.16, 0.13, "Wise\nfactor $r$", "#eadff2")
    box(ax, 0.80, 0.43, 0.16, 0.20, "$L_{box}$\n$=\\lambda L_{NWD}$\n$+(1-\\lambda)rL_{Foc}$", "#fff2cc")
    arrow(ax, (0.19, 0.67), (0.28, 0.64))
    arrow(ax, (0.19, 0.31), (0.28, 0.31))
    arrow(ax, (0.47, 0.64), (0.55, 0.37))
    arrow(ax, (0.47, 0.31), (0.55, 0.73))
    arrow(ax, (0.47, 0.31), (0.80, 0.48))
    arrow(ax, (0.71, 0.73), (0.80, 0.58))
    arrow(ax, (0.71, 0.37), (0.80, 0.48))
    ax.text(0.28, 0.08, "$\\lambda=\\sigma(-k(s-s_0))$", fontsize=7)

    save_png(fig, os.path.join(OUT, "fig2_architecture.png"))
    plt.close(fig)


if __name__ == "__main__":
    annotation_figure()
    resolution_figure()
    complexity_figure()
    validation_figure()
    standard_split_figure()
    architecture_figure()
    print("PT_FIGURES_OK")
