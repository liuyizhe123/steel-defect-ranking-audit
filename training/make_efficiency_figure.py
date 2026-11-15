# -*- coding: utf-8 -*-
"""Build the efficiency figure in a process that does not import Torch/YOLO."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SUMMARY = BASE / "efficiency_summary.json"
MEMORY_REPEATS = BASE / "efficiency_memory_repeats.json"
FIGURE = ROOT / "figs" / "fig9_efficiency.png"


def load_accuracy() -> dict[str, float]:
    yolo = json.loads((BASE / "yolo_matrix_summary.json").read_text(encoding="utf-8"))
    lsc = json.loads((BASE / "lsc_pt_summary.json").read_text(encoding="utf-8"))
    return {
        "YOLOv8n + CIoU": float(yolo["clean_640"]["mAP50"]) * 100,
        "ESCM + CIoU": float(lsc["escm_ciou_pt"]["mAP50"]) * 100,
        "YOLOv8n + FWN": float(lsc["v8n_fwn_pt_v2"]["mAP50"]) * 100,
        "ESCM+SAF + CIoU": float(lsc["lsc_ciou_pt"]["mAP50"]) * 100,
        "ESCM+SAF + FWN": float(lsc["lsc_fwn_pt_v3"]["mAP50"]) * 100,
        "SAF only + CIoU": float(lsc["saf_ciou_pt"]["mAP50"]) * 100,
    }


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    memory_repeats = json.loads(MEMORY_REPEATS.read_text(encoding="utf-8"))
    accuracy = load_accuracy()
    names = list(accuracy)
    throughput = [
        summary["models"][name]["batch1"]["throughput_images_per_s"] for name in names
    ]
    memory = [
        memory_repeats["models"][name]["mean"] for name in names
    ]
    colors = ["#315ba8", "#2b8c7e", "#77a83c", "#8a3f91", "#d9541a", "#6b4b9a"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax = axes[0]
    values = [accuracy[name] for name in names]
    ax.scatter(throughput, values, c=colors, s=58)
    for name, x, y in zip(names, throughput, values):
        ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Single-image throughput (images/s)")
    ax.set_ylabel("mAP@0.5 on seed-2026 checkpoint (%)")
    ax.set_title("(a) Throughput versus accuracy")
    ax.grid(True, color="0.85", linewidth=0.7)

    ax = axes[1]
    ax.barh(names, memory, color=colors)
    for index, value in enumerate(memory):
        ax.text(value + 5, index, f"{value:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Peak allocated CUDA memory at batch 8 (MB)")
    ax.set_title("(b) Inference memory")
    ax.grid(True, axis="x", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)

    FIGURE.parent.mkdir(exist_ok=True)
    fig.set_dpi(300)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    Image.frombuffer(
        "RGBA",
        (width, height),
        fig.canvas.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1,
    ).convert("RGB").save(FIGURE, dpi=(300, 300))
    plt.close(fig)
    print("EFFICIENCY_FIGURE_OK", FIGURE, FIGURE.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
