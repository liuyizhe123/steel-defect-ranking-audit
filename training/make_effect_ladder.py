# -*- coding: utf-8 -*-
"""Build the protocol-vs-model effect-size ladder and fragility ratios."""

from __future__ import annotations

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "figs", "fig10_effect_ladder.png")
SUMMARY = os.path.join(BASE, "effect_ladder_summary.json")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as handle:
        return json.load(handle)


def style(ax):
    ax.grid(True, color="0.82", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_png(fig):
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
    image.save(OUT, dpi=(300, 300))


def main() -> None:
    pt = load("pt_final_results.json")
    review = load("review_report.json")
    yolo = load("yolo_matrix_summary.json")

    clean_mean = pt["architecture_seed_replication"]["baseline"]["mAP50_mean"]
    split_mean = review["standard_split_results"]["v8n_ciou"]["mAP50"]["mean"]
    split_change = split_mean - clean_mean

    eval_jit10 = (
        review["evaluation_label_realizations"]["neu"]["yolov8n_clean"][
            "test_jit10"
        ]["mean"]
    )
    eval_drop20 = (
        review["evaluation_label_realizations"]["neu"]["yolov8n_clean"][
            "test_drop20"
        ]["mean"]
    )
    clean_single_seed = yolo["clean_640"]["mAP50"] * 100
    eval_jit10_change = clean_single_seed - eval_jit10
    eval_drop20_change = clean_single_seed - eval_drop20

    clean_values = pt["seed_replication"]["clean640"]["mAP50_values"]
    train_jit20_change = st.mean(
        clean - value for clean, value in zip(clean_values, pt["seed_replication"]["jit20"]["mAP50_values"])
    )
    train_drop60_change = st.mean(
        clean - value for clean, value in zip(clean_values, pt["seed_replication"]["drop60"]["mAP50_values"])
    )

    model_rows = []
    for key, label in [
        ("escm_ciou", "ESCM only"),
        ("lsc_ciou", "ESCM+SAF"),
        ("v8n_fwn", "FWN"),
        ("v8n_focaler", "Focaler"),
        ("v8n_hybrid", "NWD/Wise"),
    ]:
        row = review["architecture_seed_comparison"][key]["delta50"]
        delta = abs(float(row["mean"]))
        sd = float(row["sd"])
        model_rows.append(
            {
                "model": label,
                "delta50": delta,
                "delta_sd": sd,
                "fragility_ratio": delta / sd,
            }
        )
    model_rows.sort(key=lambda row: row["fragility_ratio"])
    model_effect_min = min(row["delta50"] for row in model_rows)
    model_effect_max = max(row["delta50"] for row in model_rows)

    ladder = [
        ("Train-label jitter 20%", train_jit20_change, "train-label", "#d9541a"),
        ("Test-label jitter 10%", eval_jit10_change, "test-label", "#315ba8"),
        ("Test-label omission 20%", eval_drop20_change, "test-label", "#315ba8"),
        ("Train-label omission 60%", train_drop60_change, "train-label", "#d9541a"),
        ("Split 6:2:2 to 7:2:1", split_change, "split", "#6b4b9a"),
        (
            f"Model effects ({model_effect_min:.2f}-{model_effect_max:.2f})",
            model_effect_max,
            "model",
            "#2b8c7e",
        ),
    ]
    ladder.sort(key=lambda row: row[1], reverse=True)

    seed_sd_min = pt["architecture_seed_replication"]["baseline"]["mAP50_sd"]
    seed_sd_max = pt["architecture_seed_replication"]["v8n_fwn"]["mAP50_sd"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax in axes:
        style(ax)

    ax = axes[0]
    labels = [row[0] for row in ladder]
    y = np.arange(len(ladder))
    ax.axvspan(seed_sd_min, seed_sd_max, color="#bdbdbd", alpha=0.25, zorder=0)
    bars = []
    for position, row in zip(y, ladder):
        if row[2] == "model":
            bar = ax.barh(
                position,
                model_effect_max - model_effect_min,
                left=model_effect_min,
                color=row[3],
                height=0.62,
            )
        else:
            bar = ax.barh(position, row[1], color=row[3], height=0.62)
        bars.append(bar[0])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 29)
    ax.set_xlabel("Absolute mAP@0.5 shift (points)")
    ax.set_title("(a) Effect-size ladder")
    for bar, row in zip(bars, ladder):
        if row[2] == "model":
            label = f"{model_effect_min:.2f}-{model_effect_max:.2f}"
            text_x = model_effect_max + 0.45
        else:
            label = f"{row[1]:.2f}"
            text_x = row[1] + 0.45
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    ax.text(
        seed_sd_max + 0.35,
        0.15,
        f"Seed SD {seed_sd_min:.2f}-{seed_sd_max:.2f}",
        color="0.32",
        fontsize=7.5,
        va="center",
    )

    ax = axes[1]
    labels = [
        f"{row['model']}\n|Δ| {row['delta50']:.2f} / SD {row['delta_sd']:.2f}"
        for row in model_rows
    ]
    ratios = [row["fragility_ratio"] for row in model_rows]
    y = np.arange(len(model_rows))
    colors = ["#315ba8" if ratio < 1 else "#2b8c7e" for ratio in ratios]
    bars = ax.barh(y, ratios, color=colors, height=0.58)
    ax.axvline(1.0, color="#b03a2e", linestyle="--", linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.4)
    ax.set_xlabel("Standardized paired effect size |$\\Delta$| / SD($\\Delta$)")
    ax.set_title("(b) Ranking fragility")
    for bar, value in zip(bars, ratios):
        ax.text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    ax.text(1.05, len(model_rows) - 0.45, "|effect| = seed SD", color="#b03a2e", fontsize=7.5)

    fig.suptitle(
        "Benchmark and annotation shifts versus model effects in the tested NEU-DET audit",
        fontsize=11,
        fontweight="bold",
    )
    save_png(fig)
    plt.close(fig)

    summary = {
        "clean_baseline_mAP50": round(clean_mean, 4),
        "standard_split_baseline_mAP50": round(split_mean, 4),
        "model_effect_range": {
            "min": round(model_effect_min, 4),
            "max": round(model_effect_max, 4),
        },
        "seed_sd_range": {
            "min": round(seed_sd_min, 4),
            "max": round(seed_sd_max, 4),
        },
        "effect_ladder": [
            {
                "factor": row[0],
                "category": row[2],
                "absolute_mAP50_shift": round(row[1], 4),
            }
            for row in ladder
        ],
        "ranking_fragility": [
            {
                **row,
                "delta50": round(row["delta50"], 4),
                "delta_sd": round(row["delta_sd"], 4),
                "fragility_ratio": round(row["fragility_ratio"], 4),
            }
            for row in model_rows
        ],
        "definition": (
            "Standardized paired effect size: abs(mean paired mAP50 difference) "
            "/ paired seed SD; analogous to Cohen's dz. Ranking fragility is an "
            "interpretive label, not a new statistic."
        ),
        "scope": (
            "Descriptive summary of heterogeneous interventions; F=1 is not a "
            "power threshold or MDES."
        ),
    }
    with open(SUMMARY, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("EFFECT_LADDER_OK", OUT, SUMMARY)


if __name__ == "__main__":
    main()
