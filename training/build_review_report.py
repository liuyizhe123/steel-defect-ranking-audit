# -*- coding: utf-8 -*-
"""Aggregate reviewer-response experiments into machine- and human-readable reports."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT_JSON = BASE / "review_report.json"
OUT_MD = BASE / "review_report.md"


def load(name: str, default=None):
    path = BASE / name
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": st.mean(values),
        "sd": st.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def kendall_tau(left, right) -> float:
    n = len(left)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (left[i] - left[j]) * (right[i] - right[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else 1.0


def main() -> None:
    yolo = load("yolo_matrix_summary.json", {})
    lsc = load("lsc_pt_summary.json", {})
    seed = load("seed_replication_summary.json", {})
    interaction = load("lsc_pt_interaction_summary.json", {})
    review = load("review_matrix_summary.json", {})
    test_realizations = load("test_annotation_realizations_summary.json", {})
    test_existing = load("test_annotation_eval_summary.json", {})
    gc10_test_existing = load("gc10_test_annotation_eval_summary.json", {})
    efficiency = load("efficiency_summary.json")
    qualitative = load("qualitative_summary.json")
    standard_split = load("standard_split_summary.json", {})
    rtdetr = load("rtdetr_summary.json", {})

    architecture_groups = {
        "baseline": [
            yolo.get("clean_640", {}).get("mAP50"),
            yolo.get("clean_640_s27", {}).get("mAP50"),
            yolo.get("clean_640_s28", {}).get("mAP50"),
        ],
        "escm_ciou": [
            lsc.get("escm_ciou_pt", {}).get("mAP50"),
            seed.get("escm_ciou_s27", {}).get("mAP50"),
            seed.get("escm_ciou_s28", {}).get("mAP50"),
        ],
        "lsc_ciou": [
            lsc.get("lsc_ciou_pt", {}).get("mAP50"),
            seed.get("lsc_ciou_s27", {}).get("mAP50"),
            seed.get("lsc_ciou_s28", {}).get("mAP50"),
        ],
        "v8n_fwn": [
            lsc.get("v8n_fwn_pt_v2", {}).get("mAP50"),
            seed.get("v8n_fwn_s27", {}).get("mAP50"),
            seed.get("v8n_fwn_s28", {}).get("mAP50"),
        ],
        "v8n_focaler": [
            lsc.get("v8n_focaler_pt", {}).get("mAP50"),
            seed.get("v8n_focaler_s27", {}).get("mAP50"),
            seed.get("v8n_focaler_s28", {}).get("mAP50"),
        ],
        "v8n_hybrid": [
            lsc.get("v8n_hybrid_pt", {}).get("mAP50"),
            seed.get("v8n_hybrid_s27", {}).get("mAP50"),
            seed.get("v8n_hybrid_s28", {}).get("mAP50"),
        ],
    }
    architecture_groups95 = {
        "baseline": [
            yolo.get("clean_640", {}).get("mAP5095"),
            yolo.get("clean_640_s27", {}).get("mAP5095"),
            yolo.get("clean_640_s28", {}).get("mAP5095"),
        ],
        "escm_ciou": [
            lsc.get("escm_ciou_pt", {}).get("mAP5095"),
            seed.get("escm_ciou_s27", {}).get("mAP5095"),
            seed.get("escm_ciou_s28", {}).get("mAP5095"),
        ],
        "lsc_ciou": [
            lsc.get("lsc_ciou_pt", {}).get("mAP5095"),
            seed.get("lsc_ciou_s27", {}).get("mAP5095"),
            seed.get("lsc_ciou_s28", {}).get("mAP5095"),
        ],
        "v8n_fwn": [
            lsc.get("v8n_fwn_pt_v2", {}).get("mAP5095"),
            seed.get("v8n_fwn_s27", {}).get("mAP5095"),
            seed.get("v8n_fwn_s28", {}).get("mAP5095"),
        ],
        "v8n_focaler": [
            lsc.get("v8n_focaler_pt", {}).get("mAP5095"),
            seed.get("v8n_focaler_s27", {}).get("mAP5095"),
            seed.get("v8n_focaler_s28", {}).get("mAP5095"),
        ],
        "v8n_hybrid": [
            lsc.get("v8n_hybrid_pt", {}).get("mAP5095"),
            seed.get("v8n_hybrid_s27", {}).get("mAP5095"),
            seed.get("v8n_hybrid_s28", {}).get("mAP5095"),
        ],
    }
    architecture = {
        name: {
            "mAP50": mean_sd([value * 100 for value in values if value is not None]),
            "mAP5095": mean_sd(
                [
                    value * 100
                    for value in architecture_groups95[name]
                    if value is not None
                ]
            ),
        }
        for name, values in architecture_groups.items()
    }
    architecture_comparison = {}
    baseline50 = architecture_groups["baseline"]
    baseline95 = architecture_groups95["baseline"]
    for name in architecture_groups:
        if name == "baseline" or any(value is None for value in architecture_groups[name]):
            continue
        delta50 = [
            (architecture_groups[name][index] - baseline50[index]) * 100
            for index in range(3)
        ]
        delta95 = [
            (architecture_groups95[name][index] - baseline95[index]) * 100
            for index in range(3)
        ]
        architecture_comparison[name] = {
            "delta50": mean_sd(delta50),
            "delta5095": mean_sd(delta95),
        }

    standard_groups = {}
    for model_name in ("v8n_ciou", "v8n_fwn", "lsc_ciou"):
        values50 = []
        values95 = []
        for seed_id in ("26", "27", "28"):
            row = review.get(f"standard721_{model_name}_s{seed_id}", {})
            if "mAP50" in row:
                values50.append(row["mAP50"] * 100)
                values95.append(row["mAP5095"] * 100)
        standard_groups[model_name] = {
            "mAP50": mean_sd(values50),
            "mAP5095": mean_sd(values95),
            "_raw50": values50,
            "_raw95": values95,
        }
    baseline_standard50 = standard_groups["v8n_ciou"]["_raw50"]
    baseline_standard95 = standard_groups["v8n_ciou"]["_raw95"]
    for model_name, row in standard_groups.items():
        if model_name == "v8n_ciou":
            row["delta50"] = None
            row["delta5095"] = None
            continue
        row["delta50"] = mean_sd(
            [
                row["_raw50"][index] - baseline_standard50[index]
                for index in range(3)
            ]
        )
        row["delta5095"] = mean_sd(
            [
                row["_raw95"][index] - baseline_standard95[index]
                for index in range(3)
            ]
        )
    for row in standard_groups.values():
        row.pop("_raw50", None)
        row.pop("_raw95", None)

    p2p5 = {
        name: {
            "mAP50": review[name]["mAP50"] * 100,
            "mAP5095": review[name]["mAP5095"] * 100,
            "p2": review[name]["variant"]["p2"],
            "p5": review[name]["variant"]["p5"],
            "params": review[name]["params"],
        }
        for name in (
            "p2off_p5on_s26",
            "p2on_p5on_s26",
            "p2off_p5off_s26",
            "p2on_p5off_s26",
        )
        if name in review
    }

    fwn_base = lsc.get("v8n_fwn_pt_v2", {})
    fwn_sensitivity = {}
    if fwn_base:
        fwn_sensitivity["base"] = {
            "mAP50": fwn_base["mAP50"] * 100,
            "mAP5095": fwn_base["mAP5095"] * 100,
            "config": {"k": 10.0, "s0": 0.06},
        }
    for name in ("fwn_k05_s26", "fwn_k20_s26", "fwn_s003_s26", "fwn_s012_s26"):
        if name in review:
            fwn_sensitivity[name] = {
                "mAP50": review[name]["mAP50"] * 100,
                "mAP5095": review[name]["mAP5095"] * 100,
                "config": review[name]["fwn_config"],
            }

    interaction_groups = {}
    interaction_raw = {}
    model_keys = {
        "baseline": "v8n_ciou",
        "fwn": "v8n_fwn",
        "lsc_ciou": "lsc_ciou",
    }
    clean_keys = {
        "baseline": ("clean_640", "clean_640_s27", "clean_640_s28"),
    }
    for label, model_name in model_keys.items():
        for condition in ("clean", "jit10", "drop40"):
            values = []
            if condition == "clean":
                if label == "baseline":
                    values = [
                        yolo.get(key, {}).get("mAP50")
                        for key in clean_keys["baseline"]
                    ]
                elif label == "fwn":
                    values = [
                        lsc.get("v8n_fwn_pt_v2", {}).get("mAP50"),
                        seed.get("v8n_fwn_s27", {}).get("mAP50"),
                        seed.get("v8n_fwn_s28", {}).get("mAP50"),
                    ]
                else:
                    values = [
                        lsc.get("lsc_ciou_pt", {}).get("mAP50"),
                        seed.get("lsc_ciou_s27", {}).get("mAP50"),
                        seed.get("lsc_ciou_s28", {}).get("mAP50"),
                    ]
            else:
                if label == "baseline":
                    values = [yolo.get(f"{condition}_640", {}).get("mAP50")]
                else:
                    original_key = f"{model_name}_{condition}_pt"
                    values = [interaction.get(original_key, {}).get("mAP50")]
                values.extend(
                    review.get(
                        f"interaction_{model_name}_{condition}_s{seed_id}",
                        {},
                    ).get("mAP50")
                    for seed_id in ("27", "28")
                )
            clean_values = [value for value in values if value is not None]
            interaction_raw[f"{label}_{condition}"] = clean_values
            interaction_groups[f"{label}_{condition}"] = mean_sd(
                [value * 100 for value in clean_values]
            )

    interaction_comparison = {}
    for condition in ("clean", "jit10", "drop40"):
        baseline_values = interaction_raw[f"baseline_{condition}"]
        comparison = {}
        for label in ("fwn", "lsc_ciou"):
            values = interaction_raw[f"{label}_{condition}"]
            comparison[label] = mean_sd(
                [
                    (values[index] - baseline_values[index]) * 100
                    for index in range(len(values))
                ]
            )
        interaction_comparison[condition] = comparison

    rank_stability = {}
    if all(
        interaction_groups.get(f"{label}_{condition}")
        and interaction_groups[f"{label}_{condition}"]["n"] >= 1
        for label in model_keys
        for condition in ("clean", "jit10", "drop40")
    ):
        labels = list(model_keys)
        for condition in ("jit10", "drop40"):
            clean_vector = [
                interaction_groups[f"{label}_clean"]["mean"] for label in labels
            ]
            condition_vector = [
                interaction_groups[f"{label}_{condition}"]["mean"] for label in labels
            ]
            margins = []
            for left in range(len(labels)):
                for right in range(left + 1, len(labels)):
                    clean_margin = clean_vector[left] - clean_vector[right]
                    condition_margin = condition_vector[left] - condition_vector[right]
                    margins.append(
                        {
                            "pair": f"{labels[left]} vs {labels[right]}",
                            "clean_margin": clean_margin,
                            "condition_margin": condition_margin,
                            "reversal": clean_margin * condition_margin < 0,
                        }
                    )
            rank_stability[condition] = {
                "kendall_tau_vs_clean": kendall_tau(clean_vector, condition_vector),
                "clean_ranking": [
                    labels[index]
                    for index in sorted(range(len(labels)), key=lambda i: -clean_vector[i])
                ],
                "condition_ranking": [
                    labels[index]
                    for index in sorted(
                        range(len(labels)), key=lambda i: -condition_vector[i]
                    )
                ],
                "pairwise_margins": margins,
                "reversals": [row for row in margins if row["reversal"]],
            }

    repeated = {}
    for dataset_name, models in {
        "neu": ("yolov8n_clean", "yolov10n_clean"),
        "gc10": ("yolov8n_clean",),
    }.items():
        repeated[dataset_name] = {}
        for model_name in models:
            repeated[dataset_name][model_name] = {}
            existing = (
                test_existing.get(model_name, {})
                if dataset_name == "neu"
                else gc10_test_existing
            )
            for condition in ("test_jit10", "test_jit20", "test_drop20", "test_drop40"):
                values = []
                if condition in existing:
                    values.append(existing[condition]["mAP50"] * 100)
                for seed_id in ("2027", "2028", "2029", "2030"):
                    row = (
                        test_realizations.get(dataset_name, {})
                        .get(model_name, {})
                        .get(seed_id, {})
                        .get(condition)
                    )
                    if row:
                        values.append(row["mAP50"] * 100)
                repeated[dataset_name][model_name][condition] = mean_sd(values)

    scale_row = [
        float(value)
        for value in (ROOT / "results" / "scale_fractions.csv")
        .read_text(encoding="utf-8")
        .strip()
        .split(",")
    ]
    scale_correction = {
        "fraction_over_25_percent_side": scale_row[3],
        "fraction_over_50_percent_side": scale_row[4],
    }

    realized_seeds = {
        seed_id
        for models in test_realizations.values()
        for seed_rows in models.values()
        for seed_id in seed_rows
    }
    report = {
        "status": {
            "seed_replication_done": len(seed),
            "seed_replication_required": 10,
            "review_matrix_done": len(review),
            "review_matrix_required": 29,
            "test_realization_seeds_done": len(realized_seeds),
            "rtdetr_done": len(rtdetr),
            "rtdetr_required": 3,
            "efficiency_done": efficiency is not None,
            "qualitative_done": qualitative is not None,
        },
        "standard_split": standard_split,
        "architecture_seed_replication": architecture,
        "architecture_seed_comparison": architecture_comparison,
        "standard_split_results": standard_groups,
        "p2_p5_ablation": p2p5,
        "fwn_sensitivity": fwn_sensitivity,
        "interaction_seed_replication": interaction_groups,
        "interaction_seed_comparison": interaction_comparison,
        "ranking_stability": rank_stability,
        "evaluation_label_realizations": repeated,
        "rtdetr": {
            name: {
                "variant": row["variant"],
                "mAP50": row["mAP50"] * 100,
                "mAP5095": row["mAP5095"] * 100,
                "params": row["params"],
            }
            for name, row in rtdetr.items()
        },
        "scale_correction": scale_correction,
        "efficiency": efficiency,
        "qualitative": qualitative,
    }
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Reviewer-response experiment report", "", "## Status", ""]
    lines.extend(f"- {key}: {value}" for key, value in report["status"].items())
    lines.extend(["", "## Architecture seed replication", ""])
    lines.append("| Model | n | mAP@0.5 mean | SD | Seed-matched Δ50 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, row in architecture.items():
        metric = row["mAP50"]
        delta = architecture_comparison.get(name, {}).get("delta50")
        if metric:
            delta_text = f"{delta['mean']:+.2f} ± {delta['sd']:.2f}" if delta else "—"
            lines.append(
                f"| {name} | {metric['n']} | {metric['mean']:.2f} | "
                f"{metric['sd']:.2f} | {delta_text} |"
            )
        else:
            lines.append(f"| {name} | 0 | pending | pending |")
    lines.extend(["", "## Standard split 70:20:10", ""])
    lines.append("| Model | n | mAP@0.5 | Δ50 | mAP@0.5:0.95 | Δ95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, row in standard_groups.items():
        m50 = row.get("mAP50")
        m95 = row.get("mAP5095")
        if m50 and m95:
            d50 = row.get("delta50")
            d95 = row.get("delta5095")
            d50_text = f"{d50['mean']:+.2f} ± {d50['sd']:.2f}" if d50 else "—"
            d95_text = f"{d95['mean']:+.2f} ± {d95['sd']:.2f}" if d95 else "—"
            lines.append(
                f"| {name} | {m50['n']} | {m50['mean']:.2f} ± {m50['sd']:.2f} | "
                f"{d50_text} | {m95['mean']:.2f} ± {m95['sd']:.2f} | {d95_text} |"
            )
        else:
            lines.append(f"| {name} | 0 | pending | pending | pending | pending |")
    lines.extend(["", "## P2/P5 ablation", ""])
    lines.append("| P2 | P5 | Detection levels | mAP@0.5 | mAP@0.5:0.95 | Parameters | Evidence |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    level_names = {
        (False, True): "P3/P4/P5",
        (True, True): "P2/P3/P4/P5",
        (False, False): "P3/P4 (no P2, no P5)",
        (True, False): "P2/P3/P4 (no P5)",
    }
    for row in p2p5.values():
        levels = level_names[(bool(row["p2"]), bool(row["p5"]))]
        lines.append(
            f"| {row['p2']} | {row['p5']} | {levels} | {row['mAP50']:.2f} | "
            f"{row['mAP5095']:.2f} | {row['params']} | Descriptive, seed 2026 |"
        )
    lines.extend(["", "## FWN sensitivity", ""])
    lines.append("| Variant | k | s0 | mAP@0.5 | mAP@0.5:0.95 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, row in fwn_sensitivity.items():
        config = row["config"]
        lines.append(
            f"| {name} | {config.get('k')} | {config.get('s0')} | "
            f"{row['mAP50']:.2f} | {row['mAP5095']:.2f} |"
        )
    lines.extend(["", "## Ranking stability", ""])
    for condition, row in rank_stability.items():
        lines.append(
            f"- {condition}: Kendall tau vs clean = "
            f"{row['kendall_tau_vs_clean']:.3f}; ranking = "
            f"{' > '.join(row['condition_ranking'])}"
        )
        for reversal in row["reversals"]:
            lines.append(
                f"  - Reversal: {reversal['pair']}; clean margin "
                f"{reversal['clean_margin']:.2f}, condition margin "
                f"{reversal['condition_margin']:.2f}"
            )
    lines.extend(["", "## Evaluation-label realization variance", ""])
    for dataset_name, models in repeated.items():
        for model_name, conditions in models.items():
            for condition, row in conditions.items():
                if row:
                    lines.append(
                        f"- {dataset_name}/{model_name}/{condition}: "
                        f"{row['mean']:.2f} +/- {row['sd']:.2f} (n={row['n']})"
                    )
    lines.extend(["", "## RT-DETR second-paradigm check", ""])
    if rtdetr:
        lines.append("| Variant | mAP@0.5 | mAP@0.5:0.95 | Parameters |")
        lines.append("|---|---:|---:|---:|")
        for row in rtdetr.values():
            lines.append(
                f"| {row['variant']} | {row['mAP50'] * 100:.2f} | "
                f"{row['mAP5095'] * 100:.2f} | {row['params']} |"
            )
    else:
        lines.append("- pending")
    lines.extend(["", "## Scale correction", ""])
    lines.append(
        f"- Fraction of NEU-DET boxes exceeding 25% of image side: "
        f"{scale_correction['fraction_over_25_percent_side']:.2f}%"
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REVIEW_REPORT_OK {OUT_JSON}")


if __name__ == "__main__":
    main()
