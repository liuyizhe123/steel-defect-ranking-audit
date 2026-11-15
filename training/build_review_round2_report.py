# -*- coding: utf-8 -*-
"""Summarise omission-draw robustness and the capacity-matched control."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8-sig"))


def label_count(root: Path, variant: str) -> int:
    label_dir = root / "ultralytics_data" / variant / "labels" / "train"
    return sum(
        len([line for line in path.read_text(encoding="ascii").splitlines() if line.strip()])
        for path in label_dir.glob("*.txt")
    )


def main() -> None:
    summary = load("review_round2_summary.json")
    yolo = load("yolo_matrix_summary.json")
    interaction = load("lsc_pt_interaction_summary.json")
    review = load("review_report.json")
    stats = json.loads(
        (ROOT / "ultralytics_data" / "drop40_realizations_stats.json").read_text(
            encoding="utf-8"
        )
    )

    clean_labels = label_count(ROOT, "clean")
    original_retention = label_count(ROOT, "drop40") / clean_labels
    draws = {
        "2026": {
            "baseline": yolo["drop40_640"]["mAP50"] * 100,
            "fwn": interaction["v8n_fwn_drop40_pt"]["mAP50"] * 100,
            "retention": original_retention,
        }
    }
    for seed in ("2027", "2028"):
        draws[seed] = {
            "baseline": summary[f"omission_r{seed}_ciou_s26"]["mAP50"] * 100,
            "fwn": summary[f"omission_r{seed}_fwn_s26"]["mAP50"] * 100,
            "retention": stats["variants"][seed]["splits"]["train"]["retention"],
        }
    for row in draws.values():
        row["fwn_minus_baseline"] = row["fwn"] - row["baseline"]

    margins = [row["fwn_minus_baseline"] for row in draws.values()]
    reversal = {
        "draws": draws,
        "n_draws": len(draws),
        "positive_count": sum(value > 0 for value in margins),
        "mean_margin": st.mean(margins),
        "sd_margin": st.stdev(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "three_seed_same_draw": review["interaction_seed_comparison"]["drop40"]["fwn"],
    }

    capacity = summary["capacity_p3p4_s26"]
    capacity_report = {
        "mAP50": capacity["mAP50"] * 100,
        "mAP5095": capacity["mAP5095"] * 100,
        "params": capacity["params"],
        "p5_absent_small": {
            "mAP50": review["p2_p5_ablation"]["p2off_p5off_s26"]["mAP50"],
            "params": review["p2_p5_ablation"]["p2off_p5off_s26"]["params"],
        },
        "p5_present_control": {
            "mAP50": review["p2_p5_ablation"]["p2off_p5on_s26"]["mAP50"],
            "params": review["p2_p5_ablation"]["p2off_p5on_s26"]["params"],
        },
    }
    memory_path = BASE / "efficiency_memory_repeats.json"
    memory_repeats = (
        json.loads(memory_path.read_text(encoding="utf-8"))
        if memory_path.is_file()
        else None
    )
    report = {
        "omission_realisation_robustness": reversal,
        "capacity_control": capacity_report,
        "memory_repeats": memory_repeats,
    }
    (BASE / "review_round2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Review round 2",
        "",
        "## Omission realisation robustness",
        "",
        "| Draw seed | Baseline | FWN | FWN - baseline | Label retention |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, row in draws.items():
        lines.append(
            f"| {seed} | {row['baseline']:.2f} | {row['fwn']:.2f} | "
            f"{row['fwn_minus_baseline']:+.2f} | {row['retention'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"- Positive reversal draws: {reversal['positive_count']}/{reversal['n_draws']}",
            f"- Margin mean +/- SD: {reversal['mean_margin']:+.2f} +/- {reversal['sd_margin']:.2f}",
            f"- Margin range: {reversal['min_margin']:+.2f} to {reversal['max_margin']:+.2f}",
            (
                "- Same-draw three-seed margin: "
                f"{reversal['three_seed_same_draw']['mean']:+.2f} +/- "
                f"{reversal['three_seed_same_draw']['sd']:.2f}"
            ),
            "",
            "## Capacity-matched P3/P4 control",
            "",
            (
                f"- Capacity control: {capacity_report['mAP50']:.2f} mAP@0.5, "
                f"{capacity_report['mAP5095']:.2f} mAP@0.5:0.95, "
                f"{capacity_report['params']:,} parameters"
            ),
            (
                f"- Small P5-absent control: "
                f"{capacity_report['p5_absent_small']['mAP50']:.2f} mAP@0.5, "
                f"{capacity_report['p5_absent_small']['params']:,} parameters"
            ),
            (
                f"- P5-present control: "
                f"{capacity_report['p5_present_control']['mAP50']:.2f} mAP@0.5, "
                f"{capacity_report['p5_present_control']['params']:,} parameters"
            ),
        ]
    )
    if memory_repeats:
        lines.extend(["", "## Fresh-process memory repeats", ""])
        for name, row in memory_repeats["models"].items():
            lines.append(
                f"- {name}: mean {row['mean']:.1f} MB, range "
                f"{row['min']:.1f}-{row['max']:.1f} MB (n={len(row['values'])})"
            )
    (BASE / "review_round2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REVIEW_ROUND2_REPORT_OK")


if __name__ == "__main__":
    main()
