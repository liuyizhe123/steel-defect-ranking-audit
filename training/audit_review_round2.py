# -*- coding: utf-8 -*-
"""Independent completeness and consistency audit for review round 2."""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
problems: list[str] = []


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def main() -> None:
    summary = load(BASE / "review_round2_summary.json")
    report = load(BASE / "review_round2_report.json")
    marker = load(BASE / "review_round2_complete.json")
    stats = load(ROOT / "ultralytics_data" / "drop40_realizations_stats.json")
    memory = load(BASE / "efficiency_memory_repeats.json")
    review = load(BASE / "review_report.json")

    required = {
        "omission_r2027_ciou_s26",
        "omission_r2027_fwn_s26",
        "omission_r2028_ciou_s26",
        "omission_r2028_fwn_s26",
        "capacity_p3p4_s26",
    }
    check(set(summary) == required, "round-2 summary key set mismatch")
    check(marker.get("status") == "complete", "round-2 completion marker missing")
    for name, row in summary.items():
        check(0 < float(row["mAP50"]) < 1, f"{name}: invalid mAP50")
        check(0 < float(row["mAP5095"]) < 1, f"{name}: invalid mAP5095")
        check(int(row["params"]) > 0, f"{name}: invalid params")

    draws = report["omission_realisation_robustness"]["draws"]
    check(set(draws) == {"2026", "2027", "2028"}, "omission draws mismatch")
    margins = []
    for seed, row in draws.items():
        expected = row["fwn"] - row["baseline"]
        check(abs(expected - row["fwn_minus_baseline"]) < 1e-9, f"draw {seed}: margin mismatch")
        margins.append(expected)
        if seed != "2026":
            check(
                abs(row["retention"] - stats["variants"][seed]["splits"]["train"]["retention"])
                < 1e-12,
                f"draw {seed}: retention mismatch",
            )
    check(
        abs(report["omission_realisation_robustness"]["mean_margin"] - st.mean(margins)) < 1e-9,
        "reversal mean mismatch",
    )
    check(
        report["omission_realisation_robustness"]["positive_count"]
        == sum(value > 0 for value in margins),
        "reversal positive-count mismatch",
    )

    capacity = summary["capacity_p3p4_s26"]
    baseline_params = review["p2_p5_ablation"]["p2off_p5on_s26"]["params"]
    check(
        abs(capacity["params"] - baseline_params) / baseline_params < 0.03,
        "capacity control is not within 3% of baseline parameter count",
    )
    check(
        report["capacity_control"]["params"] == capacity["params"],
        "capacity report parameter mismatch",
    )

    memory_models = {
        "YOLOv8n + CIoU",
        "ESCM + CIoU",
        "YOLOv8n + FWN",
        "ESCM+SAF + CIoU",
        "ESCM+SAF + FWN",
        "SAF only + CIoU",
    }
    check(set(memory["models"]) == memory_models, "memory-repeat model set mismatch")
    for name, row in memory["models"].items():
        check(len(row["values"]) == 3, f"{name}: memory repeat count is not 3")
        check(
            abs(row["mean"] - st.mean(row["values"])) < 1e-9,
            f"{name}: memory mean mismatch",
        )

    check(
        review["status"]["review_matrix_done"] == 29,
        "original review matrix is incomplete",
    )

    print("=== REVIEW_ROUND2_AUDIT ===")
    print("training jobs:", len(summary))
    print("omission draws:", len(draws))
    print("reversal draws:", report["omission_realisation_robustness"]["positive_count"])
    print("capacity params:", capacity["params"])
    if problems:
        print("REVIEW_ROUND2_AUDIT_FAILED", len(problems))
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    print("REVIEW_ROUND2_AUDIT_OK")


if __name__ == "__main__":
    main()
