# -*- coding: utf-8 -*-
"""Consolidate the reviewer-requested ablations, interaction tests and GC10 check."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

ARCH_KEYS = [
    "escm_ciou_pt",
    "v8n_nwd_pt",
    "v8n_focaler_pt",
    "v8n_wiou_pt",
    "v8n_hybrid_pt",
]
INTERACTION_KEYS = [
    "v8n_fwn_jit10_pt",
    "v8n_fwn_drop40_pt",
    "lsc_ciou_jit10_pt",
    "lsc_ciou_drop40_pt",
]
GC10_KEYS = ["gc10_v8n_ciou", "gc10_v8n_fwn", "gc10_lsc_ciou", "gc10_lsc_fwn"]
CROSS_KEYS = ["yolov10n_clean_640", "yolov10n_jit10_640", "yolov10n_drop40_640"]
TEST_EVAL_MODELS = ["yolov8n_clean", "yolov10n_clean"]
TEST_EVAL_CONDITIONS = [
    "test_jit05",
    "test_jit10",
    "test_jit20",
    "test_drop20",
    "test_drop40",
    "test_drop60",
]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def metric(record: dict) -> dict:
    return {
        "mAP50": round(record["mAP50"] * 100, 2),
        "mAP5095": round(record["mAP5095"] * 100, 2),
        "params": int(record.get("params", 0)),
    }


def delta(record: dict, control: dict) -> dict:
    return {
        "delta50": round((record["mAP50"] - control["mAP50"]) * 100, 2),
        "delta5095": round((record["mAP5095"] - control["mAP5095"]) * 100, 2),
    }


def main() -> None:
    lsc = load(BASE / "lsc_pt_summary.json")
    yolo = load(BASE / "yolo_matrix_summary.json")
    interaction = load(BASE / "lsc_pt_interaction_summary.json")
    gc10 = load(BASE / "gc10_summary.json")
    cross_model = load(BASE / "cross_model_summary.json")
    test_eval = load(BASE / "test_annotation_eval_summary.json")
    gc10_test_eval = load(BASE / "gc10_test_annotation_eval_summary.json")
    bootstrap = load(BASE / "bootstrap_ci.json")

    missing = [key for key in ARCH_KEYS if key not in lsc]
    missing += [key for key in INTERACTION_KEYS if key not in interaction]
    missing += [key for key in GC10_KEYS if key not in gc10]
    missing += [key for key in CROSS_KEYS if key not in cross_model]
    for model_name in TEST_EVAL_MODELS:
        if model_name not in test_eval or "clean" not in test_eval[model_name]:
            missing.append(f"{model_name}/clean")
        for condition in TEST_EVAL_CONDITIONS:
            if condition not in test_eval.get(model_name, {}):
                missing.append(f"{model_name}/{condition}")
    if missing:
        raise SystemExit("missing completed experiments: " + ", ".join(missing))

    report = {
        "supplemental_architecture_loss": {},
        "annotation_interaction": {},
        "architecture_interaction": {},
        "gc10_external_check": {},
        "cross_model_annotation": {},
        "test_annotation_noise": {},
        "gc10_test_annotation_noise": {},
        "bootstrap_intervals": bootstrap,
    }

    fwn = lsc["v8n_fwn_pt_v2"]
    for key in ARCH_KEYS:
        report["supplemental_architecture_loss"][key] = {
            **metric(lsc[key]),
            "delta_vs_clean_ciou": delta(lsc[key], yolo["clean_640"]),
        }
        if key.startswith("v8n_"):
            report["supplemental_architecture_loss"][key]["delta_vs_v8n_fwn"] = delta(
                lsc[key], fwn
            )

    fwn_jit10 = interaction["v8n_fwn_jit10_pt"]
    fwn_drop40 = interaction["v8n_fwn_drop40_pt"]
    report["annotation_interaction"] = {
        "jit10": {
            "ciou": metric(yolo["jit10_640"]),
            "fwn": metric(fwn_jit10),
            "fwn_minus_ciou": delta(fwn_jit10, yolo["jit10_640"]),
        },
        "drop40": {
            "ciou": metric(yolo["drop40_640"]),
            "fwn": metric(fwn_drop40),
            "fwn_minus_ciou": delta(fwn_drop40, yolo["drop40_640"]),
        },
    }
    report["architecture_interaction"] = {
        "jit10": {
            "yolov8n_ciou": metric(yolo["jit10_640"]),
            "escm_saf_ciou": metric(interaction["lsc_ciou_jit10_pt"]),
            "escm_saf_minus_yolov8n": delta(
                interaction["lsc_ciou_jit10_pt"], yolo["jit10_640"]
            ),
        },
        "drop40": {
            "yolov8n_ciou": metric(yolo["drop40_640"]),
            "escm_saf_ciou": metric(interaction["lsc_ciou_drop40_pt"]),
            "escm_saf_minus_yolov8n": delta(
                interaction["lsc_ciou_drop40_pt"], yolo["drop40_640"]
            ),
        },
    }

    baseline = gc10["gc10_v8n_ciou"]
    for key in GC10_KEYS:
        report["gc10_external_check"][key] = {
            **metric(gc10[key]),
            "delta_vs_gc10_ciou": delta(gc10[key], baseline),
        }

    cross_controls = {
        "clean_640": "clean_640",
        "jit10_640": "jit10_640",
        "drop40_640": "drop40_640",
    }
    for cross_key, yolo_key in cross_controls.items():
        report["cross_model_annotation"][cross_key] = {
            "yolov8n": metric(yolo[yolo_key]),
            "yolov10n": metric(cross_model[f"yolov10n_{cross_key}"]),
            "yolov10n_minus_yolov8n": delta(
                cross_model[f"yolov10n_{cross_key}"], yolo[yolo_key]
            ),
        }

    for model_name in TEST_EVAL_MODELS:
        clean = test_eval[model_name]["clean"]
        report["test_annotation_noise"][model_name] = {}
        for condition in TEST_EVAL_CONDITIONS:
            report["test_annotation_noise"][model_name][condition] = {
                **metric(test_eval[model_name][condition]),
                "delta_vs_clean_test_labels": delta(
                    test_eval[model_name][condition], clean
                ),
            }

    gc_clean = gc10_test_eval["clean"]
    for condition in TEST_EVAL_CONDITIONS:
        report["gc10_test_annotation_noise"][condition] = {
            **metric({**gc10_test_eval[condition], "params": 1}),
            "delta_vs_clean_test_labels": delta(gc10_test_eval[condition], gc_clean),
        }

    json_path = BASE / "revision_report.json"
    md_path = BASE / "revision_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Major-revision experiment report",
        "",
        "## Supplemental architecture and loss rows",
        "",
        "| Run | mAP@0.5 | mAP@0.5:0.95 | Params | Delta vs clean CIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in report["supplemental_architecture_loss"].items():
        d = row["delta_vs_clean_ciou"]
        lines.append(
            f"| {key} | {row['mAP50']:.2f} | {row['mAP5095']:.2f} | "
            f"{row['params']} | {d['delta50']:+.2f} / {d['delta5095']:+.2f} |"
        )
    lines += [
        "",
        "## FWN by annotation condition",
        "",
        "| Condition | CIoU mAP@0.5 | FWN mAP@0.5 | FWN minus CIoU |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["annotation_interaction"].items():
        lines.append(
            f"| {key} | {row['ciou']['mAP50']:.2f} | {row['fwn']['mAP50']:.2f} | "
            f"{row['fwn_minus_ciou']['delta50']:+.2f} |"
        )
    lines += [
        "",
        "## Architecture by annotation interaction",
        "",
        "| Condition | YOLOv8n+CIoU | ESCM+SAF+CIoU | ESCM+SAF minus YOLOv8n |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["architecture_interaction"].items():
        lines.append(
            f"| {key} | {row['yolov8n_ciou']['mAP50']:.2f} | "
            f"{row['escm_saf_ciou']['mAP50']:.2f} | "
            f"{row['escm_saf_minus_yolov8n']['delta50']:+.2f} |"
        )
    lines += [
        "",
        "## GC10-DET full annotated set",
        "",
        "| Run | mAP@0.5 | mAP@0.5:0.95 | Params | Delta vs CIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in report["gc10_external_check"].items():
        d = row["delta_vs_gc10_ciou"]
        lines.append(
            f"| {key} | {row['mAP50']:.2f} | {row['mAP5095']:.2f} | "
            f"{row['params']} | {d['delta50']:+.2f} / {d['delta5095']:+.2f} |"
        )
    lines += [
        "",
        "## Cross-model annotation interaction (NEU-DET)",
        "",
        "| Condition | YOLOv8n mAP@0.5 | YOLOv10n mAP@0.5 | YOLOv10n minus YOLOv8n |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["cross_model_annotation"].items():
        lines.append(
            f"| {key} | {row['yolov8n']['mAP50']:.2f} | "
            f"{row['yolov10n']['mAP50']:.2f} | "
            f"{row['yolov10n_minus_yolov8n']['delta50']:+.2f} |"
        )
    lines += [
        "",
        "## Test-annotation noise (no retraining)",
        "",
        "| Model | Test condition | mAP@0.5 | Delta vs clean test labels |",
        "|---|---|---:|---:|",
    ]
    for model_name, conditions in report["test_annotation_noise"].items():
        for condition, row in conditions.items():
            lines.append(
                f"| {model_name} | {condition} | {row['mAP50']:.2f} | "
                f"{row['delta_vs_clean_test_labels']['delta50']:+.2f} |"
            )
    lines += [
        "",
        "## GC10 test-annotation noise (no retraining)",
        "",
        "| Test condition | mAP@0.5 | Delta vs clean test labels |",
        "|---|---:|---:|",
    ]
    for condition, row in report["gc10_test_annotation_noise"].items():
        lines.append(
            f"| {condition} | {row['mAP50']:.2f} | "
            f"{row['delta_vs_clean_test_labels']['delta50']:+.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REVISION_REPORT_OK")


if __name__ == "__main__":
    main()
