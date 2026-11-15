# -*- coding: utf-8 -*-
"""Independent completeness and consistency checks for revision experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
problems: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def check_metric(record: dict, label: str) -> None:
    for key in ("mAP50", "mAP5095"):
        value = float(record[key])
        check(0.0 <= value <= 1.0, f"{label}: {key} out of range")
        check(value == value, f"{label}: {key} is NaN")
    check(int(record["params"]) > 0, f"{label}: invalid parameter count")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    paths = {
        "yolo": BASE / "yolo_matrix_summary.json",
        "lsc": BASE / "lsc_pt_summary.json",
        "interaction": BASE / "lsc_pt_interaction_summary.json",
        "gc10": BASE / "gc10_summary.json",
        "cross": BASE / "cross_model_summary.json",
        "test_eval": BASE / "test_annotation_eval_summary.json",
        "gc10_test_eval": BASE / "gc10_test_annotation_eval_summary.json",
        "bootstrap": BASE / "bootstrap_ci.json",
        "revision": BASE / "revision_report.json",
        "review_matrix": BASE / "review_matrix_summary.json",
        "review_report": BASE / "review_report.json",
        "realizations": BASE / "test_annotation_realizations_summary.json",
        "efficiency": BASE / "efficiency_summary.json",
        "qualitative": BASE / "qualitative_summary.json",
        "review_complete": BASE / "review_followup_complete.json",
        "gc10_stats": ROOT / "ultralytics_data" / "gc10" / "gc10_stats.json",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        print("REVISION_EXPERIMENTS_FAILED")
        for path in missing_files:
            print(" - missing file:", path)
        sys.exit(1)

    yolo = load(paths["yolo"])
    lsc = load(paths["lsc"])
    interaction = load(paths["interaction"])
    gc10 = load(paths["gc10"])
    cross = load(paths["cross"])
    test_eval = load(paths["test_eval"])
    gc10_test_eval = load(paths["gc10_test_eval"])
    bootstrap = load(paths["bootstrap"])
    revision = load(paths["revision"])
    review = load(paths["review_matrix"])
    review_report = load(paths["review_report"])
    realizations = load(paths["realizations"])
    efficiency = load(paths["efficiency"])
    qualitative = load(paths["qualitative"])
    review_complete = load(paths["review_complete"])
    gc10_stats = load(paths["gc10_stats"])

    lsc_required = [
        "v8n_fwn_pt_v2",
        "lsc_ciou_pt",
        "lsc_fwn_pt_v3",
        "saf_ciou_pt",
        "escm_ciou_pt",
        "v8n_nwd_pt",
        "v8n_focaler_pt",
        "v8n_wiou_pt",
        "v8n_hybrid_pt",
    ]
    interaction_required = [
        "v8n_fwn_jit10_pt",
        "v8n_fwn_drop40_pt",
        "lsc_ciou_jit10_pt",
        "lsc_ciou_drop40_pt",
    ]
    gc10_required = ["gc10_v8n_ciou", "gc10_v8n_fwn", "gc10_lsc_ciou", "gc10_lsc_fwn"]
    cross_required = ["yolov10n_clean_640", "yolov10n_jit10_640", "yolov10n_drop40_640"]
    test_conditions = [
        "clean",
        "test_jit05",
        "test_jit10",
        "test_jit20",
        "test_drop20",
        "test_drop40",
        "test_drop60",
    ]

    for name in lsc_required:
        check(name in lsc, f"missing LSC result: {name}")
        if name in lsc:
            check_metric(lsc[name], f"lsc/{name}")
    for name in interaction_required:
        check(name in interaction, f"missing interaction result: {name}")
        if name in interaction:
            check_metric(interaction[name], f"interaction/{name}")
    for name in gc10_required:
        check(name in gc10, f"missing GC10 result: {name}")
        if name in gc10:
            check_metric(gc10[name], f"gc10/{name}")
    for name in cross_required:
        check(name in cross, f"missing cross-model result: {name}")
        if name in cross:
            check_metric(cross[name], f"cross/{name}")
    for model_name in ("yolov8n_clean", "yolov10n_clean"):
        for condition in test_conditions:
            record = test_eval.get(model_name, {}).get(condition)
            check(record is not None, f"missing test-eval result: {model_name}/{condition}")
            if record is not None:
                check_metric({**record, "params": 1}, f"test_eval/{model_name}/{condition}")
    for condition in test_conditions:
        record = gc10_test_eval.get(condition)
        check(record is not None, f"missing GC10 test-eval result: {condition}")
        if record is not None:
            check_metric({**record, "params": 1}, f"gc10_test_eval/{condition}")

    standard_required = [
        f"standard721_{model_name}_s{seed_id}"
        for model_name in ("v8n_ciou", "v8n_fwn", "lsc_ciou")
        for seed_id in ("26", "27", "28")
    ]
    p2p5_required = [
        "p2off_p5on_s26",
        "p2on_p5on_s26",
        "p2off_p5off_s26",
        "p2on_p5off_s26",
    ]
    fwn_required = ["fwn_k05_s26", "fwn_k20_s26", "fwn_s003_s26", "fwn_s012_s26"]
    interaction_review_required = [
        f"interaction_{model_name}_{condition}_s{seed_id}"
        for model_name in ("v8n_ciou", "v8n_fwn", "lsc_ciou")
        for condition in ("jit10", "drop40")
        for seed_id in ("27", "28")
    ]
    review_required = (
        standard_required + p2p5_required + fwn_required + interaction_review_required
    )
    check(len(review_required) == 29, "review matrix requirement is not 29 jobs")
    for name in review_required:
        check(name in review, f"missing review-matrix result: {name}")
        if name in review:
            check_metric(review[name], f"review/{name}")
    check(set(review) == set(review_required), "review matrix contains unexpected or missing keys")

    status = review_report.get("status", {})
    check(status.get("review_matrix_done") == 29, "review report did not record 29 matrix jobs")
    check(status.get("test_realization_seeds_done") == 4, "review report did not record 4 extra seeds")
    check(status.get("rtdetr_done") == 3, "review report did not record 3 RT-DETR jobs")
    check(status.get("efficiency_done") is True, "review report lacks efficiency results")
    check(status.get("qualitative_done") is True, "review report lacks qualitative results")
    check(review_complete.get("status") == "complete", "review follow-up marker is incomplete")

    for dataset_name, model_names in {
        "neu": ("yolov8n_clean", "yolov10n_clean"),
        "gc10": ("yolov8n_clean",),
    }.items():
        for model_name in model_names:
            for seed_id in ("2027", "2028", "2029", "2030"):
                seed_row = (
                    realizations.get(dataset_name, {})
                    .get(model_name, {})
                    .get(seed_id, {})
                )
                for condition in ("test_jit10", "test_jit20", "test_drop20", "test_drop40"):
                    record = seed_row.get(condition)
                    check(
                        record is not None,
                        f"missing repeated realization: {dataset_name}/{model_name}/{seed_id}/{condition}",
                    )
                    if record is not None:
                        check_metric(
                            {**record, "params": 1},
                            f"realization/{dataset_name}/{model_name}/{seed_id}/{condition}",
                        )

    efficiency_models = {
        "YOLOv8n + CIoU",
        "ESCM + CIoU",
        "YOLOv8n + FWN",
        "ESCM+SAF + CIoU",
        "ESCM+SAF + FWN",
        "SAF only + CIoU",
    }
    check(
        set(efficiency.get("models", {})) == efficiency_models,
        "efficiency summary model set mismatch",
    )
    for name in efficiency_models:
        row = efficiency.get("models", {}).get(name, {})
        for batch in ("batch1", "batch8"):
            record = row.get(batch, {})
            check(
                float(record.get("throughput_images_per_s", 0)) > 0,
                f"efficiency/{name}/{batch}: invalid throughput",
            )
            check(
                float(record.get("peak_cuda_allocated_mb", 0)) > 0,
                f"efficiency/{name}/{batch}: invalid CUDA memory",
            )
    check(
        (ROOT / "figs" / "fig9_efficiency.png").is_file(),
        "efficiency Pareto figure missing",
    )

    qualitative_composite = qualitative.get("composite")
    check(len(qualitative.get("cases", [])) == 3, "qualitative summary must contain 3 cases")
    check(bool(qualitative_composite), "qualitative composite figure is not recorded")
    if qualitative_composite:
        check(
            (ROOT / "figs" / qualitative_composite).is_file(),
            "qualitative composite figure missing",
        )

    check(int(bootstrap.get("replicates", 0)) == 1000, "bootstrap replicate count is not 1000")
    for section in (
        "neu_models",
        "neu_test_annotation_noise",
        "gc10_models",
        "gc10_test_annotation_noise",
    ):
        check(section in bootstrap, f"bootstrap section missing: {section}")

    check(int(gc10_stats["used_images"]) == 2280, "GC10 does not contain 2280 images")
    check(int(gc10_stats["source_pairs"]) == 2280, "GC10 does not contain 2280 source pairs")
    check(len(gc10_stats["classes"]) == 10, "GC10 class count is not 10")
    check(
        sum(int(x) for x in gc10_stats["split_images"].values()) == 2280,
        "GC10 split image count does not sum to 2280",
    )

    for prefix in ("images", "labels"):
        count = sum(
            1
            for split in ("train", "val", "test")
            for _ in (ROOT / "ultralytics_data" / "gc10" / prefix / split).glob("*")
            if _.is_file()
        )
        check(count == 2280, f"GC10 {prefix} file count is {count}, expected 2280")

    baseline = gc10["gc10_v8n_ciou"]
    for name in gc10_required:
        expected = round((gc10[name]["mAP50"] - baseline["mAP50"]) * 100, 2)
        actual = revision["gc10_external_check"][name]["delta_vs_gc10_ciou"]["delta50"]
        check(close(expected, actual), f"GC10 delta mismatch for {name}")

    for condition, yolo_key in (
        ("clean_640", "clean_640"),
        ("jit10_640", "jit10_640"),
        ("drop40_640", "drop40_640"),
    ):
        row = revision["cross_model_annotation"][condition]
        check(
            close(
                row["yolov10n_minus_yolov8n"]["delta50"],
                round((cross[f"yolov10n_{condition}"]["mAP50"] - yolo[yolo_key]["mAP50"]) * 100, 2),
            ),
            f"cross-model delta mismatch for {condition}",
        )

    for model_name in ("yolov8n_clean", "yolov10n_clean"):
        clean = test_eval[model_name]["clean"]
        for condition in test_conditions[1:]:
            row = revision["test_annotation_noise"][model_name][condition]
            expected = round(
                (test_eval[model_name][condition]["mAP50"] - clean["mAP50"]) * 100, 2
            )
            check(
                close(row["delta_vs_clean_test_labels"]["delta50"], expected),
                f"test-eval delta mismatch for {model_name}/{condition}",
            )

    print("=== REVISION_EXPERIMENTS_AUDIT ===")
    print(f"LSC results: {len(lsc_required)}")
    print(f"Interaction results: {len(interaction_required)}")
    print(f"GC10 results: {len(gc10_required)}")
    print(f"Cross-model results: {len(cross_required)}")
    print(f"Test-evaluation results: {2 * len(test_conditions)}")
    print(f"Review-matrix results: {len(review_required)}")
    print(f"Repeated realization seeds per dataset/model: 4")
    if problems:
        print("REVISION_EXPERIMENTS_FAILED", len(problems))
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    print("REVISION_EXPERIMENTS_OK")


if __name__ == "__main__":
    main()
