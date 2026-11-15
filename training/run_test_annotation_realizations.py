# -*- coding: utf-8 -*-
"""Evaluate repeated test-label perturbations without retraining any model."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
VARIANTS = ROOT / "ultralytics_data" / "test_annotation_realizations"
SUMMARY = BASE / "test_annotation_realizations_summary.json"
LOGFILE = BASE / "test_annotation_realizations.log"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

SEEDS = [2027, 2028, 2029, 2030]
CONDITIONS = ["test_jit10", "test_jit20", "test_drop20", "test_drop40"]
DATASETS = {
    "neu": [
        (
            "yolov8n_clean",
            ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt",
        ),
        (
            "yolov10n_clean",
            ROOT
            / "cross_model_runs"
            / "yolov10n_clean_640"
            / "weights"
            / "best.pt",
        ),
    ],
    "gc10": [
        (
            "yolov8n_clean",
            ROOT / "gc10_runs" / "gc10_v8n_ciou" / "weights" / "best.pt",
        )
    ],
}


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    if not (VARIANTS / "stats.json").is_file():
        raise SystemExit(f"missing realization variants: {VARIANTS}")
    summary = {}
    if SUMMARY.is_file():
        with open(SUMMARY, encoding="utf-8") as handle:
            summary = json.load(handle)

    failures = []
    for dataset_name, models in DATASETS.items():
        dataset_summary = summary.setdefault(dataset_name, {})
        for model_name, weights in models:
            if not weights.is_file():
                failures.append(f"{dataset_name}/{model_name}:missing_weights")
                continue
            model_summary = dataset_summary.setdefault(model_name, {})
            model = YOLO(str(weights))
            for seed in SEEDS:
                seed_summary = model_summary.setdefault(str(seed), {})
                for condition in CONDITIONS:
                    if condition in seed_summary:
                        log(f"== {dataset_name}/{model_name}/{seed}/{condition} skip")
                        continue
                    data = (
                        VARIANTS
                        / dataset_name
                        / f"seed{seed}"
                        / condition
                        / "data.yaml"
                    )
                    try:
                        started = time.time()
                        metrics = model.val(
                            data=str(data),
                            split="test",
                            imgsz=640,
                            batch=16,
                            workers=0,
                            device=0,
                            plots=False,
                            verbose=False,
                        )
                        seed_summary[condition] = {
                            "mAP50": float(metrics.box.map50),
                            "mAP5095": float(metrics.box.map),
                            "perClassAP50": [
                                float(value) for value in metrics.box.ap50
                            ],
                            "seconds": time.time() - started,
                        }
                        with open(SUMMARY, "w", encoding="utf-8") as handle:
                            json.dump(
                                summary,
                                handle,
                                ensure_ascii=False,
                                indent=2,
                            )
                        log(
                            f"== {dataset_name}/{model_name}/{seed}/{condition} DONE "
                            f"mAP50={seed_summary[condition]['mAP50']:.4f} "
                            f"mAP5095={seed_summary[condition]['mAP5095']:.4f}"
                        )
                    except Exception as exc:
                        log(
                            f"== {dataset_name}/{model_name}/{seed}/{condition} "
                            f"FAILED: {exc}"
                        )
                        traceback.print_exc()
                        failures.append(
                            f"{dataset_name}/{model_name}/{seed}/{condition}"
                        )
            del model

    log("TEST ANNOTATION REALIZATION EVALUATION FINISHED")
    if failures:
        raise SystemExit("failed evaluations: " + ", ".join(failures))


if __name__ == "__main__":
    main()
