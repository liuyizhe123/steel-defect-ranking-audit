# -*- coding: utf-8 -*-
"""Evaluate clean-trained detectors against clean and perturbed test labels."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CLEAN_DATA = ROOT / "ultralytics_data" / "clean" / "data.yaml"
VARIANTS = ROOT / "ultralytics_data" / "test_annotation"
SUMMARY = BASE / "test_annotation_eval_summary.json"
LOGFILE = BASE / "test_annotation_eval.log"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

MODELS = [
    ("yolov8n_clean", ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt"),
    ("yolov10n_clean", ROOT / "cross_model_runs" / "yolov10n_clean_640" / "weights" / "best.pt"),
]
CONDITIONS = [
    "clean",
    "test_jit05",
    "test_jit10",
    "test_jit20",
    "test_drop20",
    "test_drop40",
    "test_drop60",
]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    summary = {}
    if SUMMARY.is_file():
        summary = json.load(open(SUMMARY, encoding="utf-8"))

    failures = []
    for model_name, weights in MODELS:
        if not weights.is_file():
            log(f"== {model_name} FAILED: missing {weights}")
            failures.append(f"{model_name}:missing_weights")
            continue
        model_summary = summary.setdefault(model_name, {})
        model = YOLO(str(weights))
        for condition in CONDITIONS:
            if condition in model_summary:
                log(f"== {model_name}/{condition} already finished, skip")
                continue
            data = CLEAN_DATA if condition == "clean" else VARIANTS / condition / "data.yaml"
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
                model_summary[condition] = {
                    "mAP50": float(metrics.box.map50),
                    "mAP5095": float(metrics.box.map),
                    "perClassAP50": [float(x) for x in metrics.box.ap50],
                    "seconds": time.time() - started,
                }
                with open(SUMMARY, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, ensure_ascii=False, indent=2)
                log(
                    f"== {model_name}/{condition} DONE "
                    f"mAP50={model_summary[condition]['mAP50']:.4f} "
                    f"mAP5095={model_summary[condition]['mAP5095']:.4f}"
                )
            except Exception as exc:
                log(f"== {model_name}/{condition} FAILED: {exc}")
                traceback.print_exc()
                failures.append(f"{model_name}/{condition}")
        del model

    log("TEST ANNOTATION EVALUATION FINISHED")
    if failures:
        raise SystemExit("failed evaluations: " + ", ".join(failures))


if __name__ == "__main__":
    main()
