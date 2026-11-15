# -*- coding: utf-8 -*-
"""Evaluate the clean GC10 model against independently perturbed test labels."""

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
CLEAN_DATA = ROOT / "ultralytics_data" / "gc10" / "data.yaml"
VARIANTS = ROOT / "ultralytics_data" / "gc10_test_annotation"
WEIGHTS = ROOT / "gc10_runs" / "gc10_v8n_ciou" / "weights" / "best.pt"
SUMMARY = BASE / "gc10_test_annotation_eval_summary.json"
LOGFILE = BASE / "gc10_test_annotation_eval.log"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))
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
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing GC10 baseline: {WEIGHTS}")
    summary = {}
    if SUMMARY.is_file():
        summary = json.load(open(SUMMARY, encoding="utf-8"))

    model = YOLO(str(WEIGHTS))
    failures = []
    for condition in CONDITIONS:
        if condition in summary:
            continue
        data = CLEAN_DATA if condition == "clean" else VARIANTS / condition / "data.yaml"
        try:
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
            summary[condition] = {
                "mAP50": float(metrics.box.map50),
                "mAP5095": float(metrics.box.map),
                "perClassAP50": [float(x) for x in metrics.box.ap50],
            }
            with open(SUMMARY, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            log(
                f"== {condition} DONE mAP50={summary[condition]['mAP50']:.4f} "
                f"mAP5095={summary[condition]['mAP5095']:.4f}"
            )
        except Exception as exc:
            log(f"== {condition} FAILED: {exc}")
            traceback.print_exc()
            failures.append(condition)

    log("GC10 TEST ANNOTATION EVALUATION FINISHED")
    if failures:
        raise SystemExit("failed conditions: " + ", ".join(failures))


if __name__ == "__main__":
    main()
