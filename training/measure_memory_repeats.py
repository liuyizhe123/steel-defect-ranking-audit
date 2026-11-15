# -*- coding: utf-8 -*-
"""Measure peak batch-8 CUDA memory in fresh processes and report repeats."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SUMMARY = BASE / "efficiency_memory_repeats.json"
LOGFILE = BASE / "efficiency_memory_repeats.log"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

MODELS = {
    "YOLOv8n + CIoU": ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt",
    "ESCM + CIoU": ROOT / "lsc_pt_runs" / "escm_ciou_pt" / "weights" / "best.pt",
    "YOLOv8n + FWN": ROOT / "lsc_pt_runs" / "v8n_fwn_pt_v2" / "weights" / "best.pt",
    "ESCM+SAF + CIoU": ROOT / "lsc_pt_runs" / "lsc_ciou_pt" / "weights" / "best.pt",
    "ESCM+SAF + FWN": ROOT / "lsc_pt_runs" / "lsc_fwn_pt_v3" / "weights" / "best.pt",
    "SAF only + CIoU": ROOT / "lsc_pt_runs" / "saf_ciou_pt" / "weights" / "best.pt",
}


def log(message: str) -> None:
    with LOGFILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def worker(name: str, output: Path) -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    image_path = next((ROOT / "ultralytics_data" / "clean" / "images" / "test").glob("*.jpg"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"cannot read {image_path}")
    model = YOLO(str(MODELS[name]))
    inputs = [image] * 8
    for _ in range(20):
        model.predict(inputs, imgsz=640, device=0, verbose=False, conf=0.25)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(100):
        model.predict(inputs, imgsz=640, device=0, verbose=False, conf=0.25)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    output.write_text(json.dumps({"name": name, "peak_mb": peak}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, Path(args.output))
        return

    report = {"repeats": 3, "batch": 8, "models": {}}
    for name in MODELS:
        values = []
        for repeat in range(3):
            with tempfile.NamedTemporaryFile(
                prefix="sci_mem_", suffix=".json", delete=False
            ) as handle:
                output = Path(handle.name)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        name,
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                values.append(float(json.loads(output.read_text(encoding="utf-8"))["peak_mb"]))
            finally:
                output.unlink(missing_ok=True)
        report["models"][name] = {
            "values": values,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        log(f"{name}: {values}")
    SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("MEMORY_REPEATS_OK")


if __name__ == "__main__":
    main()
