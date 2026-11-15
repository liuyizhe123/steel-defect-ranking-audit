# -*- coding: utf-8 -*-
"""Measure batched end-to-end inference latency and peak GPU memory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SUMMARY = BASE / "efficiency_summary.json"
LOGFILE = BASE / "efficiency.log"
FIGURE = ROOT / "figs" / "fig9_efficiency.png"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

MODELS = [
    ("YOLOv8n + CIoU", ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt"),
    ("ESCM + CIoU", ROOT / "lsc_pt_runs" / "escm_ciou_pt" / "weights" / "best.pt"),
    (
        "YOLOv8n + FWN",
        ROOT / "lsc_pt_runs" / "v8n_fwn_pt_v2" / "weights" / "best.pt",
    ),
    (
        "ESCM+SAF + CIoU",
        ROOT / "lsc_pt_runs" / "lsc_ciou_pt" / "weights" / "best.pt",
    ),
    (
        "ESCM+SAF + FWN",
        ROOT / "lsc_pt_runs" / "lsc_fwn_pt_v3" / "weights" / "best.pt",
    ),
    ("SAF only + CIoU", ROOT / "lsc_pt_runs" / "saf_ciou_pt" / "weights" / "best.pt"),
]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def measure(model: YOLO, image, batch: int, warmup: int = 20, runs: int = 100):
    inputs = [image] * batch
    for _ in range(warmup):
        model.predict(inputs, imgsz=640, device=0, verbose=False, conf=0.25)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(runs):
        model.predict(inputs, imgsz=640, device=0, verbose=False, conf=0.25)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return {
        "batch": batch,
        "runs": runs,
        "latency_ms_per_batch": elapsed * 1000 / runs,
        "latency_ms_per_image": elapsed * 1000 / (runs * batch),
        "throughput_images_per_s": runs * batch / elapsed,
        "peak_cuda_allocated_mb": peak_mb,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for efficiency measurement")
    image_path = next((ROOT / "ultralytics_data" / "clean" / "images" / "test").glob("*.jpg"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"cannot read image: {image_path}")

    summary = {
        "device": torch.cuda.get_device_name(0),
        "imgsz": 640,
        "confidence_threshold": 0.25,
        "measurement": "preprocess + inference + NMS/postprocess; excludes disk and camera capture",
        "models": {},
    }
    for name, weights in MODELS:
        if not weights.is_file():
            raise SystemExit(f"missing weights for {name}: {weights}")
        log(f"== measuring {name}")
        model = YOLO(str(weights))
        summary["models"][name] = {
            "weights": str(weights),
            "batch1": measure(model, image, batch=1),
            "batch8": measure(model, image, batch=8),
        }
        del model
        torch.cuda.empty_cache()

    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(BASE / "make_efficiency_figure.py")],
        check=True,
    )
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("EFFICIENCY_MEASUREMENT_OK")


if __name__ == "__main__":
    main()
