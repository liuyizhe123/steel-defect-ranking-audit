# -*- coding: utf-8 -*-
"""Check whether the annotation sensitivity is specific to YOLOv8n."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import ultralytics
import torch
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_DIR = os.path.join(ROOT, "ultralytics_data")
MODEL_YAML = str(
    Path(ultralytics.__file__).parent / "cfg" / "models" / "v10" / "yolov10n.yaml"
)
PROJ = os.path.join(ROOT, "cross_model_runs")
SUMMARY = os.path.join(BASE, "cross_model_summary.json")
LOGFILE = os.path.join(BASE, "cross_model_matrix.log")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("CROSS_MODEL_EPOCHS", "300"))
BATCH = int(os.environ.get("CROSS_MODEL_BATCH", "16"))
JOBS = [
    ("yolov10n_clean_640", "clean"),
    ("yolov10n_jit10_640", "jit10"),
    ("yolov10n_drop40_640", "drop40"),
]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    os.makedirs(PROJ, exist_ok=True)
    summary = {}
    if os.path.isfile(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as fh:
            summary = json.load(fh)

    log(f"==== cross-model matrix start (epochs={EPOCHS}, batch={BATCH}) ====")
    failures = []
    for name, variant in JOBS:
        if name in summary:
            log(f"== {name} already finished, skip")
            continue

        data = os.path.join(DATA_DIR, variant, "data.yaml")
        run_dir = os.path.join(PROJ, name)
        last_ckpt = os.path.join(run_dir, "weights", "last.pt")
        try:
            started = time.time()
            if os.path.isfile(last_ckpt):
                log(f"== {name} resuming from {last_ckpt}")
                model = YOLO(last_ckpt)
                model.train(resume=last_ckpt)
            else:
                model = YOLO(MODEL_YAML)
                model.train(
                    data=data,
                    epochs=EPOCHS,
                    imgsz=640,
                    batch=BATCH,
                    workers=0,
                    device=0,
                    seed=2026,
                    deterministic=True,
                    cos_lr=True,
                    amp=False,
                    plots=False,
                    val=True,
                    verbose=False,
                    project=PROJ,
                    name=name,
                    exist_ok=True,
                )

            best = os.path.join(run_dir, "weights", "best.pt")
            ckpt = torch.load(best, map_location="cpu", weights_only=False)
            raw_model = ckpt.get("ema") or ckpt["model"]
            params = int(sum(p.numel() for p in raw_model.parameters()))
            eval_model = YOLO(best)
            metrics = eval_model.val(
                data=data,
                split="test",
                imgsz=640,
                batch=BATCH,
                workers=0,
                device=0,
                plots=False,
                verbose=False,
            )
            summary[name] = {
                "model": MODEL_YAML,
                "variant": variant,
                "mAP50": float(metrics.box.map50),
                "mAP5095": float(metrics.box.map),
                "perClassAP50": [float(x) for x in metrics.box.ap50],
                "params": params,
                "seconds": time.time() - started,
                "epochs": EPOCHS,
                "seed": 2026,
            }
            with open(SUMMARY, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            log(
                f"== {name} DONE mAP50={summary[name]['mAP50']:.4f} "
                f"mAP5095={summary[name]['mAP5095']:.4f} "
                f"params={params} ({summary[name]['seconds']:.0f} s)"
            )
            del eval_model, raw_model, ckpt, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"== {name} FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    log("CROSS MODEL MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
