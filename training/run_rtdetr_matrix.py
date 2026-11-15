# -*- coding: utf-8 -*-
"""Validate annotation sensitivity with a second detection paradigm."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MODEL_YAML = str(
    Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8-rtdetr.yaml"
)
PROJ = ROOT / "rtdetr_runs"
SUMMARY = BASE / "rtdetr_summary.json"
LOGFILE = BASE / "rtdetr_matrix.log"
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

EPOCHS = int(os.environ.get("RTDETR_EPOCHS", "100"))
BATCH = int(os.environ.get("RTDETR_BATCH", "8"))
IMGSZ = int(os.environ.get("RTDETR_IMGSZ", "320"))
JOBS = [
    ("rtdetr_clean", "clean"),
    ("rtdetr_jit10", "jit10"),
    ("rtdetr_drop40", "drop40"),
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
        summary = json.load(open(SUMMARY, encoding="utf-8"))

    log(f"==== RT-DETR matrix start (epochs={EPOCHS}, batch={BATCH}, imgsz={IMGSZ}) ====")
    failures = []
    for name, variant in JOBS:
        if name in summary:
            log(f"== {name} already finished, skip")
            continue
        data = ROOT / "ultralytics_data" / variant / "data.yaml"
        run_dir = PROJ / name
        last_ckpt = run_dir / "weights" / "last.pt"
        try:
            started = time.time()
            if last_ckpt.is_file():
                log(f"== {name} resuming from {last_ckpt}")
                model = RTDETR(str(last_ckpt))
                model.train(resume=str(last_ckpt))
            else:
                model = RTDETR(MODEL_YAML)
                model.train(
                    data=str(data),
                    epochs=EPOCHS,
                    imgsz=IMGSZ,
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
                    project=str(PROJ),
                    name=name,
                    exist_ok=True,
                )
            best = run_dir / "weights" / "best.pt"
            ckpt = torch.load(best, map_location="cpu", weights_only=False)
            raw_model = ckpt.get("ema") or ckpt["model"]
            params = int(sum(p.numel() for p in raw_model.parameters()))
            eval_model = RTDETR(str(best))
            metrics = eval_model.val(
                data=str(data),
                split="test",
                imgsz=IMGSZ,
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
                "imgsz": IMGSZ,
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

    log("RT-DETR MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
