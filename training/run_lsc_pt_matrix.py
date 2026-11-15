# -*- coding: utf-8 -*-
"""Run the PyTorch port of the paper's architecture/loss comparison."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch
from ultralytics import YOLO

from lsc_modules import LSCDetectionTrainer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "ultralytics_data", "clean", "data.yaml")
PROJ = os.environ.get("LSC_PROJ", os.path.join(ROOT, "lsc_pt_runs"))
SUMMARY = os.environ.get("LSC_SUMMARY", os.path.join(BASE, "lsc_pt_summary.json"))
LOGFILE = os.environ.get("LSC_LOGFILE", os.path.join(BASE, "lsc_pt_matrix.log"))
SUFFIX = os.environ.get("LSC_SUFFIX", "")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("LSC_EPOCHS", "300"))
BATCH = int(os.environ.get("LSC_BATCH", "16"))
SEL = os.environ.get("LSC_ONLY", "")

JOBS = [
    # name, model yaml, loss, custom layers
    ("escm_ciou_pt", os.path.join(BASE, "lsc_models", "escm.yaml"), "ciou", "escm"),
    ("v8n_nwd_pt", "yolov8n.yaml", "nwd", False),
    ("v8n_focaler_pt", "yolov8n.yaml", "focaler", False),
    ("v8n_wiou_pt", "yolov8n.yaml", "wiou", False),
    ("v8n_hybrid_pt", "yolov8n.yaml", "hybrid", False),
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

    selected = set(x for x in SEL.split(",") if x)
    failures = []
    log(f"==== LSC PT matrix start (epochs={EPOCHS}, batch={BATCH}) ====")
    for job_name, model_cfg, loss_kind, custom_layers in JOBS:
        name = job_name + SUFFIX
        if selected and job_name not in selected:
            continue
        if name in summary:
            log(f"== {name} already finished, skip")
            continue

        run_dir = os.path.join(PROJ, name)
        last_ckpt = os.path.join(run_dir, "weights", "last.pt")
        resume = os.path.isfile(last_ckpt)
        try:
            started = time.time()
            args = dict(
                data=DATA,
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
            if resume:
                log(f"== {name} resuming from {last_ckpt}")
                # Ultralytics treats resume=True as "search the default runs dir".
                # Pass the exact checkpoint path so project-local runs resume correctly.
                args.update(model=last_ckpt, resume=last_ckpt)
            else:
                args.update(model=model_cfg)

            trainer = LSCDetectionTrainer(
                overrides=args,
                loss_kind=loss_kind,
                custom_layers=custom_layers,
            )
            trainer.train()

            best = os.path.join(run_dir, "weights", "best.pt")
            ckpt = torch.load(best, map_location="cpu", weights_only=False)
            raw_model = ckpt.get("ema") or ckpt["model"]
            params = int(sum(p.numel() for p in raw_model.parameters()))
            eval_model = YOLO(best)
            metrics = eval_model.val(
                data=DATA,
                split="test",
                imgsz=640,
                batch=16,
                workers=0,
                device=0,
                plots=False,
                verbose=False,
            )
            summary[name] = {
                "model": model_cfg,
                "loss": loss_kind,
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
                f"params={summary[name]['params']} "
                f"({summary[name]['seconds']:.0f} s)"
            )
            del trainer, eval_model, raw_model, ckpt, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"== {name} FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    log("LSC PT MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
