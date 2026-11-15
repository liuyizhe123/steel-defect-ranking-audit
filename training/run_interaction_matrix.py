# -*- coding: utf-8 -*-
"""Test whether FWN changes the detector response to annotation noise."""

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
DATA_DIR = os.path.join(ROOT, "ultralytics_data")
PROJ = os.path.join(ROOT, "lsc_pt_interaction_runs")
SUMMARY = os.path.join(BASE, "lsc_pt_interaction_summary.json")
LOGFILE = os.path.join(BASE, "lsc_pt_interaction.log")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("LSC_INTERACTION_EPOCHS", "300"))
BATCH = int(os.environ.get("LSC_INTERACTION_BATCH", "16"))

JOBS = [
    ("v8n_fwn_jit10_pt", "jit10", "yolov8n.yaml", "fwn", False),
    ("v8n_fwn_drop40_pt", "drop40", "yolov8n.yaml", "fwn", False),
    (
        "lsc_ciou_jit10_pt",
        "jit10",
        os.path.join(BASE, "lsc_models", "lsc.yaml"),
        "ciou",
        True,
    ),
    (
        "lsc_ciou_drop40_pt",
        "drop40",
        os.path.join(BASE, "lsc_models", "lsc.yaml"),
        "ciou",
        True,
    ),
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

    log(f"==== LSC PT interaction matrix start (epochs={EPOCHS}, batch={BATCH}) ====")
    failures = []
    for name, variant, model_cfg, loss_kind, custom_layers in JOBS:
        if name in summary:
            log(f"== {name} already finished, skip")
            continue

        data = os.path.join(DATA_DIR, variant, "data.yaml")
        run_dir = os.path.join(PROJ, name)
        last_ckpt = os.path.join(run_dir, "weights", "last.pt")
        resume = os.path.isfile(last_ckpt)
        try:
            started = time.time()
            args = dict(
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
            if resume:
                log(f"== {name} resuming from {last_ckpt}")
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
                "variant": variant,
                "model": model_cfg,
                "loss": loss_kind,
                "custom_layers": custom_layers,
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
            del trainer, eval_model, raw_model, ckpt, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"== {name} FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    log("LSC PT INTERACTION MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
