# -*- coding: utf-8 -*-
"""Repeat the key model comparisons with two additional training seeds."""

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
PROJ = os.path.join(ROOT, "seed_replication_runs")
SUMMARY = os.path.join(BASE, "seed_replication_summary.json")
LOGFILE = os.path.join(BASE, "seed_replication_matrix.log")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("SEED_REPLICATION_EPOCHS", "300"))
BATCH = int(os.environ.get("SEED_REPLICATION_BATCH", "16"))
JOBS = [
    (
        "escm_ciou_s27",
        os.path.join(BASE, "lsc_models", "escm.yaml"),
        "ciou",
        "escm",
        2027,
    ),
    (
        "escm_ciou_s28",
        os.path.join(BASE, "lsc_models", "escm.yaml"),
        "ciou",
        "escm",
        2028,
    ),
    (
        "lsc_ciou_s27",
        os.path.join(BASE, "lsc_models", "lsc.yaml"),
        "ciou",
        True,
        2027,
    ),
    (
        "lsc_ciou_s28",
        os.path.join(BASE, "lsc_models", "lsc.yaml"),
        "ciou",
        True,
        2028,
    ),
    ("v8n_fwn_s27", "yolov8n.yaml", "fwn", False, 2027),
    ("v8n_fwn_s28", "yolov8n.yaml", "fwn", False, 2028),
    ("v8n_focaler_s27", "yolov8n.yaml", "focaler", False, 2027),
    ("v8n_focaler_s28", "yolov8n.yaml", "focaler", False, 2028),
    ("v8n_hybrid_s27", "yolov8n.yaml", "hybrid", False, 2027),
    ("v8n_hybrid_s28", "yolov8n.yaml", "hybrid", False, 2028),
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

    log(f"==== seed replication start (epochs={EPOCHS}, batch={BATCH}) ====")
    failures = []
    for name, model_cfg, loss_kind, custom_layers, seed in JOBS:
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
                seed=seed,
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
                data=DATA,
                split="test",
                imgsz=640,
                batch=BATCH,
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
                "seed": seed,
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

    log("SEED REPLICATION MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
