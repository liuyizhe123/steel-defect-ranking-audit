# -*- coding: utf-8 -*-
"""Run omission-realisation robustness and P3/P4 capacity controls."""

from __future__ import annotations

import json
import os
import time
import traceback

import torch
from ultralytics import YOLO

from lsc_modules import LSCDetectionTrainer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CLEAN_DATA = os.path.join(ROOT, "ultralytics_data", "clean", "data.yaml")
PROJ = os.path.join(ROOT, "review_round2_runs")
SUMMARY = os.path.join(BASE, "review_round2_summary.json")
LOGFILE = os.path.join(BASE, "review_round2.log")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("ROUND2_EPOCHS", "300"))
BATCH = int(os.environ.get("ROUND2_BATCH", "16"))

JOBS = []
for realization_seed in (2027, 2028):
    data = os.path.join(
        ROOT,
        "ultralytics_data",
        f"drop40_r{realization_seed}",
        "data.yaml",
    )
    for model_name, loss_kind in (("ciou", "ciou"), ("fwn", "fwn")):
        JOBS.append(
            {
                "name": f"omission_r{realization_seed}_{model_name}_s26",
                "group": "omission_realization",
                "data": data,
                "model": "yolov8n.yaml",
                "loss": loss_kind,
                "custom_layers": False,
                "seed": 2026,
                "realization_seed": realization_seed,
            }
        )

JOBS.append(
    {
        "name": "capacity_p3p4_s26",
        "group": "capacity_control",
        "data": CLEAN_DATA,
        "model": os.path.join(
            BASE,
            "lsc_models",
            "p2off_p5off_capacity.yaml",
        ),
        "loss": "ciou",
        "custom_layers": False,
        "seed": 2026,
        "realization_seed": None,
    }
)


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def configure_fwn() -> dict[str, float]:
    keys = {
        "FWN_K": "10.0",
        "FWN_S0": "0.06",
        "FWN_U": "0.25",
        "FWN_V": "1.0",
        "FWN_WIOU_ALPHA": "1.9",
        "FWN_WIOU_DELTA": "3.0",
        "FWN_NWD_C": "12.8",
    }
    for key, value in keys.items():
        os.environ[key] = value
    return {key.removeprefix("FWN_").lower(): float(value) for key, value in keys.items()}


def main() -> None:
    os.makedirs(PROJ, exist_ok=True)
    summary = {}
    if os.path.isfile(SUMMARY):
        summary = json.loads(open(SUMMARY, encoding="utf-8").read())

    failures = []
    log(f"==== review round 2 start (epochs={EPOCHS}, batch={BATCH}) ====")
    for job in JOBS:
        name = job["name"]
        if name in summary:
            log(f"== {name} already finished, skip")
            continue
        fwn_config = configure_fwn()
        run_dir = os.path.join(PROJ, name)
        last_ckpt = os.path.join(run_dir, "weights", "last.pt")
        resume = os.path.isfile(last_ckpt)
        try:
            started = time.time()
            args = {
                "data": job["data"],
                "epochs": EPOCHS,
                "imgsz": 640,
                "batch": BATCH,
                "workers": 0,
                "device": 0,
                "seed": job["seed"],
                "deterministic": True,
                "cos_lr": True,
                "amp": False,
                "plots": False,
                "val": True,
                "verbose": False,
                "project": PROJ,
                "name": name,
                "exist_ok": True,
            }
            if resume:
                log(f"== {name} resuming from {last_ckpt}")
                args.update(model=last_ckpt, resume=last_ckpt)
            else:
                args.update(model=job["model"])

            trainer = LSCDetectionTrainer(
                overrides=args,
                loss_kind=job["loss"],
                custom_layers=job["custom_layers"],
            )
            trainer.train()

            best = os.path.join(run_dir, "weights", "best.pt")
            checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            raw_model = checkpoint.get("ema") or checkpoint["model"]
            params = int(sum(parameter.numel() for parameter in raw_model.parameters()))
            eval_model = YOLO(best)
            metrics = eval_model.val(
                data=job["data"],
                split="test",
                imgsz=640,
                batch=BATCH,
                workers=0,
                device=0,
                plots=False,
                verbose=False,
            )
            summary[name] = {
                "group": job["group"],
                "mAP50": float(metrics.box.map50),
                "mAP5095": float(metrics.box.map),
                "perClassAP50": [float(value) for value in metrics.box.ap50],
                "params": params,
                "seconds": time.time() - started,
                "epochs": EPOCHS,
                "seed": job["seed"],
                "realization_seed": job["realization_seed"],
                "data": job["data"],
                "model": job["model"],
                "loss": job["loss"],
                "fwn_config": fwn_config if job["loss"] == "fwn" else None,
            }
            with open(SUMMARY, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            log(
                f"== {name} DONE mAP50={summary[name]['mAP50']:.4f} "
                f"mAP5095={summary[name]['mAP5095']:.4f} params={params}"
            )
            del trainer, eval_model, raw_model, checkpoint, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"== {name} FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    log("REVIEW ROUND 2 FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
