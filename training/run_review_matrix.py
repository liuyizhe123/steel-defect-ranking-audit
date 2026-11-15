# -*- coding: utf-8 -*-
"""Run reviewer-requested split, P2/P5 and FWN sensitivity experiments."""

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
STANDARD_DATA = os.path.join(ROOT, "ultralytics_data", "standard_721", "data.yaml")
PROJ = os.path.join(ROOT, "review_runs")
SUMMARY = os.path.join(BASE, "review_matrix_summary.json")
LOGFILE = os.path.join(BASE, "review_matrix.log")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, ".ultralytics"))

EPOCHS = int(os.environ.get("REVIEW_EPOCHS", "300"))
BATCH = int(os.environ.get("REVIEW_BATCH", "16"))
MODELS = os.path.join(BASE, "lsc_models")

JOBS = []
STANDARD_MODELS = [
    ("v8n_ciou", "yolov8n.yaml", "ciou", False),
    ("v8n_fwn", "yolov8n.yaml", "fwn", False),
    ("lsc_ciou", os.path.join(MODELS, "lsc.yaml"), "ciou", True),
]
for seed in (2026, 2027, 2028):
    for model_name, model_cfg, loss_kind, custom_layers in STANDARD_MODELS:
        JOBS.append(
            {
                "name": f"standard721_{model_name}_s{str(seed)[-2:]}",
                "data": STANDARD_DATA,
                "model": model_cfg,
                "loss": loss_kind,
                "custom_layers": custom_layers,
                "seed": seed,
                "group": "standard_split",
                "params": {},
            }
        )

JOBS.extend(
    [
        {
            "name": "p2off_p5on_s26",
            "data": CLEAN_DATA,
            "model": "yolov8n.yaml",
            "loss": "ciou",
            "custom_layers": False,
            "seed": 2026,
            "group": "p2_p5_ablation",
            "params": {"p2": False, "p5": True},
        },
        {
            "name": "p2on_p5on_s26",
            "data": CLEAN_DATA,
            "model": os.path.join(MODELS, "p2on_p5on.yaml"),
            "loss": "ciou",
            "custom_layers": "fair",
            "seed": 2026,
            "group": "p2_p5_ablation",
            "params": {"p2": True, "p5": True},
        },
        {
            "name": "p2off_p5off_s26",
            "data": CLEAN_DATA,
            "model": os.path.join(MODELS, "p2off_p5off.yaml"),
            "loss": "ciou",
            "custom_layers": False,
            "seed": 2026,
            "group": "p2_p5_ablation",
            "params": {"p2": False, "p5": False},
        },
        {
            "name": "p2on_p5off_s26",
            "data": CLEAN_DATA,
            "model": os.path.join(MODELS, "p2on_p5off.yaml"),
            "loss": "ciou",
            "custom_layers": "fair",
            "seed": 2026,
            "group": "p2_p5_ablation",
            "params": {"p2": True, "p5": False},
        },
    ]
)

FWN_VARIANTS = [
    ("fwn_k05_s26", {"FWN_K": "5.0"}),
    ("fwn_k20_s26", {"FWN_K": "20.0"}),
    ("fwn_s003_s26", {"FWN_S0": "0.03"}),
    ("fwn_s012_s26", {"FWN_S0": "0.12"}),
]
for name, overrides in FWN_VARIANTS:
    JOBS.append(
        {
            "name": name,
            "data": CLEAN_DATA,
            "model": "yolov8n.yaml",
            "loss": "fwn",
            "custom_layers": False,
            "seed": 2026,
            "group": "fwn_sensitivity",
            "params": overrides,
        }
    )

for seed in (2027, 2028):
    for condition in ("jit10", "drop40"):
        data = os.path.join(ROOT, "ultralytics_data", condition, "data.yaml")
        for model_name, model_cfg, loss_kind, custom_layers in [
            ("v8n_ciou", "yolov8n.yaml", "ciou", False),
            ("v8n_fwn", "yolov8n.yaml", "fwn", False),
            ("lsc_ciou", os.path.join(MODELS, "lsc.yaml"), "ciou", True),
        ]:
            JOBS.append(
                {
                    "name": f"interaction_{model_name}_{condition}_s{str(seed)[-2:]}",
                    "data": data,
                    "model": model_cfg,
                    "loss": loss_kind,
                    "custom_layers": custom_layers,
                    "seed": seed,
                    "group": "interaction_seed_replication",
                    "params": {"condition": condition},
                }
            )


def fwn_overrides_for_job(job: dict) -> dict[str, str]:
    """Return only coefficient overrides; other jobs store metadata in params."""
    if job["group"] == "fwn_sensitivity":
        return job["params"]
    return {}


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOGFILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def configure_fwn(overrides: dict[str, str]) -> dict[str, float]:
    keys = {
        "FWN_K": "10.0",
        "FWN_S0": "0.06",
        "FWN_U": "0.25",
        "FWN_V": "1.0",
        "FWN_WIOU_ALPHA": "1.9",
        "FWN_WIOU_DELTA": "3.0",
        "FWN_NWD_C": "12.8",
    }
    keys.update(overrides)
    for key, value in keys.items():
        os.environ[key] = value
    return {key.removeprefix("FWN_").lower(): float(value) for key, value in keys.items()}


def main() -> None:
    if not os.path.isfile(STANDARD_DATA):
        raise SystemExit(f"missing standard split: {STANDARD_DATA}")
    os.makedirs(PROJ, exist_ok=True)
    summary = {}
    if os.path.isfile(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as handle:
            summary = json.load(handle)

    failures = []
    log(f"==== review matrix start (epochs={EPOCHS}, batch={BATCH}) ====")
    for job in JOBS:
        name = job["name"]
        if name in summary:
            log(f"== {name} already finished, skip")
            continue
        fwn_config = configure_fwn(fwn_overrides_for_job(job))
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
                "variant": job["params"],
                "model": job["model"],
                "data": job["data"],
                "loss": job["loss"],
                "mAP50": float(metrics.box.map50),
                "mAP5095": float(metrics.box.map),
                "perClassAP50": [float(value) for value in metrics.box.ap50],
                "params": params,
                "seconds": time.time() - started,
                "epochs": EPOCHS,
                "seed": job["seed"],
                "fwn_config": fwn_config if job["loss"] == "fwn" else None,
            }
            with open(SUMMARY, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            log(
                f"== {name} DONE mAP50={summary[name]['mAP50']:.4f} "
                f"mAP5095={summary[name]['mAP5095']:.4f} "
                f"params={params} ({summary[name]['seconds']:.0f} s)"
            )
            del trainer, eval_model, raw_model, checkpoint, metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"== {name} FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)

    log("REVIEW MATRIX FINISHED")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
