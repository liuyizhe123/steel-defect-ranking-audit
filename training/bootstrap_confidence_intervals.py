# -*- coding: utf-8 -*-
"""Image-level bootstrap intervals for the main and evaluation-noise results."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, ap_per_class

from lsc_modules import custom_model_classes

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT = BASE / "bootstrap_ci.json"
LOG = BASE / "bootstrap_ci.log"
REPLICATES = int(os.environ.get("BOOTSTRAP_REPLICATES", "1000"))
SEED = 2026
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

NEU_DATA = ROOT / "ultralytics_data" / "clean" / "data.yaml"
NEU_TEST_VARIANTS = ROOT / "ultralytics_data" / "test_annotation"
GC10_DATA = ROOT / "ultralytics_data" / "gc10" / "data.yaml"
GC10_TEST_VARIANTS = ROOT / "ultralytics_data" / "gc10_test_annotation"

TEST_CONDITIONS = [
    "test_jit05",
    "test_jit10",
    "test_jit20",
    "test_drop20",
    "test_drop40",
    "test_drop60",
]


class CaptureMetrics(DetMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.image_stats: list[dict[str, np.ndarray]] = []

    def update_stats(self, stat: dict) -> None:
        super().update_stats(stat)
        self.image_stats.append(
            {
                key: np.asarray(stat[key]).copy()
                for key in ("tp", "conf", "pred_cls", "target_cls")
            }
        )


class CaptureValidator(DetectionValidator):
    last: "CaptureValidator | None" = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = CaptureMetrics()
        self.captured_image_stats: list[dict[str, np.ndarray]] | None = None
        CaptureValidator.last = self

    def get_stats(self):
        self.metrics.process(save_dir=self.save_dir, plot=False, on_plot=None)
        self.captured_image_stats = list(self.metrics.image_stats)
        self.metrics.clear_stats()
        return self.metrics.results_dict


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def load_yolo(path: Path, custom: bool = False) -> YOLO:
    if custom:
        with custom_model_classes():
            return YOLO(str(path))
    return YOLO(str(path))


def evaluate(weights: Path, data: Path, custom: bool = False):
    model = load_yolo(weights, custom)
    CaptureValidator.last = None
    model.val(
        data=str(data),
        split="test",
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        plots=False,
        verbose=False,
        validator=CaptureValidator,
    )
    validator = CaptureValidator.last
    if validator is None or validator.captured_image_stats is None:
        raise RuntimeError(f"failed to capture image statistics for {weights}")
    stats = validator.captured_image_stats
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def metric_once(image_stats: list[dict[str, np.ndarray]], indices: np.ndarray) -> tuple[float, float]:
    selected = [image_stats[int(i)] for i in indices]
    tp = np.concatenate([x["tp"] for x in selected], axis=0)
    conf = np.concatenate([x["conf"] for x in selected], axis=0)
    pred_cls = np.concatenate([x["pred_cls"] for x in selected], axis=0)
    target_cls = np.concatenate([x["target_cls"] for x in selected], axis=0)
    ap = ap_per_class(tp, conf, pred_cls, target_cls)[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def bootstrap_values(
    image_stats: list[dict[str, np.ndarray]], indices_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = [metric_once(image_stats, idx) for idx in indices_matrix]
    arr = np.asarray(values, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def interval(values: np.ndarray) -> list[float]:
    return [round(float(x) * 100, 2) for x in np.percentile(values, [2.5, 97.5])]


def summarize_model(values50, values95, baseline50=None, baseline95=None) -> dict:
    row = {
        "mAP50": round(float(values50.mean()) * 100, 2),
        "mAP50_ci95": interval(values50),
        "mAP5095": round(float(values95.mean()) * 100, 2),
        "mAP5095_ci95": interval(values95),
    }
    if baseline50 is not None:
        delta50 = (values50 - baseline50) * 100
        delta95 = (values95 - baseline95) * 100
        row["delta50_vs_clean"] = round(float(delta50.mean()), 2)
        row["delta50_ci95"] = [
            round(float(x), 2) for x in np.percentile(delta50, [2.5, 97.5])
        ]
        row["delta95_vs_clean"] = round(float(delta95.mean()), 2)
        row["delta95_ci95"] = [
            round(float(x), 2) for x in np.percentile(delta95, [2.5, 97.5])
        ]
    return row


def main() -> None:
    report = {"seed": SEED, "replicates": REPLICATES}
    neu_specs = {
        "baseline": (ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt", False),
        "v8n_fwn": (ROOT / "lsc_pt_runs" / "v8n_fwn_pt_v2" / "weights" / "best.pt", False),
        "v8n_focaler": (ROOT / "lsc_pt_runs" / "v8n_focaler_pt" / "weights" / "best.pt", False),
        "v8n_hybrid": (ROOT / "lsc_pt_runs" / "v8n_hybrid_pt" / "weights" / "best.pt", False),
        "escm_ciou": (ROOT / "lsc_pt_runs" / "escm_ciou_pt" / "weights" / "best.pt", True),
        "lsc_ciou": (ROOT / "lsc_pt_runs" / "lsc_ciou_pt" / "weights" / "best.pt", True),
        "lsc_fwn": (ROOT / "lsc_pt_runs" / "lsc_fwn_pt_v3" / "weights" / "best.pt", True),
        "yolov10n": (
            ROOT / "cross_model_runs" / "yolov10n_clean_640" / "weights" / "best.pt",
            False,
        ),
    }
    neu_stats = {}
    for name, (weights, custom) in neu_specs.items():
        log(f"capturing NEU stats: {name}")
        neu_stats[name] = evaluate(weights, NEU_DATA, custom)

    n_neu = len(neu_stats["baseline"])
    rng = np.random.default_rng(SEED)
    neu_indices = rng.integers(0, n_neu, size=(REPLICATES, n_neu))
    neu_values = {}
    for name, stats in neu_stats.items():
        log(f"bootstrapping NEU: {name}")
        neu_values[name] = bootstrap_values(stats, neu_indices)
    base50, base95 = neu_values["baseline"]
    report["neu_models"] = {
        name: summarize_model(v50, v95, base50, base95)
        for name, (v50, v95) in neu_values.items()
    }

    log("capturing NEU evaluation-label statistics")
    neu_noise_stats = {}
    for condition in TEST_CONDITIONS:
        neu_noise_stats[condition] = evaluate(
            neu_specs["baseline"][0],
            NEU_TEST_VARIANTS / condition / "data.yaml",
            False,
        )
    neu_noise_values = {
        condition: bootstrap_values(stats, neu_indices)
        for condition, stats in neu_noise_stats.items()
    }
    report["neu_test_annotation_noise"] = {
        condition: summarize_model(v50, v95, base50, base95)
        for condition, (v50, v95) in neu_noise_values.items()
    }

    gc10_specs = {
        "baseline": (ROOT / "gc10_runs" / "gc10_v8n_ciou" / "weights" / "best.pt", False),
        "fwn": (ROOT / "gc10_runs" / "gc10_v8n_fwn" / "weights" / "best.pt", False),
        "lsc_ciou": (ROOT / "gc10_runs" / "gc10_lsc_ciou" / "weights" / "best.pt", True),
        "lsc_fwn": (ROOT / "gc10_runs" / "gc10_lsc_fwn" / "weights" / "best.pt", True),
    }
    gc10_stats = {}
    for name, (weights, custom) in gc10_specs.items():
        log(f"capturing GC10 stats: {name}")
        gc10_stats[name] = evaluate(weights, GC10_DATA, custom)
    n_gc10 = len(gc10_stats["baseline"])
    gc10_indices = rng.integers(0, n_gc10, size=(REPLICATES, n_gc10))
    gc10_values = {
        name: bootstrap_values(stats, gc10_indices) for name, stats in gc10_stats.items()
    }
    gc_base50, gc_base95 = gc10_values["baseline"]
    report["gc10_models"] = {
        name: summarize_model(v50, v95, gc_base50, gc_base95)
        for name, (v50, v95) in gc10_values.items()
    }

    log("capturing GC10 evaluation-label statistics")
    gc10_noise_stats = {}
    for condition in TEST_CONDITIONS:
        gc10_noise_stats[condition] = evaluate(
            gc10_specs["baseline"][0],
            GC10_TEST_VARIANTS / condition / "data.yaml",
            False,
        )
    gc10_noise_values = {
        condition: bootstrap_values(stats, gc10_indices)
        for condition, stats in gc10_noise_stats.items()
    }
    report["gc10_test_annotation_noise"] = {
        condition: summarize_model(v50, v95, gc_base50, gc_base95)
        for condition, (v50, v95) in gc10_noise_values.items()
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    log("BOOTSTRAP_CI_OK")


if __name__ == "__main__":
    main()
