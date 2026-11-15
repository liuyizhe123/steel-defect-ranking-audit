# -*- coding: utf-8 -*-
"""Create side-by-side qualitative comparisons selected by matched-detection loss."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
IMAGE_DIR = ROOT / "ultralytics_data" / "clean" / "images" / "test"
LABEL_DIR = ROOT / "ultralytics_data" / "clean" / "labels" / "test"
OUT = ROOT / "figs"
SUMMARY = BASE / "qualitative_summary.json"
LOGFILE = BASE / "qualitative.log"
PANEL_SIZE = 800
PANEL_HEADER = 76
CASE_HEADER = 44
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))

MODELS = [
    ("YOLOv8n+CIoU", ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt"),
    (
        "YOLOv8n+FWN",
        ROOT / "lsc_pt_runs" / "v8n_fwn_pt_v2" / "weights" / "best.pt",
    ),
    (
        "ESCM+SAF+CIoU",
        ROOT / "lsc_pt_runs" / "lsc_ciou_pt" / "weights" / "best.pt",
    ),
]


def log(message: str) -> None:
    line = f"{message}"
    with open(LOGFILE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_targets(image_path: Path, width: int, height: int):
    rows = []
    for line in (LABEL_DIR / f"{image_path.stem}.txt").read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = [float(value) for value in line.split()]
        rows.append(
            (
                int(cls),
                np.array(
                    [
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return rows


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def match_boxes(boxes, classes, targets):
    used = set()
    matched_targets = set()
    matched_predictions = set()
    for target_index, (class_id, target_box) in enumerate(targets):
        best_index = None
        best_iou = 0.0
        for index, (box, pred_class) in enumerate(zip(boxes, classes)):
            if index in used or pred_class != class_id:
                continue
            overlap = iou(target_box, box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= 0.5:
            used.add(best_index)
            matched_targets.add(target_index)
            matched_predictions.add(best_index)
    return len(matched_targets), matched_targets, matched_predictions


def match_counts(result, targets) -> tuple[int, int]:
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    matched, _, _ = match_boxes(boxes, classes, targets)
    return matched, len(targets)


def predict_result(model: YOLO, image_path: Path):
    return model.predict(
        str(image_path),
        imgsz=640,
        device=0,
        conf=0.25,
        verbose=False,
    )[0]


def class_code(class_names, class_id: int) -> str:
    if isinstance(class_names, dict):
        name = str(class_names[class_id])
    else:
        name = str(class_names[class_id])
    abbreviations = {
        "crazing": "Cr",
        "inclusion": "In",
        "patches": "Pa",
        "pitted surface": "PS",
        "pitted_surface": "PS",
        "rolled-in scale": "RS",
        "rolled_in_scale": "RS",
        "scratches": "Sc",
    }
    return abbreviations.get(name.lower(), name[:3])


def dashed_segment(draw, start, end, fill, width, dash=10, gap=7):
    import math

    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    offset = 0.0
    while offset < length:
        segment_end = min(offset + dash, length)
        draw.line(
            (
                x1 + dx * offset,
                y1 + dy * offset,
                x1 + dx * segment_end,
                y1 + dy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        offset += dash + gap


def dashed_rectangle(draw, box, fill, width):
    x1, y1, x2, y2 = box
    dashed_segment(draw, (x1, y1), (x2, y1), fill, width)
    dashed_segment(draw, (x2, y1), (x2, y2), fill, width)
    dashed_segment(draw, (x2, y2), (x1, y2), fill, width)
    dashed_segment(draw, (x1, y2), (x1, y1), fill, width)


def draw_box_label(draw, x, y, text, fill, font):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = min(max(2, x), PANEL_SIZE - text_width - 4)
    y = min(max(2, y), PANEL_SIZE - text_height - 4)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=2,
        stroke_fill="white",
    )


def render_panel(result, name, image_path, targets, matched, total):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    panel = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize(
        (PANEL_SIZE, PANEL_SIZE),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(panel)
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
    prediction_color = "#d62728"
    ground_truth_color = "#00a6a6"
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    _, matched_targets, _ = match_boxes(boxes, classes, targets)

    for target_index, (class_id, target_box) in enumerate(targets):
        scaled = (
            float(target_box[0]) * PANEL_SIZE / width,
            float(target_box[1]) * PANEL_SIZE / height,
            float(target_box[2]) * PANEL_SIZE / width,
            float(target_box[3]) * PANEL_SIZE / height,
        )
        dashed_rectangle(draw, scaled, ground_truth_color, 4)
        if target_index not in matched_targets:
            draw_box_label(
                draw,
                scaled[0] + 3,
                scaled[1] + 3,
                f"MISS {class_code(result.names, class_id)}",
                ground_truth_color,
                label_font,
            )

    for box in boxes:
        scaled = (
            float(box[0]) * PANEL_SIZE / width,
            float(box[1]) * PANEL_SIZE / height,
            float(box[2]) * PANEL_SIZE / width,
            float(box[3]) * PANEL_SIZE / height,
        )
        draw.rectangle(scaled, outline=prediction_color, width=4)

    canvas = Image.new(
        "RGB",
        (PANEL_SIZE, PANEL_SIZE + PANEL_HEADER),
        "white",
    )
    canvas.paste(panel, (0, PANEL_HEADER))
    canvas_draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 40)
    match_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 30)
    for text, y, font in (
        (name, 5, title_font),
        (f"matched {matched}/{total} at IoU 0.5", 45, match_font),
    ):
        text_width = canvas_draw.textlength(text, font=font)
        canvas_draw.text(
            ((PANEL_SIZE - text_width) / 2, y),
            text,
            fill="black",
            font=font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-selection",
        action="store_true",
        help="redraw the cases already frozen in qualitative_summary.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for qualitative inference")
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    models = []
    for name, weights in MODELS:
        if not weights.is_file():
            raise SystemExit(f"missing weights for {name}: {weights}")
        models.append((name, YOLO(str(weights))))

    if args.reuse_selection:
        prior = json.loads(SUMMARY.read_text(encoding="utf-8"))
        selected = prior["cases"]
    else:
        candidates = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            targets = load_targets(image_path, width, height)
            if not targets:
                continue
            counts = {}
            for name, model in models:
                result = predict_result(model, image_path)
                counts[name] = match_counts(result, targets)
            baseline = counts[MODELS[0][0]][0]
            alternative_loss = max(
                counts[MODELS[1][0]][0],
                counts[MODELS[2][0]][0],
            )
            candidates.append(
                {
                    "image": image_path.name,
                    "targets": len(targets),
                    "scores": {
                        name: {"matched": value[0], "total": value[1]}
                        for name, value in counts.items()
                    },
                    "selection_margin": baseline - alternative_loss,
                }
            )

        candidates.sort(
            key=lambda row: (
                row["selection_margin"],
                row["targets"],
                row["image"],
            ),
            reverse=True,
        )
        selected = [row for row in candidates if row["selection_margin"] > 0][:3]
        if len(selected) < 3:
            selected = candidates[:3]

    case_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 30)
    OUT.mkdir(exist_ok=True)
    cases = []
    for index, row in enumerate(selected, start=1):
        image_path = IMAGE_DIR / row["image"]
        panels = []
        for name, model in models:
            result = predict_result(model, image_path)
            targets = load_targets(
                image_path,
                *cv2.imread(str(image_path)).shape[1::-1],
            )
            matched, total = match_counts(result, targets)
            expected = row.get("scores", {}).get(name)
            if expected and (matched, total) != (
                expected["matched"],
                expected["total"],
            ):
                raise SystemExit(
                    f"selection drift for {row['image']} / {name}: "
                    f"{matched}/{total} != {expected['matched']}/{expected['total']}"
                )
            panel = render_panel(
                result,
                name,
                image_path,
                targets,
                matched,
                total,
            )
            panels.append(panel)
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        output = Image.new("RGB", (width, height), "white")
        x = 0
        for panel in panels:
            output.paste(panel, (x, 0))
            x += panel.width
        output_name = f"fig8_qualitative_case{index}.png"
        output.save(OUT / output_name, dpi=(300, 300))
        row["output"] = output_name
        cases.append(row)
        log(f"== case {index}: {row['image']} -> {output_name}")

    if cases:
        case_images = [Image.open(OUT / row["output"]).convert("RGB") for row in cases]
        width = max(image.width for image in case_images)
        header_height = CASE_HEADER
        height = sum(image.height + header_height for image in case_images)
        composite = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(composite)
        y = 0
        for row, image in zip(cases, case_images):
            draw.rectangle((0, y, width, y + header_height), fill="#ececec")
            draw.text(
                (8, y + 8),
                f"Case {cases.index(row) + 1}: {row['image']}",
                fill="black",
                font=case_font,
            )
            y += header_height
            composite.paste(image, ((width - image.width) // 2, y))
            y += image.height
        composite_name = "fig8_qualitative.png"
        composite.save(OUT / composite_name, dpi=(300, 300))
    else:
        composite_name = None

    summary = {
        "selection": "largest matched-detection loss relative to baseline",
        "composite": composite_name,
        "render": {
            "panel_pixels": PANEL_SIZE,
            "prediction_boxes": True,
            "ground_truth_boxes": True,
            "labels": "unmatched ground truth (MISS) only",
        },
        "cases": cases,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("QUALITATIVE_COMPARISON_OK")


if __name__ == "__main__":
    main()
