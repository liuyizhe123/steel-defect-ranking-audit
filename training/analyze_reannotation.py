# -*- coding: utf-8 -*-
"""Compare completed human reannotations with the original NEU-DET boxes."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REANNOTATION = ROOT / "reannotation_round1" / "labels_to_fill"
ORIGINAL = ROOT / "ultralytics_data" / "clean" / "labels" / "test"
SUMMARY = BASE / "reannotation_summary.json"


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            rows.append([float(value) for value in line.split()])
    return rows


def box(row):
    cls, cx, cy, width, height = row
    return int(cls), np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float64,
    )


def iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-12)


def main() -> None:
    label_paths = sorted(path for path in REANNOTATION.glob("*.txt") if path.stat().st_size > 0)
    if not label_paths:
        raise SystemExit("no completed reannotation label files found")

    center_offsets = []
    width_ratios = []
    height_ratios = []
    original_instances = 0
    annotated_instances = 0
    matched_instances = 0
    image_rows = []

    for path in label_paths:
        original_path = ORIGINAL / path.name
        if not original_path.is_file():
            raise FileNotFoundError(f"original label missing for {path.name}")
        original = load(original_path)
        revised = load(path)
        original_instances += len(original)
        annotated_instances += len(revised)
        used = set()
        matched = 0
        for original_row in original:
            original_class, original_box = box(original_row)
            best_index = None
            best_iou = 0.0
            for index, revised_row in enumerate(revised):
                revised_class, revised_box = box(revised_row)
                if index in used or revised_class != original_class:
                    continue
                overlap = iou(original_box, revised_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                used.add(best_index)
                matched += 1
                revised_box = box(revised[best_index])[1]
                original_center = np.array(
                    [
                        (original_box[0] + original_box[2]) / 2,
                        (original_box[1] + original_box[3]) / 2,
                    ]
                )
                revised_center = np.array(
                    [
                        (revised_box[0] + revised_box[2]) / 2,
                        (revised_box[1] + revised_box[3]) / 2,
                    ]
                )
                center_offsets.append(float(np.linalg.norm(revised_center - original_center)))
                original_width = original_box[2] - original_box[0]
                original_height = original_box[3] - original_box[1]
                width_ratios.append((revised_box[2] - revised_box[0]) / max(original_width, 1e-12))
                height_ratios.append((revised_box[3] - revised_box[1]) / max(original_height, 1e-12))
        matched_instances += matched
        image_rows.append(
            {
                "image": path.name,
                "original_instances": len(original),
                "reannotated_instances": len(revised),
                "matched_iou50": matched,
            }
        )

    summary = {
        "images": len(label_paths),
        "original_instances": original_instances,
        "reannotated_instances": annotated_instances,
        "matched_iou50": matched_instances,
        "matched_fraction_of_original": matched_instances / max(original_instances, 1),
        "original_instances_absent_from_reannotation": original_instances - matched_instances,
        "extra_reannotated_instances": annotated_instances - matched_instances,
        "center_offset_median": st.median(center_offsets) if center_offsets else None,
        "center_offset_p90": (
            float(np.percentile(center_offsets, 90)) if center_offsets else None
        ),
        "width_ratio_median": st.median(width_ratios) if width_ratios else None,
        "height_ratio_median": st.median(height_ratios) if height_ratios else None,
        "images_detail": image_rows,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REANNOTATION_ANALYSIS_OK")


if __name__ == "__main__":
    main()
