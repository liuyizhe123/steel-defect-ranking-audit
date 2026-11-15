# -*- coding: utf-8 -*-
"""Build a blind 120-image reannotation package without exposing original labels."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SOURCE = ROOT / "ultralytics_data" / "clean" / "images" / "test"
OUT = ROOT / "reannotation_round1"
SEED = 2026
PER_CLASS = 20
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    images_out = OUT / "images"
    labels_out = OUT / "labels_to_fill"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    rng = random.Random(SEED)
    rows = []
    for class_name in CLASSES:
        candidates = sorted(SOURCE.glob(f"{class_name}_*.jpg"))
        rng.shuffle(candidates)
        selected = candidates[:PER_CLASS]
        if len(selected) != PER_CLASS:
            raise SystemExit(
                f"expected {PER_CLASS} {class_name} images, found {len(selected)}"
            )
        for image_path in selected:
            shutil.copy2(image_path, images_out / image_path.name)
            (labels_out / f"{image_path.stem}.txt").write_text("", encoding="ascii")
            with Image.open(image_path) as image:
                width, height = image.size
            rows.append(
                {
                    "image": image_path.name,
                    "label_to_fill": f"labels_to_fill/{image_path.stem}.txt",
                    "class_hint": class_name,
                    "width": width,
                    "height": height,
                }
            )

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    instructions = """# Blind reannotation instructions

1. Open each image in `images/` and annotate every visible defect instance.
2. Write one line per box in the matching file under `labels_to_fill/`.
3. Use YOLO format: `class_id center_x center_y width height`, all normalized to 0-1.
4. Do not inspect or open the original dataset labels.
5. Include touching or partially visible instances when their extent is identifiable.
6. Use a tight box around the defect evidence, not a region of uniformly textured background.
7. If an image contains no visible target, leave its label file empty and note it in the review log.

Class ids:

| id | class |
|---:|---|
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |

This package is intentionally identity-blind and does not contain original boxes. The
`class_hint` column in `manifest.csv` may be hidden from annotators if a fully blind
class assignment is required.
"""
    (OUT / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print(f"REANNOTATION_PACKAGE_OK images={len(rows)}")


if __name__ == "__main__":
    main()
