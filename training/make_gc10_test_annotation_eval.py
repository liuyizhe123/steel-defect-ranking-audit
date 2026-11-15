# -*- coding: utf-8 -*-
"""Create test-only annotation variants for the full annotated GC10-DET set."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from make_test_annotation_eval import SPECS, link_or_copy, load_label, perturb

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "ultralytics_data" / "gc10"
OUT = ROOT / "ultralytics_data" / "gc10_test_annotation"
SEED = 2026
CLASSES = [
    "1_chongkong",
    "2_hanfeng",
    "3_yueyawan",
    "4_shuiban",
    "5_youban",
    "6_siban",
    "7_yiwu",
    "8_yahen",
    "9_zhehen",
    "10_yaozhe",
]


def main() -> None:
    clean_images = CLEAN / "images" / "test"
    clean_labels = CLEAN / "labels" / "test"
    image_paths = sorted(clean_images.glob("*.jpg"))
    if len(image_paths) != 340:
        raise SystemExit(f"expected 340 GC10 test images, found {len(image_paths)}")

    if OUT.exists():
        shutil.rmtree(OUT)

    stats = {"seed": SEED, "clean_images": len(image_paths), "variants": {}}
    for name, mode, strength in SPECS:
        variant = OUT / name
        image_dst = variant / "images" / "test"
        label_dst = variant / "labels" / "test"
        image_dst.mkdir(parents=True)
        label_dst.mkdir(parents=True)
        rng = np.random.default_rng(SEED)
        before = 0
        after = 0
        for image_path in image_paths:
            label_path = clean_labels / f"{image_path.stem}.txt"
            labels = load_label(label_path)
            changed = perturb(labels, mode, strength, rng)
            before += len(labels)
            after += len(changed)
            link_or_copy(image_path, image_dst / image_path.name)
            (label_dst / label_path.name).write_text(
                "\n".join(
                    f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
                    for row in changed
                )
                + "\n",
                encoding="ascii",
            )

        yaml_lines = [
            f"path: {variant.as_posix()}",
            f"train: {(CLEAN / 'images' / 'train').as_posix()}",
            f"val: {(CLEAN / 'images' / 'val').as_posix()}",
            "test: images/test",
            f"nc: {len(CLASSES)}",
            "names:",
        ]
        yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
        (variant / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        stats["variants"][name] = {
            "mode": mode,
            "strength": strength,
            "instances_before": before,
            "instances_after": after,
        }

    (OUT / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("GC10_TEST_ANNOTATION_VARIANTS_OK")


if __name__ == "__main__":
    main()
