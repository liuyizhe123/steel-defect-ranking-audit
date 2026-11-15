# -*- coding: utf-8 -*-
"""Create repeated NEU-DET and GC10-DET evaluation-label perturbation draws."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from make_test_annotation_eval import link_or_copy, load_label, perturb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ultralytics_data" / "test_annotation_realizations"
SEEDS = [2027, 2028, 2029, 2030]
SPECS = [
    ("test_jit10", "jitter", 0.10),
    ("test_jit20", "jitter", 0.20),
    ("test_drop20", "drop", 0.20),
    ("test_drop40", "drop", 0.40),
]
DATASETS = {
    "neu": {
        "root": ROOT / "ultralytics_data" / "clean",
        "classes": [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ],
    },
    "gc10": {
        "root": ROOT / "ultralytics_data" / "gc10",
        "classes": [
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
        ],
    },
}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    summary = {"seeds": SEEDS, "variants": {}}

    for dataset_name, config in DATASETS.items():
        clean = config["root"]
        clean_images = clean / "images" / "test"
        clean_labels = clean / "labels" / "test"
        image_paths = sorted(clean_images.glob("*.jpg"))
        if not image_paths:
            raise SystemExit(f"no test images for {dataset_name}")
        summary["variants"][dataset_name] = {}

        for seed in SEEDS:
            for condition, mode, strength in SPECS:
                variant = OUT / dataset_name / f"seed{seed}" / condition
                image_dst = variant / "images" / "test"
                label_dst = variant / "labels" / "test"
                image_dst.mkdir(parents=True)
                label_dst.mkdir(parents=True)
                rng = np.random.default_rng(seed)
                instances_before = 0
                instances_after = 0
                for image_path in image_paths:
                    label_path = clean_labels / f"{image_path.stem}.txt"
                    labels = load_label(label_path)
                    changed = perturb(labels, mode, strength, rng)
                    instances_before += len(labels)
                    instances_after += len(changed)
                    link_or_copy(image_path, image_dst / image_path.name)
                    (label_dst / label_path.name).write_text(
                        "\n".join(
                            f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} "
                            f"{row[3]:.6f} {row[4]:.6f}"
                            for row in changed
                        )
                        + "\n",
                        encoding="ascii",
                    )

                yaml_lines = [
                    f"path: {variant.as_posix()}",
                    f"train: {(clean / 'images' / 'train').as_posix()}",
                    f"val: {(clean / 'images' / 'val').as_posix()}",
                    "test: images/test",
                    f"nc: {len(config['classes'])}",
                    "names:",
                ]
                yaml_lines.extend(
                    f"  {index}: {name}"
                    for index, name in enumerate(config["classes"])
                )
                (variant / "data.yaml").write_text(
                    "\n".join(yaml_lines) + "\n",
                    encoding="utf-8",
                )
                summary["variants"][dataset_name][f"{seed}/{condition}"] = {
                    "seed": seed,
                    "condition": condition,
                    "mode": mode,
                    "strength": strength,
                    "instances_before": instances_before,
                    "instances_after": instances_after,
                    "images": len(image_paths),
                }

    (OUT / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("TEST_ANNOTATION_REALIZATIONS_OK")


if __name__ == "__main__":
    main()
