# -*- coding: utf-8 -*-
"""Create test-only annotation variants for evaluation-side noise experiments."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "ultralytics_data" / "clean"
OUT = ROOT / "ultralytics_data" / "test_annotation"
SEED = 2026
SPECS = [
    ("test_jit05", "jitter", 0.05),
    ("test_jit10", "jitter", 0.10),
    ("test_jit20", "jitter", 0.20),
    ("test_drop20", "drop", 0.20),
    ("test_drop40", "drop", 0.40),
    ("test_drop60", "drop", 0.60),
]


def load_label(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            values = [float(x) for x in line.split()]
            if len(values) != 5:
                raise ValueError(f"invalid label row in {path}: {line}")
            rows.append(values)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def perturb(labels: np.ndarray, mode: str, strength: float, rng: np.random.Generator) -> np.ndarray:
    if len(labels) == 0:
        return labels
    cls = labels[:, :1]
    boxes = labels[:, 1:]
    if mode == "drop":
        keep = rng.random(len(labels)) >= strength
        if not keep.any():
            keep[rng.integers(len(keep))] = True
        return labels[keep]
    if mode != "jitter":
        raise ValueError(f"unknown mode: {mode}")

    scale = np.clip(1.0 + strength * rng.standard_normal(len(boxes)), 0.5, 2.0)
    width = boxes[:, 2] * scale
    height = boxes[:, 3] * scale
    center_x = boxes[:, 0] + strength * boxes[:, 2] * rng.standard_normal(len(boxes))
    center_y = boxes[:, 1] + strength * boxes[:, 3] * rng.standard_normal(len(boxes))
    x = np.clip(center_x - width / 2, 0.0, 1.0 - 4.0 / 200.0)
    y = np.clip(center_y - height / 2, 0.0, 1.0 - 4.0 / 200.0)
    width = np.clip(width, 4.0 / 200.0, 1.0 - x)
    height = np.clip(height, 4.0 / 200.0, 1.0 - y)
    out = np.concatenate(
        [cls, (x + width / 2)[:, None], (y + height / 2)[:, None], width[:, None], height[:, None]],
        axis=1,
    )
    return out


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    clean_images = CLEAN / "images" / "test"
    clean_labels = CLEAN / "labels" / "test"
    image_paths = sorted(clean_images.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"no clean test images in {clean_images}")

    out_resolved = OUT.resolve()
    root_resolved = ROOT.resolve()
    if out_resolved == root_resolved or root_resolved not in out_resolved.parents:
        raise RuntimeError(f"refusing to replace unexpected path: {out_resolved}")
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
        instances_before = 0
        instances_after = 0
        for image_path in image_paths:
            label_path = clean_labels / f"{image_path.stem}.txt"
            labels = load_label(label_path)
            perturbed = perturb(labels, mode, strength, rng)
            instances_before += len(labels)
            instances_after += len(perturbed)
            link_or_copy(image_path, image_dst / image_path.name)
            (label_dst / label_path.name).write_text(
                "\n".join(
                    f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
                    for row in perturbed
                )
                + "\n",
                encoding="ascii",
            )

        yaml = variant / "data.yaml"
        yaml.write_text(
            "\n".join(
                [
                    f"path: {variant.as_posix()}",
                    f"train: {(CLEAN / 'images' / 'train').as_posix()}",
                    f"val: {(CLEAN / 'images' / 'val').as_posix()}",
                    "test: images/test",
                    "nc: 6",
                    "names:",
                    "  0: crazing",
                    "  1: inclusion",
                    "  2: patches",
                    "  3: pitted_surface",
                    "  4: rolled-in_scale",
                    "  5: scratches",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stats["variants"][name] = {
            "mode": mode,
            "strength": strength,
            "instances_before": instances_before,
            "instances_after": instances_after,
        }

    (OUT / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("TEST_ANNOTATION_VARIANTS_OK")


if __name__ == "__main__":
    main()
