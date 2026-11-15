# -*- coding: utf-8 -*-
"""Create independent 40% training-label realisation datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CLEAN = ROOT / "ultralytics_data" / "clean"
OUT_ROOT = ROOT / "ultralytics_data"
SEEDS = (2027, 2028)
OMISSION_RATE = 0.40


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def retained_lines(lines: list[str], rng: random.Random) -> list[str]:
    kept = [line for line in lines if rng.random() >= OMISSION_RATE]
    if lines and not kept:
        kept = [lines[rng.randrange(len(lines))]]
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = {"omission_rate": OMISSION_RATE, "seeds": list(SEEDS), "variants": {}}
    for seed in SEEDS:
        out = OUT_ROOT / f"drop40_r{seed}"
        if out.exists():
            if not args.force:
                raise SystemExit(f"output already exists: {out}")
            if out.parent.resolve() != OUT_ROOT.resolve():
                raise RuntimeError(f"unsafe output path: {out}")
            shutil.rmtree(out)

        stats = {"seed": seed, "splits": {}}
        for split in ("train", "val", "test"):
            source_images = CLEAN / "images" / split
            source_labels = CLEAN / "labels" / split
            target_images = out / "images" / split
            target_labels = out / "labels" / split
            target_images.mkdir(parents=True, exist_ok=True)
            target_labels.mkdir(parents=True, exist_ok=True)

            original_instances = 0
            retained_instances = 0
            for image_path in sorted(source_images.iterdir()):
                if image_path.is_file():
                    link_or_copy(image_path, target_images / image_path.name)

            rng = random.Random(seed + 100 * len(split))
            for label_path in sorted(source_labels.glob("*.txt")):
                lines = [line for line in label_path.read_text(encoding="ascii").splitlines() if line.strip()]
                original_instances += len(lines)
                output_lines = retained_lines(lines, rng) if split == "train" else lines
                retained_instances += len(output_lines)
                (target_labels / label_path.name).write_text(
                    "\n".join(output_lines) + ("\n" if output_lines else ""),
                    encoding="ascii",
                )
            stats["splits"][split] = {
                "images": len(list(target_images.glob("*"))),
                "original_instances": original_instances,
                "retained_instances": retained_instances,
                "retention": retained_instances / original_instances if original_instances else 1.0,
            }

        data_yaml = (
            f"path: {out.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 6\n"
            "names:\n"
            "  0: crazing\n"
            "  1: inclusion\n"
            "  2: patches\n"
            "  3: pitted_surface\n"
            "  4: rolled-in_scale\n"
            "  5: scratches\n"
        )
        (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
        manifest["variants"][str(seed)] = stats

    (OUT_ROOT / "drop40_realizations_stats.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("TRAINING_OMISSION_REALIZATIONS_OK")


if __name__ == "__main__":
    main()
