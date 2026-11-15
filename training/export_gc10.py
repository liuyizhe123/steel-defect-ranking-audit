# -*- coding: utf-8 -*-
"""Convert the complete annotated GC10-DET VOC mirror to a YOLO dataset."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "gc10_det_data" / "VOC2007"
OUT = ROOT / "ultralytics_data" / "gc10"
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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 2026
MIN_PAIRS = int(os.environ.get("GC10_MIN_PAIRS", "2280"))


def canonical_name(name: str) -> str | None:
    name = name.strip()
    if name == "10_yaozhed":
        return "10_yaozhe"
    if name == "d":
        return None
    return name if name in CLASS_TO_ID else None


def parse_annotation(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = float(size.findtext("width")) if size is not None else 0.0
    height = float(size.findtext("height")) if size is not None else 0.0
    objects = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="")
        class_name = canonical_name(raw_name)
        box = obj.find("bndbox")
        if class_name is None or box is None:
            continue
        xmin = max(0.0, min(width, float(box.findtext("xmin"))))
        ymin = max(0.0, min(height, float(box.findtext("ymin"))))
        xmax = max(0.0, min(width, float(box.findtext("xmax"))))
        ymax = max(0.0, min(height, float(box.findtext("ymax"))))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append(
            (
                CLASS_TO_ID[class_name],
                ((xmin + xmax) / 2) / width,
                ((ymin + ymax) / 2) / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return objects


def split_counts(total: int) -> dict[str, int]:
    n_train = int(round(total * SPLIT_RATIOS["train"]))
    n_val = int(round(total * SPLIT_RATIOS["val"]))
    n_test = total - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"missing GC10-DET source: {SRC}")

    image_dir = SRC / "JPEGImages"
    ann_dir = SRC / "Annotations"
    image_stems = {p.stem: p for p in image_dir.glob("*.jpg")}
    ann_stems = {p.stem: p for p in ann_dir.glob("*.xml")}
    common = sorted(image_stems.keys() & ann_stems.keys())
    if len(common) < MIN_PAIRS:
        raise SystemExit(
            f"only {len(common)} image/annotation pairs found; "
            f"expected at least {MIN_PAIRS}"
        )

    parsed = {}
    by_primary = defaultdict(list)
    raw_label_counts = Counter()
    dropped_labels = Counter()
    for stem in common:
        root = ET.parse(ann_stems[stem]).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            raw_label_counts[name] += 1
            if canonical_name(name) is None:
                dropped_labels[name] += 1
        objects = parse_annotation(ann_stems[stem])
        if not objects:
            continue
        parsed[stem] = objects
        by_primary[CLASSES[objects[0][0]]].append(stem)

    rng = random.Random(SEED)
    assigned = {}
    for class_name in CLASSES:
        stems = sorted(by_primary[class_name])
        rng.shuffle(stems)
        counts = split_counts(len(stems))
        offset = 0
        for split in ("train", "val", "test"):
            for stem in stems[offset : offset + counts[split]]:
                assigned[stem] = split
            offset += counts[split]

    out_resolved = OUT.resolve()
    root_resolved = ROOT.resolve()
    if out_resolved == root_resolved or root_resolved not in out_resolved.parents:
        raise RuntimeError(f"refusing to replace unexpected path: {out_resolved}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLIT_RATIOS:
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    split_images = Counter()
    split_instances = Counter()
    split_classes = {split: Counter() for split in SPLIT_RATIOS}
    split_class_images = {split: Counter() for split in SPLIT_RATIOS}
    for stem, split in assigned.items():
        objects = parsed[stem]
        primary_class = CLASSES[objects[0][0]]
        shutil.copy2(image_stems[stem], OUT / "images" / split / image_stems[stem].name)
        label_lines = [
            f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for cls, x, y, w, h in objects
        ]
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="ascii"
        )
        split_images[split] += 1
        split_instances[split] += len(objects)
        split_classes[split].update(CLASSES[obj[0]] for obj in objects)
        split_class_images[split][primary_class] += 1

    yaml_lines = [
        f"path: {OUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (OUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    stats = {
        "source_pairs": len(common),
        "used_images": len(assigned),
        "classes": CLASSES,
        "seed": SEED,
        "split_images": dict(split_images),
        "split_instances": dict(split_instances),
        "split_class_instances": {
            split: dict(counts) for split, counts in split_classes.items()
        },
        "split_class_images": {
            split: dict(counts) for split, counts in split_class_images.items()
        },
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "normalized_labels": {"10_yaozhed": "10_yaozhe", "d": "dropped"},
        "dropped_labels": dict(sorted(dropped_labels.items())),
    }
    (OUT / "gc10_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("GC10_EXPORT_OK")


if __name__ == "__main__":
    main()
