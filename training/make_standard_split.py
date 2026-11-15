# -*- coding: utf-8 -*-
"""Create a leakage-aware 70:20:10 NEU-DET split from the full annotated set."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CLEAN = ROOT / "ultralytics_data" / "clean"
OUT = ROOT / "ultralytics_data" / "standard_721"
SUMMARY = BASE / "standard_split_summary.json"
SEED = 2026
NEAR_DUP_THRESHOLD = 2
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> np.uint64:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return np.uint64(value)


def class_name(path: Path) -> str:
    stem = path.stem
    name = stem.rsplit("_", 1)[0]
    if name not in CLASS_NAMES:
        raise ValueError(f"cannot infer class from {path.name}")
    return name


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    clean_images = []
    for split in ("train", "val", "test"):
        clean_images.extend(sorted((CLEAN / "images" / split).glob("*.jpg")))
    clean_images = sorted(clean_images)
    if len(clean_images) != 1800:
        raise SystemExit(f"expected 1800 NEU-DET images, found {len(clean_images)}")

    records = []
    for index, image_path in enumerate(clean_images):
        label_path = CLEAN / "labels" / "train" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            label_path = CLEAN / "labels" / "val" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            label_path = CLEAN / "labels" / "test" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label for {image_path.name}")
        records.append(
            {
                "index": index,
                "class": class_name(image_path),
                "image": image_path,
                "label": label_path,
                "sha256": file_sha256(image_path),
                "dhash": dhash64(image_path),
            }
        )

    groups_by_class = {}
    exact_duplicates = defaultdict(list)
    for record in records:
        exact_duplicates[record["sha256"]].append(record["image"].name)
    exact_duplicate_groups = [
        names for names in exact_duplicates.values() if len(names) > 1
    ]

    rng = random.Random(SEED)
    for name in CLASS_NAMES:
        members = [record for record in records if record["class"] == name]
        uf = UnionFind(len(members))
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                distance = int(members[left]["dhash"] ^ members[right]["dhash"]).bit_count()
                if distance <= NEAR_DUP_THRESHOLD:
                    uf.union(left, right)
        grouped = defaultdict(list)
        for position, record in enumerate(members):
            grouped[uf.find(position)].append(record)
        groups = list(grouped.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        groups_by_class[name] = groups

    assigned = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": 0.70, "val": 0.20, "test": 0.10}
    for name in CLASS_NAMES:
        total = sum(len(group) for group in groups_by_class[name])
        target_counts = {split: total * ratio for split, ratio in targets.items()}
        current = {"train": 0, "val": 0, "test": 0}
        for group in groups_by_class[name]:
            split = max(
                targets,
                key=lambda key: (
                    target_counts[key] - current[key],
                    -current[key] / max(target_counts[key], 1.0),
                ),
            )
            for record in group:
                assigned.append({**record, "split": split})
            current[split] += len(group)
            split_counts[split] += len(group)

    out_resolved = OUT.resolve()
    root_resolved = ROOT.resolve()
    if out_resolved == root_resolved or root_resolved not in out_resolved.parents:
        raise RuntimeError(f"refusing to replace unexpected path: {out_resolved}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    for record in assigned:
        split = record["split"]
        link_or_copy(record["image"], OUT / "images" / split / record["image"].name)
        link_or_copy(record["label"], OUT / "labels" / split / record["label"].name)

    data_yaml = "\n".join(
        [
            f"path: {OUT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 6",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
        ]
    )
    (OUT / "data.yaml").write_text(data_yaml + "\n", encoding="utf-8")

    near_groups = [
        [record["image"].name for record in group]
        for groups in groups_by_class.values()
        for group in groups
        if len(group) > 1
    ]
    cross_split_exact_duplicates = 0
    cross_split_near_duplicates = 0
    for left in range(len(assigned)):
        for right in range(left + 1, len(assigned)):
            if (
                assigned[left]["split"] == assigned[right]["split"]
                or assigned[left]["class"] != assigned[right]["class"]
            ):
                continue
            if assigned[left]["sha256"] == assigned[right]["sha256"]:
                cross_split_exact_duplicates += 1
            if (
                int(assigned[left]["dhash"] ^ assigned[right]["dhash"]).bit_count()
                <= NEAR_DUP_THRESHOLD
            ):
                cross_split_near_duplicates += 1

    summary = {
        "seed": SEED,
        "near_duplicate_dhash_threshold": NEAR_DUP_THRESHOLD,
        "total_images": len(assigned),
        "split_counts": split_counts,
        "per_class_split_counts": {
            name: {
                split: sum(
                    1
                    for record in assigned
                    if record["class"] == name and record["split"] == split
                )
                for split in ("train", "val", "test")
            }
            for name in CLASS_NAMES
        },
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_groups": near_groups,
        "cross_split_exact_duplicates": cross_split_exact_duplicates,
        "cross_split_near_duplicates": cross_split_near_duplicates,
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("STANDARD_SPLIT_OK")


if __name__ == "__main__":
    main()
