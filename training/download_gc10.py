# -*- coding: utf-8 -*-
"""Download and validate the public GC10-DET VOC mirror used for external checks."""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "gc10_det_data" / "VOC2007"
TREE_API = "https://api.github.com/repos/wyx19980727/GC10-DET-VOC/git/trees/main"
RAW = "https://raw.githubusercontent.com/wyx19980727/GC10-DET-VOC/main/VOC2007"
PER_CLASS = int(os.environ.get("GC10_PER_CLASS", "0"))
WORKERS = int(os.environ.get("GC10_WORKERS", "32"))
HEADERS = {"User-Agent": "Codex-GC10-downloader/1.0"}


def list_tree():
    response = requests.get(
        TREE_API,
        params={"recursive": "1"},
        headers=HEADERS,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["tree"]


def tree_items(tree, name, suffix):
    prefix = f"VOC2007/{name}/"
    return [
        {
            "name": Path(x["path"]).name,
            "size": x.get("size", 0),
            "download_url": f"{RAW}/{name}/{Path(x['path']).name}",
        }
        for x in tree
        if x.get("type") == "blob"
        and x["path"].startswith(prefix)
        and x["path"].endswith(suffix)
    ]


def select(items):
    if not PER_CLASS:
        return items
    counts = {}
    selected = []
    for item in items:
        match = re.match(r"([^_]+)_", item["name"])
        group = match.group(1) if match else "unknown"
        if counts.get(group, 0) >= PER_CLASS:
            continue
        counts[group] = counts.get(group, 0) + 1
        selected.append(item)
    return selected


def download_one(item, subdir):
    destination = OUT / subdir / item["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("size", 0)
    if destination.exists() and destination.stat().st_size == expected:
        return destination, 0
    url = item["download_url"] or f"{RAW}/{subdir}/{item['name']}"
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(20, 120))
            response.raise_for_status()
            data = response.content
            if expected and len(data) != expected:
                raise IOError(f"size mismatch {len(data)} != {expected}")
            destination.write_bytes(data)
            return destination, 1
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    print("Listing GC10-DET files...", flush=True)
    tree = list_tree()
    images = select(tree_items(tree, "JPEGImages", ".jpg"))
    annotations = select(tree_items(tree, "Annotations", ".xml"))
    image_stems = {Path(x["name"]).stem for x in images}
    annotation_stems = {Path(x["name"]).stem for x in annotations}
    common = sorted(image_stems & annotation_stems)
    images = [x for x in images if Path(x["name"]).stem in common]
    annotations = [x for x in annotations if Path(x["name"]).stem in common]
    print(
        f"Selected {len(images)} images, {len(annotations)} annotations, "
        f"missing pairs={len(image_stems ^ annotation_stems)}",
        flush=True,
    )

    tasks = [(x, "Annotations") for x in annotations] + [(x, "JPEGImages") for x in images]
    downloaded = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(download_one, item, subdir) for item, subdir in tasks]
        for index, future in enumerate(cf.as_completed(futures), 1):
            _, created = future.result()
            downloaded += created
            if index % 100 == 0 or index == len(futures):
                print(f"progress {index}/{len(futures)} downloaded_this_run={downloaded}", flush=True)

    bad = []
    for item in annotations:
        path = OUT / "Annotations" / item["name"]
        try:
            root = ET.parse(path).getroot()
            if root.find("object") is None:
                bad.append(item["name"])
        except Exception as exc:
            bad.append(f"{item['name']}: {exc}")
    if bad:
        print("INVALID_ANNOTATIONS", bad[:20], flush=True)
        sys.exit(1)
    print("GC10_DOWNLOAD_OK", flush=True)


if __name__ == "__main__":
    main()
