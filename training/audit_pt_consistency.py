# -*- coding: utf-8 -*-
"""Fail fast when manuscript values or reported training settings drift."""

import json
import os
import statistics as st
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
report = json.load(open(os.path.join(BASE, "pt_final_results.json"), encoding="utf-8"))
yolo = json.load(open(os.path.join(BASE, "yolo_matrix_summary.json"), encoding="utf-8"))
lsc = json.load(open(os.path.join(BASE, "lsc_pt_summary.json"), encoding="utf-8"))
seed = json.load(open(os.path.join(BASE, "seed_replication_summary.json"), encoding="utf-8"))
manuscript = open(os.path.join(ROOT, "manuscript.md"), encoding="utf-8").read()
cover = open(os.path.join(ROOT, "cover_letter.md"), encoding="utf-8").read()

problems = []
groups = {
    "baseline": [
        yolo["clean_640"]["mAP50"],
        yolo["clean_640_s27"]["mAP50"],
        yolo["clean_640_s28"]["mAP50"],
    ],
    "escm_ciou": [
        lsc["escm_ciou_pt"]["mAP50"],
        seed["escm_ciou_s27"]["mAP50"],
        seed["escm_ciou_s28"]["mAP50"],
    ],
    "lsc_ciou": [
        lsc["lsc_ciou_pt"]["mAP50"],
        seed["lsc_ciou_s27"]["mAP50"],
        seed["lsc_ciou_s28"]["mAP50"],
    ],
    "v8n_fwn": [
        lsc["v8n_fwn_pt_v2"]["mAP50"],
        seed["v8n_fwn_s27"]["mAP50"],
        seed["v8n_fwn_s28"]["mAP50"],
    ],
    "v8n_focaler": [
        lsc["v8n_focaler_pt"]["mAP50"],
        seed["v8n_focaler_s27"]["mAP50"],
        seed["v8n_focaler_s28"]["mAP50"],
    ],
    "v8n_hybrid": [
        lsc["v8n_hybrid_pt"]["mAP50"],
        seed["v8n_hybrid_s27"]["mAP50"],
        seed["v8n_hybrid_s28"]["mAP50"],
    ],
}
for name, values in groups.items():
    token = f"{st.mean(values) * 100:.2f} ± {st.stdev(values) * 100:.2f}"
    if token not in manuscript:
        problems.append(f"manuscript missing {name} multi-seed token {token}")

for token in [
    "76.79 ± 0.64",
    "77.29 ± 0.33",
    "75.22 ± 0.06",
    "+1.03 ± 0.18",
    "−9.66 ± 0.49",
    "70.16 ± 0.32",
    "+1.08 ± 0.64",
    "63.67 ± 0.98",
    "56.18 ± 0.82",
    "52.19 ± 1.45",
    "54.06 ± 1.34",
    "1.000",
    "0.333",
    "2.11",
    "7.47",
    "−0.51 ± 0.78",
    "−1.06",
    "−0.86",
    "345.8",
    "345.9",
    "177.2",
    "Harmful/5",
    "0.01–0.91",
    "2,924,140",
]:
    if token not in manuscript:
        problems.append(f"manuscript missing standard-split token: {token}")

for name in ["v8n_nwd", "v8n_wiou", "lsc_fwn", "saf"]:
    row = report["architecture_loss"][name]
    if f"{row['mAP50']:.2f}" not in manuscript:
        problems.append(f"manuscript missing {name} mAP50 {row['mAP50']:.2f}")
    if f"{row['mAP5095']:.2f}" not in manuscript:
        problems.append(f"manuscript missing {name} mAP5095 {row['mAP5095']:.2f}")

for text, label in [(manuscript, "manuscript"), (cover, "cover letter")]:
    if "AdamW" not in text and label == "manuscript":
        problems.append("manuscript does not report AdamW")
    if "0.001" not in text and label == "manuscript":
        problems.append("manuscript does not report lr 0.001")

forbidden = [
    "70.14",
    "30.45",
    "−3.19",
    "−9.39",
    "SGD at the full-schedule iteration count",
    "81.55",
    "82.52",
    "19.7 mAP",
    "20.7 points",
]
for token in forbidden:
    if token in manuscript:
        problems.append(f"forbidden stale token in manuscript: {token}")

overclaims = [
    "dominant factor",
    "strongest controllable",
    "binding constraint",
    "correct target for improvement",
    "where should effort be spent",
    "central result",
    "reversal is a stable",
    "annotation-dependent rather than universal",
]
for text, label in [(manuscript, "manuscript"), (cover, "cover letter")]:
    lower = text.lower()
    for token in overclaims:
        if token in lower:
            problems.append(f"overclaim in {label}: {token}")

for token in ["−1.44", "−6.47", "−2.35", "−3.48", "−25.83", "−7.60"]:
    if token not in manuscript:
        problems.append(f"seed-matched paired value missing: {token}")

for token in ["98.3", "110.7", "154.7", "177.1", "177.2", "345.8", "345.9"]:
    if token not in manuscript:
        problems.append(f"efficiency token missing: {token}")

if "73.86 ± 0.69%" not in cover:
    problems.append("cover letter missing three-seed baseline")
if "72.57 ± 1.58%" not in cover or "71.31 ± 0.90%" not in cover:
    problems.append("cover letter missing replicated architecture results")
if "A Case Study of Annotation-Dependent Model Rankings" not in manuscript:
    problems.append("manuscript title is not scoped as a case study")
if "one reversal in three" not in manuscript:
    problems.append("manuscript does not state draw-level reversal frequency")

if problems:
    print("PT_CONSISTENCY_FAILED")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)

print("PT_CONSISTENCY_OK")
