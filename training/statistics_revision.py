# -*- coding: utf-8 -*-
"""Seed-matched paired analysis for the revised NEU-DET study."""

from __future__ import annotations

import json
import os
import statistics as st

from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
YOLO = json.load(open(os.path.join(BASE, "yolo_matrix_summary.json"), encoding="utf-8"))


def paired(clean_keys, condition_keys):
    diffs50 = [
        (YOLO[c]["mAP50"] - YOLO[b]["mAP50"]) * 100
        for b, c in zip(clean_keys, condition_keys)
    ]
    diffs95 = [
        (YOLO[c]["mAP5095"] - YOLO[b]["mAP5095"]) * 100
        for b, c in zip(clean_keys, condition_keys)
    ]
    result = {
        "delta50_values": [round(x, 4) for x in diffs50],
        "delta50_mean": round(st.mean(diffs50), 4),
        "delta50_sd": round(st.stdev(diffs50), 4) if len(diffs50) > 1 else None,
        "delta95_values": [round(x, 4) for x in diffs95],
        "delta95_mean": round(st.mean(diffs95), 4),
        "delta95_sd": round(st.stdev(diffs95), 4) if len(diffs95) > 1 else None,
    }
    if len(diffs50) > 1:
        t50 = stats.ttest_rel(
            [YOLO[c]["mAP50"] for c in condition_keys],
            [YOLO[b]["mAP50"] for b in clean_keys],
        )
        t95 = stats.ttest_rel(
            [YOLO[c]["mAP5095"] for c in condition_keys],
            [YOLO[b]["mAP5095"] for b in clean_keys],
        )
        result["ttest50"] = {"statistic": float(t50.statistic), "pvalue": float(t50.pvalue)}
        result["ttest95"] = {"statistic": float(t95.statistic), "pvalue": float(t95.pvalue)}
    return result


clean3 = ["clean_640", "clean_640_s27", "clean_640_s28"]
report = {
    "jitter05_vs_clean_seed2026": paired(["clean_640"], ["jit05_640"]),
    "jitter10_vs_clean_seed2026": paired(["clean_640"], ["jit10_640"]),
    "drop20_vs_clean_seed2026": paired(["clean_640"], ["drop20_640"]),
    "drop40_vs_clean_seed2026": paired(["clean_640"], ["drop40_640"]),
    "jitter20_paired_3seeds": paired(clean3, ["jit20_640", "jit20_640_s27", "jit20_640_s28"]),
    "drop60_paired_3seeds": paired(clean3, ["drop60_640", "drop60_640_s27", "drop60_640_s28"]),
    "resolution_320_paired_3seeds": paired(clean3, ["clean_320", "clean_320_s27", "clean_320_s28"]),
}

json_path = os.path.join(BASE, "statistics_revision.json")
md_path = os.path.join(BASE, "statistics_revision.md")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)

lines = ["# Seed-matched paired analysis", ""]
labels = {
    "jitter05_vs_clean_seed2026": "5% jitter vs seed-matched clean",
    "jitter10_vs_clean_seed2026": "10% jitter vs seed-matched clean",
    "drop20_vs_clean_seed2026": "20% instance omission vs seed-matched clean",
    "drop40_vs_clean_seed2026": "40% instance omission vs seed-matched clean",
    "jitter20_paired_3seeds": "20% jitter, three-seed paired",
    "drop60_paired_3seeds": "60% instance omission, three-seed paired",
    "resolution_320_paired_3seeds": "320 vs 640 clean, three-seed paired",
}
for key, label in labels.items():
    row = report[key]
    text = f"- {label}: ΔmAP50={row['delta50_mean']:+.2f}"
    if row["delta50_sd"] is not None:
        text += f" ± {row['delta50_sd']:.2f}"
    if "ttest50" in row:
        pvalue = row["ttest50"]["pvalue"]
        ptext = "< 0.001" if pvalue < 0.001 else f"= {pvalue:.3f}"
        text += f", paired t-test p {ptext}"
    lines.append(text)
with open(md_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print(json.dumps(report, ensure_ascii=False, indent=2))
