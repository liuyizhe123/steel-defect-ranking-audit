# -*- coding: utf-8 -*-
"""Independent final audit for data, manuscript, DOCX, checkpoint and package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics as st
import sys
import zipfile

import torch
from docx import Document
from PIL import Image
from pypdf import PdfReader

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PKG = os.path.join(os.path.dirname(ROOT), "论文投稿包")
problems = []


def check(condition, message):
    if not condition:
        problems.append(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_text(path):
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


yolo = json.load(open(os.path.join(BASE, "yolo_matrix_summary.json"), encoding="utf-8"))
lsc = json.load(open(os.path.join(BASE, "lsc_pt_summary.json"), encoding="utf-8"))
report = json.load(open(os.path.join(BASE, "pt_final_results.json"), encoding="utf-8"))
gc10 = json.load(open(os.path.join(BASE, "gc10_summary.json"), encoding="utf-8"))
cross = json.load(open(os.path.join(BASE, "cross_model_summary.json"), encoding="utf-8"))
test_eval = json.load(open(os.path.join(BASE, "test_annotation_eval_summary.json"), encoding="utf-8"))
gc10_test_eval = json.load(
    open(os.path.join(BASE, "gc10_test_annotation_eval_summary.json"), encoding="utf-8")
)
bootstrap = json.load(open(os.path.join(BASE, "bootstrap_ci.json"), encoding="utf-8"))
revision = json.load(open(os.path.join(BASE, "revision_report.json"), encoding="utf-8"))
review_report = json.load(open(os.path.join(BASE, "review_report.json"), encoding="utf-8"))
review_complete = json.load(
    open(os.path.join(BASE, "review_followup_complete.json"), encoding="utf-8-sig")
)
efficiency = json.load(open(os.path.join(BASE, "efficiency_summary.json"), encoding="utf-8"))
qualitative = json.load(open(os.path.join(BASE, "qualitative_summary.json"), encoding="utf-8"))
round2_summary = json.load(open(os.path.join(BASE, "review_round2_summary.json"), encoding="utf-8"))
round2_report = json.load(open(os.path.join(BASE, "review_round2_report.json"), encoding="utf-8"))
round2_post_complete = json.load(
    open(os.path.join(BASE, "review_round2_post_complete.json"), encoding="utf-8")
)
memory_repeats = json.load(
    open(os.path.join(BASE, "efficiency_memory_repeats.json"), encoding="utf-8")
)
effect_ladder = json.load(
    open(os.path.join(BASE, "effect_ladder_summary.json"), encoding="utf-8")
)
gc10_stats = json.load(
    open(os.path.join(ROOT, "ultralytics_data", "gc10", "gc10_stats.json"), encoding="utf-8")
)

# 1. Raw architecture results and independent delta checks.
expected_arch = {
    "baseline": (73.33, 39.84),
    "v8n_fwn": (69.42, 37.53),
    "v8n_nwd": (71.54, 39.46),
    "v8n_focaler": (73.39, 40.01),
    "v8n_wiou": (72.01, 39.39),
    "v8n_hybrid": (73.42, 40.78),
    "escm_ciou": (70.77, 40.15),
    "lsc_ciou": (70.27, 31.11),
    "lsc_fwn": (70.10, 30.51),
    "saf": (66.64, 28.33),
}
for name, (m50, m95) in expected_arch.items():
    row = report["architecture_loss"][name]
    check(abs(row["mAP50"] - m50) < 1e-9, f"{name}: report mAP50 mismatch")
    check(abs(row["mAP5095"] - m95) < 1e-9, f"{name}: report mAP5095 mismatch")
    expected_delta50 = round(m50 - expected_arch["baseline"][0], 2)
    expected_delta95 = round(m95 - expected_arch["baseline"][1], 2)
    check(row["delta50"] == expected_delta50, f"{name}: delta50 mismatch")
    check(row["delta5095"] == expected_delta95, f"{name}: delta5095 mismatch")

check(
    report["architecture_loss"]["escm_ciou"]["params"] == 3034898,
    "ESCM-only parameter count mismatch",
)
for name in ["baseline", "v8n_fwn", "v8n_nwd", "v8n_focaler", "v8n_wiou", "v8n_hybrid"]:
    check(
        report["architecture_loss"][name]["params"] == 3012018,
        f"{name}: parameter count mismatch",
    )

v3 = lsc["lsc_fwn_pt_v3"]
check(abs(v3["mAP50"] * 100 - 70.10) < 0.006, "v3 raw mAP50 mismatch")
check(abs(v3["mAP5095"] * 100 - 30.51) < 0.006, "v3 raw mAP5095 mismatch")
check(v3["params"] == 977778, "v3 parameter count mismatch")

expected_gc10 = {
    "gc10_v8n_ciou": (59.80, 29.74, 3012798),
    "gc10_v8n_fwn": (58.55, 29.75, 3012798),
    "gc10_lsc_ciou": (58.24, 28.16, 978430),
    "gc10_lsc_fwn": (53.41, 25.91, 978430),
}
for name, (m50, m95, params) in expected_gc10.items():
    row = gc10[name]
    check(abs(row["mAP50"] * 100 - m50) < 0.006, f"{name}: GC10 mAP50 mismatch")
    check(abs(row["mAP5095"] * 100 - m95) < 0.006, f"{name}: GC10 mAP5095 mismatch")
    check(row["params"] == params, f"{name}: GC10 parameter count mismatch")

expected_cross = {
    "yolov10n_clean_640": (70.36, 38.73),
    "yolov10n_jit10_640": (59.33, 29.32),
    "yolov10n_drop40_640": (63.56, 34.73),
}
for name, (m50, m95) in expected_cross.items():
    row = cross[name]
    check(abs(row["mAP50"] * 100 - m50) < 0.006, f"{name}: cross-model mAP50 mismatch")
    check(abs(row["mAP5095"] * 100 - m95) < 0.006, f"{name}: cross-model mAP5095 mismatch")
    check(row["params"] == 2709380, f"{name}: cross-model parameter count mismatch")

check(gc10_stats["used_images"] == 2280, "GC10 full set does not contain 2280 images")
check(sum(gc10_stats["split_images"].values()) == 2280, "GC10 split mismatch")
check(gc10_stats["split_images"] == {"train": 1596, "val": 344, "test": 340}, "GC10 split counts mismatch")
check(len(gc10_stats["classes"]) == 10, "GC10 class count mismatch")
check(
    abs(test_eval["yolov8n_clean"]["test_drop40"]["mAP50"] * 100 - 56.34) < 0.006,
    "test-label omission result mismatch",
)
check(
    abs(test_eval["yolov8n_clean"]["test_jit20"]["mAP50"] * 100 - 31.91) < 0.006,
    "test-label jitter result mismatch",
)
check(
    abs(gc10_test_eval["test_jit20"]["mAP50"] * 100 - 24.31) < 0.006,
    "GC10 test-label jitter result mismatch",
)
check(
    abs(gc10_test_eval["test_drop40"]["mAP50"] * 100 - 52.96) < 0.006,
    "GC10 test-label omission result mismatch",
)
check(bootstrap["replicates"] == 1000, "bootstrap replicate count mismatch")
check(set(revision) == {
    "supplemental_architecture_loss",
    "annotation_interaction",
    "architecture_interaction",
    "gc10_external_check",
    "cross_model_annotation",
    "test_annotation_noise",
    "gc10_test_annotation_noise",
    "bootstrap_intervals",
}, "revision report section mismatch")
review_status = review_report.get("status", {})
check(review_status.get("review_matrix_done") == 29, "review matrix is incomplete")
check(review_status.get("review_matrix_required") == 29, "review matrix requirement mismatch")
check(review_status.get("test_realization_seeds_done") == 4, "test realization seeds are incomplete")
check(review_status.get("rtdetr_done") == 3, "RT-DETR matrix is incomplete")
check(review_status.get("efficiency_done") is True, "efficiency results are incomplete")
check(review_status.get("qualitative_done") is True, "qualitative results are incomplete")
check(review_complete.get("status") == "complete", "review follow-up completion marker missing")
check(round2_post_complete.get("status") == "complete", "review round-2 postprocessing incomplete")
check(len(round2_summary) == 5, "review round-2 summary must contain five training jobs")
check(
    round2_report["omission_realisation_robustness"]["positive_count"] == 1,
    "review round-2 reversal count mismatch",
)
check(
    abs(
        round2_report["omission_realisation_robustness"]["mean_margin"]
        - (-0.51)
    )
    < 0.01,
    "review round-2 mean margin mismatch",
)
check(
    abs(round2_summary["capacity_p3p4_s26"]["params"] - 2924140) == 0,
    "review round-2 capacity parameter count mismatch",
)
for name, row in memory_repeats["models"].items():
    check(len(row["values"]) == 3, f"memory repeat count mismatch: {name}")

# 2. Independent seed statistics from raw results.
seed_groups = {
    "clean640": ["clean_640", "clean_640_s27", "clean_640_s28"],
    "drop60": ["drop60_640", "drop60_640_s27", "drop60_640_s28"],
    "jit20": ["jit20_640", "jit20_640_s27", "jit20_640_s28"],
    "clean320": ["clean_320", "clean_320_s27", "clean_320_s28"],
}
for name, keys in seed_groups.items():
    vals = [yolo[k]["mAP50"] * 100 for k in keys]
    vals95 = [yolo[k]["mAP5095"] * 100 for k in keys]
    mean50, sd50 = round(st.mean(vals), 2), round(st.stdev(vals), 2)
    mean95, sd95 = round(st.mean(vals95), 2), round(st.stdev(vals95), 2)
    stored = report["seed_replication"][name]
    check(stored["mAP50_mean"] == mean50, f"{name}: mean50 mismatch")
    check(stored["mAP50_sd"] == sd50, f"{name}: sd50 mismatch")
    check(stored["mAP5095_mean"] == mean95, f"{name}: mean95 mismatch")
    check(stored["mAP5095_sd"] == sd95, f"{name}: sd95 mismatch")

# Single-seed annotation effects must use the same seed's clean control.
clean50 = yolo["clean_640"]["mAP50"] * 100
clean95 = yolo["clean_640"]["mAP5095"] * 100
for key in ["jit05_640", "jit10_640", "jit20_640", "drop20_640", "drop40_640", "drop60_640"]:
    row = report["annotation"][key]
    expected50 = round(yolo[key]["mAP50"] * 100 - clean50, 2)
    expected95 = round(yolo[key]["mAP5095"] * 100 - clean95, 2)
    check(row["delta50"] == expected50, f"{key}: seed-matched delta50 mismatch")
    check(row["delta5095"] == expected95, f"{key}: seed-matched delta5095 mismatch")

# 3. Raw checkpoint integrity and persisted FWN state.
checkpoint = os.path.join(ROOT, "lsc_pt_runs", "lsc_fwn_pt_v3", "weights", "best.pt")
ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = ckpt.get("ema") or ckpt["model"]
check(sum(p.numel() for p in model.parameters()) == 977778, "v3 checkpoint parameter count mismatch")
check("_fwn_lbar" in model.state_dict(), "v3 checkpoint missing _fwn_lbar")
check(torch.isfinite(model.state_dict()["_fwn_lbar"]).all(), "v3 _fwn_lbar is non-finite")

# 4. Manuscript text, exact final table, optimizer and stale-token checks.
manuscript = open(os.path.join(ROOT, "manuscript.md"), encoding="utf-8").read()
cover = open(os.path.join(ROOT, "cover_letter.md"), encoding="utf-8").read()
highlights = open(os.path.join(ROOT, "highlights.md"), encoding="utf-8").read()
supplement_text = open(
    os.path.join(ROOT, "supplementary_material.md"),
    encoding="utf-8",
).read()
abstract_match = re.search(
    r"## Abstract\s+(.+?)\n\n\*\*Keywords", manuscript, flags=re.DOTALL
)
check(abstract_match is not None, "manuscript abstract block missing")
if abstract_match:
    abstract = re.sub(r"[#*_`]", "", abstract_match.group(1)).strip()
    check(len(abstract.split()) <= 250, "manuscript abstract exceeds 250 words")
highlight_lines = [
    line.lstrip("-* ").strip()
    for line in highlights.splitlines()
    if re.match(r"^\s*[-*]\s+", line)
]
check(3 <= len(highlight_lines) <= 5, "highlights must contain 3 to 5 bullets")
check(all(len(line) <= 85 for line in highlight_lines), "highlight exceeds 85 characters")
check(
    "| ESCM+SAF + FWN | 1 | 0.978 | 7.3 | 70.10 | −3.23 | 30.51 | −9.33 |" in manuscript,
    "manuscript final Table 1 row mismatch",
)
check("AdamW" in manuscript and "0.001" in manuscript, "manuscript optimizer description missing")
for token in [
    "70.14",
    "30.45",
    "−3.19",
    "−9.39",
    "81.55",
    "82.52",
    "NEU-DET only",
]:
    check(token not in manuscript, f"manuscript contains stale token: {token}")
for token in [
    "dominant factor",
    "strongest controllable determinant",
    "annotation quality, not architecture, is the correct target",
]:
    check(token.lower() not in manuscript.lower(), f"manuscript contains overclaim: {token}")
check(
    "Annotation-Dependent Model Rankings and Evaluation-Label Sensitivity" in cover,
    "cover letter title is not updated",
)
check(
    "one in three" in cover and "−0.51 ± 0.78" in cover,
    "cover letter missing draw-level reversal result",
)
check("70.14%" not in cover, "cover letter still contains v2 mAP50")
for token in ["59.80", "53.41", "70.36", "10.32 ± 0.76", "56.18 ± 0.82", "54.06 ± 1.34", "2,280", "1,000"]:
    check(token in manuscript, f"manuscript missing final GC10/interaction token: {token}")
for token in [
    "70.0%",
    "exceed 25%",
    "2401.10525",
    "7:2:1",
    "one primary plus four additional realizations",
    "72.57 ± 1.58",
    "71.31 ± 0.90",
    "72.07 ± 2.72",
    "73.63 ± 0.30",
    "72.79 ± 1.10",
    "−1.29 ± 1.32",
    "−2.55 ± 0.65",
    "76.79 ± 0.64",
    "77.29 ± 0.33",
    "75.22 ± 0.06",
    "+1.03 ± 0.18",
    "−9.66 ± 0.49",
]:
    check(token in manuscript, f"manuscript missing reviewer-response token: {token}")
for token in [
    "No architecture variant improved",
    "rankings are stable",
    "full annotated GC10-DET",
]:
    check(token not in manuscript, f"manuscript contains stale claim: {token}")
check("DRAFT NOTE" not in manuscript, "source manuscript contains DRAFT NOTE")
check(
    "A Case Study of Annotation-Dependent Model Rankings and Evaluation-Label Sensitivity" in manuscript,
    "manuscript title is not updated",
)
check("https://github.com/xxx/" not in manuscript, "manuscript contains placeholder repository URL")
for token in ["73.9%", "2301.02586", "use one seeded realization"]:
    check(token not in manuscript, f"manuscript contains stale reviewer-response token: {token}")
figure_sources = [
    ("figs/fig_scale_distribution.png", "Fig1_缺陷尺度分布.png"),
    ("figs/fig2_architecture.png", "Fig2_模块结构.png"),
    ("figs/fig7_complexity.png", "Fig3_复杂度与精度.png"),
    ("figs/fig5_annotation.png", "Fig4_标注扰动曲线.png"),
    ("figs/fig6_resolution.png", "Fig5_输入分辨率对比.png"),
    ("figs/fig5_validation.png", "Fig6_综合验证.png"),
    ("figs/fig8_standard_split.png", "Fig7_标准切分复核.png"),
    ("figs/fig8_qualitative.png", "Fig8_定性失败案例.png"),
    ("figs/fig9_efficiency.png", "Fig9_效率与显存.png"),
    ("figs/fig10_effect_ladder.png", "Fig10_效应梯与脆弱度.png"),
]
for source, _ in figure_sources:
    check(os.path.isfile(os.path.join(ROOT, source)), f"figure source missing: {source}")

# Reviewer-response results must be present verbatim in the manuscript.
for row in review_report.get("p2_p5_ablation", {}).values():
    token = f"{row['mAP50']:.2f}"
    check(token in manuscript, f"P2/P5 result missing from manuscript: {token}")
for row in review_report.get("fwn_sensitivity", {}).values():
    token = f"{row['mAP50']:.2f}"
    check(
        token in supplement_text,
        f"FWN sensitivity result missing from supplement: {token}",
    )
for condition, row in review_report.get("ranking_stability", {}).items():
    token = f"{row['kendall_tau_vs_clean']:.3f}"
    check(token in manuscript, f"ranking tau missing from manuscript: {condition}={token}")
for models in review_report.get("evaluation_label_realizations", {}).values():
    for conditions in models.values():
        for condition, row in conditions.items():
            if row:
                token = f"{row['mean']:.2f} ± {row['sd']:.2f}"
                check(
                    token in manuscript,
                    f"repeated realization result missing from manuscript: {condition}={token}",
                )
for name, row in efficiency.get("models", {}).items():
    token = f"{row['batch1']['throughput_images_per_s']:.1f}"
    check(token in manuscript, f"efficiency result missing from manuscript: {name}={token}")
check(
    qualitative.get("composite") == "fig8_qualitative.png",
    "qualitative composite filename mismatch",
)
qualitative_image = Image.open(os.path.join(ROOT, "figs", "fig8_qualitative.png"))
check(
    qualitative_image.size == (2400, 2760),
    "qualitative composite must be rendered at 2400x2760",
)
check(
    qualitative.get("render", {}).get("ground_truth_boxes") is True,
    "qualitative summary does not record ground-truth boxes",
)
check(
    "single-seed model-annotation interaction results" not in manuscript.lower(),
    "manuscript still labels the replicated interaction study as single-seed",
)
for token in ["one reversal in three", "−0.51 ± 0.78", "−1.06", "−0.86", "67.29", "2.11", "7.47"]:
    check(token in manuscript, f"round-2 result missing from manuscript: {token}")
for token in ["177.2", "345.8", "345.9"]:
    check(token in manuscript, f"fresh-process memory result missing from manuscript: {token}")
for token in ["Harmful/5", "0.01–0.91", "fresh-process repeats", "2,924,140"]:
    check(token in manuscript, f"third-round consistency token missing: {token}")
for token in [
    "31.71 ± 1.08 and 56.18 ± 0.82%",
    "rather than the primary seed-2026 values of 31.91 and 56.34%",
]:
    check(token in manuscript, f"Fig. 6 caption consistency token missing: {token}")
for token in [
    "realized omissions of 16.7%, 30.4% and 43.3%",
    "rendered at 2400×2760 px",
    "only missed ground-truth targets (MISS) are labeled",
    "Artificial Intelligence Review 59 (2026) Article 107",
    "Effect-size ladder and ranking fragility",
    "ranking-fragility ratio",
    "ESCM+SAF+CIoU has a ratio of 3.91",
]:
    check(token in manuscript, f"fifth-round consistency token missing: {token}")
check(
    len(effect_ladder.get("effect_ladder", [])) == 6,
    "effect ladder must contain six principal interventions",
)
check(
    len(effect_ladder.get("ranking_fragility", [])) == 5,
    "ranking fragility must contain five replicated model comparisons",
)

# 5. DOCX text must agree with Markdown and exclude stale values.
docx_paths = {
    "manuscript": os.path.join(ROOT, "steel-defect-empirical_manuscript.docx"),
    "cover": os.path.join(ROOT, "cover_letter.docx"),
    "highlights": os.path.join(ROOT, "highlights.docx"),
    "response": os.path.join(ROOT, "response_to_reviewers.docx"),
    "supplement": os.path.join(ROOT, "supplementary_material.docx"),
}
doc_texts = {name: doc_text(path) for name, path in docx_paths.items()}
normalized_docx_text = " ".join(doc_texts["manuscript"].split())
check("DRAFT NOTE" not in doc_texts["manuscript"], "DOCX manuscript contains DRAFT NOTE")
for token in [
    "70.10",
    "30.51",
    "AdamW",
    "0.001",
    "31.71 ± 1.08 and 56.18 ± 0.82%",
    "rather than the primary seed-2026 values of 31.91 and 56.34%",
    "realized omissions of 16.7%, 30.4% and 43.3%",
    "rendered at 2400×2760 px",
    "only missed ground-truth targets (MISS) are labeled",
    "Artificial Intelligence Review 59 (2026) Article 107",
    "Effect-size ladder and ranking fragility",
    "ranking-fragility ratio",
    "ESCM+SAF+CIoU has a ratio of 3.91",
    "112.4",
    "1,000",
]:
    check(token in doc_texts["manuscript"], f"DOCX manuscript missing: {token}")
for token in [
    "70.0%",
    "exceed 25%",
    "2401.10525",
    "7:2:1",
    "one primary plus four additional realizations",
    "72.57 ± 1.58",
    "71.31 ± 0.90",
    "72.07 ± 2.72",
    "76.79 ± 0.64",
    "77.29 ± 0.33",
    "75.22 ± 0.06",
]:
    check(token in normalized_docx_text, f"DOCX manuscript missing reviewer-response token: {token}")
for token in ["70.14", "30.45", "81.55", "82.52"]:
    check(token not in doc_texts["manuscript"], f"DOCX manuscript contains stale token: {token}")
for token in ["73.9%", "2301.02586", "use one seeded realization"]:
    check(
        token not in doc_texts["manuscript"],
        f"DOCX manuscript contains stale reviewer-response token: {token}",
    )
check(
    "Annotation-Dependent Model Rankings and Evaluation-Label Sensitivity" in doc_texts["cover"],
    "DOCX cover letter title is not updated",
)
check("70.14%" not in doc_texts["cover"], "DOCX cover letter contains 70.14%")
for token in [
    "Third-Round Response",
    "Fig. 9(b)",
    "0.01-0.91",
    "LSCDetectionTrainer",
]:
    check(token in doc_texts["response"], f"DOCX response letter missing: {token}")
for token in [
    "Supplementary Tables S1-S3",
    "2400x2760",
]:
    check(token in doc_texts["response"], f"DOCX response letter missing: {token}")
for token in ["Table S1", "Table S2", "Table S3"]:
    check(token in doc_texts["supplement"], f"DOCX supplement missing: {token}")
check(
    "No new original steel images were acquired" in doc_texts["cover"],
    "DOCX cover letter contains the stale data statement",
)

pdf_path = os.path.join(ROOT, "render", "manuscript.pdf")
pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
normalized_pdf_text = " ".join(pdf_text.split())
for token in [
    "70.0%",
    "exceed 25%",
    "2401.10525",
    "7:2:1",
    "one primary plus four additional realizations",
    "72.57 ± 1.58",
    "71.31 ± 0.90",
    "72.07 ± 2.72",
    "76.79 ± 0.64",
    "77.29 ± 0.33",
    "75.22 ± 0.06",
]:
    check(token in normalized_pdf_text, f"PDF manuscript missing reviewer-response token: {token}")
for token in ["73.9%", "2301.02586", "use one seeded realization"]:
    check(
        token not in pdf_text,
        f"PDF manuscript contains stale reviewer-response token: {token}",
    )

# 6. Package must contain byte-identical source documents and final PDF.
package_pairs = [
    (os.path.join(ROOT, "README.md"), os.path.join(PKG, "00_说明", "README.md")),
    (
        os.path.join(ROOT, "requirements.txt"),
        os.path.join(PKG, "00_说明", "requirements.txt"),
    ),
    (
        os.path.join(ROOT, "reproduce.ps1"),
        os.path.join(PKG, "00_说明", "reproduce.ps1"),
    ),
    (os.path.join(ROOT, "manuscript.md"), os.path.join(PKG, "01_论文正文", "manuscript.md")),
    (docx_paths["manuscript"], os.path.join(PKG, "01_论文正文", "论文正文.docx")),
    (os.path.join(ROOT, "render", "manuscript.pdf"), os.path.join(PKG, "01_论文正文", "论文正文.pdf")),
    (docx_paths["cover"], os.path.join(PKG, "02_投稿材料", "cover_letter.docx")),
    (docx_paths["highlights"], os.path.join(PKG, "02_投稿材料", "highlights.docx")),
    (os.path.join(ROOT, "cover_letter.md"), os.path.join(PKG, "02_投稿材料", "cover_letter.md")),
    (os.path.join(ROOT, "highlights.md"), os.path.join(PKG, "02_投稿材料", "highlights.md")),
    (
        os.path.join(ROOT, "response_to_reviewers.md"),
        os.path.join(PKG, "02_投稿材料", "response_to_reviewers.md"),
    ),
    (
        docx_paths["response"],
        os.path.join(PKG, "02_投稿材料", "response_to_reviewers.docx"),
    ),
    (
        os.path.join(ROOT, "render_aux_round3", "response_to_reviewers.pdf"),
        os.path.join(PKG, "02_投稿材料", "response_to_reviewers.pdf"),
    ),
    (
        os.path.join(ROOT, "render_aux_round3", "cover_letter.pdf"),
        os.path.join(PKG, "02_投稿材料", "cover_letter.pdf"),
    ),
    (
        os.path.join(ROOT, "render_aux_round3", "highlights.pdf"),
        os.path.join(PKG, "02_投稿材料", "highlights.pdf"),
    ),
    (
        docx_paths["supplement"],
        os.path.join(PKG, "02_投稿材料", "Supplementary_Material.docx"),
    ),
    (
        os.path.join(ROOT, "supplementary_material.md"),
        os.path.join(PKG, "02_投稿材料", "Supplementary_Material.md"),
    ),
    (
        os.path.join(ROOT, "render_aux_round3", "supplementary_material.pdf"),
        os.path.join(PKG, "02_投稿材料", "Supplementary_Material.pdf"),
    ),
    (
        os.path.join(BASE, "review_round2_report.json"),
        os.path.join(PKG, "04_实测数据", "第二轮审稿汇总.json"),
    ),
    (
        os.path.join(BASE, "review_round2_report.md"),
        os.path.join(PKG, "04_实测数据", "第二轮审稿汇总.md"),
    ),
    (
        os.path.join(BASE, "efficiency_memory_repeats.json"),
        os.path.join(PKG, "04_实测数据", "效率显存重复测量.json"),
    ),
    (
        os.path.join(BASE, "reproduce_project.py"),
        os.path.join(PKG, "05_代码", "reproduce_project.py"),
    ),
    (
        os.path.join(BASE, "README.md"),
        os.path.join(PKG, "05_代码", "README.md"),
    ),
    (
        os.path.join(BASE, "make_effect_ladder.py"),
        os.path.join(PKG, "05_代码", "make_effect_ladder.py"),
    ),
    (
        os.path.join(BASE, "effect_ladder_summary.json"),
        os.path.join(PKG, "04_实测数据", "效应梯与脆弱度.json"),
    ),
]
for source, packaged_name in figure_sources:
    package_pairs.append(
        (
            os.path.join(ROOT, source),
            os.path.join(PKG, "03_图表", packaged_name),
        )
    )
for source, packaged in package_pairs:
    check(os.path.isfile(packaged), f"package missing: {packaged}")
    if os.path.isfile(packaged):
        check(sha256(source) == sha256(packaged), f"package hash mismatch: {packaged}")

packaged_manuscript = os.path.join(PKG, "01_论文正文", "manuscript.md")
if os.path.isfile(packaged_manuscript):
    packaged_text = open(packaged_manuscript, encoding="utf-8").read()
    check("DRAFT NOTE" not in packaged_text, "packaged manuscript contains DRAFT NOTE")

manifest_path = os.path.join(PKG, "manifest.json")
if os.path.isfile(manifest_path):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    check(not manifest.get("missing"), "submission package manifest reports missing files")
else:
    problems.append("submission package manifest missing")
check(
    not os.path.exists(os.path.join(PKG, "STALE_BUILD.txt")),
    "submission package still contains STALE_BUILD.txt",
)
check(
    not os.path.exists(os.path.join(os.path.dirname(ROOT), "论文投稿包_STALE_DO_NOT_SUBMIT.txt")),
    "stale ZIP marker still exists",
)

archive_path = os.path.join(os.path.dirname(ROOT), "论文投稿包.zip")
check(os.path.isfile(archive_path), "submission ZIP missing")
if os.path.isfile(archive_path):
    check(os.path.getsize(archive_path) > 0, "submission ZIP is empty")
    with zipfile.ZipFile(archive_path) as archive:
        check(archive.testzip() is None, "submission ZIP contains a corrupt entry")
        names = set(archive.namelist())
        for required in (
            "论文投稿包/01_论文正文/论文正文.docx",
            "论文投稿包/01_论文正文/论文正文.pdf",
            "论文投稿包/00_说明/README.md",
            "论文投稿包/00_说明/reproduce.ps1",
            "论文投稿包/05_代码/reproduce_project.py",
            "论文投稿包/05_代码/README.md",
            "论文投稿包/02_投稿材料/response_to_reviewers.md",
            "论文投稿包/02_投稿材料/response_to_reviewers.docx",
            "论文投稿包/02_投稿材料/response_to_reviewers.pdf",
            "论文投稿包/02_投稿材料/Supplementary_Material.docx",
            "论文投稿包/02_投稿材料/Supplementary_Material.pdf",
            "论文投稿包/03_图表/Fig8_定性失败案例.png",
            "论文投稿包/03_图表/Fig9_效率与显存.png",
            "论文投稿包/03_图表/Fig10_效应梯与脆弱度.png",
            "论文投稿包/04_实测数据/第二轮审稿汇总.json",
            "论文投稿包/04_实测数据/效率显存重复测量.json",
        ):
            check(required in names, f"submission ZIP missing: {required}")

print("=== FINAL_AUDIT ===")
print("architecture:", expected_arch)
print("v3 checkpoint:", checkpoint)
print("seed groups checked:", len(seed_groups))
print("DOCX text checked:", len(doc_texts))
print("package hashes checked:", len(package_pairs))
if problems:
    print("FINAL_AUDIT_FAILED", len(problems))
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("FINAL_AUDIT_OK")
