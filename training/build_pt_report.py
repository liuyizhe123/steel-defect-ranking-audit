# -*- coding: utf-8 -*-
"""Build the final PyTorch-only quantitative report used by the manuscript."""

from __future__ import annotations

import json
import os
import statistics as st

BASE = os.path.dirname(os.path.abspath(__file__))
YOLO = json.load(open(os.path.join(BASE, "yolo_matrix_summary.json"), encoding="utf-8"))
PT = json.load(open(os.path.join(BASE, "lsc_pt_summary.json"), encoding="utf-8"))
SEED = json.load(open(os.path.join(BASE, "seed_replication_summary.json"), encoding="utf-8"))


def pct(value: float) -> float:
    return round(value * 100, 2)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return round(st.mean(values), 2), round(st.stdev(values), 2)


base = YOLO["clean_640"]
rows = {
    "baseline": {
        "mAP50": pct(base["mAP50"]),
        "mAP5095": pct(base["mAP5095"]),
        "params": 3012018,
        "perClassAP50": [pct(x) for x in base["perClassAP50"]],
    },
    "v8n_fwn": {
        "mAP50": pct(PT["v8n_fwn_pt_v2"]["mAP50"]),
        "mAP5095": pct(PT["v8n_fwn_pt_v2"]["mAP5095"]),
        "params": PT["v8n_fwn_pt_v2"]["params"],
        "perClassAP50": [pct(x) for x in PT["v8n_fwn_pt_v2"]["perClassAP50"]],
    },
    "v8n_nwd": {
        "mAP50": pct(PT["v8n_nwd_pt"]["mAP50"]),
        "mAP5095": pct(PT["v8n_nwd_pt"]["mAP5095"]),
        "params": PT["v8n_nwd_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["v8n_nwd_pt"]["perClassAP50"]],
    },
    "v8n_focaler": {
        "mAP50": pct(PT["v8n_focaler_pt"]["mAP50"]),
        "mAP5095": pct(PT["v8n_focaler_pt"]["mAP5095"]),
        "params": PT["v8n_focaler_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["v8n_focaler_pt"]["perClassAP50"]],
    },
    "v8n_wiou": {
        "mAP50": pct(PT["v8n_wiou_pt"]["mAP50"]),
        "mAP5095": pct(PT["v8n_wiou_pt"]["mAP5095"]),
        "params": PT["v8n_wiou_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["v8n_wiou_pt"]["perClassAP50"]],
    },
    "v8n_hybrid": {
        "mAP50": pct(PT["v8n_hybrid_pt"]["mAP50"]),
        "mAP5095": pct(PT["v8n_hybrid_pt"]["mAP5095"]),
        "params": PT["v8n_hybrid_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["v8n_hybrid_pt"]["perClassAP50"]],
    },
    "escm_ciou": {
        "mAP50": pct(PT["escm_ciou_pt"]["mAP50"]),
        "mAP5095": pct(PT["escm_ciou_pt"]["mAP5095"]),
        "params": PT["escm_ciou_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["escm_ciou_pt"]["perClassAP50"]],
    },
    "lsc_ciou": {
        "mAP50": pct(PT["lsc_ciou_pt"]["mAP50"]),
        "mAP5095": pct(PT["lsc_ciou_pt"]["mAP5095"]),
        "params": PT["lsc_ciou_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["lsc_ciou_pt"]["perClassAP50"]],
    },
    "lsc_fwn": {
        "mAP50": pct(PT["lsc_fwn_pt_v3"]["mAP50"]),
        "mAP5095": pct(PT["lsc_fwn_pt_v3"]["mAP5095"]),
        "params": PT["lsc_fwn_pt_v3"]["params"],
        "perClassAP50": [pct(x) for x in PT["lsc_fwn_pt_v3"]["perClassAP50"]],
    },
    "saf": {
        "mAP50": pct(PT["saf_ciou_pt"]["mAP50"]),
        "mAP5095": pct(PT["saf_ciou_pt"]["mAP5095"]),
        "params": PT["saf_ciou_pt"]["params"],
        "perClassAP50": [pct(x) for x in PT["saf_ciou_pt"]["perClassAP50"]],
    },
}
for row in rows.values():
    row["delta50"] = round(row["mAP50"] - rows["baseline"]["mAP50"], 2)
    row["delta5095"] = round(row["mAP5095"] - rows["baseline"]["mAP5095"], 2)

groups = {
    "clean640": ["clean_640", "clean_640_s27", "clean_640_s28"],
    "drop60": ["drop60_640", "drop60_640_s27", "drop60_640_s28"],
    "jit20": ["jit20_640", "jit20_640_s27", "jit20_640_s28"],
    "clean320": ["clean_320", "clean_320_s27", "clean_320_s28"],
}
seeds = {}
for name, keys in groups.items():
    vals50_raw = [YOLO[k]["mAP50"] * 100 for k in keys]
    vals95_raw = [YOLO[k]["mAP5095"] * 100 for k in keys]
    vals50 = [round(x, 2) for x in vals50_raw]
    vals95 = [round(x, 2) for x in vals95_raw]
    m50, sd50 = mean_sd(vals50_raw)
    m95, sd95 = mean_sd(vals95_raw)
    seeds[name] = {
        "mAP50_values": vals50,
        "mAP50_mean": m50,
        "mAP50_sd": sd50,
        "mAP5095_values": vals95,
        "mAP5095_mean": m95,
        "mAP5095_sd": sd95,
    }

seed_arch_keys = {
    "baseline": ("clean_640", "clean_640_s27", "clean_640_s28"),
    "escm_ciou": ("escm_ciou_pt", "escm_ciou_s27", "escm_ciou_s28"),
    "lsc_ciou": ("lsc_ciou_pt", "lsc_ciou_s27", "lsc_ciou_s28"),
    "v8n_fwn": ("v8n_fwn_pt_v2", "v8n_fwn_s27", "v8n_fwn_s28"),
    "v8n_focaler": ("v8n_focaler_pt", "v8n_focaler_s27", "v8n_focaler_s28"),
    "v8n_hybrid": ("v8n_hybrid_pt", "v8n_hybrid_s27", "v8n_hybrid_s28"),
}


def seed_row(key: str):
    for source in (YOLO, PT, SEED):
        if key in source:
            return source[key]
    raise KeyError(key)


seed_arch = {}
for name, keys in seed_arch_keys.items():
    vals50 = [seed_row(key)["mAP50"] * 100 for key in keys]
    vals95 = [seed_row(key)["mAP5095"] * 100 for key in keys]
    m50, sd50 = mean_sd(vals50)
    m95, sd95 = mean_sd(vals95)
    seed_arch[name] = {
        "seeds": 3,
        "mAP50_values": [round(x, 2) for x in vals50],
        "mAP50_mean": m50,
        "mAP50_sd": sd50,
        "mAP5095_values": [round(x, 2) for x in vals95],
        "mAP5095_mean": m95,
        "mAP5095_sd": sd95,
        "_raw50": vals50,
        "_raw95": vals95,
    }
baseline50 = seed_arch["baseline"]["_raw50"]
baseline95 = seed_arch["baseline"]["_raw95"]
for name, row in seed_arch.items():
    if name == "baseline":
        continue
    delta50 = [row["_raw50"][i] - baseline50[i] for i in range(3)]
    delta95 = [row["_raw95"][i] - baseline95[i] for i in range(3)]
    row["delta50_mean"], row["delta50_sd"] = mean_sd(delta50)
    row["delta5095_mean"], row["delta5095_sd"] = mean_sd(delta95)
for row in seed_arch.values():
    row.pop("_raw50", None)
    row.pop("_raw95", None)

clean50 = YOLO["clean_640"]["mAP50"] * 100
clean95 = YOLO["clean_640"]["mAP5095"] * 100
annotation = {}
for key in ["jit05_640", "jit10_640", "jit20_640", "drop20_640", "drop40_640", "drop60_640"]:
    annotation[key] = {
        "mAP50": pct(YOLO[key]["mAP50"]),
        "mAP5095": pct(YOLO[key]["mAP5095"]),
        # Single-seed perturbation rows must be compared with the same seed's
        # clean run, not with the three-seed clean mean.
        "delta50": round(YOLO[key]["mAP50"] * 100 - clean50, 2),
        "delta5095": round(YOLO[key]["mAP5095"] * 100 - clean95, 2),
    }

resolution = {}
for a, b in [("clean_640", "clean_320"), ("jit10_640", "jit10_320"), ("drop40_640", "drop40_320")]:
    resolution[a] = {
        "mAP50_640": pct(YOLO[a]["mAP50"]),
        "mAP50_320": pct(YOLO[b]["mAP50"]),
        "delta50": round(pct(YOLO[b]["mAP50"]) - pct(YOLO[a]["mAP50"]), 2),
        "mAP5095_640": pct(YOLO[a]["mAP5095"]),
        "mAP5095_320": pct(YOLO[b]["mAP5095"]),
        "delta5095": round(pct(YOLO[b]["mAP5095"]) - pct(YOLO[a]["mAP5095"]), 2),
    }

report = {
    "architecture_loss": rows,
    "architecture_seed_replication": seed_arch,
    "seed_replication": seeds,
    "annotation": annotation,
    "resolution": resolution,
}

json_path = os.path.join(BASE, "pt_final_results.json")
md_path = os.path.join(BASE, "pt_final_results.md")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)

lines = [
    "# PyTorch unified-protocol final results",
    "",
    "| Configuration | Params (M) | mAP@0.5 (%) | Δ50 | mAP@0.5:0.95 (%) | Δ95 |",
    "|---|---:|---:|---:|---:|---:|",
]
labels = {
    "baseline": "YOLOv8n + CIoU",
    "v8n_fwn": "YOLOv8n + FWN",
    "v8n_nwd": "YOLOv8n + NWD",
    "v8n_focaler": "YOLOv8n + Focaler",
    "v8n_wiou": "YOLOv8n + Wise-IoU",
    "v8n_hybrid": "YOLOv8n + NWD/Wise-IoU hybrid",
    "escm_ciou": "ESCM-only + CIoU",
    "lsc_ciou": "ESCM+SAF + CIoU",
    "lsc_fwn": "ESCM+SAF + FWN",
    "saf": "SAF only + CIoU",
}
for key in [
    "baseline",
    "v8n_fwn",
    "v8n_nwd",
    "v8n_focaler",
    "v8n_wiou",
    "v8n_hybrid",
    "escm_ciou",
    "lsc_ciou",
    "lsc_fwn",
    "saf",
]:
    row = rows[key]
    lines.append(
        f"| {labels[key]} | {row['params']/1e6:.3f} | {row['mAP50']:.2f} | "
        f"{row['delta50']:+.2f} | {row['mAP5095']:.2f} | {row['delta5095']:+.2f} |"
    )
lines += ["", "## Seed replication", ""]
for name, row in seeds.items():
    lines.append(
        f"- {name}: mAP50 {row['mAP50_mean']:.2f} ± {row['mAP50_sd']:.2f}; "
        f"mAP5095 {row['mAP5095_mean']:.2f} ± {row['mAP5095_sd']:.2f} "
        f"(seeds {row['mAP50_values']})."
    )
lines += ["", "## Architecture and loss seed replication", ""]
lines.append("| Configuration | Seeds | mAP@0.5 | Δ50 | mAP@0.5:0.95 | Δ95 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for name, row in seed_arch.items():
    if name == "baseline":
        delta50_text = "—"
        delta95_text = "—"
    else:
        delta50_text = f"{row['delta50_mean']:+.2f} ± {row['delta50_sd']:.2f}"
        delta95_text = f"{row['delta5095_mean']:+.2f} ± {row['delta5095_sd']:.2f}"
    lines.append(
        f"| {name} | {row['seeds']} | {row['mAP50_mean']:.2f} ± "
        f"{row['mAP50_sd']:.2f} | {delta50_text} | "
        f"{row['mAP5095_mean']:.2f} ± {row['mAP5095_sd']:.2f} | "
        f"{delta95_text} |"
    )
with open(md_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print(json.dumps(report, ensure_ascii=False, indent=2))
