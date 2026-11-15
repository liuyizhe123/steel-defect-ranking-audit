# -*- coding: utf-8 -*-
"""Run the reproducible paper build from the project root."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
STATUS_PATH = ROOT / "REPRODUCTION_VERIFY_OK.json"
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env.setdefault("YOLO_CONFIG_DIR", str(TRAINING / ".ultralytics"))
    pandoc = ROOT.parent / ".tools" / "pandoc-3.12.1" / "pandoc.exe"
    if pandoc.is_file():
        env["PANDOC"] = str(pandoc)
    return env


ENV = build_environment()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_step(label: str, command: list[str]) -> None:
    started = time.time()
    print(f"\n=== {label} ===", flush=True)
    print("> " + " ".join(str(item) for item in command), flush=True)
    result = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=ENV,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    print(f"OK {label} ({time.time() - started:.1f}s)", flush=True)


def check_runtime() -> None:
    print("=== Runtime ===", flush=True)
    modules = [
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "cv2",
        "matplotlib",
        "PIL",
        "docx",
        "pypdf",
        "yaml",
    ]
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "missing Python modules: "
            + ", ".join(missing)
            + "; install with: python -m pip install -r requirements.txt"
        )
    pandoc = Path(ENV.get("PANDOC", ""))
    if not pandoc.is_file():
        raise RuntimeError("pandoc not found; expected parent/.tools/pandoc-3.12.1/pandoc.exe")
    if not POWERSHELL.is_file():
        raise RuntimeError(f"Windows PowerShell not found: {POWERSHELL}")
    winword = Path(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE")
    if not winword.is_file():
        raise RuntimeError(
            "Microsoft Word automation is required for PDF export on this host"
        )
    print(f"python: {sys.executable}", flush=True)
    print(f"version: {platform.python_version()}", flush=True)
    print(f"pandoc: {pandoc}", flush=True)
    print("runtime check: OK", flush=True)


def check_data_and_inputs() -> None:
    print("\n=== Data and input integrity ===", flush=True)
    expected = {"train": 1077, "val": 361, "test": 362}
    for split, count in expected.items():
        image_dir = ROOT / "ultralytics_data" / "clean" / "images" / split
        label_dir = ROOT / "ultralytics_data" / "clean" / "labels" / split
        images = len(list(image_dir.glob("*.jpg")))
        labels = len(list(label_dir.glob("*.txt")))
        if images != count or labels != count:
            raise RuntimeError(
                f"NEU-DET {split} mismatch: images={images}, labels={labels}, expected={count}"
            )
        print(f"NEU-DET {split}: {images} images / {labels} labels", flush=True)

    gc10_stats = json.loads(
        (ROOT / "ultralytics_data" / "gc10" / "gc10_stats.json").read_text(
            encoding="utf-8"
        )
    )
    if gc10_stats.get("used_images") != 2280:
        raise RuntimeError("GC10-DET mirror does not contain 2,280 matched pairs")
    print("GC10-DET mirror: 2,280 matched pairs", flush=True)

    required_checkpoints = [
        ROOT / "ultralytics_runs" / "clean_640" / "weights" / "best.pt",
        ROOT / "lsc_pt_runs" / "v8n_fwn_pt_v2" / "weights" / "best.pt",
        ROOT / "lsc_pt_runs" / "lsc_ciou_pt" / "weights" / "best.pt",
        ROOT / "lsc_pt_runs" / "lsc_fwn_pt_v3" / "weights" / "best.pt",
        ROOT / "gc10_runs" / "gc10_v8n_ciou" / "weights" / "best.pt",
        ROOT / "cross_model_runs" / "yolov10n_clean_640" / "weights" / "best.pt",
        ROOT / "rtdetr_runs" / "rtdetr_clean" / "weights" / "best.pt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_checkpoints if not path.is_file()]
    if missing:
        raise RuntimeError("missing checkpoints: " + ", ".join(missing))
    print(f"required checkpoints: {len(required_checkpoints)} present", flush=True)

    required_json = [
        TRAINING / "yolo_matrix_summary.json",
        TRAINING / "lsc_pt_summary.json",
        TRAINING / "seed_replication_summary.json",
        TRAINING / "review_matrix_summary.json",
        TRAINING / "review_round2_summary.json",
        TRAINING / "review_round2_report.json",
        TRAINING / "efficiency_memory_repeats.json",
        TRAINING / "bootstrap_ci.json",
        TRAINING / "qualitative_summary.json",
    ]
    for path in required_json:
        if not path.is_file():
            raise RuntimeError(f"missing required result: {path.relative_to(ROOT)}")
        json.loads(path.read_text(encoding="utf-8-sig"))
    print(f"result JSON files: {len(required_json)} valid", flush=True)

    pandoc = Path(ENV["PANDOC"])
    if not pandoc.is_file():
        raise RuntimeError("pandoc executable missing")
    if not (ROOT / "manuscript.md").is_file():
        raise RuntimeError("manuscript.md missing")
    if not (ROOT / "supplementary_material.md").is_file():
        raise RuntimeError("supplementary_material.md missing")
    print("data and input integrity: OK", flush=True)


def compile_python_sources() -> int:
    print("\n=== Python syntax compilation ===", flush=True)
    source_files = [
        path
        for path in ROOT.rglob("*.py")
        if not any(
            part in {".mimosa", ".tools", "__pycache__", ".ultralytics"}
            for part in path.parts
        )
    ]
    for path in sorted(source_files):
        if not compileall.compile_file(str(path), quiet=1):
            raise RuntimeError(f"Python syntax compilation failed: {path.relative_to(ROOT)}")
    print(f"compiled Python files: {len(source_files)}", flush=True)
    return len(source_files)


def export_pdfs() -> None:
    jobs = [
        ("steel-defect-empirical_manuscript.docx", "render/manuscript.pdf"),
        ("cover_letter.docx", "render_aux_round3/cover_letter.pdf"),
        ("highlights.docx", "render_aux_round3/highlights.pdf"),
        ("response_to_reviewers.docx", "render_aux_round3/response_to_reviewers.pdf"),
        ("supplementary_material.docx", "render_aux_round3/supplementary_material.pdf"),
    ]
    for document, pdf in jobs:
        run_step(
            f"PDF export {document}",
            [
                str(POWERSHELL),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(TRAINING / "export_docx_pdf.ps1"),
                "-DocumentPath",
                str(ROOT / document),
                "-PdfPath",
                str(ROOT / pdf),
            ],
        )


def run_verify(include_bootstrap: bool) -> None:
    started = time.time()
    check_runtime()
    check_data_and_inputs()
    python_file_count = compile_python_sources()

    if include_bootstrap:
        run_step(
            "Recompute image-level bootstrap intervals",
            [sys.executable, str(TRAINING / "bootstrap_confidence_intervals.py")],
        )

    run_step(
        "Recompute paired statistics",
        [sys.executable, str(TRAINING / "statistics_revision.py")],
    )
    run_step(
        "Build final quantitative report",
        [sys.executable, str(TRAINING / "build_pt_report.py")],
    )
    run_step(
        "Build manuscript figures",
        [sys.executable, str(TRAINING / "make_pt_figures.py")],
    )
    run_step(
        "Build efficiency figure",
        [sys.executable, str(TRAINING / "make_efficiency_figure.py")],
    )
    run_step(
        "Render frozen qualitative cases",
        [
            sys.executable,
            str(TRAINING / "make_qualitative_comparison.py"),
            "--reuse-selection",
        ],
    )
    run_step(
        "Build revision report",
        [sys.executable, str(TRAINING / "build_revision_report.py")],
    )
    run_step(
        "Build review report",
        [sys.executable, str(TRAINING / "build_review_report.py")],
    )
    run_step(
        "Build effect-size ladder",
        [sys.executable, str(TRAINING / "make_effect_ladder.py")],
    )
    run_step(
        "Build round-2 report",
        [sys.executable, str(TRAINING / "build_review_round2_report.py")],
    )
    run_step(
        "Test FWN reference implementation",
        [sys.executable, str(TRAINING / "test_fwn_loss_reference.py")],
    )
    run_step(
        "Test review-matrix routing",
        [sys.executable, str(TRAINING / "test_review_matrix_routing.py")],
    )
    run_step(
        "Build manuscript DOCX",
        [sys.executable, str(ROOT / "build_docx.py")],
    )
    run_step(
        "Build auxiliary DOCX files",
        [sys.executable, str(ROOT / "build_aux_docx.py")],
    )
    export_pdfs()
    run_step(
        "Audit manuscript structure",
        [sys.executable, str(ROOT / "audit_manuscript.py")],
    )
    run_step(
        "Audit revision experiments",
        [sys.executable, str(TRAINING / "audit_revision_experiments.py")],
    )
    run_step(
        "Audit paper-text consistency",
        [sys.executable, str(TRAINING / "audit_pt_consistency.py")],
    )
    run_step(
        "Audit round-2 experiments",
        [sys.executable, str(TRAINING / "audit_review_round2.py")],
    )
    run_step(
        "Build submission package",
        [sys.executable, str(ROOT / "build_submission_package.py")],
    )
    run_step(
        "Final independent audit",
        [sys.executable, str(TRAINING / "final_audit.py")],
    )

    status = {
        "status": "ok",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_sources_compiled": python_file_count,
        "bootstrap_recomputed": include_bootstrap,
        "outputs": {
            "manuscript_docx": {
                "path": "steel-defect-empirical_manuscript.docx",
                "sha256": sha256(ROOT / "steel-defect-empirical_manuscript.docx"),
            },
            "manuscript_pdf": {
                "path": "render/manuscript.pdf",
                "sha256": sha256(ROOT / "render" / "manuscript.pdf"),
            },
            "qualitative_figure": {
                "path": "figs/fig8_qualitative.png",
                "sha256": sha256(ROOT / "figs" / "fig8_qualitative.png"),
            },
            "effect_ladder_figure": {
                "path": "figs/fig10_effect_ladder.png",
                "sha256": sha256(ROOT / "figs" / "fig10_effect_ladder.png"),
            },
            "effect_ladder_summary": {
                "path": "training/effect_ladder_summary.json",
                "sha256": sha256(TRAINING / "effect_ladder_summary.json"),
            },
            "submission_zip": {
                "path": "../论文投稿包.zip",
                "sha256": sha256(ROOT.parent / "论文投稿包.zip"),
            },
        },
    }
    STATUS_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nREPRODUCTION_VERIFY_OK", flush=True)
    print(str(STATUS_PATH), flush=True)


def print_training_plan() -> None:
    steps = [
        ("Base YOLO matrix", "training/run_yolo_matrix.py"),
        ("Architecture and loss matrix", "training/run_lsc_pt_matrix.py"),
        ("Model-annotation interactions", "training/run_interaction_matrix.py"),
        ("Seed replication", "training/run_seed_replication_matrix.py"),
        ("GC10 export", "training/export_gc10.py"),
        ("GC10 matrix", "training/run_gc10_matrix.py"),
        ("Cross-model check", "training/run_cross_model_matrix.py"),
        ("RT-DETR check", "training/run_rtdetr_matrix.py"),
        ("Leakage-aware 7:2:1 split", "training/make_standard_split.py"),
        ("Review matrix", "training/run_review_matrix.py"),
        ("Test-label realizations", "training/make_test_annotation_realizations.py"),
        ("Test-label evaluation", "training/run_test_annotation_realizations.py"),
        ("Efficiency measurement", "training/measure_efficiency.py"),
        ("Fresh-process memory repeats", "training/measure_memory_repeats.py"),
        ("Training omission realizations", "training/make_training_omission_realizations.py"),
        ("Round-2 training", "training/run_review_round2.py"),
        ("Bootstrap intervals", "training/bootstrap_confidence_intervals.py"),
    ]
    print("Full long-running experiment order. No step is executed by this plan.\n")
    for index, (name, script) in enumerate(steps, start=1):
        print(f"{index:02d}. {name}: python {script}")
    print("\nAfter the experiment summaries are complete, run:")
    print("  .\\reproduce.ps1 verify")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "plan"], nargs="?", default="verify")
    parser.add_argument("--include-bootstrap", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "plan":
            print_training_plan()
        else:
            run_verify(args.include_bootstrap)
    except Exception as exc:
        print(f"\nREPRODUCTION_FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
