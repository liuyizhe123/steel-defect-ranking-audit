# Training and Reproduction

Run all project-level commands from the repository root. The root
`reproduce.ps1` is the canonical entry point.

```powershell
.\reproduce.ps1 verify
.\reproduce.ps1 plan
```

`verify` checks the environment and input counts, compiles every Python
source file, rebuilds reports and figures from the retained experiment
summaries and checkpoints, regenerates all DOCX/PDF materials, runs the
audits and rebuilds the submission package.

`verify` does not start 300-epoch training. `plan` prints the long-running
experiment order. The training drivers are resumable and skip jobs already
recorded in their summary JSON files.

## Training drivers

| Driver | Purpose |
|---|---|
| `run_yolo_matrix.py` | YOLOv8n, annotation perturbations and seed replication |
| `run_lsc_pt_matrix.py` | ESCM/SAF and localization-loss comparisons |
| `run_interaction_matrix.py` | Model-by-annotation interaction runs |
| `run_seed_replication_matrix.py` | Additional seeds for key configurations |
| `run_gc10_matrix.py` | Full 2,280-pair GC10-DET comparison |
| `run_cross_model_matrix.py` | Bounded YOLOv10n check |
| `run_rtdetr_matrix.py` | Bounded YOLOv8-RTDETR check |
| `run_review_matrix.py` | Seven-two-one split, P2/P5 and FWN sensitivity matrix |
| `run_review_round2.py` | Additional omission draws and capacity control |

## Derived-data scripts

`build_pt_report.py`, `build_revision_report.py`, `build_review_report.py` and
`build_review_round2_report.py` aggregate the raw run summaries.
`make_pt_figures.py`, `make_efficiency_figure.py` and
`make_qualitative_comparison.py` produce the paper figures.
`make_effect_ladder.py` produces the protocol-versus-model effect ladder and
ranking-fragility ratio from the retained summaries.

The high-resolution qualitative figure reuses the frozen selection and labels
only missed ground-truth targets (`MISS`):

```powershell
python training\make_qualitative_comparison.py --reuse-selection
```

## Audits

```powershell
python audit_manuscript.py
python training\audit_revision_experiments.py
python training\audit_pt_consistency.py
python training\audit_review_round2.py
python training\final_audit.py
```

The final audit validates the packaged DOCX/PDF files, checkpoint state,
reported statistics and package hashes.
