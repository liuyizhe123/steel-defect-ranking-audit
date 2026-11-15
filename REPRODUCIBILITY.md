# Reproducibility Notes

## Environment

- Python 3.14
- PyTorch 2.12.1
- Ultralytics 8.4.72
- Matplotlib 3.11.0 (PNG output uses the Agg canvas through Pillow)
- NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB memory
- `amp=False`, `plots=False`, and `workers=0` for all reported training runs

## NEU-DET

The fixed per-class stratified split contains 1,077 training, 361 validation,
and 362 test images. Every experiment uses the same clean validation and test
images. Training-annotation variants change only selected labels in the training
split.

Training variants:

- Box jitter with sigma = 5%, 10%, and 20% of box width and height
- Training-time instance omission with probabilities 20%, 40%, and 60%
- Input resolution 320 or 640

Unless stated otherwise, training uses 300 epochs, batch size 16, seed 2026,
cosine learning-rate decay, deterministic training, and the Ultralytics
automatic AdamW selection. Key conditions are repeated with seeds 2027 and
2028.

`training/make_standard_split.py` builds a second leakage-aware 70:20:10 split
with 1,260/360/180 images. It groups exact duplicates and images with
same-class dHash distance at most 2 before assigning splits. The baseline, FWN
and ESCM+SAF models are each repeated over seeds 2026-2028 on this split.

## Models and Losses

- YOLOv8n with CIoU
- YOLOv8n with FWN
- ESCM-only with the standard P3/P4/P5 detection head and CIoU
- ESCM+SAF with CIoU or FWN
- SAF-only with CIoU
- YOLOv10n with its standard training and inference path

FWN component variants use NWD, Focaler, Wise-IoU, and an equal-weight hybrid
of NWD and Wise-IoU. The reduction is checked independently by
`training/test_fwn_loss_reference.py`.

## GC10-DET Full Annotated Set

`training/export_gc10.py` uses every image/annotation pair in the public
`wyx19980727/GC10-DET-VOC` mirror: 2,280 matched images and annotations. The
mirror also contains 26 images without annotations; these cannot be used for
supervised detection and are excluded. Raw label `10_yaozhed` is normalized to
`10_yaozhe`; one invalid annotation with label `d` is removed. The split is
generated deterministically from all matched pairs in this selected mirror;
it is not a claim of the complete official release.

## Test-Annotation Noise

`training/make_test_annotation_eval.py` creates test-only box-jitter and
instance-omission variants. A model trained only on clean training labels is
then evaluated against each variant without retraining. These results isolate
the effect of noisy evaluation ground truth from the effect of noisy training
labels.

The same procedure is applied to the test split of all matched pairs in the
selected GC10-DET VOC mirror with
`training/make_gc10_test_annotation_eval.py`. Image-level uncertainty is
estimated with 1,000 bootstrap resamples using the same image indices for
paired model comparisons (`training/bootstrap_confidence_intervals.py`).

Four additional independent test-label realizations, seeds 2027-2030, are
created by `training/make_test_annotation_realizations.py` and evaluated by
`training/run_test_annotation_realizations.py`.

## Reviewer-Response Experiments

`training/run_review_matrix.py` runs 29 additional 300-epoch jobs: nine
standard-split runs, four P2/P5 2x2 controls, four FWN sensitivity runs and
twelve interaction-seed replications. All 29 jobs completed successfully. The
P2/P5 controls use standard C2f blocks with depths matched to the effective
YOLOv8n scaling. Four-scale variants use `FairDetect` to keep the
classification branch at 64 channels; without this control, placing P2 first
would silently reduce that branch to 32 channels.

`training/test_review_matrix_routing.py` is a regression test for the job
metadata boundary between annotation conditions and numeric FWN overrides.
`training/run_review_followup_sequence.ps1` records each Python step's complete
stdout/stderr in `training/<step>_console.log`.

The three standard-split ESCM+SAF+CIoU runs use seeds 2026, 2027 and 2028.
Their best-checkpoint SHA-256 digests are:

- `305BF50C6B5C09BA5104205B27C4C3AE5F3AB5EF1A06EFF104428223F74BD280`
- `A4F4B0137B04D73265D7BE58C9210C595D26762812AE96EB52AD0FC2D750EF3A`
- `A076E7B77A005A6796F792396A950F569875DFF864F4DE34CE6520B158B29D00`

These files are distinct; the low seed variance is not caused by repeated
evaluation of one checkpoint.

Efficiency is measured by `training/measure_efficiency.py` after all training
jobs stop. Its figure is produced by `training/make_efficiency_figure.py` in a
separate process that does not import PyTorch or Ultralytics.
`training/make_qualitative_comparison.py` selects deterministic
matched-detection failure cases. `training/build_review_report.py` aggregates
seed means, Kendall rank stability, pairwise reversal margins, sensitivity,
efficiency and qualitative results.
`training/make_effect_ladder.py` builds Fig. 10 and
`training/effect_ladder_summary.json` from the same seed-matched and
realization summaries.

The qualitative figure is rendered at 2400x2760 using 800x800 panels. Solid
red boxes show model predictions and dashed cyan boxes show the original
ground-truth annotations; only missed ground-truth targets are labeled. Use
`python training/make_qualitative_comparison.py --reuse-selection` to redraw
the frozen failure cases without rerunning the full selection scan.

`supplementary_material.md` contains the complete bootstrap-interval,
FWN-sensitivity and omission-draw tables moved from the main manuscript. The
same data remain in `training/bootstrap_ci.json`, `review_report.json` and
`review_round2_report.json`.

The second review round adds two independent 40% training-omission
realisations and a 2.924 M-parameter P3/P4 control without P5:
`training/make_training_omission_realizations.py`,
`training/run_review_round2.py` and `training/build_review_round2_report.py`.
Fresh-process memory repeats are produced by
`training/measure_memory_repeats.py`.

The omission margins are +0.39, -1.06 and -0.86 points for draws 2026-2028,
respectively: one reversal in three draws, mean -0.51 +/- 0.78 points. The
capacity-enriched P3/P4 control reaches 67.29% mAP@0.5 and remains 7.47 points
below the P5-present control. Three fresh-process memory repeats give about
177 MB for YOLOv8n/ESCM baseline/FWN and 345.8-345.9 MB for the P2/P3/P4
models.

The optional real-annotation pilot is generated by
`training/make_reannotation_package.py`. After annotators fill every file in
`reannotation_round1/labels_to_fill`, run
`training/analyze_reannotation.py`. No real-annotation calibration claim is
made until that analysis has completed.

## Commands

Run the experiments from the repository root:

```powershell
python training/run_yolo_matrix.py
python training/run_lsc_pt_matrix.py
python training/run_seed_replication_matrix.py
python training/run_interaction_matrix.py
python training/export_gc10.py
python training/run_gc10_matrix.py
python training/run_cross_model_matrix.py
python training/run_rtdetr_matrix.py
python training/make_test_annotation_eval.py
python training/run_test_annotation_eval.py
python training/make_gc10_test_annotation_eval.py
python training/run_gc10_test_annotation_eval.py
python training/bootstrap_confidence_intervals.py
python training/make_standard_split.py
python training/run_review_matrix.py
python training/make_test_annotation_realizations.py
python training/run_test_annotation_realizations.py
python training/measure_efficiency.py
python training/make_efficiency_figure.py
python training/make_qualitative_comparison.py --reuse-selection
python training/build_review_report.py
python training/make_effect_ladder.py
python training/test_review_matrix_routing.py
python training/make_training_omission_realizations.py
python training/run_review_round2.py
python training/measure_memory_repeats.py
python training/build_review_round2_report.py
python training/make_reannotation_package.py
python training/analyze_reannotation.py
python training/build_pt_report.py
python training/build_revision_report.py
python training/test_fwn_loss_reference.py
python build_aux_docx.py
```

The principal summaries are:

- `training/yolo_matrix_summary.json`
- `training/lsc_pt_summary.json`
- `training/seed_replication_summary.json`
- `training/lsc_pt_interaction_summary.json`
- `training/rtdetr_summary.json`
- `training/gc10_summary.json`
- `training/cross_model_summary.json`
- `training/test_annotation_eval_summary.json`
- `training/gc10_test_annotation_eval_summary.json`
- `training/standard_split_summary.json`
- `training/review_matrix_summary.json`
- `training/test_annotation_realizations_summary.json`
- `training/efficiency_summary.json`
- `training/qualitative_summary.json`
- `training/review_report.json`
- `training/review_report.md`
- `training/review_followup_complete.json`
- `training/bootstrap_ci.json`
- `training/revision_report.json`

`training/reannotation_summary.json` is produced only if the optional human
reannotation pilot is completed. It is not a principal result and is not
included in the submission package until those labels exist and are analyzed.

## Rebuilding the Submission Package

Run document and package commands from the repository root:

```powershell
python build_docx.py
python build_aux_docx.py
python audit_manuscript.py
python build_submission_package.py
```

`build_submission_package.py` first verifies every planned source file. If any
result is missing, it exits without modifying the existing package. Successful
builds are staged and then swapped into `论文投稿包`, so the package can never
be left half-deleted after a failed rebuild.

All perturbation strengths are synthetic design parameters. Because the
strengths are not physically matched across families, the results should be
interpreted as sensitivity curves under the tested conditions rather than a
universal ranking of annotation factors.
