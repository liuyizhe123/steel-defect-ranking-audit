# Steel Defect Ranking Audit

This repository contains the code, fixed split manifests, structured result
files, and figure-generation workflows for the manuscript:

**A Case Study of Annotation-Dependent Model Rankings and Evaluation-Label
Sensitivity in Steel Surface Defect Detection**

The study is an empirical benchmark audit. It does not introduce a new
detector. Its purpose is to measure how model rankings and reported mAP change
when training annotations, evaluation labels, data splits, and detector
structures change.

## Repository contents

- `training/`: training drivers, perturbation generators, statistical scripts,
  figure generators, and audits.
- `results/`: structured summaries used to produce the reported tables and
  figures.
- `figures/`: manuscript figures and source figure files.
- `data/splits/`: fixed image-level train/validation/test manifests.
- `configs/data/`: Ultralytics-style dataset configuration templates.
- `REPRODUCIBILITY.md`: environment, commands, and result provenance.

Raw NEU-DET and GC10-DET images, model checkpoints, and long training logs are
not included in this repository. Dataset sources and the expected directory
layout are documented in `data/README.md`.

## Environment

The reported environment is Python 3.14 with PyTorch 2.12.1 and Ultralytics
8.4.72. Install the Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

An NVIDIA GPU is required for retraining. Stored-result verification and figure
regeneration do not start new 300-epoch training jobs.

## Reproduction

Run commands from the repository root:

```powershell
.\reproduce.ps1 plan
.\reproduce.ps1 verify
```

`plan` prints the long-running experiment order. `verify` checks the stored
inputs and summaries, compiles the Python sources, regenerates reports and
figures, and runs consistency audits. Full commands and expected outputs are
listed in `REPRODUCIBILITY.md`.

## Data and code availability

NEU-DET and the selected GC10-DET VOC mirror remain available from their
original public sources. This repository distributes the fixed split
manifests, generated perturbation metadata, analysis code, and structured
results. It does not redistribute the original image archives.

Model checkpoints are available from the corresponding author on reasonable
request. A permanent archival DOI will be added after the manuscript is
accepted.

## Citation

Citation information will be updated when the manuscript is published.
