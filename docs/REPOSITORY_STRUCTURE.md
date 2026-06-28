# Repository Structure and Maintenance Guide

## Purpose

This file defines clear ownership for each folder so the repository stays clean and reproducible.

## Directory Responsibilities

- `ecgiddb/`
  - Local ECG-ID dataset mirror.
  - Considered data cache, not source code.
  - Do not add new large binaries to git history.

- `notebook/`
  - Interactive research workflow.
  - `ecg_01_signal_processing.ipynb`: signal preprocessing and beat extraction.
  - `ecg_02_identity_classification.ipynb`: model training/evaluation.
  - Subfolders are generated artifacts and should generally be ignored unless a small curated sample is needed.

- `paper/`
  - Manuscript and paper-specific pipeline code.
  - `main.tex`: manuscript source.
  - `pipleline.py`: script used to regenerate benchmark figures.

- `paper_figures/`
  - Exported publication figures and compact benchmark tables.
  - Keep final figures and summary CSV files.
  - Avoid committing intermediate debug plots.

- `docs/`
  - Human-readable project documentation.
  - Keep process and reproducibility notes here.

## Commit Policy

Commit:

- source code (`.py`, `.ipynb`, `.tex`),
- concise CSV summaries needed for claims,
- final publication figures used in the manuscript,
- environment and dependency definitions.

Do not commit:

- virtual environments,
- notebook checkpoints,
- temporary caches,
- large regenerated datasets,
- local editor/OS artifacts.

## Reproducibility Workflow

1. Create environment from `environment.yml`.
2. Run notebooks in sequence (01 then 02), or run `paper/pipleline.py`.
3. Validate updated outputs in `paper_figures/`.
4. Commit only stable artifacts needed for paper traceability.

## Suggested Long-Term Refactor (Optional)

If you want to further harden the repo later, move toward this target layout:

- `src/` for reusable code,
- `notebooks/` for exploratory work,
- `data/raw`, `data/processed` for local-only data,
- `results/` for metrics,
- `figures/` for publication assets,
- `tests/` for automated checks.

This is optional and can be done incrementally without breaking current experiments.
