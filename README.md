# ECG Biometric Verification with WTMM and KAN

Companion repository for ECG biometric verification experiments based on the ECG-ID dataset.

## What This Repo Contains

This workspace combines:

- reproducible signal-processing and identity-classification notebooks,
- a script pipeline used for paper figure generation,
- manuscript files and publication-ready figures,
- local ECG-ID dataset copies for offline experimentation.

The main experiment flow is:

1. Load ECG-ID recordings.
2. Detect R-peaks and segment heartbeat windows.
3. Build WTMM-based representations and engineered features.
4. Train one-class models (including OC-SVM and neural baselines).
5. Export benchmark metrics and paper figures.

## Current Repository Layout

```
ecg-kan/
├── README.md
├── LICENSE
├── environment.yml
├── ecg.code-workspace
├── .gitignore
│
├── ecgiddb/                    # ECG-ID local dataset mirror (large, local use)
│   └── Person_01..Person_90/
│
├── notebook/
│   ├── ecg_01_signal_processing.ipynb
│   ├── ecg_02_identity_classification.ipynb
│   ├── ecg_morlet_wtmm/         # generated intermediates
│   ├── ecg_wtmm_images/         # generated images
│   ├── ecg_ml_results/          # generated metrics
│   └── ecgiddb/                 # notebook-local dataset copy/cache
│
├── paper/
│   ├── main.tex
│   └── pipleline.py
│
├── paper_figures/
│   ├── benchmark_results.csv
│   └── fig*.png
│
└── docs/
    ├── REPOSITORY_STRUCTURE.md
    └── RESEARCH_REFERENCE.md
```

## Quick Start

### 1. Environment setup

```bash
conda env create -f environment.yml
conda activate ecg-kan
```

If you use venv instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run notebooks

```bash
jupyter lab
```

Run in order:

1. `notebook/ecg_01_signal_processing.ipynb`
2. `notebook/ecg_02_identity_classification.ipynb`

### 3. Run paper pipeline script

```bash
python paper/pipleline.py
```

This generates/updates benchmark artifacts in `paper_figures/`.

## Documentation Map

- `docs/REPOSITORY_STRUCTURE.md`: folder responsibilities, what to commit, what to ignore.
- `docs/RESEARCH_REFERENCE.md`: experiment references, key parameters, and reproducibility checklist.

## Data and Reproducibility Notes

- ECG-ID originates from PhysioNet: https://physionet.org/content/ecgiddb/
- Large raw data and generated derivatives should stay out of version control.
- Keep only compact, publication-relevant artifacts (for example selected figures or benchmark CSV summaries).

## Citation

```bibtex
@inproceedings{hichou2026ecg,
  title     = {{ECG} Biometric Verification Using {WTMM} Representations:
               A Comparison of One-Class {SVM} and {Kolmogorov--Arnold} Networks},
  author    = {Hichou, Oussama},
  booktitle = {Proceedings of J2DI 2026},
  year      = {2026}
}
```

## License

MIT License. See `LICENSE`.
