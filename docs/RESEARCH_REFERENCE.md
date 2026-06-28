# Research Reference and Reproducibility Notes

## Study Scope

ECG biometric verification using ECG-ID records, with waveform and WTMM-derived representations for one-class identity modeling.

## Dataset

- Source: ECG-ID Database (PhysioNet)
- URL: https://physionet.org/content/ecgiddb/
- Typical local location in this repo: `ecgiddb/`
- Notebook-local cache may also exist in: `notebook/ecgiddb/`

## Core Processing Stages

1. ECG loading (Lead I).
2. R-peak detection and cleaning.
3. Beat segmentation around each R peak.
4. WTMM/CWT feature construction.
5. One-class model training and per-subject evaluation.
6. Figure and benchmark export.

## Key Parameters (Current Defaults)

Parameters below should be verified against the latest notebook/script cells before publication:

- `NUM_SUBJECTS`: number of subjects included.
- `PRE_MS` / `POST_MS`: beat window around R peaks.
- `FREQ_MIN`, `FREQ_MAX`, `N_FREQS`: wavelet frequency sampling.
- `TRAIN_RATIO`: train/test split for subject-level runs.
- model-specific hyperparameters (OC-SVM, neural autoencoder/KAN settings).

## Outputs Used in Reporting

- `paper_figures/benchmark_results.csv`
- publication figures under `paper_figures/fig*.png`

## Reproduction Checklist

- Confirm environment is activated.
- Ensure dataset path is valid and complete.
- Run preprocessing before classification.
- Save seeds/config values used for each run.
- Regenerate benchmark CSV before updating paper claims.
- Check figure labels/axis units before submission.

## Recommended Citation Set

- Lugovaya (ECG-ID thesis).
- Goldberger et al. (PhysioNet).
- Mallat and Hwang (WTMM/wavelet singularities).
- Sch\u00f6lkopf et al. (One-Class SVM).
- Liu et al. (KAN).

Keep this file updated when model families, datasets, or evaluation protocol change.
