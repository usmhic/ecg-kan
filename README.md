# ECG verification with wavelet features and a compact KAN

This research repository compares six enrolment-only ECG verifiers on the
[PhysioNet ECG-ID database](https://physionet.org/content/ecgiddb/1.0.0/).
Each verifier learns one person's genuine beats. It then scores individual
beats from that person and from other people.

The two inputs are normalised waveforms with enrolment-fitted PCA, and a
26-value wavelet ridge summary. The models are OC-SVM, Isolation Forest, LOF,
Elliptic Envelope, an MLP autoencoder and a Kolmogorov-Arnold network (KAN)
autoencoder with learned B-spline connections.

## What the saved experiments show

The main experiment uses 30 people and beats from one recording per person.
These values come from the saved experiments, rather than a new training run.

| Model, using wavelet features | AUROC (higher) | F1 (higher) | EER (lower) | Trainable parameters |
|---|---:|---:|---:|---:|
| OC-SVM | 0.979 | 0.321 | 0.054 | N/A |
| MLP autoencoder | 0.977 | 0.848 | 0.057 | 8,422 |
| KAN autoencoder | 0.977 | 0.875 | 0.054 | 3,248 |

KAN has similar observed AUROC and EER with fewer parameters than this MLP.
Its F1 difference from MLP is uncertain. The intervals do not establish
statistical equivalence. Parameter counts do not establish faster execution.

All reported experiments are **exploratory**: the saved design comparisons
include both participant groups. The ten-person second-recording test uses
separate recordings made on the **same day**. It also changes cohort size,
enrolment size and the impostor pool relative to the main test.
See [the results guide](results/README.md), [manuscript](paper/main.tex) and
[release notes](research/RELEASE_NOTES.md) for the evidence and remaining gaps.

## Set up

Run commands from the repository root. Conda provides Python 3.10 and notebook tools:

```text
conda env create -f environment.yml
conda activate ecg-kan
```

Alternatively, create a virtual environment and install the runtime packages:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then run:

```text
python -m pip install -r requirements.txt
```

These files specify compatible dependency ranges. They are **not** the lost
historical environment. New runs record their actual versions and source hashes.
For notebooks with the pip setup, also install `jupyterlab` and `ipykernel`.

## Run an experiment

ECG-ID downloads into the ignored `ecgiddb/` folder on the first real-data run.
If download or record loading fails, the experiment stops with an error.
Use `--db-dir PATH` for an existing ECG-ID directory containing `Person_01/`, etc.

```text
python pipeline.py --num-subjects 30
```

By default, new outputs go to the ignored `results/local_run/`, including its
`figures/` subfolder. The saved paper results are preserved. A short real-data
check uses four people, fewer epochs and no figures:

```text
python pipeline.py --smoke-test
```

For an offline code demonstration, explicitly request synthetic signals:

```text
python pipeline.py --synthetic-demo
```

Both checks default to `.tmp/smoke-test/`. Synthetic results are labelled and
must not be used as biometric evidence. `python pipeline.py --help` lists the
other options. The historical fixed split seeds remain documented in the paper;
`--seed` does not change every source of randomness.

To repeat the paper's three configurations into new, separate folders:

```text
python pipeline.py --num-subjects 30 --out-dir .tmp/reproduction/main
python pipeline.py --num-subjects 10 --probe-record 2 --no-figures --out-dir .tmp/reproduction/gaussian
python pipeline.py --num-subjects 10 --probe-record 2 --wavelet morl --no-figures --out-dir .tmp/reproduction/morlet
```

These repeat the documented protocol, but exact historical numbers are not
guaranteed because the original environment and split manifests were not archived.
Use `--fig-dir paper/figures` only when intentionally replacing manuscript figures.
Inspect all warnings: covariance estimates can be unstable with so few beats.
An incomplete benchmark records its failures and exits with an error.

## Read and build the paper

[paper/main.tex](paper/main.tex) contains the full methods, results and limitations.
Its four numerical tables are generated from the saved CSVs:

```text
python research/build_paper_tables.py
python research/build_paper_tables.py --check
```

The seven figures are redrawn from the same saved runs, without training:

```text
python research/make_paper_figures.py                 # all seven; needs ECG-ID
python research/make_paper_figures.py --results-only  # the four result figures
```

The result figures (mean performance, AUROC heatmap, KAN versus MLP, ROC) read the
run's metric tables and its local `data/reconstruction_errors.json`. The signal
figures (processing steps, example beats, t-SNE) re-extract beats from `ecgiddb/`
and stop if the regenerated wavelet features differ from the saved feature file.

With a TeX installation containing `elsarticle`, compile from `paper/`:

For Overleaf, upload `paper/main.tex` and the `paper/figures/` folder together,
keeping the folder name `figures`, and select `main.tex` as the main document.
Numerical tables are embedded in `main.tex`; uploading `paper/tables/` is optional.

```text
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
```

The resulting `paper/main.pdf` is ignored by Git. Tectonic also works:
`tectonic paper/main.tex` from the repository root.

## Check the repository

```text
python -m unittest discover -s tests -v
python research/build_paper_tables.py --check
python research/audit_artifacts.py
```

Tests cover feature scaling, spline grids, parameter counts, split separation,
score-based metric reconstruction and separate-recording PCA dimensions.
The artifact audit checks saved arithmetic and local provenance without training.
Missing optional raw headers and feature files are reported explicitly.
GitHub Actions runs the tests, table check, audit and LaTeX build after publication.

## Files

| File or folder | Purpose |
|---|---|
| `pipeline.py` | Configuration, evaluation and command-line runner |
| `ecg_models.py` | Model definitions, factories and shared autoencoder training |
| `ecg_features.py` | Data loading, beat extraction and shared feature calculations |
| `ecg_plotting.py` | Figures and table formatting |
| `paper/` | LaTeX, generated tables and seven retained figures |
| `results/` | Original saved metrics, configurations and design comparisons |
| `notebook/` | Signal-processing and verification walkthroughs |
| `research/` | Evidence review, arithmetic audit and release notes |
| `tests/` | Regression checks |

## Citation and licence

See [CITATION.cff](CITATION.cff) for citation metadata. The manuscript is an
unpublished research paper; no accepted publication or DOI is claimed here.

Oussama Hichou, Abdelhak Tali and Abderrahim El Inaji (2026),
*One-class ECG verification using wavelet features and a compact
Kolmogorov-Arnold autoencoder*.

Code is MIT-licensed; see [LICENSE](LICENSE). ECG-ID is separately licensed under
[Open Data Commons Attribution 1.0](https://physionet.org/content/ecgiddb/1.0.0/).
Cite its creator, Tatiana Lugovaya, and PhysioNet when using the database.
Raw recordings are not distributed in this repository.
