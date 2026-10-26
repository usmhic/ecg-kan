# Repository and manuscript revision

## Revision of 20 October 2026: presentation and wording

Methods, saved results and section structure are unchanged; no model was retrained.

- **Similarity report (Turnitin, 14%).** Matches were almost entirely the
  reference list, author and journal boilerplate, dataset names and hyperparameter
  strings. The only prose overlap (the NeuroKit2 filter description) was
  rephrased, and the detector and training settings moved into Table 2. The
  "replaced characters" integrity flag points to Greek symbols in equations
  (tau, psi, sigma, nu); the LaTeX source is plain ASCII with no substituted
  letters.
- **AI-writing report (Turnitin, 21%, 18 passages).** Flagged passages were
  short uniform sentences, repeated "X; it does not Y" hedges and line-by-line
  descriptions of the code. All 18 were rewritten as ordinary scientific prose
  with the same facts.
- **New, checkable statements from saved data.** The KAN and MLP give identical
  AUROC for 27 of 30 participants and identical EER for 29, so the positive
  KAN-MLP EER interval rests on one person. With balanced probes,
  F1 >= 2R/(2+R) bounds genuine acceptance R; the low F1 of Elliptic Envelope
  and OC-SVM therefore comes from rejecting genuine beats (at most 9% and 43%
  accepted on average), not from accepting impostors.
- **Figures** are redrawn by `research/make_paper_figures.py` from saved metrics
  and reconstruction errors; regenerated wavelet features match the saved
  feature file (relative difference < 1e-7). Bars became interval plots with
  95% CIs, the 30-panel ROC grid became two autoencoder panels (classical
  detector scores were not saved), the duplicate per-person bar chart became a
  per-person KAN-versus-MLP comparison, and the t-SNE highlights three people
  instead of using 30 indistinguishable colours. A colour-blind-safe palette is
  used throughout.
- The abstract was shortened to 249 words for the journal limit.

## Revision of 10 October 2026

The manuscript has complete method and result sections in simple English.
This is a research paper prepared for public review, not a claim of journal
acceptance or an independently confirmed biometric system.

## Manuscript changes

- Overleaf follow-up: all four generated tables are embedded in `main.tex`,
  removing the missing `tables/primary.tex` dependency. The table generator
  updates both embedded tables and separate reference copies; its check mode
  verifies both. An upload needs only `main.tex` and the seven-image `figures/`
  folder.
- Follow-up LaTeX repair: explicit Times-compatible fonts for Unicode TeX
  engines, correctly formatted monospaced email addresses, and a title-page
  footer without overflow. Results tables use readable two-line cells instead
  of scaling all text; confidence intervals are separated from their estimates.
  Float barriers keep participant figures before the next results subsection.
  The PDF includes working internal links, bookmarks and author metadata.
  The 18-page PDF was rebuilt with Tectonic and its rendered pages inspected;
  the final LaTeX log has no warnings, unresolved references or overflowing boxes.
  The local compiler still prints a Fontconfig configuration message outside
  the LaTeX log; all requested text fonts are embedded in the PDF.
- Checked the beat count against the local feature file: 747 beats, 17-39 per
  person, with 11-27 enrolment beats and 6-12 genuine probes in the main test.
- Confirmed that all ten second-recording pairs have the same calendar date.
  The experiment tests separate same-day recordings, rather than later visits.
- Explained that the saved design comparisons include both participant groups.
  Neither group is described as an untouched validation cohort.
- Generated four numerical tables directly from the saved CSVs. The main and
  paired tables retain the existing measurements. The second-recording table
  now includes all six detectors, and a descriptive table separates the two
  participant groups. Some last displayed digits differ from the previous
  hand-written tables because the CSV values are now formatted directly.
- Reported valid counts: 30 people for every main-test model and ten for every
  second-recording model, in both feature branches.
- Added earlier KAN autoencoder and anomaly-detection work, including
  [Moradi et al.](https://arxiv.org/abs/2410.02077),
  [KAN-AD](https://arxiv.org/abs/2411.00278), and
  [Lomoio et al.](https://arxiv.org/abs/2507.19524). The latter uses heart sounds,
  rather than ECG. The contribution is framed around small enrolment sets and
  this particular wavelet summary, rather than KAN autoencoders in general.
- Added reproducibility, data-licence and AI-assistance statements. Dataset
  details and licensing were checked against
  [the ECG-ID provider](https://physionet.org/content/ecgiddb/1.0.0/).

The central conclusion remains cautious: the selected KAN has fewer parameters
than the selected MLP and similar observed AUROC and EER. Its F1 difference from
MLP is uncertain. These results do not establish statistical equivalence,
lower latency, attack resistance or performance over longer time gaps.

## Code and public repository

`pipeline.py` now focuses on configuration, evaluation and running the
experiment. `ecg_models.py` contains the models and shared training loop;
`ecg_features.py` contains shared signal and feature processing;
`ecg_plotting.py` contains figures and table formatting. Repeated feature
extraction and notebook training code were removed. Model construction is
shared between evaluation and parameter-count reporting.

The follow-up cleanup moved the model code out of the runner while keeping
existing imports from `pipeline` available to notebooks and scripts. New run
and audit hashes include `ecg_models.py`. Removed a broad scoring fallback:
all four supported baseline detectors provide `score_samples`, and scoring
failures now reach the existing error audit directly. The CLI example writes
to an ignored local folder instead of a historical result folder.

The feature calculations, architectures, training order and historical split
seeds were preserved. Intentional runner changes are:

- Normal runs default to 30 people and `results/local_run/`. Smoke checks use
  `.tmp/smoke-test/`, with figures disabled unless explicitly requested.
- A failed download never silently switches to synthetic signals. Synthetic
  data requires `--synthetic-demo` and is labelled in the outputs.
- Missing requested records stop a run. Model errors are recorded, warnings
  remain visible and enter the log, and incomplete benchmarks exit with an error.
- New outputs include beat indices, impostor identities, thresholds and scores.
  Run metadata includes package versions, source hashes and actual participant
  IDs. Benchmark tables include valid participant counts.
- Raw-branch complexity counts use the fitted input dimensions, including all
  enrolment beats in the separate-recording protocol. The incorrect historical
  raw counts remain archived and are identified in the results guide.
- Ordinary runs do not rebuild the historical combined summary. Explicit
  summary generation excludes synthetic data.

The README, contribution guide, citation metadata, dependency files and ignore
rules were updated. GitHub Actions is configured to run tests on Python 3.10
and 3.12 and compile the manuscript. Those remote jobs have not run in this
local revision. Existing staged deletions and other author edits were retained.

## Validation performed

See [validation.json](validation.json) for the local environment and source hashes.

- All 15 unit and artifact tests passed. They cover feature amplitude handling,
  spline grids, parameter counts, split separation, metric reconstruction,
  separate-recording PCA, saved arithmetic and manuscript references.
- After the model-module cleanup, all 15 tests passed again, both notebooks
  executed with their existing settings, and a two-person real-data smoke test
  completed for all six models and both branches. Controlled before/after
  comparisons gave exactly matching metrics, reconstruction errors and ROC
  arrays for both feature branches under both recording protocols. The actual
  Overleaf ZIP also compiled without an external table directory.
- All 198 saved bootstrap intervals were rechecked with the documented
  resampling rule. Endpoints agree within 0.0001. This checks implementation
  consistency, not whether the intervals cover all scientific uncertainty.
- Both notebooks executed from top to bottom during the revision; their final
  JSON schemas and code cells were also validated.
- The extracted Person 01 beats and both feature branches match the pre-refactor
  code exactly. On controlled inputs, all six models returned identical AUROC,
  F1 and EER for both branches under both recording protocols.
- Real ECG-ID smoke checks completed for four people within one recording and
  across two recordings, using three training epochs. A two-person run generated
  all seven figures. An explicit synthetic demonstration also completed. These
  checks validate execution, not the headline performance estimates.
- The manuscript compiled with Tectonic 0.17.0. Tables and rendered page
  layouts were inspected. The final LaTeX log has no warnings or overflowing
  boxes. A local Fontconfig configuration message does not affect rendering.
- A scan of candidate public files found no common credential patterns or raw
  database, virtual-environment or cache paths. This was a limited file check.

The original metric CSVs and seven manuscript figures were preserved. The main
30-person experiment and full ten-person secondary experiments were not rerun.
The local validation environment uses Python 3.14.7; it is not the missing
historical environment and does not substitute for the configured CI matrix.

The historical package versions, selection chronology and original figure
provenance remain unavailable; the paper states these limits explicitly.
