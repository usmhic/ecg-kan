# Saved experiment results

These are the original experiment artifacts. Repository cleanup has not replaced
them with new training results. `summary_all_experiments.csv` collects their
reported means, standard deviations and participant-bootstrap intervals.

## The three protocols

| Run | People | Enrolment | Genuine probes | Wavelet |
|---|---:|---|---|---|
| `1_same_recording_gaussian` | 30 | 70% of each person's `rec_1` beats | Remaining `rec_1` beats | First Gaussian derivative (`gaus1`) |
| `2_new_recording_gaussian` | 10 | All usable `rec_1` beats | Usable `rec_2` beats | `gaus1` |
| `3_new_recording_morlet` | 10 | All usable `rec_1` beats | Usable `rec_2` beats | Real Morlet (`morl`) |

Each verifier also receives as many impostor probes as genuine probes, sampled
from other people's recordings. Impostor beats do not train that verifier.

Run 1 contains 747 usable beats, with 17-39 per person. Enrolment uses
`floor(0.70 * beat_count)`: Person 01 has 23 beats, so 16 are used for enrolment
and seven for genuine probes, alongside seven impostor probes.

Runs 2 and 3 contain 239 enrolment beats and 241 genuine probes in total.
All ten pairs have identical dates in their ECG-ID headers; see
[record_dates.csv](../research/audit/record_dates.csv). They are **separate
same-day recordings**. The metadata does not establish electrode removal,
posture changes or an exact time interval.

Run 1 versus run 2 changes the cohort, enrolment fraction, probe counts and
impostor pool as well as the recording. Their difference does not isolate
elapsed time or a recording effect. Runs 2 and 3 use the same people and
protocol with different wavelets. Morlet has better observed KAN means
(AUROC 0.952 versus 0.938); no independent wavelet-selection test is available.

All three runs are exploratory. `kan_design_ablation.csv` includes results for
Persons 01-10 and 11-30, so it does not establish an untouched follow-up cohort
or a dated sequence of model choices. Most results use one training seed.

## Published files

| File in each run's `metrics/` | Contents |
|---|---|
| `benchmark_results.csv` | Means, sample standard deviations and 95% bootstrap intervals |
| `per_subject_long.csv` | One value per participant, feature branch, model and metric |
| `paired_comparisons.csv` | KAN advantage over each comparator; positive favours KAN |
| `model_complexity.csv` | Historical parameter-count export |
| `run_config.json` | Historical settings and data-source label |

There are finite entries for all six models and both branches: 30 people in
run 1 and ten in runs 2 and 3. Missing historical metadata includes exact
package versions and sample-level split manifests. CPU execution is recorded
in the local logs, but the CPU model is not recorded.

The historical raw-branch complexity export in runs 2 and 3 incorrectly uses
70% enrolment counts; those counts should not be quoted as the evaluated raw
model sizes. The 26-feature counts are correct: KAN 3,248 and MLP 8,422.
The original files are retained to preserve their history.

Local `data/` and `logs/` folders are ignored by Git. When present, they contain
per-beat features, reconstruction errors and console logs. The old feature
column `partition=enrollment` labels **every** `rec_1` beat, including held-out
probes in run 1; it is not a split manifest. The committed audit records the
checks made when those local files were available.

## New run outputs

The updated runner writes an explicit recording name and shared feature names
in `data/wtmm_features_long.csv`. It also writes `data/evaluation_details.json`
with zero-based beat indices, impostor identities, scores, enrolment thresholds,
training information and failures. Beat indices refer to complete extracted
windows, rather than raw sample offsets. `run_config.json` records package
versions, source hashes and actual participant IDs; benchmark tables include
valid participant counts (`N`). These additions do not reconstruct lost metadata
for the original experiments.

The [root README](../README.md) gives commands for repeating configurations in
separate folders. Replace the original metric files only after reviewing a
complete rerun and updating the manuscript consistently. The table generator
reads the original three run folders. `python pipeline.py --summarise-runs`
explicitly rebuilds the combined CSV from real-data run folders; ordinary runs
do not rebuild that file.
