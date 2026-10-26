# Research integrity, methodology and submission audit

> Historical review dated 30 September 2026. Several issues below were addressed in the 10 October revision; see [current release notes](RELEASE_NOTES.md). This review is retained as a record of the original findings.

Date: 30 September 2026. **Assessment: the current draft is exploratory and is not ready for submission in its present form.** There is a potentially useful study here, but several claims conflict with sources or implementation, and performance has not been independently reproduced. This assessment does not imply fabricated data: the available evidence establishes specific inconsistencies and missing validation, not how they arose.

This document distinguishes external publications (reviewed in [PAPER_REVIEW.md](PAPER_REVIEW.md)), previously generated local results, computations performed in this audit, and proposed experiments. The working manuscript and historical result files were left unchanged.

## What was actually checked

Read the manuscript, pipeline, environment specification, existing tests, configurations and result tables. Checked primary literature and official software documentation. Executed the new standard-library [artifact audit](audit_artifacts.py), which recomputes aggregate means, sample standard deviations and paired mean differences from existing per-person values, checks duplicates/nonfinite values, extracts recording dates and hashes input files. It also computes descriptive cohort subsets without retraining.

Outputs:

- [artifact_checks.json](audit/artifact_checks.json): arithmetic checks, counts and SHA-256 inventory.
- [cohort_summary.csv](audit/cohort_summary.csv): descriptive reaggregation of existing measurements, not new model runs.
- [record_dates.csv](audit/record_dates.csv): local header dates for the ten recording pairs.

The three runs contain 1,080, 360 and 360 per-person metric rows, respectively, corresponding to 30/10/10 people, two representations, six models and three metrics. No duplicate metric keys or nonfinite values were found. Recomputed sample standard deviations agree at reported precision; aggregate means differ by at most 0.0001, consistent with rounding/floating-point boundary effects. Paired means agree to numerical precision. This is **internal consistency**, not validation of labels, splits, training, thresholds or confidence-interval coverage. Bootstrap interval endpoints were not independently regenerated.

The original test command was attempted using `.venv/Scripts/python.exe -m unittest discover -s tests -v`. Test import failed with `ModuleNotFoundError: No module named 'torch'`; no model tests passed in this session. The local interpreter is Python 3.14.7 while `environment.yml` requests Python 3.10. Training, complete preprocessing reproduction and manuscript compilation were not performed. No new empirical performance is claimed.

## Findings requiring correction before submission

| Priority | Claim or issue | Evidence | Required response |
|---|---|---|---|
| Critical | Ozaltin/Yeniay cited as 99.21% ECG-ID identification | The source is cardiac-type classification; see review item 9 | Remove the biometric benchmark attribution; correct pages and scope |
| Critical | El Boujnouni journal article described as using ECG-ID | Its accessible publisher abstract names PTB, MIT-BIH, STDB and NSRDB | Separate it from the conference chapter, which reports ECG-ID |
| Critical | `rec_2` described as a return on another day | All ten local header date pairs are equal | Describe separate same-day recordings; remove unsupported posture/electrode chronology |
| Critical | “Only one thing changes” from run 1 to run 2 | Cohort changes 30→10; enrollment fraction 70%→100%; probe counts and impostor pool also change | Treat this as a protocol contrast, not an isolated effect of elapsed time |
| Critical | KAN beats MLP on F1 / equivalent ranking quality established | Paired F1 difference CI crosses zero; no equivalence margin/test | Report point estimates and uncertainty without a superiority or equivalence conclusion |
| Critical | Novel KAN anomaly-detection setting in broad terms | Existing KAN-AD, AutoKAN and biomedical KAN-AE work | Narrow the research question and update related work |
| Major | Preprocessing text does not match code | Default NeuroKit method; raw-only per-beat standardization | Document actual historical configuration, or change code and rerun |
| Major | All models improve with wavelet representation | Elliptic Envelope AUROC/EER worsen | State heterogeneous effects; assess paired representation changes |
| Major | Compact model implies lower cost | Parameter counts only; no latency/FLOP/memory measurements | Limit to parameter count or benchmark full computation |
| Major | Strong confirmatory inference from pooled table | Development people are included and multiple comparisons are made | Separate development from locked evaluation; label present results exploratory |
| Major | Single command fully reproduces the paper | No historical lockfile, exact code snapshot or split manifest | Archive immutable experiment state and demonstrate a clean rerun |

## Existing numerical evidence and what it supports

Values below come from saved `benchmark_results.csv` files, not a fresh experiment. AUROC/F1/EER are fractions; multiply EER by 100 to express a percentage. EER is the pipeline's discrete approximation.

| Protocol and representation | People | Model | AUROC | F1 | EER |
|---|---:|---|---:|---:|---:|
| Within `rec_1`, Gaussian summary | 30 | OC-SVM | 0.9785 | 0.3206 | 0.0536 |
| Within `rec_1`, Gaussian summary | 30 | MLP AE | 0.9774 | 0.8478 | 0.0566 |
| Within `rec_1`, Gaussian summary | 30 | KAN AE | 0.9771 | 0.8747 | 0.0539 |
| `rec_1`→`rec_2`, Gaussian summary | 10 | MLP AE | 0.9462 | 0.6712 | 0.1183 |
| `rec_1`→`rec_2`, Gaussian summary | 10 | KAN AE | 0.9380 | 0.6698 | 0.1056 |
| `rec_1`→`rec_2`, Morlet summary | 10 | MLP AE | 0.9532 | 0.7009 | 0.0904 |
| `rec_1`→`rec_2`, Morlet summary | 10 | KAN AE | 0.9516 | 0.7106 | 0.0860 |

The 26-input networks have reported parameter counts 3,248 (KAN) and 8,422 (MLP), a ratio of approximately 38.6%. Static inspection supports the KAN count: 464 edges, each containing six spline coefficients and one base-path weight. The implementation has no separate learned spline multiplier `w_s`, despite the manuscript equation describing one. It can be absorbed into the coefficients mathematically; the parameterization should be described accurately.

The saved paired KAN-minus-MLP differences on the 30-person wavelet representation are:

- AUROC: −0.000280, 95% interval [−0.003770, 0.002470].
- F1: +0.026820, 95% interval [−0.019910, 0.093604].
- EER advantage (MLP minus KAN): +0.002777, interval [0, 0.008330].

These intervals do not establish KAN superiority over MLP. Inclusion of zero means the comparison is inconclusive under this analysis; it does not prove equality. An equivalence/noninferiority conclusion requires a justified margin specified before evaluation and an appropriate analysis. The parameter advantage is against this particular MLP, not every possible MLP of comparable capacity.

KAN's smaller within-recording F1 standard deviation is descriptive (0.0831 versus 0.1898). Across-recording Morlet standard deviations are 0.3349 versus 0.3290. A general claim of steadier performance therefore does not follow. Report uncertainty on dispersion if consistency is a central contribution.

Saved KAN intervals against Isolation Forest, LOF and Elliptic Envelope favor KAN across the three wavelet metrics, but these are exploratory, unadjusted comparisons on a partly selected cohort. “Clear universal wins” overstates that evidence.

The abstract's “best conventional detector improves from 0.930 to 0.979” uses the raw OC-SVM result, although Elliptic Envelope has the highest raw AUROC, 0.9543. Use a same-model comparison, or explicitly compare the best observed conventional scores: 0.9543 to 0.9785. Furthermore, Elliptic Envelope changes from AUROC 0.9543/EER 0.0685 on raw/PCA to 0.9519/0.0983 on wavelet features. OC-SVM, LOF and Elliptic Envelope also have lower F1 on the wavelet branch. “All six benefit” must specify a metric and acknowledge exceptions.

## Protocol and dataset analysis

**Dates.** `rec_1` and `rec_2` have identical dates for Persons 01–10. The headers establish a zero calendar-day gap, not the exact elapsed minutes or electrode-removal history. The [dataset documentation](https://physionet.org/content/ecgiddb/1.0.0/) explicitly allows multiple recordings collected within a day. Use “recording-disjoint, same-day” unless stronger metadata is obtained.

**Confounding.** Run 1 uses 30 people with randomly split beats, whereas runs 2/3 use ten people with all first-recording beats for enrollment. Restricting saved run-1 values to Persons 01–10 yields KAN AUROC 0.987905, F1 0.877210, EER 0.036806; run 2 yields 0.937976, 0.669803, 0.105587. These recalculations reduce the target-cohort difference, but the impostor pool, enrollment/probe counts and random training stream still differ. They are not an estimate of a pure session effect.

**Beat accounting.** The local feature file contains 747 first-recording beats, ranging 17–39 per person. The code uses `int(n * 0.7)`, yielding 11–27 enrollment beats, not the draft's 12–27. Person 01 has 23 beats, giving 16 enrollment and seven genuine probes; `results/README.md` instead says 17 and six. The feature CSV labels every `rec_1` beat `enrollment`, even when some are genuine probes. This prevents that column from functioning as a split manifest.

**Temporal dependence.** Cleaning occurs on the complete recording before within-recording splitting. The 700 ms beat windows can overlap when adjacent R peaks are closer than 700 ms. Verify actual overlap using saved peak indices and raw sample spans; those are not currently exported. Whole-record filtering and overlapping windows can share information across the within-recording split. Their magnitude was not measured here. Recording-disjoint evaluation addresses that particular sharing, but not long-term generalization.

**Impostors.** Per verifier, negatives are sampled from other people's beats; they are not used to fit that verifier. This supports a genuine-only training statement at the verifier level. It does not establish generalization to people untouched during system development. The sampled negative pool is also weighted by the number of available beats per person. Compare with an identity-balanced sampling strategy and report impostor identity coverage.

**Development reuse.** The manuscript acknowledges tuning on Persons 01–10, but still includes them in the pooled headline table. The ablation CSV also records results for Persons 11–30 for two wavelets. Without a dated selection record, “untouched follow-up cohort” is not demonstrated. Existing ablations should remain exploratory; select genuinely unused identities/recordings or an external dataset for confirmation.

## Methodology versus implementation

**Peak detection and filtering.** `detect_rpeaks()` calls `nk.ecg_process(sig, sampling_rate=fs)` with no method override. Official [pipeline source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_process.html) distinguishes default NeuroKit processing from Pan–Tompkins; [cleaning source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_clean.html) documents its filter choices. These current pages do not prove the historical installed version. Recover that version and document its exact behavior before choosing whether text or code should change.

**Amplitude normalization.** `main()` standardizes each raw beat, but passes the original cleaned beat to `wtmm_feature_vector()`. That function computes CWT without per-beat normalization. Later enrollment-fitted feature scaling is not equivalent. By CWT linearity, scaling a beat by a constant scales its wavelet power by the square of that constant; the power summaries therefore are not intrinsically amplitude invariant. The manuscript's mechanistic explanation is unsupported by the implemented transformation.

**Dimensionality.** Raw waveforms have 350 samples, but their model input is PCA-reduced to at most the number of enrollment beats, ordinarily 11–27 in the main run. The wavelet input is 26-dimensional. Thus a simple “350 dimensions versus 26” explanation of model performance does not describe the actual inputs. Enrollment-centered PCA has rank at most `n_train − 1`, while the code permits `n_train` components. Inspect degenerate directions, covariance rank and reconstruction residuals. Elliptic Envelope's covariance estimate is particularly questionable when features approach or exceed the sample count.

**Ridges and boundaries.** Document strict local-maxima tests, accepted edge maxima, greedy linking, tie handling and inclusive band limits. The frequency/power moments pool ridge points within each band; they do not average one value per ridge. At 1 Hz, a 700 ms window spans less than a full period. Low-frequency and boundary behavior needs analysis rather than an assumption that coefficients represent stable P/T physiology. Compare summary features with ordinary band energies to establish whether ridge extraction adds value.

**Training and seeds.** Global seeds are set once, while split RNG and autoencoder validation use hard-coded `42 + idx`; conventional random states are also fixed at 42. Neural initialization consumes a shared PyTorch stream across subjects/models. Consequently, changing subject count, run order or model list can change later neural fits, and `--seed` does not vary every randomness source. A subset rerun need not reproduce its values from a larger run. Record independent split, initialization, loader and bootstrap seeds; initialize each model locally.

**Failure handling.** Global warning suppression and broad exception handling can hide singular covariance, convergence failures or skipped models/subjects. Missing probe records trigger skips. Report intended and achieved cohort/model counts explicitly and fail clearly on unsupported scientific runs. A mean over successful fits alone can be selectively optimistic.

## Metrics, calibration and inference

**Thresholding.** The minimum training inlier score / maximum enrollment reconstruction error accepts all observed enrollment scores. It does not guarantee equal false-rejection rates or equal calibration quality across methods. The manuscript's explanation that the minimum threshold rejects many probes because a model marks a fixed training fraction as outliers is not demonstrated by this code: the explicit threshold overrides the default decision offset. Inspect held-out genuine and impostor score distributions, margins and fit quality before proposing a mechanism.

**Calibration separation.** Prefer genuine calibration beats not used to fit the scoring model, or a carefully validated cross-fitting method. With so few beats, empirical tail quantiles are unstable; report that uncertainty. A positive-only calibration rule can target genuine coverage under assumptions, but cannot promise an impostor FAR without representative impostor validation. If population negatives are used for development calibration, disclose that as a separate learning regime.

**F1 and prevalence.** Tests are artificially balanced, making F1 conditional on a 50% genuine prevalence. Report FAR, FRR and counts at the actual chosen threshold. Do not characterize high balanced-test F1 as deployment security. Low-FAR claims need enough independent impostor trials and uncertainty intervals.

**EER.** `eer_score()` selects a discrete ROC point closest to equal false-positive/false-negative rates, then averages the rates. They need not actually be equal. Report this as an approximation or prespecify interpolation and regenerate all tables. EER uses test labels to characterize discrimination, so it is not the performance of the enrollment threshold. Without ties, AUROC granularity is `1/(n_genuine*n_impostor)`; a universal step size of 0.02 is incorrect.

**Uncertainty.** Subject bootstrap is preferable to treating beats as independent, but shared impostor identities induce dependence between verifiers. The current interval is conditional on one split, one impostor draw and one model-training realization. Plan repeated runs plus inference that respects subject/recording structure. Average seeds within subject for the prespecified subject-level estimand; do not turn seeds into extra independent people. Declare a primary contrast and address multiple testing. Recompute on full precision, rounding only for presentation.

**Visualization.** Neither t-SNE nor a heatmap demonstrates that recording quality causes a subject's difficulty. That requires a measured quality variable and a suitable analysis or intervention. Similarly, a cluster plot cannot establish information content, explain a causal advantage or justify selecting a model.

## A defensible research program — all proposed, not completed

The main research question should be: **under genuine-only enrollment and documented acquisition change, do wavelet ridge statistics improve verification over simpler representations, and do spline autoencoders offer a useful accuracy/size/latency tradeoff after fair model selection?** This can support an original contribution without depending on an unverified first-use claim.

1. **Freeze evidence and repair provenance.** Archive this exploratory snapshot; record code revision and dirty diff, dataset version/file checksums, environment lock, hardware and all seeds. Export peak/sample indices, enrollment/calibration/probe IDs, impostor identity IDs, raw scores, thresholds and exclusion reasons. Explicitly separate synthetic demos from scientific output directories and prohibit synthetic fallback for production experiments. The current fallback is labelled in config/logs, but summary tables and figures do not preserve that label reliably.

2. **Define cohorts before further selection.** Reserve development identities for preprocessing, architecture and threshold-policy choices. Lock a separate evaluation cohort and its impostor identities; use all eligible subjects only under a predefined design. Audit date gaps and identify genuinely different-day recordings. If some recording pairs are missing, publish eligibility rules and counts. A ten-person same-day study can remain a pilot, but cannot stand in for long-term evaluation.

3. **Control recording comparisons.** On the same eligible people, keep enrollment beat budget, impostor pool and probe aggregation fixed while comparing held-out within-recording blocks, separate same-day recordings and different-day recordings. Use non-overlapping sample intervals and report elapsed days. Balance or explicitly model variable probe counts. A factorial design separates representation and acquisition effects.

4. **Use serious baselines.** Include tuned OC-SVM, nearest-template/correlation or distance scoring, regularized covariance, PCA reconstruction, Isolation Forest, properly calibrated novelty LOF and compact MLP autoencoders. Match neural latent dimensions and compare several parameter budgets. Tune all models with equivalent development resources. Population-pretrained ECGXtractor or contrastive features may be added as a clearly labelled external-data regime, not silently mixed with enrollment-only models.

5. **Isolate representation and architecture effects.** Compare standardized raw/PCA, ordinary band powers, DWT summaries and WTMM-inspired summaries at matched effective dimension where possible. Ablate ridge linking, power statistics, band counts, normalization, low-frequency boundary treatment and Gaussian/Morlet choice. For KAN, test base-path-only, spline component, fixed versus fitted grids, clipping, grid size, latent size and regularization. Each ablation should answer a hypothesis, not generate configurations for retrospective best-score selection.

6. **Measure robustness and operating behavior.** Prespecify a justified number of seeds (for example ten as a starting design choice, not a power guarantee), enrollment-size learning curves and a primary metric/contrast. Report macro metrics, distributions, ROC/DET, FAR/FRR, uncertainty and calibration failure. Assess controlled amplitude/noise/timing perturbations on held-out recordings; do not infer clinical robustness from synthetic perturbations alone. Test one-beat and fixed multi-beat decisions separately.

7. **Measure actual resources and external validity.** Benchmark model and full-pipeline latency, peak memory, serialized state including scaler/PCA/grids, training time and parameter count on specified hardware. Account for per-person storage scaling. Evaluate a second appropriate multi-recording dataset with a locked protocol and documented acquisition differences. Do not claim liveness/attack resistance without experiments matching an explicit threat model.

8. **Publish the evidence, including negative results.** Generate tables from immutable score files and compare them automatically with manuscript values. Archive code, manifests, environment, evaluation scripts and permitted derived artifacts in a release with a persistent identifier. Report failed runs and limitations. Record author contributions, data-license compliance, and any applicable ethics/publication declarations based on actual author information; do not invent approvals or contribution statements.

## Language currently supported by the artifacts

> In exploratory ECG-ID experiments, the 26-feature wavelet representation yielded high mean discrimination for several genuine-only detectors. The selected KAN autoencoder had fewer trainable parameters than the selected MLP and a higher observed within-recording F1, but the paired F1 interval included zero. A separate ten-person evaluation used different recordings collected on the same day. These results do not establish equivalence, superiority over capacity-matched alternatives, longitudinal robustness, or lower computational cost.

This wording remains conditional on the stored experiments until the implementation/provenance issues are resolved and the final pipeline is reproduced. A publication-quality submission needs those experiments and corrected citations; polished prose alone cannot supply the missing evidence.
