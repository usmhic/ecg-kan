# Evidence-based review of the manuscript bibliography

> Historical review dated 30 September 2026. Several issues below were addressed in the 10 October revision; see [current release notes](RELEASE_NOTES.md). This review is retained as a record of the original findings.

Review date: 30 September 2026. Scope: all 26 references in `paper/main.tex`, plus selected missing related work. This is a critical narrative review, not a systematic review or an independent replication of the cited studies. The bibliography was used as the working corpus because no separate paper list was supplied.

**Evidence labels.** “Published” means a result reported by an external source, not verified experimentally here. “Preprint” identifies the version actually consulted. “Local” means existing repository artifacts. “Assessment” means our analysis. “Proposed” means work that has not been performed. Full-text access means relevant methods/results passages were inspected; it does not mean every experiment was reproduced. Abstract-only and source-access limitations are recorded rather than filled with assumptions. Books, software papers and infrastructure references do not necessarily have a single dataset or performance experiment.

The companion [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) assesses our implementation and results. It should be read before reusing claims from the manuscript.

## 1. Bhuva and Kumar — continuous authentication (`bhuva2023`)

**Source/access:** Publisher abstract and section previews. The correct venue is *Internet of Things* 24 (2023), 100927, DOI [10.1016/j.iot.2023.100927](https://www.sciencedirect.com/science/article/pii/S2542660523002500), not the journal/volume/article number in the draft.

**Methodology:** Continuous authentication combining ECG recognition with EMG hand-gesture authentication when the ECG decision is insufficient; CNN adaptation is part of the ECG branch.

**Dataset:** The preview describes PhysioNet ECG data for 295 patients and a separate EMG source. Exact record identifiers, split manifests and overlap checks were not accessible.

**Experiments/results — published:** The authors report approximately 99% average ECG accuracy and investigate impersonation robustness. This is not an ECG-ID one-class EER result.

**Limitations — assessment:** Without the complete protocol, accuracy cannot establish performance for unseen impostors or separate visits. Security assertions require the actual attack model and attack data.

**Comparison/reproducibility:** Relevant to multimodal continuous authentication, but not a direct baseline for a single-beat, genuine-only verifier. Obtain the full protocol before extracting a numerical comparison. Do not use this citation to assert that ECG cannot be replayed.

## 2. Melzi, Tolosana and Vera-Rodriguez — ECGXtractor (`melzi2022`)

**Source/access:** [Author preprint, sections III–VI and Tables III–VII](https://arxiv.org/html/2204.03992), compared with the journal metadata: *IEEE Access* 11 (2023), 15555–15566. The consulted full text is the 2022 preprint; journal-version agreement remains to be checked.

**Methodology/dataset:** An autoencoder representation trained using heartbeats from 55,967 people, followed by biometric matching; evaluation uses in-house data, PTB, ECG-ID and CYBHi.

**Experiments/results — preprint:** Both verification and identification are studied. ECG-ID verification EER is 1.52% within recording and 0.15% across same-day recordings, with different segment/template aggregation. PTB verification EER is 0.14% and 2.06% in single/multiple-session settings, respectively.

**Limitations — assessment:** Pretraining data access limits full reproduction. The ECG-ID contrast changes aggregation as well as recording, so it does not isolate time drift.

**Comparison/reproducibility:** This is a central verification comparator, not merely a review of closed-set identification. Its population-pretrained feature extractor and multi-beat templates differ from our per-person training. Reproduce its protocol or adapt its released [code and weights](https://github.com/BiDAlab/ECGXtractor) under a separately labelled external-pretraining regime. Do not rank its EER directly against our macro per-person EER.

## 3. Clifford, Azuaje and McSharry — ECG analysis textbook (`clifford2006`)

**Source/access:** [Publisher record](https://us.artechhouse.com/Advanced-Methods-and-Tools-for-ECG-Data-Analysis-P957.aspx) and [author-hosted preface](https://www.mit.edu/~gari/ecgbook/preface.pdf). *Advanced Methods and Tools for ECG Data Analysis*, Artech House, 2006. The relevant technical chapters were not fully reviewed.

**Methodology:** A reference work on physiological interpretation, signal processing and computational ECG analysis, rather than a proposed biometric classifier.

**Dataset/experiments/results:** No single dataset, split or biometric score applies to the entire book. Chapter-specific experimental evidence must be cited at chapter/page level.

**Limitations — assessment:** A general textbook citation cannot substantiate every numerical assertion about P/T/QRS frequency bands, filter cutoffs or T-wave duration. Those statements also need to match the actual implementation and its intended population.

**Comparison/reproducibility:** Retain for established background after locating the exact supporting passages. It does not establish that the chosen 700 ms window or five frequency bands are optimal. Those are study design choices that need empirical sensitivity analysis. The draft's claim of earlier WTMM use in ECG denoising/classification needs a directly relevant primary study, not an unlocated book citation.

## 4. Byeon, Pan and Kwak — scalogram biometrics (`byeon2019`)

**Source/access:** [Full publisher text, sections 3–4 and Table 4](https://www.mdpi.com/1424-8220/19/4/935/htm), *Sensors* 19(4), 935 (2019), DOI 10.3390/s19040935.

**Methodology:** Filter ECG, locate beats, construct Morse-wavelet CWT scalograms, resize them for CNNs, and compare AlexNet, GoogLeNet and ResNet with transfer/non-transfer training.

**Dataset:** PTB and a Chosun University collection. The PTB experiment retains 211 of 290 people with 120 samples each; CU acquisition includes 100 people.

**Experiments/results — published:** Architecture, optimizer and training settings are compared. Table 4 reports 98.10% PTB test accuracy for a ResNet configuration. The paper also discusses verification metrics; describing all its experiments as exclusively closed-set is too broad.

**Limitations — assessment:** Exclusions alter cohort composition. Segment-level success does not establish longitudinal robustness. The source has unusual sample-rate/count wording that should be checked against acquisition documentation rather than copied.

**Comparison/reproducibility:** A useful full-scalogram alternative to our statistical summary. It uses **Morse**, so it cannot support the draft's attribution of Morlet to this study. Match subject sets, recordings and scoring units before comparison.

## 5. Mallat and Hwang — wavelet singularities (`mallat1992`)

**Source/access:** [Author-hosted paper](https://www.di.ens.fr/~mallat/papiers/MallatSingDet92.pdf), *IEEE Transactions on Information Theory* 38(2), 617–643 (1992), DOI 10.1109/18.119727. Indexed primary-source text was accessible; direct PDF retrieval failed. This is a limited-access assessment.

**Methodology:** Relates wavelet modulus maxima across scales to local signal regularity and investigates processing/reconstruction from maxima information.

**Dataset/experiments/results:** Mathematical analysis and signal-processing demonstrations; no ECG-ID identity-verification benchmark. No numerical biometric outcome is attributable to this work.

**Limitations — assessment:** Our greedy nearest-maxima linking and five-band statistics are implementation choices, not established as equivalent to the paper's theoretical construction. Boundary maxima and finite-record effects need explicit treatment.

**Comparison/reproducibility:** Cite as the conceptual origin of modulus-maxima analysis. Use “WTMM-inspired features” for our method, and specify linking order, tolerance, minimum chain length, ties and band boundaries. Do not claim multifractal estimation, a regularity theorem or information preservation merely because maxima are retained. Complete the theoretical comparison using the full paper before submission.

## 6. Mallat — wavelet textbook (`mallat1999`)

**Source/access:** [Publisher table of contents and bibliographic record](https://www.sciencedirect.com/book/9780124666061/a-wavelet-tour-of-signal-processing), *A Wavelet Tour of Signal Processing*, second edition, 1999. Chapter contents were not fully accessible.

**Methodology:** General treatment of time-frequency analysis, wavelets, frames, bases, approximation and estimation.

**Dataset/experiments/results:** A theoretical and instructional book, not a single empirical study. No ECG verification score should be assigned to it.

**Limitations — assessment:** Citation of the book does not validate the chosen PyWavelets scale-to-frequency mapping, ridge summary or Morlet/Gaussian comparison.

**Comparison/reproducibility:** Useful for definitions and assumptions. Locate exact pages supporting each mathematical statement. Document the discrete transform and finite support separately from the continuous integral. In particular, low-frequency coefficients on a 700 ms beat can depend heavily on boundaries; the existence of a CWT formula does not establish reliable measurement at every requested frequency.

## 7. El Boujnouni et al. — wavelet capsule network (`elboujnouni2022`)

**Source/access:** [Publisher-indexed abstract/highlights](https://www.sciencedirect.com/science/article/pii/S1746809422002142), *Biomedical Signal Processing and Control* 76, 103692 (2022), DOI 10.1016/j.bspc.2022.103692. Full methods were not accessible.

**Methodology:** CWT/DWT processing and a capsule network for ECG identity classification; the abstract emphasizes recognition using few heartbeat segments.

**Dataset/results — published abstract:** Reported accuracies are 99.5% on PTB, 98.1% on MIT-BIH arrhythmia, 98.2% on STDB and 100% on NSRDB. **These are the four datasets named in the abstract; ECG-ID is not one of them.**

**Experiments/limitations:** Exact cohort sizes, split units, repeated trials and hyperparameter selection are unresolved without the full text. A high identification accuracy is not a calibrated verification operating point.

**Comparison/reproducibility:** Relevant wavelet/CapsNet prior work, but the draft's ECG-ID attribution is unsupported by the inspected source. Do not transfer the 100% figure to ECG-ID. Obtain the paper and distinguish any shared methods/data from the related conference chapter before making originality claims.

## 8. El Boujnouni, Tali and Bentaleb — scalogram capsule chapter (`elboujnouni2020`)

**Source/access:** [Publisher chapter abstract](https://link.springer.com/chapter/10.1007/978-3-030-90639-9_3), *Advanced Intelligent Systems for Sustainable Development (AI2SD’2020)*, AISC 1418, pp. 29–42, first online 10 February **2022**. DOI 10.1007/978-3-030-90639-9_3.

**Methodology:** Noise preprocessing, QRS–T segmentation, wavelet scalograms and CapsNet classification.

**Dataset/results — published:** The abstract reports 94.59% identification accuracy on PTB-ECG and 93.75% on ECG-ID.

**Experiments/limitations:** Full cohort sizes, recording splits and tuning protocol were unavailable. The abstract's training-efficiency assertion is not enough to quantify a matched learning curve.

**Comparison/reproducibility:** This chapter actually supports an ECG-ID CapsNet citation. It must not be conflated with the 2022 journal paper or its 100% NSRDB result. The citation key may remain a historical identifier, but prose should use the publication year accurately. Compare identity training requirements and beat aggregation before discussing our genuine-only protocol as a contribution.

## 9. Ozaltin and Yeniay — CNN–SVM scalograms (`ozaltin2023`)

**Source/access:** [Journal abstract/metadata](https://link.springer.com/article/10.1007/s00500-022-07729-x) and [author preprint, dataset section](https://assets-eu.researchsquare.com/files/rs-1364437/v1_covered.pdf). Correct pages: *Soft Computing* 27, **4639–4658** (2023), DOI 10.1007/s00500-022-07729-x.

**Methodology:** A 34-layer CNN produces image features for an SVM; CWT settings and alternative architectures are compared. The journal abstract also reports an STFT comparison.

**Dataset:** The preprint describes arrhythmia, congestive-heart-failure and normal-sinus-rhythm groups from MIT-BIH Arrhythmia, BIDMC CHF and MIT-BIH NSR. This is **cardiac-type classification**, not ECG-ID identity recognition.

**Experiments/results — published:** The journal reports a maximum 99.21% accuracy for CNN–SVM with CWT.

**Limitations — assessment:** Disease labels aligned with source databases may permit source-specific cues; this is a risk to test, not a demonstrated error. Subject grouping and all feature-learning steps must be checked within folds.

**Comparison/reproducibility:** Retain only as signal-representation/classification background. Remove its use as a 99.21% ECG-ID biometric benchmark. Journal/preprint protocol differences require version-aware citation.

## 10. Schölkopf et al. — one-class SVM (`scholkopf2001`)

**Source/access:** [Author-affiliated publication record and abstract](https://research.google/pubs/estimating-the-support-of-a-high-dimensional-distribution/), *Neural Computation* 13, 1443–1471 (2001). Full experiment tables were not inspected.

**Methodology:** Kernel support estimation by learning a separating function through a regularized quadratic optimization problem; a positive region describes the estimated support of the distribution.

**Dataset/experiments/results:** The source presents an algorithm and statistical analysis. No ECG-ID result is reported in the inspected evidence; individual benchmark details remain unextracted.

**Limitations — assessment:** Kernel bandwidth, scaling and regularization matter especially with very few positive examples. The training-distribution support is not necessarily the support under a new recording condition.

**Comparison/reproducibility:** This is a justified genuine-only baseline. Our `nu=0.1`, library `gamma` default and minimum-training-score threshold are specific choices. Resetting the operating threshold does not make poor F1 an intrinsic limitation of OC-SVM. Compare both ranking performance and a separately calibrated acceptance rule before attributing an advantage to neural reconstruction.

## 11. Liu, Ting and Zhou — Isolation Forest (`liu2008`)

**Source/access:** [Author-hosted full paper, sections 4–6](https://www.lamda.nju.edu.cn/publication/icdm08b.pdf), ICDM 2008, pp. 413–422.

**Methodology:** Random feature/split trees isolate unusual points; normalized average path length supplies an anomaly score. Subsampling controls computation and influences masking/swamping.

**Dataset:** Twelve benchmark datasets in Table 2 include KDDCUP99 subsets, ForestCover, Mulcross, Shuttle, Mammography and Arrhythmia.

**Experiments/results — published:** AUC and processing time are compared with ORCA, LOF and Random Forests, alongside parameter/scalability analyses. The reported strength is efficient anomaly ranking across these tasks, including training without anomalies; there is no biometric verification finding.

**Limitations — assessment:** A dozen enrollment beats provides little support for learning tails. Axis-aligned isolation can respond differently to raw PCA coordinates and heterogeneous wavelet features.

**Comparison/reproducibility:** Report tree count, subsample size, random seed and score orientation. Our 200-tree setting is a baseline configuration, not a faithful reproduction of every original benchmark. Tuning and repeated seeds are required for a model-family comparison.

## 12. Breunig et al. — LOF (`breunig2000`)

**Source/access:** [SIGMOD full paper](https://sigmodrecord.org/?download_id=6631&smd_process_download=1), SIGMOD 2000, pp. 93–104.

**Methodology:** Reachability-based local densities are compared with neighboring densities to assign a continuous outlier factor; theoretical bounds describe behavior in clusters.

**Dataset/experiments:** Synthetic examples and real sports data, including hockey and 375 Bundesliga players, illustrate local anomalies that global criteria miss. Runtime and parameter behavior are also examined.

**Results — published:** Demonstrates why local outlier scoring can identify meaningful unusual observations; it does not establish ECG verification performance.

**Limitations — assessment:** Neighborhood size approaches the whole enrollment cohort in our small-sample setting, reducing meaningful locality. Sensitive scaling and unreliable distances are additional concerns.

**Comparison/reproducibility:** Distinguish original outlier detection on the fitted sample from novelty detection on unseen probes. Our `novelty=True` use and training-score threshold need an explicit implementation justification. In particular, scoring fitted points through a novelty API is not automatically equivalent to leave-one-out calibration. Use genuine calibration samples or a validated cross-fitting procedure.

## 13. Tax and Duin — SVDD (`tax2004`)

**Source/access:** [Publisher abstract](https://link.springer.com/article/10.1023/B:MACH.0000008084.60811.49), *Machine Learning* 54, 45–66 (2004).

**Methodology:** Finds a compact enclosing sphere, with slack for outliers; kernels allow flexible boundaries. Negative examples can optionally tighten the description.

**Dataset/experiments/results:** Algorithmic data-description study. Specific numerical experiments were not extracted from accessible material; no ECG-ID result is claimed here.

**Limitations — assessment:** Kernel and regularization choices determine the learned region. Optional use of negatives must not be confused with strictly genuine-only training.

**Comparison/reproducibility:** Appropriate background and a possible additional baseline, but our repository does not implement SVDD. Label it as related work, not one of the six evaluated methods. If added, describe the exact implementation and its relation to the RBF OC-SVM configuration so that closely related kernel models are not presented as independent evidence of broad coverage.

## 14. Ruff et al. — anomaly detection review (`ruff2021`)

**Source/access:** [Full author preprint, sections IV–VIII](https://arxiv.org/html/2009.11732), corresponding to *Proceedings of the IEEE* 109(5), 756–795 (2021).

**Methodology:** Unifies density estimation, one-class objectives and reconstruction under their assumptions; distinguishes anomaly scores from decision thresholds.

**Dataset/experiments:** Empirical comparisons on MNIST-C and MVTec-AD, with worked examples including thyroid disease detection and industrial inspection.

**Results — published framework:** Demonstrates that method choice depends on data and anomaly assumptions; it is not evidence of a universal deep-learning winner or our specific ECG setup.

**Limitations — assessment:** Image/corruption benchmarks do not recreate positive-only identity enrollment or same-person physiological drift.

**Comparison/reproducibility:** Strong justification for comparing shallow and reconstruction methods and reporting operating thresholds separately. A low reconstruction error is a proxy for normality, not a guaranteed likelihood or identity probability. The proposed study should test whether errors remain discriminative under recording change rather than assuming reconstruction fidelity implies a useful verifier.

## 15. Kiyasseh, Zhu and Clifton — CLOCS (`kiyasseh2021`)

**Source/access:** [ICML proceedings and full paper](https://proceedings.mlr.press/v139/kiyasseh21a.html), PMLR 139, 5606–5615 (2021), especially sections 4–6 and Tables 1–2.

**Methodology:** Patient-aware contrastive representation learning across temporal segments and leads. This is **not** reconstruction-error autoencoder anomaly detection.

**Dataset:** Chapman, PhysioNet 2020, PhysioNet 2017 and Cardiology datasets.

**Experiments/results — published:** Linear evaluation, transfer and label-efficiency experiments with five seeds. Table 1 reports Chapman AUROC 0.896 ± 0.005 for CMSC versus 0.738 ± 0.034 for SimCLR with 50% labelled data and four leads. These are downstream clinical classification outcomes.

**Limitations — assessment:** Patient-related representations do not themselves validate biometric impostor rejection. Multi-person pretraining differs from enrollment using one person's data alone.

**Comparison/reproducibility:** Correct the draft's use of CLOCS as evidence for reconstruction-based ECG anomaly detection. It could motivate an externally pretrained encoder comparison, with pretraining/test identities and use of identity information disclosed. Results from that regime must remain separate from strictly enrollment-only representation learning.

## 16. Liu et al. — original KAN (`liu2024`)

**Source/access:** [Full author manuscript, v5](https://arxiv.org/html/2404.19756v5). Preprint first posted in 2024; the draft cites ICLR 2025. The venue page encountered a browser challenge, so final proceedings metadata should be imported before submission.

**Methodology:** Learnable univariate edge functions, spline parametrization, grid refinement and regularization/pruning provide a different parameterization from fixed node activations in MLPs.

**Dataset/experiments:** Function approximation, scientific/mathematical examples and comparisons addressing accuracy, scaling and interpretability. No ECG-ID genuine-only experiment is established by this paper.

**Results — author report:** KANs can yield compact accurate representations in selected tasks and inspectable learned functions. Those observations do not establish general superiority.

**Limitations — assessment:** Benefits depend on task, capacity, optimization and compute; a spline function being inspectable does not prove a physiological explanation.

**Comparison/reproducibility:** Our fixed-grid, clipped-input, reconstruction model is an adaptation. Report its differences from the reference implementation. Match MLP capacity and tuning budgets, and benchmark real latency before making efficiency claims. Cite the exact version used rather than silently mixing the original preprint and later author lists.

## 17. Yu, Yu and Wang — fair KAN/MLP comparison (`yu2024`)

**Source/access:** [Full technical report](https://arxiv.org/html/2407.16674), [version record](https://arxiv.org/abs/2407.16674). The inspected record identifies a 2024 technical report, not a verified journal publication.

**Methodology:** Controls parameter count or FLOPs while comparing KAN and MLP; ablates spline activations and examines continual learning.

**Dataset/experiments:** Multiple benchmarks spanning tabular learning, vision, language, audio and symbolic formula representation. This is not an ECG-ID benchmark.

**Results — preprint:** MLP generally performs better outside symbolic formula tasks; adding B-splines to MLP can recover or exceed the symbolic advantage. KAN exhibits worse forgetting under the examined class-incremental setup.

**Limitations — assessment:** Its task suite cannot establish how a small one-class ECG autoencoder will behave; neither does it support assuming a KAN benefit in that setting.

**Comparison/reproducibility:** The clearest methodological warning against comparing one tuned KAN with one larger default MLP. Use capacity/compute curves, equal development budgets and multiple seeds. Architecture, spline activation and representation should be separated through ablations. Authors provide [comparison code](https://github.com/yu-rp/KANbeFair).

## 18. Jahin et al. — KACQ-DCNN (`jahin2025`)

**Source/access:** [Full author manuscript](https://arxiv.org/html/2410.07446) and [publisher record](https://www.sciencedirect.com/science/article/pii/S0010482525013289). *Computers in Biology and Medicine* 197, 110976 (2025), DOI 10.1016/j.compbiomed.2025.110976.

**Methodology:** Classical/quantum hybrid channels with KAN components; ablations, LIME/SHAP and conformal prediction assess model components, explanations and uncertainty.

**Dataset:** Five merged heart-disease datasets yield 1,190 observations before removing 272 duplicates, leaving 918. These are clinical tabular observations, not our ECG waveform identity task.

**Experiments/results — author report:** A four-qubit, one-layer variant reports 92.03% accuracy and 94.77% ROC-AUC against 37 benchmarks.

**Limitations — assessment:** Merged-source evaluation requires checking duplicates, source effects, feature preprocessing and validation nesting. Hybrid gains do not isolate KAN alone. Explanations are not causal validation, and quantum components do not by themselves demonstrate practical quantum speedup.

**Comparison/reproducibility:** Context for KAN in cardiovascular research, not direct support for ECG biometric verification. Complete the bibliography's author list and article number; preserve manuscript/journal version differences during extraction.

## 19. Zhao et al. — MAK-Net (`zhao2025`)

**Source/access:** [Original article PDF, archived copy](https://pdfs.semanticscholar.org/e373/f8e38990d7a76ff4baaa6779dbf42abebd6b.pdf); publisher DOI [10.3390/s25133928](https://doi.org/10.3390/s25133928). Authors: Cong Zhao, Bingwei Lai, Yongzheng Xu, Yiping Wang and Haorong Dong; *Sensors* 25(13), 3928 (2025).

**Methodology:** Multiscale convolutions, attention, BiGRU and KAN layers, with SMOTE/focal-loss imbalance handling.

**Dataset/experiments/results — published:** MIT-BIH arrhythmia classification; component/baseline comparisons. Abstract/Table 10 report accuracy 0.9980 and F1 0.9888.

**Limitations — source conflict:** Section 2 describes inter-patient training, but the discussion on page 16 states intra-patient evaluation and defers inter-patient validation. The latter also acknowledges absent explicit spline visualizations. Treat subject-independent generalization and interpretability claims as unresolved. Verify oversampling occurs only after splitting.

**Comparison/reproducibility:** This supervised hybrid does not test one-class identity verification or establish an isolated KAN effect. Cite the protocol conflict explicitly if relying on the results. Do not import its accuracy as evidence that our architecture is clinically validated or biometric state of the art.

## 20. Israel et al. — early ECG identification (`israel2005`)

**Source/access:** [Original article PDF, university-hosted copy](https://cse.buffalo.edu/~wenyaoxu/courses/fall2017/Paper/ECG_Biometrics.pdf), *Pattern Recognition* 38(1), 133–142 (2005), DOI 10.1016/j.patcog.2004.05.014.

**Methodology:** Filtering, fiducial feature extraction and discriminant classification; aggregates heartbeat decisions to identify people. Experiments examine sensor placement and anxiety/task conditions.

**Dataset:** A study-specific collection, not ECG-ID. Exact cohort accounting across individual experiments needs further extraction.

**Experiments/results — published:** Figure 12 separates heartbeat classification (82%/79%) from person identification (100%) for neck/chest conditions. Figure 13 reports 97–98% person identification for its anxiety experiments, with lower heartbeat-level rates.

**Limitations — assessment:** Aggregation changes the evaluation unit substantially; “over 90%” without naming that unit is misleading. Specific stress/placement experiments do not prove universal invariance.

**Comparison/reproducibility:** Useful historical motivation and a reason to evaluate multi-beat voting. A comparable replication requires the exact feature definitions, split structure, number of beats per decision and accessible original data. Do not compare person-level voting accuracy with our individual-beat F1.

## 21. Pan and Tompkins — QRS detection (`pan1985`)

**Source/access:** [IEEE primary abstract/metadata](https://ieeexplore.ieee.org/document/4122029/), *IEEE Transactions on Biomedical Engineering* BME-32(3), 230–236 (1985), DOI 10.1109/TBME.1985.325532.

**Methodology:** Real-time QRS detection using slope, amplitude and width information, filtering and adaptive thresholds.

**Dataset/results — published:** The abstract reports 99.3% correctly detected QRS complexes on the standard 24-hour MIT/BIH arrhythmia database. This is a peak-detection result, not identity accuracy.

**Experiments/limitations:** Full error accounting was not extracted. Performance in a historical benchmark does not establish segmentation quality on our noisier recordings or with a different implementation.

**Comparison/reproducibility:** Cite only if that algorithm is actually selected, or as historical background. Our code calls `nk.ecg_process` without a method argument; the documented default is `neurokit`, distinct from `pantompkins1985`. Specify the original run's package version and method before repairing the manuscript's attribution. Changing the detector would require new experiments, not merely a citation change.

## 22. Makowski et al. — NeuroKit2 (`makowski2021`)

**Source/access:** [Publisher article](https://link.springer.com/article/10.3758/s13428-020-01516-y), *Behavior Research Methods* 53, 1689–1696 (2021), and [official ECG pipeline source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_process.html).

**Methodology:** Software for physiological signal simulation, preprocessing and analysis; ECG processing combines several configurable stages.

**Dataset/experiments/results:** A software/toolbox contribution, not a trained biometric detector with one headline ECG-ID result. No such result is inferred here.

**Limitations — assessment:** A software citation cannot identify which release, defaults or filters generated an old run. Current online development documentation establishes present behavior, not the unrecorded historical environment.

**Comparison/reproducibility:** Pin the actual release and explicitly record method, artifact correction, filters and segmentation. Current [`ecg_clean` documentation](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_clean.html) describes the NeuroKit route as a high-pass plus powerline filtering procedure; do not label it an unspecified 0.5–40 Hz bandpass. Save processing reports and peak indices so that changes in software do not silently change the dataset.

## 23. Lugovaya — thesis and ECG-ID (`lugovaya2005`)

**Source/access:** [Author's thesis summary hosted by PhysioNet](https://physionet.org/content/ecgiddb/1.0.0/biometric.shtml) and [versioned dataset](https://physionet.org/content/ecgiddb/1.0.0/), DOI 10.13026/C2J01F. The complete thesis was not inspected.

**Methodology:** Segmented QRS/P/T-containing features, PCA, linear discriminant analysis and majority voting for identification.

**Dataset:** 310 lead-I recordings from 90 volunteers, 20 seconds at 500 Hz; headers include recording dates. Raw and filtered channels coexist.

**Experiments/results — thesis summary:** Reports 96% correct identification for a predetermined group of 90 people. This is not positive-only verification.

**Limitations — assessment:** Short recordings, uneven recording counts and nonuniform time gaps limit temporal conclusions. A later record number does not establish a different day or documented electrode replacement.

**Comparison/reproducibility:** Cite both the thesis and exact database version; specify raw channel 0 versus filtered channel 1. Report subject exclusions and actual dates. Our local header audit shows all ten stress-test pairs have zero calendar-day gaps. That local observation is separate from the thesis's published identification result.

## 24. Goldberger et al. — PhysioNet infrastructure (`goldberger2000`)

**Source/access:** [PhysioNet's official citation record](https://physionet.org/about/), *Circulation* 101(23), e215–e220 (2000), DOI 10.1161/01.CIR.101.23.e215. Original full article not inspected in this pass.

**Methodology:** Infrastructure integrating physiological datasets, analysis software and network access.

**Dataset/experiments/results:** A resource description spanning datasets; it has no single ECG-ID experiment or performance claim. ECG-ID's dataset details require its own versioned source.

**Limitations — assessment:** Repository access does not by itself make an analysis reproducible. Dataset versions, file integrity, selections and transformations remain the experimenter's responsibility.

**Comparison/reproducibility:** Keep as infrastructure credit. The current official page also requests the 2026 Pollard et al. platform citation, DOI [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z); this is a citation-policy update, not a reason to replace the ECG-ID thesis citation or an experimental result.

## 25. van der Maaten and Hinton — t-SNE (`maaten2008`)

**Source/access:** [JMLR primary record](https://www.jmlr.org/papers/v9/vandermaaten08a.html), *JMLR* 9, 2579–2605 (2008). Abstract/metadata inspected; full experiment tables were not extracted.

**Methodology:** Nonlinear low-dimensional visualization based on neighborhood probability matching, with a heavy-tailed embedding distribution.

**Dataset/experiments/results:** Visual comparisons on high-dimensional benchmark data; the method is a visualization algorithm, not an ECG verification model.

**Limitations — assessment:** Displayed cluster spacing, sizes and apparent separation depend on projection settings and cannot quantify preserved identity information. Fitting a plot on every labelled subject is descriptive, not an out-of-sample evaluation.

**Comparison/reproducibility:** Record seed, perplexity, preprocessing and sample selection. Treat our plots as illustrations only. Claims that the summary discards exactly within-person variation, or that raw beats contain more identity information, require quantitative controlled experiments; a t-SNE plot does not establish either. Keep visualization outside all model/threshold selection.

## 26. Hichou et al. — companion repository (`hichou2026repo`)

**Source/access:** Local `pipeline.py`, `paper/main.tex`, result tables/configurations, feature files and headers. The public [repository URL](https://github.com/usmhic/ecg-kan) did not load through the browsing tool; public accessibility of this exact snapshot was not verified.

**Methodology/dataset:** Per-person one-class verification using raw/PCA or 26 wavelet features; six models; 30 people within recording and ten across same-day recordings.

**Experiments/results — local:** KAN wavelet AUROC 0.9771, F1 0.8747 and approximate EER 0.0539 in the 30-person stored table. Across-recording Morlet values are 0.9516, 0.7106 and 0.0860. These are existing artifacts, not newly replicated findings.

**Limitations:** See the [full audit](RESEARCH_AUDIT.md): development/test reuse, unmatched model capacity, preprocessing discrepancies, incomplete provenance and no independent training reproduction.

**Comparison/reproducibility:** A repository citation is not an accepted paper or external validation. Archive an immutable release and dataset/split manifests. The audit script checks stored arithmetic and hashes; the original test suite currently fails to import because PyTorch is missing.

## Missing related work and the limits of originality claims

The searches below were exploratory, completed 29–30 September 2026 through web search with exact titles and combinations including `Kolmogorov-Arnold biometric ECG`, `KAN anomaly detection autoencoder`, `wavelet modulus maxima biometric ECG`, and `ECG one-class verification autoencoder`. Publisher pages, proceedings and author manuscripts were preferred. Search engine coverage and inaccessible full texts prevent an exhaustive novelty determination. No absence claim should be based on these searches alone.

**KAN-AD — Zhou et al.** [Author record](https://arxiv.org/abs/2411.00278) identifies ICML 2025, PMLR 267, 79136–79149. The accessible abstract describes Fourier-based univariate functions for time-series anomaly detection, motivated by limitations of localized B-splines. Four benchmarks are reported; exact dataset-level results were not extracted. This is directly relevant to function-basis and parameter-efficiency arguments, though not biometric verification. Read the proceedings version before implementing a fair baseline.

**AutoKAN.** [IEEE article](https://ieeexplore.ieee.org/document/11115154/), *IEEE Transactions on Consumer Electronics* 71(4), 11303–11314 (2025), DOI 10.1109/TCE.2025.3596250. Its abstract describes a federated KAN autoencoder with adaptive thresholds for IoT healthcare anomaly detection and reports fewer parameters than conventional autoencoders. Dataset-level split and threshold details were not accessible here. It is prior art against broad claims that KAN autoencoders for anomaly detection are new; it is not an ECG-ID performance comparator.

**Biomedical KAN/MLP autoencoder comparison.** [Publisher abstract](https://doi.org/10.1016/j.jocs.2026.102809) describes *Comparing Kolmogorov-Arnold Network Autoencoders versus MLP Autoencoders for the analysis of biomedical data*: AbnormalHeartbeat stethoscope audio, reconstruction/generation/denoising/inpainting/anomaly detection, and matched or reduced KAN parameter counts. Convolutional variants are reported as more robust. The [2025 related preprint](https://arxiv.org/abs/2507.19524) should be checked for version continuity. Heart sounds are not electrical ECG signals. Nevertheless, this study is important to our architecture, capacity and novelty discussion.

**KAN in EEG biometrics.** [IEEE conference abstract](https://doi.org/10.1109/ICBSII65145.2025.11013938) for *Waves Don’t Lie* (2025) compares EEG authentication models including KAN and test-time training. Full datasets and splits were not inspected. It prevents an unqualified “first KAN biometric system” claim, while leaving the narrower ECG task distinct.

**Defensible candidate contribution — proposed framing:** a transparent evaluation of WTMM-inspired statistical features and spline autoencoders under explicitly genuine-only enrollment, with controlled capacity, threshold calibration and documented recording intervals. The contribution must rest on reproducible evidence and a clear comparison with prior work. A first-use claim is optional and presently unproven.

## Comparison rules for the manuscript

| Research setting | Typical decision | What is needed for a fair comparison with this project |
|---|---|---|
| Identity classification | Choose among trained people | Separate accuracy table; no numerical ranking against verification EER |
| Population-pretrained verification | Compare embeddings/templates | Declare pretraining identities and data budget; match probe aggregation |
| Genuine-only per-person verification | Accept/reject a claimed identity | Same enrollment/probe recordings, impostors, threshold policy and capacity search |
| Clinical ECG classification | Assign a rhythm/disease class | Background relevance only; identity outcomes cannot be inferred |
| General anomaly detection | Rank departures from normality | Adapt with explicit normal class, score, calibration and comparable data |
| QRS detection / software / theory | Locate peaks or define processing | Methodological support, not biometric performance evidence |

For every numerical literature table retain dataset/version, actual cohort size, decision unit, learning regime, recording/date split, metric definition, threshold source and source table/page. Use “not reported” only when the full source was checked; use “not verified in accessible material” otherwise. Missing details must not become guessed values.
