# Research review and reproducibility audit

The original review was prepared on 30 September 2026. See [release notes](RELEASE_NOTES.md) for the 10 October revision.

- [Paper-by-paper review](PAPER_REVIEW.md): methodology, datasets, experiments, reported results, limitations, relevance and source-access status for every reference; selected missing related work; limits of novelty claims.
- [Research audit](RESEARCH_AUDIT.md): submission blockers, code/manuscript discrepancies, rechecked local results and a proposed confirmatory experiment design.
- [Audit script](audit_artifacts.py): standard-library-only arithmetic/provenance checks. It does not import the training pipeline, download data or run models.
- [Machine-readable checks](audit/artifact_checks.json), [cohort reaggregation](audit/cohort_summary.csv), and [recording dates](audit/record_dates.csv).

Re-run the artifact audit from the repository root:

```powershell
.\.venv\Scripts\python.exe research/audit_artifacts.py
```

Or use `python research/audit_artifacts.py` in any Python 3.10+ environment. Outputs are written only under `research/audit/`. Optional local feature files and raw headers are ignored by the repository; when absent, the audit reports the missing evidence rather than claiming to have checked it.

The audit was executed successfully. The September review could not run model tests because PyTorch was missing. The October revision installs the required dependencies and validates the refactor; it does not independently reproduce the headline experiments. All numerical results are explicitly labelled as published reports, existing local artifacts, or reaggregation. Proposed experiments are not completed work. The original metric CSVs were preserved; the manuscript was revised and its tables are now generated from those CSVs.

This is a critical narrative review, not an exhaustive systematic review or a guarantee of acceptance. Some sources were accessible only through their abstracts, previews or preprints; outstanding verification is recorded in each entry.

Generate or check the manuscript tables without importing the training stack:

```text
python research/build_paper_tables.py
python research/build_paper_tables.py --check
```
