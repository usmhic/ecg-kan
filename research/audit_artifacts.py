"""Audit existing research artifacts without training or changing source results.

Standard library only. Run from any directory with Python 3.10+:
    python research/audit_artifacts.py

Outputs describe arithmetic and local provenance checks, not model reproduction.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "audit"
RUNS = ["1_same_recording_gaussian", "2_new_recording_gaussian",
        "3_new_recording_morlet"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auroc_from_errors(genuine, impostor):
    """Probability that a genuine sample has lower error, with half credit for ties."""
    if not genuine or not impostor:
        raise ValueError("Both score classes must be nonempty")
    if not all(math.isfinite(v) for v in genuine + impostor):
        raise ValueError("Scores must be finite")
    return sum((g < i) + 0.5 * (g == i) for g in genuine for i in impostor) / (
        len(genuine) * len(impostor))


def main():
    OUT.mkdir(exist_ok=True)
    report = {"scope": "Existing artifact audit; no training or signal processing",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs": {}, "record_dates": [], "input_sha256": {},
              "missing_optional_inputs": [],
              "bibliography_keys": re.findall(r"\\bibitem\{([^}]+)\}",
                  (ROOT / "paper/main.tex").read_text(encoding="utf-8"))}
    summary_rows = []
    for run in RUNS:
        base = ROOT / "results" / run
        metrics = base / "metrics"
        cfg = json.loads((metrics / "run_config.json").read_text())
        rows = read_csv(metrics / "per_subject_long.csv")
        groups = defaultdict(dict)
        duplicate_keys, nonfinite = [], []
        for row in rows:
            key = (row["feature_set"], row["model"], row["metric"])
            subject, value = int(row["subject"]), float(row["value"])
            if subject in groups[key]:
                duplicate_keys.append([*key, subject])
            groups[key][subject] = value
            if not math.isfinite(value):
                nonfinite.append([*key, subject])
        deviations = []
        for row in read_csv(metrics / "benchmark_results.csv"):
            feature = "raw" if row["Feature set"].startswith("Raw") else "wtmm"
            key = (feature, row["Model"], row["Metric"].lower())
            values = [v for v in groups[key].values() if math.isfinite(v)]
            # Pipeline rounds subject values to 4 decimals before aggregation.
            rounded = [round(v, 4) for v in values]
            mean, sd = statistics.mean(rounded), statistics.stdev(rounded)
            deviations.append({"key": list(key),
                "mean_absolute_error": abs(round(mean, 4) - float(row["Mean"])),
                "std_absolute_error": abs(round(sd, 4) - float(row["Std"]))})
        paired_deviations = []
        for row in read_csv(metrics / "paired_comparisons.csv"):
            fk, metric, model = row["feature_set"], row["metric"], row["comparator"]
            kan, comp = groups[(fk, "KAN Autoencoder", metric)], groups[(fk, model, metric)]
            common = sorted(set(kan) & set(comp))
            deltas = [(round(kan[s], 4) - round(comp[s], 4)) *
                      (-1 if metric == "eer" else 1) for s in common
                      if math.isfinite(kan[s]) and math.isfinite(comp[s])]
            paired_deviations.append({"feature_set": fk, "metric": metric,
                "comparator": model, "n_pairs": len(deltas),
                "mean_absolute_error": abs(statistics.mean(deltas) -
                                           float(row["mean_KAN_advantage"]))})
        counts = defaultdict(int)
        features_path = base / "data/wtmm_features_long.csv"
        if features_path.exists():
            for row in read_csv(features_path):
                counts[(int(row["subject"]), row["partition"])] += 1
        else:
            report["missing_optional_inputs"].append(features_path.relative_to(ROOT).as_posix())
        score_checks = []
        errors_path = base / "data/reconstruction_errors.json"
        if errors_path.exists():
            errors = json.loads(errors_path.read_text(encoding="utf-8"))
            for feature, subjects in errors.items():
                for zero_based_subject, models in subjects.items():
                    subject = int(zero_based_subject) + 1
                    for model, scores in models.items():
                        auc = auroc_from_errors(scores["genuine"], scores["impostor"])
                        score_checks.append({"subject": subject, "feature_set": feature,
                            "model": model, "genuine_count": len(scores["genuine"]),
                            "impostor_count": len(scores["impostor"]),
                            "recomputed_auroc": auc,
                            "absolute_error": abs(auc - groups[(feature, model, "auroc")][subject])})
        else:
            report["missing_optional_inputs"].append(errors_path.relative_to(ROOT).as_posix())
        run_report = {"data_source_label": cfg.get("data_source"),
            "timestamp_recorded": cfg.get("timestamp"),
            "configured_subject_count": cfg["num_subjects"],
            "actual_subject_count": len({int(r["subject"]) for r in rows}),
            "row_count": len(rows), "duplicate_keys": duplicate_keys,
            "nonfinite_values": nonfinite,
            "max_mean_error": max(d["mean_absolute_error"] for d in deviations),
            "max_std_error": max(d["std_absolute_error"] for d in deviations),
            "max_paired_mean_error": max(d["mean_absolute_error"] for d in paired_deviations),
            "score_auroc_checks": score_checks,
            "score_auroc_check_count": len(score_checks),
            "max_score_auroc_error": max((d["absolute_error"] for d in score_checks), default=None),
            "checks": deviations, "paired_checks": paired_deviations,
            "feature_counts": [{"subject": s, "partition": p, "beats": n}
                               for (s, p), n in sorted(counts.items())]}
        expected_subjects = set(range(cfg.get("subject_start", 1),
                                      cfg.get("subject_start", 1) + cfg["num_subjects"]))
        run_report["passed"] = (
            not duplicate_keys and not nonfinite and len(groups) == 36
            and all(set(values) == expected_subjects for values in groups.values())
            and run_report["max_mean_error"] <= 0.00011
            and run_report["max_std_error"] <= 0.00011
            and run_report["max_paired_mean_error"] <= 1e-10
            and (run_report["max_score_auroc_error"] or 0) <= 1e-10)
        report["runs"][run] = run_report
        for (feature, model, metric), by_subject in sorted(groups.items()):
            for cohort, start, stop in [("all", 1, 90), ("persons_01_10", 1, 10),
                                        ("persons_11_30", 11, 30)]:
                values = [v for s, v in by_subject.items()
                          if start <= s <= stop and math.isfinite(v)]
                if values:
                    summary_rows.append({"run": run, "cohort": cohort,
                        "feature_set": feature, "model": model, "metric": metric,
                        "n_subjects": len(values), "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else None})
        for path in sorted(base.rglob("*")):
            if path.is_file():
                report["input_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path)
    for subject in range(1, 11):
        dates = []
        for record in (1, 2):
            path = ROOT / "ecgiddb" / f"Person_{subject:02d}" / f"rec_{record}.hea"
            if path.exists():
                match = re.search(r"ECG date:\s*(\d{2}\.\d{2}\.\d{4})", path.read_text())
                dates.append(match.group(1) if match else None)
                report["input_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path)
            else:
                dates.append(None)
                report["missing_optional_inputs"].append(path.relative_to(ROOT).as_posix())
        gap = ((datetime.strptime(dates[1], "%d.%m.%Y") -
                datetime.strptime(dates[0], "%d.%m.%Y")).days
               if all(dates) else None)
        report["record_dates"].append({"subject": subject, "rec_1": dates[0],
                                        "rec_2": dates[1], "gap_days": gap})
    for name in ("research/audit_artifacts.py", "research/build_paper_tables.py",
                 "pipeline.py", "ecg_features.py", "ecg_models.py", "ecg_plotting.py",
                 "paper/main.tex", "environment.yml", "requirements.txt",
                 "results/kan_design_ablation.csv", "results/summary_all_experiments.csv"):
        report["input_sha256"][name] = sha256(ROOT / name)
    for path in sorted((ROOT / "paper").glob("*/*.png")) + sorted((ROOT / "paper/tables").glob("*.tex")):
        report["input_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path)
    (OUT / "artifact_checks.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUT / "cohort_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT / "record_dates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["record_dates"][0]))
        writer.writeheader()
        writer.writerows(report["record_dates"])
    print(json.dumps({"references": len(report["bibliography_keys"]),
        "runs": {k: {key: v[key] for key in ("passed", "actual_subject_count", "row_count",
            "max_mean_error", "max_std_error", "max_paired_mean_error",
            "score_auroc_check_count", "max_score_auroc_error")}
            for k, v in report["runs"].items()},
        "record_dates": report["record_dates"]}, indent=2))
    if not all(run["passed"] for run in report["runs"].values()):
        raise SystemExit("Artifact checks failed; see research/audit/artifact_checks.json")


if __name__ == "__main__":
    main()
