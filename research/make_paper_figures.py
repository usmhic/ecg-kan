"""Rebuild the manuscript figures from saved results, without training.

Result figures (mean performance, AUROC heatmap, KAN versus MLP, ROC) use only
results/1_same_recording_gaussian. Signal figures (processing steps, example
beats, t-SNE) also need the local ECG-ID copy; they are skipped when it is
absent. The regenerated wavelet features are compared with the saved feature
file so that a package change cannot silently alter the t-SNE input.

    python research/make_paper_figures.py                 # all figures
    python research/make_paper_figures.py --results-only  # no ECG-ID needed
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pywt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg_features import load_person  # noqa: E402
from ecg_plotting import (  # noqa: E402
    load_summary, make_result_figures, make_signal_figures,
)
from pipeline import PipelineConfig  # noqa: E402

RUN = ROOT / "results" / "1_same_recording_gaussian"


def check_features(people):
    """Compare regenerated wavelet features with the saved per-beat file."""
    path = RUN / "data" / "wtmm_features_long.csv"
    if not path.exists():
        print("Saved feature file absent; regenerated features not compared.")
        return
    saved = pd.read_csv(path)
    worst = 0.0
    for pid, data in people.items():
        old = saved[saved.subject == pid].iloc[:, 3:].to_numpy(float)
        new = np.asarray(data["wtmm"], float)
        if old.shape != new.shape:
            raise SystemExit(f"Person {pid:02d}: saved {old.shape} beats/features, "
                             f"regenerated {new.shape}")
        worst = max(worst, float(np.max(np.abs(old - new) / (np.abs(old) + 1e-6))))
    print(f"Regenerated features match the saved file "
          f"(largest relative difference {worst:.2e}).")
    if worst > 1e-3:
        raise SystemExit("Regenerated features differ from the saved file")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fig-dir", default=str(ROOT / "paper" / "figures"))
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()

    metrics = pd.read_csv(RUN / "metrics" / "per_subject_long.csv")
    summary = load_summary(RUN / "metrics" / "benchmark_results.csv")
    with (RUN / "data" / "reconstruction_errors.json").open(encoding="utf-8") as f:
        recon = json.load(f)
    make_result_figures(metrics, summary, recon, args.fig_dir)

    cfg = PipelineConfig()
    cfg.db_dir = str(ROOT / "ecgiddb")
    if args.results_only:
        return
    if not (ROOT / "ecgiddb" / "Person_01").is_dir():
        print("ECG-ID not found; signal figures skipped.")
        return
    people = {p: load_person(p, cfg) for p in range(1, cfg.num_subjects + 1)}
    print(f"Loaded {sum(len(d['beats']) for d in people.values())} beats "
          f"from {len(people)} people.")
    check_features(people)
    make_signal_figures(people, pywt.central_frequency(cfg.wavelet_name),
                        cfg, args.fig_dir)


if __name__ == "__main__":
    main()
