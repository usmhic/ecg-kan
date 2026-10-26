import csv
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as PIPELINE


class PipelineTests(unittest.TestCase):
    def test_wavelet_power_keeps_amplitude_while_raw_branch_normalises(self):
        cfg = PIPELINE.PipelineConfig(n_freqs=12)
        t = np.arange(350) / 500
        beats = np.array([np.sin(2 * np.pi * 8 * t) + 0.2 * np.cos(2 * np.pi * 23 * t)])
        original = PIPELINE.beat_features(beats, 500, cfg)
        doubled = PIPELINE.beat_features(2 * beats, 500, cfg)
        np.testing.assert_allclose(original['raw'], doubled['raw'], atol=1e-6)
        # Linear CWT means doubling amplitude multiplies power moments by four.
        expected = original['wtmm'].copy()
        for band in range(cfg.n_bands):
            expected[:, 1 + 5 * band + 3:1 + 5 * band + 5] *= 4
        np.testing.assert_allclose(expected, doubled['wtmm'], rtol=1e-5, atol=1e-6)

    def test_unavailable_database_never_silently_uses_demo_data(self):
        import ecg_features
        with patch.object(ecg_features, 'load_ecg_id_from_physionet', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'ECG-ID is unavailable'):
                ecg_features.load_subject_records(PIPELINE.PipelineConfig())

    def test_evaluation_manifest_reconstructs_metrics_and_keeps_probes_out_of_enrolment(self):
        rng = np.random.default_rng(9)
        features = {0: rng.normal(size=(20, 4)), 1: rng.normal(3, size=(20, 4))}
        cfg = PIPELINE.PipelineConfig(max_epochs=2)
        PIPELINE.set_all_seeds(cfg.seed)
        audit = {}
        results, _, _ = PIPELINE.evaluate_subject(
            0, 'wtmm', features, features, cfg, audit=audit)
        self.assertEqual(audit['effective_input_dim'], 4)
        self.assertFalse(set(audit['enrollment_beats']) & set(audit['genuine_beats']))
        self.assertEqual(len(audit['genuine_beats']), len(audit['impostor_beats']))
        self.assertTrue(all(person == 2 for person, _ in audit['impostor_beats']))
        labels = np.array([1] * 6 + [0] * 6)
        for name, details in audit['models'].items():
            with self.subTest(model=name):
                self.assertNotIn('error', details)
                scores = np.asarray(details['probe_scores'])
                higher = details['score_direction'] == 'higher_is_genuine'
                ranking = scores if higher else -scores
                decisions = scores >= details['threshold'] if higher else scores <= details['threshold']
                self.assertAlmostEqual(results[name]['auroc'], PIPELINE.roc_auc_score(labels, ranking))
                self.assertAlmostEqual(results[name]['f1'], PIPELINE.f1_score(labels, decisions))

    def test_second_recording_uses_all_enrolment_beats_for_raw_pca(self):
        rng = np.random.default_rng(12)
        enrolled = {i: rng.normal(i, 1, size=(12, 16)) for i in range(2)}
        probes = {i: rng.normal(i, 1, size=(8, 16)) for i in range(2)}
        audit = {}
        # One conventional detector suffices here; this check concerns the split and PCA.
        with patch.object(PIPELINE, 'build_traditional_models', return_value={}):
            PIPELINE.evaluate_subject(0, 'raw', enrolled, enrolled,
                PIPELINE.PipelineConfig(probe_record=2, max_epochs=1),
                probes, probes, audit=audit)
        self.assertEqual(len(audit['enrollment_beats']), 12)
        self.assertEqual(audit['effective_input_dim'], 12)
        self.assertEqual(audit['genuine_record'], 2)
        self.assertEqual(audit['genuine_beats'], list(range(8)))

    def test_per_feature_grid_follows_observed_ranges(self):
        edge = PIPELINE.BSplineEdge(2, 3, G=3, k=3)
        x = torch.tensor([
            [-2.0, 100.0], [-1.0, 110.0], [0.0, 120.0], [1.0, 130.0], [2.0, 140.0]
        ])
        edge.fit_grid(x)
        low = edge.grid[:, edge.k]
        high = edge.grid[:, -(edge.k + 1)]
        self.assertLess(float(high[0]), float(low[1]))
        self.assertTrue(torch.isfinite(edge(x)).all())

    def test_compact_kan_has_fewer_parameters_than_mlp(self):
        mlp = PIPELINE.MLPAutoencoder(26, hidden=64, latent=12)
        kan = PIPELINE.KANAutoencoder(26, hidden=8, latent=3, G=3, k=3)
        mlp_n = sum(p.numel() for p in mlp.parameters())
        kan_n = sum(p.numel() for p in kan.parameters())
        self.assertEqual(mlp_n, 8422)
        self.assertEqual(kan_n, 3248)
        self.assertLess(kan_n, mlp_n)

    def test_notebooks_are_valid_json(self):
        for name in ("ecg_01_signal_processing.ipynb",
                     "ecg_02_identity_verification.ipynb"):
            notebook = json.loads((ROOT / "notebook" / name).read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertGreater(len(notebook["cells"]), 0)

    def test_results_folders_match_the_documented_runs(self):
        """The three run folders and the combined summary are what
        results/README.md and the paper point at."""
        results = ROOT / "results"
        for name in ("1_same_recording_gaussian", "2_new_recording_gaussian",
                     "3_new_recording_morlet"):
            with self.subTest(run=name):
                self.assertTrue((results / name / "metrics" /
                                 "benchmark_results.csv").is_file())
        self.assertTrue((results / "summary_all_experiments.csv").is_file())
        self.assertTrue((results / "README.md").is_file())

    def test_every_figure_the_paper_includes_exists(self):
        tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        included = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
        self.assertGreater(len(included), 0)
        for name in included:
            with self.subTest(figure=name):
                self.assertTrue((ROOT / "paper" / "figures" / name).is_file(),
                                f"{name} is included by main.tex but missing; "
                                f"re-run pipeline.py to regenerate the figures")

    def test_result_figures_draw_from_committed_metric_tables(self):
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import pandas as pd
        import ecg_plotting
        metrics_dir = ROOT / "results" / "1_same_recording_gaussian" / "metrics"
        metrics = pd.read_csv(metrics_dir / "per_subject_long.csv")
        summary = ecg_plotting.load_summary(metrics_dir / "benchmark_results.csv")
        self.assertEqual(set(summary.feature_set), {"raw", "wtmm"})
        ecg_plotting.use_paper_style()
        with tempfile.TemporaryDirectory() as tmp:
            ecg_plotting.plot_mean_summary(summary, tmp)
            ecg_plotting.plot_auroc_heatmap(metrics, tmp)
            ecg_plotting.plot_kan_vs_mlp(metrics, tmp)
            for name in ("fig_bars.png", "fig_heatmap.png", "fig_perperson.png"):
                self.assertTrue((Path(tmp) / name).stat().st_size > 0)

    def test_mean_roc_uses_best_rate_at_or_below_each_false_acceptance(self):
        import ecg_plotting
        fpr, tpr = ecg_plotting._person_roc({"genuine": [1.0, 3.0],
                                             "impostor": [2.0, 4.0]})
        mean = ecg_plotting._mean_roc([(fpr, tpr)], np.array([0.0, 0.49, 0.5, 1.0]))
        np.testing.assert_allclose(mean, [0.5, 0.5, 1.0, 1.0])

    def test_bootstrap_ci_is_deterministic_and_ordered(self):
        values = np.array([0.8, 0.9, 1.0])
        first = PIPELINE.bootstrap_mean_ci(values, seed=7, n_boot=200)
        second = PIPELINE.bootstrap_mean_ci(values, seed=7, n_boot=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], values.mean())
        self.assertGreaterEqual(first[1], values.mean())

    def test_saved_bootstrap_intervals_follow_the_documented_rule(self):
        for folder in sorted((ROOT / 'results').glob('*/metrics')):
            groups = defaultdict(dict)
            with (folder / 'per_subject_long.csv').open() as handle:
                for row in csv.DictReader(handle):
                    groups[row['feature_set'], row['metric'], row['model']][int(row['subject'])] = round(float(row['value']), 4)
            cfg = json.loads((folder / 'run_config.json').read_text())
            for filename in ('benchmark_results.csv', 'paired_comparisons.csv'):
                with (folder / filename).open() as handle:
                    for row in csv.DictReader(handle):
                        if filename.startswith('benchmark'):
                            fk = 'raw' if row['Feature set'].startswith('Raw') else 'wtmm'
                            metric, model = row['Metric'].lower(), row['Model']
                            by_person = groups[fk, metric, model]
                            values = [by_person[s] for s in sorted(by_person)]
                            seed_label = f'{fk}-{metric}-{model}'
                            expected = [float(row[k]) for k in ('CI95 low', 'CI95 high')]
                        else:
                            fk, metric, model = row['feature_set'], row['metric'], row['comparator']
                            kan, other = groups[fk, metric, 'KAN Autoencoder'], groups[fk, metric, model]
                            values = [(kan[s] - other[s]) * (-1 if metric == 'eer' else 1) for s in sorted(kan)]
                            seed_label = f'paired-{fk}-{metric}-{model}'
                            expected = [float(row[k]) for k in ('CI95_low', 'CI95_high')]
                        actual = PIPELINE.bootstrap_mean_ci(
                            values, seed=cfg['seed'] + sum(map(ord, seed_label)))
                        np.testing.assert_allclose(actual, expected, atol=0.00011, rtol=0)


if __name__ == "__main__":
    unittest.main()
