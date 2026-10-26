"""Run the ECG-ID one-class verification experiment.

The model settings are the fixed settings reported in the paper. The script
loads ECG beats, builds raw and 26-value wavelet feature sets, compares six
models, and saves the seven paper figures plus reproducible metric tables.

Metric tables, per-beat features, logs and figures go to results/local_run by
default. Use --fig-dir paper/figures explicitly to replace manuscript figures.

Examples
--------
Primary experiment:   python pipeline.py --num-subjects 30
Cross-session test:   python pipeline.py --num-subjects 10 --probe-record 2 \
                        --out-dir results/local_run/cross_recording --no-figures
Quick check:          python pipeline.py --smoke-test
Rebuild the summary:  python pipeline.py --summarise-runs

See results/README.md for what separates the three runs the paper reports.
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
import platform
import hashlib
from importlib.metadata import version
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import pywt
import wfdb

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from ecg_features import (
    detect_rpeaks, extract_beats, load_subject_records, load_person,
    compute_wtmm_lines, wtmm_feature_vector, scales_for, feature_names,
    process_record, beat_features,
)
from ecg_plotting import (
    MODEL_ORDER, FEAT_LABELS, to_df, load_summary, make_signal_figures,
    make_result_figures,
)

# Re-export model helpers for existing notebook and script imports.
from ecg_models import (
    DEVICE, MLPAutoencoder, BSplineEdge, KANAutoencoder,
    build_traditional_models, make_autoencoder, train_autoencoder,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    seed: int = 42

    db_dir: str = "./ecgiddb"
    out_dir: str = "results/local_run"
    fig_dir: str = "results/local_run/figures"
    make_figures: bool = True
    synthetic_demo: bool = False

    subject_start: int = 1          # 1-based first ECG-ID subject
    num_subjects: int = 30          # first N subjects of ECG-ID
    probe_record: int | None = None # e.g. 2: enroll rec_1, test on rec_2
    pre_ms: int = 250               # ms kept before the R-peak
    post_ms: int = 450              # ms kept after the R-peak
    fs_default: int = 500           # fallback sampling rate

    wavelet_name: str = "gaus1"     # first Gaussian derivative: classical WTMM choice
    freq_min: float = 1.0           # Hz
    freq_max: float = 60.0          # Hz
    n_freqs: int = 80               # number of CWT scales
    cut_freq: float = 5.0           # lowest frequency shown in ridge-line plots
    n_bands: int = 5                # frequency bands in the WTMM feature vector

    train_ratio: float = 0.70
    pca_dim_raw: int = 40           # PCA dimensionality for the raw-waveform branch

    # -- autoencoder training (shared by MLP and KAN) --
    max_epochs: int = 160
    patience: int = 20
    batch_size: int = 32
    lr_mlp: float = 1e-3
    lr_kan: float = 1e-3
    weight_decay: float = 1e-4
    kan_l1: float = 1e-4            # spline-weight L1 penalty (KAN only)
    kan_grid_size: int = 3          # G, kept small for 12-20 enrollment beats
    kan_spline_degree: int = 3      # k, cubic B-splines
    mlp_hidden_width: int = 64
    kan_hidden_width: int = 8       # capacity-controlled for the small-n regime
    threshold_percentile: float = 100.0  # genuine-only enrollment threshold

    mlp_latent: int = 12
    kan_latent: int = 3

    grid_subjects: int = 6          # columns in the beat / ridge-line grids
    grid_beats: int = 4             # rows in the beat / ridge-line grids
    tsne_perplexity: int = 30
    tsne_max_points: int = 1500


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------
# Output-folder scaffolding
# --------------------------------------------------------------------------
def make_output_dirs(cfg: PipelineConfig) -> dict:
    paths = {
        "root": cfg.out_dir,
        "figures": cfg.fig_dir,
        "metrics": os.path.join(cfg.out_dir, "metrics"),
        "data": os.path.join(cfg.out_dir, "data"),
        "logs": os.path.join(cfg.out_dir, "logs"),
    }
    for key, p in paths.items():
        if key == "figures" and not cfg.make_figures:
            continue
        os.makedirs(p, exist_ok=True)
    return paths


class Tee:
    """Mirrors stdout to a log file so the console transcript is saved too."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr
        self.log = open(path, "a", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.flush()
        self.log.close()


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def eer_score(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2.0


def normalise(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def bootstrap_mean_ci(values, seed=42, n_boot=10000):
    """Subject-level percentile bootstrap CI for a macro mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def evaluate_subject(idx, feat_key, feats_raw, feats_wtmm, cfg: PipelineConfig,
                     probe_feats_raw=None, probe_feats_wtmm=None, verbose=False,
                     audit=None):
    """Evaluate one verifier; optionally record splits, scores and failures.

    Beat indices are zero-based within the arrays from extract_beats. The
    historical fixed split seeds and model order are preserved.
    """
    if audit is None:
        audit = {}
    feat_dict = feats_raw if feat_key == "raw" else feats_wtmm
    probe_dict = probe_feats_raw if feat_key == "raw" else probe_feats_wtmm
    pca_dim = cfg.pca_dim_raw if feat_key == "raw" else None

    pos = feat_dict[idx]
    eval_dict = probe_dict if probe_dict is not None else feat_dict
    if len(eval_dict) < 2:
        raise ValueError("Verification needs at least two participants")
    negs = np.vstack([eval_dict[i] for i in eval_dict if i != idx])
    neg_ids = [(i + 1, beat) for i in eval_dict if i != idx
               for beat in range(len(eval_dict[i]))]

    rng = np.random.default_rng(42 + idx)
    n_pos = len(pos)
    if n_pos < 4:
        raise ValueError(f"Person {idx + 1:02d}: too few beats ({n_pos}).")

    perm = rng.permutation(n_pos)
    if probe_dict is None:
        n_train = min(max(3, int(n_pos * cfg.train_ratio)), n_pos - 1)
        pos_test = pos[perm[n_train:]]
    else:
        # Session-disjoint evaluation: every rec_1 beat is enrollment data;
        # every beat from the selected separate recording is a genuine probe.
        n_train = n_pos
        pos_test = probe_dict[idx]
    n_test = len(pos_test)
    # The protocol specifies a balanced test set. Sampling len(pos) here
    # (as an earlier draft did) silently creates 3-4x more impostors than
    # held-out genuine beats and makes thresholded metrics incomparable.
    if n_test == 0:
        raise ValueError("No genuine probes are available")
    neg_indices = rng.choice(len(negs), size=n_test, replace=len(negs) < n_test)
    neg = negs[neg_indices]

    sc = StandardScaler().fit(pos[perm[:n_train]])
    X_tr = sc.transform(pos[perm[:n_train]])
    X_pos = sc.transform(pos_test)
    X_neg = sc.transform(neg)

    if feat_key == "raw" and pca_dim:
        n_comp = min(int(pca_dim), X_tr.shape[0], X_tr.shape[1])
        if n_comp >= 1 and X_tr.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_pos = pca.transform(X_pos)
            X_neg = pca.transform(X_neg)

    X_test = np.vstack([X_pos, X_neg])
    y_test = np.array([1] * len(X_pos) + [0] * len(X_neg))
    dim = X_tr.shape[1]
    audit.update(
        split_seed=42 + idx, effective_input_dim=dim,
        enrollment_beats=perm[:n_train].tolist(),
        genuine_record=cfg.probe_record or 1,
        genuine_beats=(perm[n_train:].tolist() if probe_dict is None
                       else list(range(n_test))),
        impostor_beats=[list(neg_ids[j]) for j in neg_indices],
        models={},
    )

    results, recon_errors, rocs = {}, {}, {}

    for name, clf in build_traditional_models(n_train=len(X_tr)).items():
        try:
            clf.fit(X_tr)
            s_test = clf.score_samples(X_test)
            s_tr = clf.score_samples(X_tr)
            norm = normalise(s_test)
            # Match the autoencoder policy: the operating boundary accepts
            # every enrollment score. Traditional models assign larger
            # scores to inliers, so this is the complementary percentile of
            # the maximum-error rule used below.
            thr = np.percentile(s_tr, 100.0 - cfg.threshold_percentile)
            auc = roc_auc_score(y_test, norm)
            results[name] = dict(
                auroc=auc,
                f1=f1_score(y_test, (s_test >= thr).astype(int), zero_division=0),
                eer=eer_score(y_test, norm),
            )
            fpr, tpr, _ = roc_curve(y_test, norm)
            rocs[name] = (fpr, tpr, auc)
            audit["models"][name] = dict(
                score_direction="higher_is_genuine", threshold=float(thr),
                enrollment_scores=s_tr.tolist(), probe_scores=s_test.tolist())
        except Exception as e:
            audit["models"][name] = {"error": f"{type(e).__name__}: {e}"}
            warnings.warn(f"Person {idx + 1:02d}, {feat_key}, {name}: {e}")
            if verbose:
                print(f"[{name} failed: {type(e).__name__}]", end=" ")
            results[name] = dict(auroc=np.nan, f1=np.nan, eer=np.nan)

    Xtr_t = torch.tensor(X_tr.astype(np.float32)).to(DEVICE)
    Xtest_t = torch.tensor(X_test.astype(np.float32)).to(DEVICE)

    ae_specs = [
        ("MLP Autoencoder", cfg.lr_mlp, 0.0, cfg.mlp_latent),
        ("KAN Autoencoder", cfg.lr_kan, cfg.kan_l1, cfg.kan_latent),
    ]
    for name, lr, l1, latent in ae_specs:
        try:
            model = make_autoencoder(name, dim, cfg).to(DEVICE)
            model, train_info = train_autoencoder(
                model, X_tr.astype(np.float32),
                max_epochs=cfg.max_epochs, patience=cfg.patience,
                lr=lr, weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
                l1_lambda=l1, seed=42 + idx,
            )
            err = model.recon_error(Xtest_t)
            norm = 1 - normalise(err)
            thr = np.percentile(model.recon_error(Xtr_t), cfg.threshold_percentile)
            auc = roc_auc_score(y_test, norm)
            results[name] = dict(
                auroc=auc,
                f1=f1_score(y_test, (err <= thr).astype(int), zero_division=0),
                eer=eer_score(y_test, norm),
            )
            fpr, tpr, _ = roc_curve(y_test, norm)
            rocs[name] = (fpr, tpr, auc)
            recon_errors[name] = {
                "genuine": err[y_test == 1].tolist(),
                "impostor": err[y_test == 0].tolist(),
                "epochs_run": train_info["epochs_run"],
                "latent": latent,
            }
            audit["models"][name] = dict(
                score_direction="lower_is_genuine", threshold=float(thr),
                enrollment_scores=model.recon_error(Xtr_t).tolist(),
                probe_scores=err.tolist(), training=train_info,
                trainable_parameters=sum(p.numel() for p in model.parameters()
                                         if p.requires_grad))
        except Exception as e:
            audit["models"][name] = {"error": f"{type(e).__name__}: {e}"}
            warnings.warn(f"Person {idx + 1:02d}, {feat_key}, {name}: {e}")
            if verbose:
                print(f"[{name} failed: {type(e).__name__}]", end=" ")
            results[name] = dict(auroc=np.nan, f1=np.nan, eer=np.nan)

    return results, recon_errors, rocs


# --------------------------------------------------------------------------
# Command-line runner and result exports
# --------------------------------------------------------------------------
def summarise_runs(results_dir="results"):
    """Collect every run's benchmark_results.csv into one table.

    Each run folder answers a different question (see results/README.md); this
    puts their headline numbers side by side so the runs can be compared
    without opening three folders.
    """
    rows = []
    for run in sorted(os.listdir(results_dir)):
        cfg_path = os.path.join(results_dir, run, "metrics", "run_config.json")
        bench_path = os.path.join(results_dir, run, "metrics", "benchmark_results.csv")
        if not (os.path.isfile(cfg_path) and os.path.isfile(bench_path)):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            run_cfg = json.load(f)
        if run_cfg.get("data_source") != "physionet":
            continue
        df = pd.read_csv(bench_path).rename(columns={
            "Feature set": "feature_set", "Model": "model", "Metric": "metric",
            "Mean": "mean", "Std": "std",
            "CI95 low": "ci95_low", "CI95 high": "ci95_high"})
        df.insert(0, "experiment", run)
        df.insert(1, "protocol",
                  "same recording" if run_cfg.get("probe_record") is None
                  else "new recording")
        df.insert(2, "wavelet", run_cfg.get("wavelet_name"))
        df.insert(3, "people", run_cfg.get("num_subjects"))
        rows.append(df)

    if not rows:
        print(f"No completed runs found under {results_dir}/")
        return None
    out = pd.concat(rows, ignore_index=True)
    dest = os.path.join(results_dir, "summary_all_experiments.csv")
    out.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(out)} rows from {len(rows)} runs)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="ECG-ID biometric verification pipeline.")
    p.add_argument("--summarise-runs", action="store_true",
                   help="Rebuild results/summary_all_experiments.csv from the "
                        "existing run folders and exit; runs no experiment.")
    p.add_argument("--num-subjects", type=int, default=None)
    p.add_argument("--subject-start", type=int, default=None,
                   help="1-based first ECG-ID subject; useful for untouched-cohort validation.")
    p.add_argument("--probe-record", type=int, default=None,
                   help="Cross-session protocol: train on rec_1 and test on this record number.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Metric tables, per-beat features and logs (default: results/local_run).")
    p.add_argument("--fig-dir", type=str, default=None,
                   help="Figure folder (default: OUT_DIR/figures); explicitly enables figures.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip all figures; use for secondary runs such as the cross-session test.")
    p.add_argument("--db-dir", type=str, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--wavelet", type=str, default=None,
                   help="PyWavelets CWT name (primary: gaus1; ablation: morl).")
    p.add_argument("--kan-hidden", type=int, default=None)
    p.add_argument("--kan-grid-size", type=int, default=None)
    p.add_argument("--kan-latent", type=int, default=None,
                   help="KAN bottleneck width (default: 3).")
    p.add_argument("--threshold-percentile", type=float, default=None)
    p.add_argument("--smoke-test", action="store_true",
                    help="Small subject count + small epoch budget, for a fast sanity run.")
    p.add_argument("--synthetic-demo", action="store_true",
                   help="Use simulated ECG explicitly; implies isolated smoke-test defaults.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    args = parse_args()
    if args.summarise_runs:
        summarise_runs()
        return
    cfg = PipelineConfig()
    if args.smoke_test or args.synthetic_demo:
        cfg.num_subjects = min(cfg.num_subjects, 4)
        cfg.max_epochs = 40
        cfg.patience = 8
        cfg.out_dir = ".tmp/smoke-test"
        cfg.make_figures = False
    cfg.synthetic_demo = args.synthetic_demo
    if cfg.synthetic_demo and args.probe_record is not None:
        raise ValueError("Synthetic demos cannot use the second-recording protocol")
    if args.num_subjects is not None:
        if not 2 <= args.num_subjects <= 90:
            raise ValueError("--num-subjects must be in [2, 90]")
        cfg.num_subjects = args.num_subjects
    if args.subject_start is not None:
        if args.subject_start < 1 or args.subject_start > 90:
            raise ValueError("--subject-start must be in [1, 90]")
        cfg.subject_start = args.subject_start
    if args.probe_record is not None:
        if args.probe_record < 2:
            raise ValueError("--probe-record must be 2 or greater")
        cfg.probe_record = args.probe_record
    if cfg.subject_start - 1 + cfg.num_subjects > 90:
        raise ValueError("Requested subject range exceeds ECG-ID's 90 subjects")
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.fig_dir is not None:
        cfg.fig_dir = args.fig_dir
        cfg.make_figures = True
    else:
        cfg.fig_dir = os.path.join(cfg.out_dir, "figures")
    if args.no_figures:
        cfg.make_figures = False
    if args.db_dir is not None:
        cfg.db_dir = args.db_dir
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be positive")
        cfg.max_epochs = args.max_epochs
    if args.wavelet is not None:
        cfg.wavelet_name = args.wavelet
    if args.kan_hidden is not None:
        if args.kan_hidden < 1:
            raise ValueError("--kan-hidden must be positive")
        cfg.kan_hidden_width = args.kan_hidden
    if args.kan_grid_size is not None:
        if args.kan_grid_size < 1:
            raise ValueError("--kan-grid-size must be positive")
        cfg.kan_grid_size = args.kan_grid_size
    if args.kan_latent is not None:
        if args.kan_latent < 1:
            raise ValueError("--kan-latent must be positive")
        cfg.kan_latent = args.kan_latent
    if args.threshold_percentile is not None:
        if not 0 < args.threshold_percentile <= 100:
            raise ValueError("--threshold-percentile must be in (0, 100]")
        cfg.threshold_percentile = args.threshold_percentile
    if args.seed is not None:
        cfg.seed = args.seed

    set_all_seeds(cfg.seed)
    paths = make_output_dirs(cfg)
    sys.stdout = Tee(os.path.join(paths["logs"],
                                   f"run_{datetime.now():%Y%m%d_%H%M%S}.log"))
    sys.stderr = sys.stdout

    print(f"Device: {DEVICE}")
    print(f"Config: {asdict(cfg)}\n")

    # -- 1. Load data -----------------------------------------------------
    records, data_source = load_subject_records(cfg)
    cf = pywt.central_frequency(cfg.wavelet_name)

    processed_by_subj, beats_by_subj, fs_by_subj = {}, {}, {}
    probe_beats_by_subj, probe_fs_by_subj = {}, {}
    for idx, rec in records:
        data = process_record(rec, cfg)
        fs, beats = data["fs"], data["beats"]
        print(f"{len(data['r_peaks'])} R-peaks -> {len(beats)} beats ({fs} Hz)")
        processed_by_subj[idx] = data
        beats_by_subj[idx] = beats
        fs_by_subj[idx] = fs
        if cfg.probe_record is not None:
            if data_source != "physionet":
                raise RuntimeError("Cross-session evaluation requires the real ECG-ID data")
            probe_name = (f"{cfg.db_dir}/Person_{idx + 1:02d}/"
                          f"rec_{cfg.probe_record}")
            try:
                probe_rec = wfdb.rdrecord(probe_name)
            except Exception as e:
                raise RuntimeError(f"Cannot read probe {probe_name}") from e
            probe = process_record(probe_rec, cfg)
            probe_beats, probe_fs = probe["beats"], probe["fs"]
            probe_beats_by_subj[idx] = probe_beats
            probe_fs_by_subj[idx] = probe_fs
            print(f"   probe rec_{cfg.probe_record}: {len(probe_beats)} beats")
    print(f"All subjects loaded ({data_source}).\n")

    # -- 2. Feature extraction --------------------------------------------
    print("Extracting features ...")
    feats_raw, feats_wtmm = {}, {}
    probe_feats_raw, probe_feats_wtmm = {}, {}
    feature_rows = []  # for the tidy WTMM feature CSV
    partitions = [("rec_1", beats_by_subj, fs_by_subj, feats_raw, feats_wtmm)]
    if cfg.probe_record is not None:
        partitions.append((f"rec_{cfg.probe_record}", probe_beats_by_subj,
                           probe_fs_by_subj, probe_feats_raw, probe_feats_wtmm))
    for record, beat_dict, fs_dict, raw_dict, wtmm_dict in partitions:
        for idx in sorted(beat_dict):
            features = beat_features(beat_dict[idx], fs_dict[idx], cfg)
            raw_dict[idx], wtmm_dict[idx] = features["raw"], features["wtmm"]
            for beat, vector in enumerate(features["wtmm"]):
                feature_rows.append({"subject": idx + 1, "record": record,
                                     "beat": beat,
                                     **dict(zip(feature_names(cfg), vector))})
    print("Features ready.\n")

    pd.DataFrame(feature_rows).to_csv(
        os.path.join(paths["data"], "wtmm_features_long.csv"), index=False)

    # -- 3. Benchmark -------------------------------------------------------
    if len(feats_raw) != cfg.num_subjects:
        raise RuntimeError(f"Expected {cfg.num_subjects} participants, loaded {len(feats_raw)}")
    results_by_feat, recon_by_feat, audit_by_feat = {}, {}, {}
    for feat_key in ["raw", "wtmm"]:
        print(f"-- Benchmark: {feat_key.upper()} --")
        results_by_feat[feat_key] = {}
        recon_by_feat[feat_key] = {}
        audit_by_feat[feat_key] = {}
        for idx in sorted(feats_raw):
            print(f"  Person {idx + 1:02d} ...", end=" ", flush=True)
            audit_by_feat[feat_key][idx] = {}
            try:
                res, recon, _ = evaluate_subject(
                    idx, feat_key, feats_raw, feats_wtmm, cfg,
                    probe_feats_raw if cfg.probe_record is not None else None,
                    probe_feats_wtmm if cfg.probe_record is not None else None,
                    verbose=True, audit=audit_by_feat[feat_key][idx])
                results_by_feat[feat_key][idx] = res
                recon_by_feat[feat_key][idx] = recon
                best = max(res, key=lambda m: res[m].get("auroc", -1)
                           if not np.isnan(res[m].get("auroc", np.nan)) else -1)
                print(f"-> best {best} AUROC={res[best].get('auroc', 0):.3f}")
            except Exception as e:
                audit_by_feat[feat_key][idx]["error"] = f"{type(e).__name__}: {e}"
                warnings.warn(f"Person {idx + 1:02d}, {feat_key}: {e}")
    print("Benchmark complete.\n")

    # save reconstruction-error distributions (for notebook 2, no retraining needed)
    with open(os.path.join(paths["data"], "reconstruction_errors.json"), "w", encoding="utf-8") as f:
        json.dump({fk: {str(k): v for k, v in d.items()} for fk, d in recon_by_feat.items()}, f)

    with open(os.path.join(paths["data"], "evaluation_details.json"), "w", encoding="utf-8") as f:
        json.dump({fk: {str(i + 1): details for i, details in subjects.items()}
                   for fk, subjects in audit_by_feat.items()}, f, indent=2, allow_nan=False)

    # -- 4. Metrics: CSVs + console summary ---------------------------------
    print("\nSaving results tables ...")
    rows_wide, rows_long = [], []
    for fk, res_fk in results_by_feat.items():
        for metric in ["auroc", "f1", "eer"]:
            df = to_df(res_fk, metric)
            if "MEAN" not in df.index:
                continue
            std_r = df.drop(index="MEAN").std(numeric_only=True)
            for model in df.columns:
                lo, hi = bootstrap_mean_ci(
                    df.drop(index="MEAN")[model].to_numpy(),
                    seed=cfg.seed + sum(ord(ch) for ch in f"{fk}-{metric}-{model}"),
                )
                rows_wide.append({
                    "Feature set": FEAT_LABELS.get(fk, fk), "Model": model,
                    "Metric": metric.upper(),
                    "Mean": round(float(df.loc["MEAN", model]), 4),
                    "Std": round(float(std_r.get(model, np.nan)), 4),
                    "CI95 low": round(float(lo), 4),
                    "CI95 high": round(float(hi), 4),
                    "N": int(df.drop(index="MEAN")[model].notna().sum()),
                })
        for idx, res in sorted(res_fk.items()):
            for model, vals in res.items():
                for metric in ("auroc", "f1", "eer"):
                    rows_long.append({
                        "subject": idx + 1, "feature_set": fk, "model": model,
                        "metric": metric, "value": vals.get(metric, np.nan),
                    })
    pd.DataFrame(rows_wide).to_csv(os.path.join(paths["metrics"], "benchmark_results.csv"), index=False)
    pd.DataFrame(rows_long).to_csv(
        os.path.join(paths["metrics"], "per_subject_long.csv"), index=False)

    paired_rows = []
    for fk, res_fk in results_by_feat.items():
        for metric in ["auroc", "f1", "eer"]:
            df = to_df(res_fk, metric).drop(index="MEAN")
            kan = df["KAN Autoencoder"].to_numpy(float)
            for comparator in ["MLP Autoencoder", "OC-SVM", "Isolation Forest",
                               "LOF", "Elliptic Envelope"]:
                comp = df[comparator].to_numpy(float)
                # Positive always means that KAN is better.
                delta = kan - comp if metric != "eer" else comp - kan
                lo, hi = bootstrap_mean_ci(
                    delta, seed=cfg.seed + sum(ord(ch) for ch in f"paired-{fk}-{metric}-{comparator}"))
                paired_rows.append({
                    "feature_set": fk, "metric": metric,
                    "comparator": comparator,
                    "mean_KAN_advantage": float(np.nanmean(delta)),
                    "CI95_low": lo, "CI95_high": hi,
                })
    pd.DataFrame(paired_rows).to_csv(
        os.path.join(paths["metrics"], "paired_comparisons.csv"), index=False)

    # Equal KAN/MLP widths are not equal-capacity comparisons: every KAN edge
    # owns G+k spline coefficients plus a residual coefficient. Record actual
    # neural parameter counts for every effective input dimension encountered.
    complexity_rows = []
    for fk in ("raw", "wtmm"):
        dims = sorted({details["effective_input_dim"]
                       for details in audit_by_feat[fk].values()
                       if "effective_input_dim" in details})
        for dim in dims:
            for model_name in ("MLP Autoencoder", "KAN Autoencoder"):
                model = make_autoencoder(model_name, dim, cfg)
                complexity_rows.append({
                    "feature_set": fk,
                    "effective_input_dim": dim,
                    "model": model_name,
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad),
                })
    pd.DataFrame(complexity_rows).to_csv(
        os.path.join(paths["metrics"], "model_complexity.csv"), index=False)
    print(f"  -> {paths['metrics']}/benchmark_results.csv")
    print(f"  -> {paths['metrics']}/per_subject_long.csv")
    print(f"  -> {paths['metrics']}/paired_comparisons.csv")
    print(f"  -> {paths['metrics']}/model_complexity.csv")

    # -- 5. Figures -----------------------------------------------------------
    # Signal figures show what the processing does; result figures are drawn
    # from the tables just written, as research/make_paper_figures.py draws
    # them from the saved runs.
    if cfg.make_figures:
        people = {idx + 1: {**processed_by_subj[idx], "raw": feats_raw[idx],
                            "wtmm": feats_wtmm[idx]}
                  for idx in sorted(feats_raw)}
        make_signal_figures(people, cf, cfg, paths["figures"])
        make_result_figures(
            pd.DataFrame(rows_long),
            load_summary(os.path.join(paths["metrics"], "benchmark_results.csv")),
            {fk: {str(k): v for k, v in d.items()} for fk, d in recon_by_feat.items()},
            paths["figures"])
    else:
        print("Figures skipped (--no-figures).\n")

    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("SUMMARY - mean metrics across subjects")
    summary_lines.append("=" * 65)
    for fk in ["raw", "wtmm"]:
        ad, fd, ed = (to_df(results_by_feat[fk], m) for m in ["auroc", "f1", "eer"])
        if "MEAN" not in ad.index:
            continue
        n_s = len(ad) - 1
        summary_lines.append(f"\n-- {FEAT_LABELS[fk]} ({n_s} subjects) --")
        for m in MODEL_ORDER:
            if m not in ad.columns:
                continue
            a, f, e = ad.loc["MEAN", m], fd.loc["MEAN", m], ed.loc["MEAN", m]
            a_s = ad.drop(index="MEAN")[m].std()
            f_s = fd.drop(index="MEAN")[m].std()
            e_s = ed.drop(index="MEAN")[m].std()
            summary_lines.append(f"  {m:<25s} AUROC={a:.3f}+/-{a_s:.3f}  "
                                  f"F1={f:.3f}+/-{f_s:.3f}  EER={e:.3f}+/-{e_s:.3f}")
    print("\n".join(summary_lines))

    # -- 6. run_config.json + auto README -----------------------------------
    run_config = asdict(cfg)
    run_config["data_source"] = data_source
    run_config["timestamp"] = datetime.now(timezone.utc).isoformat()
    run_config["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": DEVICE, "processor": platform.processor(),
        "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in (
            "numpy", "scipy", "pandas", "PyWavelets", "scikit-learn",
            "torch", "wfdb", "neurokit2", "matplotlib")},
    }
    run_config["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("pipeline.py", "ecg_features.py", "ecg_models.py", "ecg_plotting.py")
    }
    run_config["actual_subjects"] = [idx + 1 for idx in sorted(feats_raw)]
    with open(os.path.join(paths["metrics"], "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    # The per-run README states only what makes THIS run different; the file
    # layout, shared by every run, is documented once in results/README.md.
    wavelet_label = {"gaus1": "Gaussian derivative (gaus1)",
                     "gaus2": "second Gaussian derivative (gaus2)",
                     "morl": "Morlet (morl)"}.get(cfg.wavelet_name, cfg.wavelet_name)
    if cfg.probe_record is None:
        protocol = (f"**same recording** - learn each person from "
                    f"{cfg.train_ratio:.0%} of their `rec_1` beats, then test "
                    f"on the other {1 - cfg.train_ratio:.0%} of the same "
                    f"recording (plus an equal number of other people's "
                    f"beats).")
    else:
        protocol = (f"**new recording** - learn each person from *all* of "
                    f"their `rec_1` beats, then test on `rec_{cfg.probe_record}`, "
                    f"a separate recording. Recording numbers do not establish "
                    f"a gap in calendar days.")
    first_subj = cfg.subject_start
    last_subj = cfg.subject_start - 1 + cfg.num_subjects
    run_name = os.path.basename(os.path.normpath(paths["root"]))

    readme = f"""# {run_name}

Generated {datetime.now():%Y-%m-%d %H:%M} | data source: **{data_source}**
{"" if data_source == "physionet" else (
"> **SYNTHETIC DEMO DATA** - Simulated signals were explicitly requested. "
"These numbers are a code demonstration and must not be used as biometric "
"evidence in the paper.")}

| | |
|---|---|
| Protocol | {protocol} |
| Wavelet | {wavelet_label} |
| People | {cfg.num_subjects} (Person_{first_subj:02d}-Person_{last_subj:02d}) |
| Seed | {cfg.seed} |
| Figures | {f"written to `{cfg.fig_dir}/`" if cfg.make_figures else "skipped (`--no-figures`)"} |

See [`../README.md`](../README.md) for how this run compares with the others
and for what each file in `metrics/`, `data/` and `logs/` contains.
"""
    with open(os.path.join(paths["root"], "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # Rebuilding the historical cross-run table is an explicit --summarise-runs action.
    failures = [f"{fk}/Person_{i + 1:02d}/{name}"
                for fk, subjects in audit_by_feat.items()
                for i, details in subjects.items()
                for name in MODEL_ORDER
                if "error" in details or "error" in details.get("models", {}).get(name, {})]
    if failures:
        raise RuntimeError("Incomplete benchmark; inspect evaluation_details.json: "
                           + ", ".join(failures))

    print(f"\nDone. All outputs -> {paths['root']}/")


if __name__ == "__main__":
    try:
        main()
    finally:
        if isinstance(sys.stdout, Tee):
            transcript = sys.stdout
            sys.stdout = transcript.terminal
            sys.stderr = transcript.error_terminal
            transcript.close()
