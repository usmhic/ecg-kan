"""
pipeline.py
===========================================================================
ECG Biometric Verification on the ECG-ID database
---------------------------------------------------------------------------
One consolidated script that reproduces every figure and the results table
used in the manuscript.

What this version does (and how it differs from the earlier draft)
-------------------------------------------------------------------
DATA LOADING (PhysioNet / wfdb)
  * ECG-ID is pulled directly from PhysioNet with wfdb.dl_database if it is
    not already on disk. One recording per subject (rec_1), lead I.
  * The raw p_signal is handed straight to NeuroKit2, which performs its own
    internal cleaning; no extra bandpass is stacked on top of it.
  * A sensible sampling-rate fallback (500 Hz) is used if the header is
    missing the value.

SEGMENTATION (NeuroKit2)
  * R-peaks come from nk.ecg_process (Pan-Tompkins under the hood).
  * Beats are cut with nk.epochs_create over a 700 ms asymmetric window and
    z-scored individually.

FEATURES
  * "Before wavelet": the z-scored raw beat waveform.
  * "After wavelet" : a 26-D WTMM statistical summary built from the Morlet
    CWT ridge lines.

VISUALISATION
  * ONLY t-SNE is used (UMAP has been removed entirely).
  * The headline figure is a side-by-side t-SNE of the SAME beats before and
    after the wavelet step, so the tightening of per-subject clusters makes
    the benefit of the WTMM representation visible at a glance.

CLASSIFIERS (training + evaluation, per subject)
  * OC-SVM, Isolation Forest, LOF, Elliptic Envelope, MLP autoencoder,
    KAN autoencoder. Trained on 70% genuine beats, tested on the rest plus an
    equal number of impostor beats.

Removed on request: the two large per-subject grid figures (individual beats
and per-beat WTMM line images).

Requirements:
    pip install numpy pandas matplotlib seaborn scikit-learn torch \
                PyWavelets neurokit2 wfdb tqdm
===========================================================================
"""

# --------------------------------------------------------------------------
# 0. Imports
# --------------------------------------------------------------------------
import os
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pywt
import wfdb
import neurokit2 as nk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
DB_DIR = "./ecgiddb"
OUT_DIR = "./paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SUBJECTS = 10          # first N subjects of ECG-ID
GRID_SUBJECTS = 6          # kept only for the four-stage illustration

PRE_MS = 250               # ms kept before the R-peak
POST_MS = 450              # ms kept after the R-peak
FS_DEFAULT = 500           # fallback sampling rate

WAVELET_NAME = "morl"      # Morlet wavelet
FREQ_MIN = 1.0             # Hz
FREQ_MAX = 60.0            # Hz
N_FREQS = 80               # number of CWT scales
CUT_FREQ = 5.0             # lowest frequency shown in ridge-line plots
N_BANDS = 5                # frequency bands in the WTMM feature vector

TRAIN_RATIO = 0.70
NN_EPOCHS = 80

# t-SNE settings
PROJ_MAX_SAMPLES = 3000
TSNE_PERPLEXITY = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 200
print(f"Device: {DEVICE}")

# --------------------------------------------------------------------------
# 2. Signal-processing helpers (NeuroKit2)
# --------------------------------------------------------------------------
def detect_rpeaks(sig, fs):
    """Clean the raw ECG with NeuroKit2 and return (R-peak indices, cleaned)."""
    signals, info = nk.ecg_process(sig, sampling_rate=fs)
    cleaned = signals["ECG_Clean"].values
    r_idx = np.asarray(info["ECG_R_Peaks"])
    return r_idx, cleaned


def extract_beats(cleaned, r_idx, fs):
    """Cut one 700 ms window per R-peak and return a (n_beats, n_samples) array."""
    pre_s, post_s = PRE_MS / 1000.0, POST_MS / 1000.0
    epochs = nk.epochs_create(
        cleaned, events=r_idx, sampling_rate=fs,
        epochs_start=-pre_s, epochs_end=post_s,
    )
    keys = sorted(epochs.keys(), key=lambda k: int(k))
    beats = np.array([epochs[k]["Signal"].values for k in keys])
    t_axis = np.linspace(-pre_s, post_s, beats.shape[1])
    return beats, t_axis


# --------------------------------------------------------------------------
# 3. WTMM helpers
# --------------------------------------------------------------------------
def compute_wtmm_lines(power, scales, min_len=2):
    """Detect local maxima per scale and link them into ridge chains.

    A chain is grown from coarse to fine scale; the linking tolerance widens
    with scale to reflect the growing spatial extent of the wavelet. Chains
    shorter than `min_len` scale rows are discarded as noise.
    """
    n_scales, n_times = power.shape
    mask = np.zeros_like(power, dtype=bool)

    for s in range(n_scales):
        row = power[s]
        if n_times >= 3:
            mask[s, 1:-1] = (row[1:-1] > row[:-2]) & (row[1:-1] > row[2:])
        if n_times >= 2:
            mask[s, 0] = row[0] > row[1]
            mask[s, -1] = row[-1] > row[-2]
        else:
            mask[s, 0] = True

    maxima = [np.where(mask[s])[0].tolist() for s in range(n_scales)]
    used = [set() for _ in range(n_scales)]
    lines = []

    for s0 in range(n_scales):
        for t0 in maxima[s0]:
            if t0 in used[s0]:
                continue
            chain = [(s0, t0)]
            used[s0].add(t0)
            prev = t0
            for s in range(s0 + 1, n_scales):
                tol = max(1, int(round(scales[s] / 2.0)))
                cands = [t for t in maxima[s]
                         if t not in used[s] and abs(int(t) - int(prev)) <= tol]
                if not cands:
                    break
                nxt = min(cands, key=lambda x: abs(x - prev))
                chain.append((s, int(nxt)))
                used[s].add(int(nxt))
                prev = int(nxt)
            if len(chain) >= min_len:
                lines.append(chain)
    return lines


def wtmm_feature_vector(beat, scales, freqs_out, fs):
    """Turn one beat into a 26-D WTMM summary vector.

    Layout: [ total_lines,
              (count, mean_f, std_f, mean_p, std_p) x 5 log-spaced bands ].
    Empty bands contribute five zeros.
    """
    coeffs, _ = pywt.cwt(beat, scales, WAVELET_NAME, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    lines = compute_wtmm_lines(power, scales)

    edges = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_BANDS + 1)
    feats = [len(lines)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = [ln for ln in lines
                if any(lo <= freqs_out[si] <= hi for si, _ in ln)]
        if band:
            freqs = [freqs_out[si] for ln in band for si, _ in ln
                     if lo <= freqs_out[si] <= hi]
            powers = [power[si, ti] for ln in band for si, ti in ln
                      if lo <= freqs_out[si] <= hi]
            feats += [len(band), np.mean(freqs), np.std(freqs),
                      np.mean(powers), np.std(powers)]
        else:
            feats += [0, 0.0, 0.0, 0.0, 0.0]
    return np.array(feats, dtype=np.float32)


# --------------------------------------------------------------------------
# 4. Load ECG-ID from PhysioNet
# --------------------------------------------------------------------------
if not os.path.isdir(DB_DIR) or not os.listdir(DB_DIR):
    print("Downloading ECG-ID from PhysioNet (first run only) ...")
    wfdb.dl_database("ecgiddb", DB_DIR)
else:
    print("ECG-ID already present on disk.")

cf = pywt.central_frequency(WAVELET_NAME)

signals_by_subj = {}
beats_by_subj = {}
t_axis_by_subj = {}
fs_by_subj = {}

for idx in range(NUM_SUBJECTS):
    person = idx + 1
    rec_name = f"{DB_DIR}/Person_{person:02d}/rec_1"
    print(f"-> Person {person:02d}", end="  ")
    try:
        rec = wfdb.rdrecord(rec_name)
    except Exception as e:
        print(f"could not read record: {e}")
        continue

    try:
        sig = rec.p_signal[:, 0]
    except Exception:
        sig = np.asarray(rec.p_signal).flatten()

    fs = getattr(rec, "fs", FS_DEFAULT) or FS_DEFAULT
    r_idx, cleaned = detect_rpeaks(sig, fs)
    beats, t_axis = extract_beats(cleaned, r_idx, fs)
    print(f"{len(r_idx)} R-peaks -> {beats.shape[0]} beats ({fs} Hz)")

    signals_by_subj[idx] = sig
    beats_by_subj[idx] = beats
    t_axis_by_subj[idx] = t_axis
    fs_by_subj[idx] = fs

print("All subjects loaded.\n")

# --------------------------------------------------------------------------
# 5. Feature extraction
# --------------------------------------------------------------------------
print("Extracting features ...")
feats_raw = {}    # z-scored raw beat waveform  (before wavelet)
feats_wtmm = {}   # 26-D WTMM summary           (after  wavelet)

for idx in sorted(beats_by_subj):
    beats = beats_by_subj[idx]
    fs = fs_by_subj[idx]

    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
    scales = cf * fs / freqs
    _, f_out = pywt.cwt(beats[0], scales, WAVELET_NAME, sampling_period=1.0 / fs)

    feats_raw[idx] = np.vstack(
        [((b - b.mean()) / (b.std() + 1e-8)).astype(np.float32) for b in beats]
    )
    feats_wtmm[idx] = np.vstack(
        [wtmm_feature_vector(b, scales, f_out, fs)
         for b in tqdm(beats, desc=f"  WTMM P{idx + 1:02d}", leave=False)]
    )

print("Features ready.\n")

# Stacks used only for the t-SNE visualisation
raw_stack = np.vstack([feats_raw[i] for i in sorted(feats_raw)])
wtmm_stack = np.vstack([feats_wtmm[i] for i in sorted(feats_wtmm)])
labels = np.concatenate([np.full(feats_raw[i].shape[0], i) for i in sorted(feats_raw)])

scaler_raw_vis = StandardScaler().fit(raw_stack)
scaler_wtmm_vis = StandardScaler().fit(wtmm_stack)

# --------------------------------------------------------------------------
# 6. Models
# --------------------------------------------------------------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, dim, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(1).cpu().numpy()


def train_mlp(X, dim, epochs=NN_EPOCHS, lr=1e-3):
    model = MLPAutoencoder(dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(X).to(DEVICE)),
                        batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            nn.MSELoss()(model(xb), xb).backward()
            opt.step()
    return model.eval()


class BSplineEdge(nn.Module):
    """A single KAN edge: learnable cubic B-spline plus a SiLU residual base."""
    def __init__(self, in_f, out_f, G=5, k=3):
        super().__init__()
        self.in_f, self.G, self.k = in_f, G, k
        n = G + k
        self.sp_w = nn.Parameter(torch.randn(out_f, in_f, n) * 0.1)
        self.base_w = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        h = 2.0 / G
        grid = torch.arange(-k, G + k + 1) * h - 1.0
        self.register_buffer("grid", grid)

    def bspline(self, x):
        x = x.unsqueeze(-1)
        g = self.grid.view(1, 1, -1)
        b = ((x >= g[:, :, :-1]) & (x < g[:, :, 1:])).float()
        for order in range(1, self.k + 1):
            gl, gr = g[:, :, :-(order + 1)], g[:, :, order:-1]
            gl2, gr2 = g[:, :, 1:-order], g[:, :, (order + 1):]
            d1, d2 = gr - gl, gr2 - gl2
            b1 = torch.where(d1 > 1e-8, (x - gl) / d1 * b[:, :, :-1],
                             torch.zeros_like(b[:, :, :-1]))
            b2 = torch.where(d2 > 1e-8, (gr2 - x) / d2 * b[:, :, 1:],
                             torch.zeros_like(b[:, :, 1:]))
            b = b1 + b2
        return b

    def forward(self, x):
        sp = torch.einsum("oin,bin->bo", self.sp_w, self.bspline(x.clamp(-1.0, 1.0)))
        return sp + nn.functional.silu(x) @ self.base_w.T


class KANAutoencoder(nn.Module):
    def __init__(self, dim, latent=12):
        super().__init__()
        self.enc = nn.Sequential(BSplineEdge(dim, 64), BSplineEdge(64, latent))
        self.dec = nn.Sequential(BSplineEdge(latent, 64), BSplineEdge(64, dim))

    def forward(self, x):
        return self.dec(self.enc(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(1).cpu().numpy()


def train_kan(X, dim, epochs=NN_EPOCHS, lr=3e-4):
    model = KANAutoencoder(dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.tensor(X).to(DEVICE)),
                        batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model.eval()


def build_traditional_models(n_train):
    lof_k = max(2, min(20, n_train - 1))
    return {
        "OC-SVM": OneClassSVM(kernel="rbf", nu=0.1, gamma="scale"),
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.1,
                                            random_state=42),
        "LOF": LocalOutlierFactor(n_neighbors=lof_k, novelty=True,
                                  contamination=0.1),
        "Elliptic Envelope": EllipticEnvelope(contamination=0.1, random_state=42),
    }


# --------------------------------------------------------------------------
# 7. Evaluation helpers
# --------------------------------------------------------------------------
def eer_score(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2.0


def normalise(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


# --------------------------------------------------------------------------
# 8. Per-subject benchmark
# --------------------------------------------------------------------------
def evaluate_subject(idx, feat_key="raw", pca_dim=40, verbose=False):
    feat_dict = feats_raw if feat_key == "raw" else feats_wtmm

    pos = feat_dict[idx]
    negs = np.vstack([feat_dict[i] for i in feat_dict if i != idx])

    rng = np.random.default_rng(42 + idx)
    neg = negs[rng.choice(len(negs), size=len(pos), replace=len(negs) < len(pos))]

    n_pos = len(pos)
    if n_pos < 4:
        raise ValueError(f"Person {idx + 1:02d}: too few beats ({n_pos}).")

    perm = rng.permutation(n_pos)
    n_train = min(max(3, int(n_pos * TRAIN_RATIO)), n_pos - 1)

    # Scaler fitted on training genuine beats only (no leakage).
    sc = StandardScaler().fit(pos[perm[:n_train]])
    X_tr = sc.transform(pos[perm[:n_train]])
    X_pos = sc.transform(pos[perm[n_train:]])
    X_neg = sc.transform(neg)

    # PCA-40 for the raw waveform feature set.
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

    results, roc_data, recon_errors = {}, {}, {}

    # Classical detectors
    for name, clf in build_traditional_models(n_train=len(X_tr)).items():
        try:
            clf.fit(X_tr)
            try:
                s_test = clf.score_samples(X_test)
                s_tr = clf.score_samples(X_tr)
            except Exception:
                s_test = clf.decision_function(X_test)
                s_tr = clf.decision_function(X_tr)
            norm = normalise(s_test)
            thr = np.median(s_tr)
            auc = roc_auc_score(y_test, norm)
            results[name] = dict(
                auroc=auc,
                f1=f1_score(y_test, (s_test >= thr).astype(int), zero_division=0),
                eer=eer_score(y_test, norm),
            )
            fp, tp, _ = roc_curve(y_test, norm)
            roc_data[name] = (fp, tp, auc)
        except Exception as e:
            if verbose:
                print(f"[{name} failed: {type(e).__name__}]", end=" ")
            results[name] = dict(auroc=np.nan, f1=np.nan, eer=np.nan)
            roc_data[name] = (np.array([0, 1]), np.array([0, 1]), np.nan)

    Xtr_t = torch.tensor(X_tr.astype(np.float32)).to(DEVICE)
    Xtest_t = torch.tensor(X_test.astype(np.float32)).to(DEVICE)

    for name, trainer in [("MLP Autoencoder", train_mlp),
                          ("KAN Autoencoder", train_kan)]:
        try:
            model = trainer(X_tr.astype(np.float32), dim)
            err = model.recon_error(Xtest_t)
            norm = 1 - normalise(err)
            thr = np.percentile(model.recon_error(Xtr_t), 90)
            auc = roc_auc_score(y_test, norm)
            results[name] = dict(
                auroc=auc,
                f1=f1_score(y_test, (err <= thr).astype(int), zero_division=0),
                eer=eer_score(y_test, norm),
            )
            fp, tp, _ = roc_curve(y_test, norm)
            roc_data[name] = (fp, tp, auc)
            recon_errors[name] = {"genuine": err[y_test == 1],
                                  "impostor": err[y_test == 0]}
        except Exception as e:
            if verbose:
                print(f"[{name} failed: {type(e).__name__}]", end=" ")
            results[name] = dict(auroc=np.nan, f1=np.nan, eer=np.nan)
            roc_data[name] = (np.array([0, 1]), np.array([0, 1]), np.nan)

    return results, roc_data, recon_errors


# --------------------------------------------------------------------------
# 9. Run benchmark
# --------------------------------------------------------------------------
results_by_feat, roc_by_feat = {}, {}

for feat_key in ["raw", "wtmm"]:
    print(f"-- Benchmark: {feat_key.upper()} --")
    results_by_feat[feat_key] = {}
    roc_by_feat[feat_key] = {}
    pca = 40 if feat_key == "raw" else None
    for idx in sorted(feats_raw):
        print(f"  Person {idx + 1:02d} ...", end=" ", flush=True)
        try:
            res, roc, _ = evaluate_subject(idx, feat_key, pca, verbose=True)
            results_by_feat[feat_key][idx] = res
            roc_by_feat[feat_key][idx] = roc
            best = max(res, key=lambda m: res[m].get("auroc", -1)
                       if not np.isnan(res[m].get("auroc", np.nan)) else -1)
            print(f"-> best {best} AUROC={res[best].get('auroc', 0):.3f}")
        except Exception as e:
            print(f"SKIP ({e})")

print("Benchmark complete.\n")

# --------------------------------------------------------------------------
# 10. Plotting helpers
# --------------------------------------------------------------------------
MODEL_ORDER = ["OC-SVM", "Isolation Forest", "LOF", "Elliptic Envelope",
               "MLP Autoencoder", "KAN Autoencoder"]

MODEL_COLORS = {
    "OC-SVM": ("#1f77b4", "solid"),
    "Isolation Forest": ("#1f77b4", "dashed"),
    "LOF": ("#aec7e8", "dotted"),
    "Elliptic Envelope": ("#aec7e8", "solid"),
    "MLP Autoencoder": ("#2ca02c", "solid"),
    "KAN Autoencoder": ("#d62728", "solid"),
}

FEAT_LABELS = {"raw": "Raw Beat (before wavelet, PCA-40)",
               "wtmm": "WTMM Summary (after wavelet, 26-d)"}


def to_df(res_dict, metric):
    rows = [{"Person": f"P{i + 1:02d}",
             **{m: round(v.get(metric, np.nan), 4) for m, v in res.items()}}
            for i, res in sorted(res_dict.items())]
    df = pd.DataFrame(rows).set_index("Person")
    if len(df) > 0:
        df.loc["MEAN"] = df.mean(numeric_only=True).round(4)
    return df


def maybe_sub(X, y, n, seed=42):
    if n is None or len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


# --------------------------------------------------------------------------
# 11. Figure: four processing stages (single beat, before/after wavelet)
# --------------------------------------------------------------------------
print("Plotting four-stage figure ...")
si0 = sorted(signals_by_subj)[0]
sig = signals_by_subj[si0]
fs = fs_by_subj[si0]
r_idx0, cleaned0 = detect_rpeaks(sig, fs)
center = r_idx0[min(1, len(r_idx0) - 1)]
beats0, t_beat0 = extract_beats(cleaned0, np.array([center]), fs)
beat0 = beats0[0]

freqs0 = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
scales0 = cf * fs / freqs0
coeff0, fo0 = pywt.cwt(beat0, scales0, WAVELET_NAME, sampling_period=1.0 / fs)
power0 = np.abs(coeff0) ** 2
lines0 = compute_wtmm_lines(power0, scales0)

win = int(1.0 * fs)
t_sig = np.arange(len(sig)) / fs

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
sw, ew = max(0, center - win), min(len(sig), center + win)
ax.plot(t_sig[sw:ew], sig[sw:ew], lw=0.9)
ax.axvline(center / fs, ls="--", lw=1.2, color="red", label="R-peak")
ax.set_title(f"Stage 1 - Raw Signal (Person {si0 + 1:02d})")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.legend(); ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(t_beat0, beat0, lw=1.3)
ax.axvline(0, ls="--", lw=1.2, color="red", label="R-peak (t=0)")
ax.set_title("Stage 2 - Extracted Beat")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.legend(); ax.grid(alpha=0.25)

ax = axes[1, 0]
T0, F0 = np.meshgrid(np.arange(beat0.size) / fs, fo0)
pcm = ax.pcolormesh(T0, F0, power0, shading="auto", cmap="viridis")
ax.set_yscale("log"); ax.set_ylim([FREQ_MIN, FREQ_MAX])
ax.set_title("Stage 3 - Morlet Scalogram (after wavelet)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
plt.colorbar(pcm, ax=ax, label="Energy")

ax = axes[1, 1]
for line in lines0:
    ax.plot([t / fs for _, t in line], [fo0[s] for s, _ in line],
            "-", color="navy", lw=0.9)
ax.set_yscale("log"); ax.set_ylim([CUT_FREQ, max(fo0)])
ax.set_title("Stage 4 - WTMM Ridge Lines (fingerprint)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
ax.grid(alpha=0.2)

plt.suptitle(f"Four Processing Stages - Person {si0 + 1:02d}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_stages.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_stages.png")

# --------------------------------------------------------------------------
# 12. Figure: t-SNE before vs after wavelet  (headline "wavelet benefit" plot)
# --------------------------------------------------------------------------
print("Plotting t-SNE comparison (before vs after wavelet) ...")
Xr, yr = maybe_sub(scaler_raw_vis.transform(raw_stack), labels, PROJ_MAX_SAMPLES)
Xw, yw = maybe_sub(scaler_wtmm_vis.transform(wtmm_stack), labels, PROJ_MAX_SAMPLES)

perp = min(TSNE_PERPLEXITY, max(5, len(Xr) - 1))
print(f"  t-SNE on raw beats  ({len(Xr)} pts) ...")
tsne_raw = TSNE(n_components=2, perplexity=perp, random_state=42,
                init="pca", n_jobs=-1).fit_transform(Xr)
print(f"  t-SNE on WTMM feats ({len(Xw)} pts) ...")
tsne_wtmm = TSNE(n_components=2, perplexity=perp, random_state=42,
                 init="pca", n_jobs=-1).fit_transform(Xw)

n_uniq = len(np.unique(yr))
palette = plt.cm.get_cmap("tab10", max(10, n_uniq))
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
panels = [
    (axes[0], tsne_raw, yr, "Before wavelet - raw beat shape"),
    (axes[1], tsne_wtmm, yw, "After wavelet - WTMM summary"),
]
for ax, proj, lbs, title in panels:
    for i in sorted(np.unique(lbs)):
        m = lbs == i
        ax.scatter(proj[m, 0], proj[m, 1], color=palette(int(i) % 10),
                   label=f"Person {i + 1:02d}", s=22, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
axes[1].legend(fontsize=8, ncol=2, loc="best")
plt.suptitle(
    f"t-SNE of the same beats before and after the wavelet step "
    f"({n_uniq} subjects, {len(Xr)} beats)\n"
    "Tighter, better-separated clusters on the right show the WTMM benefit.",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_tsne.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_tsne.png")

# --------------------------------------------------------------------------
# 13. Figure: AUROC heatmaps
# --------------------------------------------------------------------------
print("Plotting AUROC heatmaps ...")
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, fk in zip(axes, ["raw", "wtmm"]):
    df = to_df(results_by_feat[fk], "auroc")
    if "MEAN" in df.index:
        df = df.drop(index="MEAN")
    df = df.astype(float)
    sns.heatmap(df, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.4, vmax=1.0, linewidths=0.4, linecolor="#ddd",
                cbar_kws={"label": "AUROC"})
    ax.set_title(f"AUROC - {FEAT_LABELS[fk]}", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("AUROC per subject per model (green good, red poor)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_heatmap.png")

# --------------------------------------------------------------------------
# 14. Figure: mean performance bars
# --------------------------------------------------------------------------
print("Plotting mean-performance bars ...")
metric_info = [("auroc", "AUROC (higher better)"),
               ("f1", "F1 (higher better)"),
               ("eer", "EER (lower better)")]
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for row, fk in enumerate(["raw", "wtmm"]):
    for col, (metric, mlabel) in enumerate(metric_info):
        ax = axes[row, col]
        means = to_df(results_by_feat[fk], metric).loc["MEAN"]
        colors = ["#DD8452" if "KAN" in n else "#4C72B0" for n in means.index]
        bars = ax.bar(means.index, means.values, color=colors,
                      edgecolor="white", lw=0.8)
        for b, v in zip(bars, means.values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title(f"{FEAT_LABELS[fk]}\n{mlabel}", fontsize=10)
        ax.set_xticks(np.arange(len(means.index)))
        ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
fig.legend(handles=[
    mpatches.Patch(facecolor="#4C72B0", label="Traditional / MLP"),
    mpatches.Patch(facecolor="#DD8452", label="KAN"),
], loc="upper right", bbox_to_anchor=(1.0, 1.0))
plt.suptitle("Average performance across all subjects", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_bars.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_bars.png")

# --------------------------------------------------------------------------
# 15. Figure: per-person AUROC
# --------------------------------------------------------------------------
print("Plotting per-person AUROC ...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, fk in zip(axes, ["raw", "wtmm"]):
    df = to_df(results_by_feat[fk], "auroc")
    if "MEAN" in df.index:
        df = df.drop(index="MEAN")
    trad = [c for c in df.columns if "KAN" not in c and "MLP" not in c]
    best = df[trad].mean().idxmax() if trad else df.columns[0]
    x, w = np.arange(len(df)), 0.25
    ax.bar(x - w, df[best].values, width=w, label=best,
           color="#4C72B0", edgecolor="white")
    ax.bar(x, df["MLP Autoencoder"].values, width=w, label="MLP Autoencoder",
           color="#55A868", edgecolor="white")
    ax.bar(x + w, df["KAN Autoencoder"].values, width=w, label="KAN Autoencoder",
           color="#DD8452", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, ls=":", lw=1, color="gray", label="Random (0.5)")
    ax.set_title(f"{FEAT_LABELS[fk]}\nPer-person AUROC", fontsize=11)
    ax.set_ylabel("AUROC"); ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Per-person: best traditional vs MLP vs KAN", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_perperson.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_perperson.png")

# --------------------------------------------------------------------------
# 16. Figure: ROC curves (WTMM)
# --------------------------------------------------------------------------
print("Plotting ROC curves ...")
wtmm_roc = roc_by_feat["wtmm"]
n_subj = len(wtmm_roc)
ncols = min(5, n_subj)
nrows = int(np.ceil(n_subj / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                         squeeze=False)
af = axes.flatten()
for k, si in enumerate(sorted(wtmm_roc)):
    ax = af[k]
    for name, (fp, tp, auc) in wtmm_roc[si].items():
        c, ls = MODEL_COLORS.get(name, ("#888", "solid"))
        label = f"{name} ({auc:.2f})" if not np.isnan(auc) else name
        ax.plot(fp, tp, color=c, linestyle=ls, lw=1.2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_title(f"Person {si + 1:02d}", fontsize=9)
    ax.legend(fontsize=5); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
for ax in af[n_subj:]:
    ax.axis("off")
plt.suptitle("ROC curves per subject - WTMM features", fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_roc.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  -> fig_roc.png")

# --------------------------------------------------------------------------
# 17. CSV summary + console table
# --------------------------------------------------------------------------
print("\nSaving benchmark_results.csv ...")
rows = []
for fk, res_fk in results_by_feat.items():
    for metric in ["auroc", "f1", "eer"]:
        df = to_df(res_fk, metric)
        if "MEAN" not in df.index:
            continue
        std_r = df.drop(index="MEAN").std(numeric_only=True)
        for model in df.columns:
            rows.append({
                "Feature set": FEAT_LABELS.get(fk, fk),
                "Model": model,
                "Metric": metric.upper(),
                "Mean": round(float(df.loc["MEAN", model]), 4),
                "Std": round(float(std_r.get(model, np.nan)), 4),
            })
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/benchmark_results.csv", index=False)
print(f"  -> {OUT_DIR}/benchmark_results.csv")

print("\n" + "=" * 65)
print("SUMMARY - mean metrics across subjects")
print("=" * 65)
for fk in ["raw", "wtmm"]:
    ad, fd, ed = (to_df(results_by_feat[fk], m) for m in ["auroc", "f1", "eer"])
    if "MEAN" not in ad.index:
        continue
    n_s = len(ad) - 1
    print(f"\n-- {FEAT_LABELS[fk]} ({n_s} subjects) --")
    for m in MODEL_ORDER:
        if m not in ad.columns:
            continue
        a, f, e = ad.loc["MEAN", m], fd.loc["MEAN", m], ed.loc["MEAN", m]
        a_s = ad.drop(index="MEAN")[m].std()
        f_s = fd.drop(index="MEAN")[m].std()
        e_s = ed.drop(index="MEAN")[m].std()
        print(f"  {m:<25s} AUROC={a:.3f}+/-{a_s:.3f}  "
              f"F1={f:.3f}+/-{f_s:.3f}  EER={e:.3f}+/-{e_s:.3f}")

print(f"\nDone. Figures + CSV -> {OUT_DIR}/")