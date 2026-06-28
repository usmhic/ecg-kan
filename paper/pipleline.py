"""
pipeline.py
======================
ECG Biometric Verification — ECG-ID Database
---------------------------------------------
Pipeline matched to the two companion notebooks (ECG Signal Processing +
ECG Identity) so that results are directly comparable.

Key alignment changes vs v4
────────────────────────────
DATA LOADING
  • No bandpass pre-filter  — notebooks feed the raw p_signal straight into
    nk.ecg_process, which applies its own internal cleaning.  Bandpass is
    removed here to match.
  • Single recording per subject  — rec_1 only, matching notebook Step 4.
  • No beat quality gate  — notebooks keep every beat returned by
    nk.epochs_create without SNR / R-peak plausibility filtering.
  • No min-max normalisation  — notebooks use z-score per beat (subtract
    mean, divide by std) for the raw feature set, not min-max.
  • fs fallback  — notebooks use FS_DEFAULT=500 when fs is missing/zero.

FEATURE EXTRACTION
  • WTMM lines: min_len=2, tolerance = floor(scale/2) with NO upper cap —
    identical to both notebooks.
  • WTMM band statistics: raw power (not log1p) — identical to notebook 2.
  • No fiducial (9-d) feature set — not present in either notebook;
    removed to keep the benchmark comparable.

CLASSIFIER EVALUATION
  • Per-subject scaler fitted on training genuine beats only (kept from v4
    — this is good practice and does not conflict with the notebooks which
    fit a global scaler on all subjects for visualisation only).
  • PCA-40 applied to raw features after per-subject scaling (same as
    notebook 2 Step 6).
  • Models: OC-SVM, Isolation Forest, LOF, Elliptic Envelope,
    MLP Autoencoder, KAN Autoencoder — same six as notebook 2.
    (VAE removed: not present in either notebook.)

VISUALISATION
  • Figures 2, 4, 5, 6, 7, 8, 9, 10, 11, 12 retained with minor label
    updates to reflect two feature sets (raw / WTMM) instead of three.

Requirements:
    pip install numpy matplotlib PyWavelets neurokit2 wfdb scikit-learn \\
                torch tqdm pandas seaborn umap-learn scipy
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
import umap

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

# ── 1. Settings ───────────────────────────────────────────────────────────────
DB_DIR  = './ecgiddb'
OUT_DIR = './paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Matched to notebook 1 Step 2
NUM_SUBJECTS  = 10
GRID_SUBJECTS = 6
GRID_BEATS    = 6

PRE_MS     = 250          # ms before R-peak (notebook 1 default)
POST_MS    = 450          # ms after  R-peak (notebook 1 default)
FS_DEFAULT = 500          # fallback sample rate when not in file

WAVELET_NAME = 'morl'     # Morlet wavelet (notebook 1 WAVELET_NAME)
FREQ_MIN     = 1.0        # Hz (notebook 1 FREQ_MIN)
FREQ_MAX     = 60.0       # Hz (notebook 1 FREQ_MAX)
N_FREQS      = 80         # frequency steps (notebook 1 N_FREQS)
CUT_FREQ     = 5.0        # lowest freq shown in line images (notebook 1 CUT_FREQ)
N_OCTAVE_BANDS = 5        # frequency bands in WTMM feature vector (notebook 2)

TRAIN_RATIO = 0.70        # notebook 2 TRAIN_RATIO
NN_EPOCHS   = 80          # notebook 2 NN_EPOCHS

# Projection visualisation
PROJ_MAX_SAMPLES = 3000
TSNE_PERPLEXITY  = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
DPI = 200

# ── 2. Signal processing helpers  (matched to notebook 1 Step 3) ──────────────
#
# IMPORTANT: notebooks pass the raw p_signal directly to nk.ecg_process.
# nk.ecg_process applies its own cleaning pipeline internally.
# No external bandpass filter is applied here.

def detect_rpeaks(sig, fs):
    """Clean the ECG and return heartbeat peak locations.
    Identical to notebook 1 detect_rpeaks."""
    signals, info = nk.ecg_process(sig, sampling_rate=fs)
    cleaned = signals['ECG_Clean'].values
    r_idx   = info['ECG_R_Peaks']
    return np.array(r_idx), cleaned


def extract_beats(cleaned, r_idx, fs):
    """Cut one window per heartbeat and return them as a 2D array.
    Identical to notebook 1 extract_beats."""
    pre_s, post_s = PRE_MS / 1000., POST_MS / 1000.
    epochs = nk.epochs_create(cleaned, events=r_idx, sampling_rate=fs,
                               epochs_start=-pre_s, epochs_end=post_s)
    keys  = sorted(epochs.keys(), key=lambda k: int(k))
    beats = np.array([epochs[k]['Signal'].values for k in keys])
    return beats, np.linspace(-pre_s, post_s, beats.shape[1])


# ── 3. WTMM helpers  (matched to notebook 1 Step 3 + notebook 2 Step 3) ───────
#
# min_len=2  and  tol = max(1, floor(scale/2))  with NO upper cap —
# identical to both notebooks.

def compute_wtmm_lines(power, scales, min_len=2):
    """Find and connect the strongest ridgelines in the time-frequency image.
    Identical to notebook 1 compute_wtmm_lines (min_len=2, uncapped tol)."""
    n_scales, n_times = power.shape
    mask = np.zeros_like(power, dtype=bool)

    for s in range(n_scales):
        r = power[s]
        if n_times >= 3:
            mask[s, 1:-1] = (r[1:-1] > r[:-2]) & (r[1:-1] > r[2:])
        if n_times >= 2:
            mask[s, 0]  = r[0] > r[1]
            mask[s, -1] = r[-1] > r[-2]
        else:
            mask[s, 0] = True

    maxima_idx = [np.where(mask[s])[0].tolist() for s in range(n_scales)]
    used, lines = [set() for _ in range(n_scales)], []

    for s0 in range(n_scales):
        for t0 in maxima_idx[s0]:
            if t0 in used[s0]:
                continue
            line = [(s0, t0)]
            used[s0].add(t0)
            prev_t = t0
            for s in range(s0 + 1, n_scales):
                # Uncapped tolerance — matches both notebooks exactly
                tol   = max(1, int(round(scales[s] / 2.0)))
                cands = [tt for tt in maxima_idx[s]
                         if tt not in used[s] and abs(int(tt) - int(prev_t)) <= tol]
                if not cands:
                    break
                nxt = min(cands, key=lambda x: abs(x - prev_t))
                line.append((s, int(nxt)))
                used[s].add(int(nxt))
                prev_t = int(nxt)
            if len(line) >= min_len:
                lines.append(line)

    return lines


def wtmm_feature_vector(beat, scales, freqs_out, fs):
    """Summarise the line image as 26 numbers a model can work with.
    Matched to notebook 2 wtmm_feature_vector:
      - min_len=2 (via compute_wtmm_lines default)
      - raw power (not log1p)
      - same band statistics: [n_lines, n_band, mean_freq, std_freq,
                                mean_power, std_power] × N_OCTAVE_BANDS"""
    coeffs, _ = pywt.cwt(beat, scales, WAVELET_NAME, sampling_period=1. / fs)
    power  = np.abs(coeffs) ** 2
    lines  = compute_wtmm_lines(power, scales)
    edges  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_OCTAVE_BANDS + 1)
    feats  = [len(lines)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = [l for l in lines if any(lo <= freqs_out[si] <= hi for si, _ in l)]
        if band:
            pf = [freqs_out[si] for l in band for si, ti in l if lo <= freqs_out[si] <= hi]
            pp = [power[si, ti]  for l in band for si, ti in l if lo <= freqs_out[si] <= hi]
            feats += [len(band), np.mean(pf), np.std(pf), np.mean(pp), np.std(pp)]
        else:
            feats += [0, 0., 0., 0., 0.]
    return np.array(feats, dtype=np.float32)


# ── 4. Load ECG-ID  (matched to notebook 1 Step 4 + notebook 2 Step 3) ────────
#
# • One recording per subject: rec_1 only
# • Raw p_signal[:,0] → detect_rpeaks → extract_beats
# • fs fallback: FS_DEFAULT=500 when missing or zero
# • No quality gate, no min-max normalisation

if not os.path.exists(DB_DIR):
    print('Downloading ECG-ID database — this may take a few minutes ...')
    wfdb.dl_database('ecgiddb', DB_DIR)
else:
    print('Database already on disk — skipping download.')

cf = pywt.central_frequency(WAVELET_NAME)

all_subjects_signals = {}   # subj_idx → raw signal array
all_subjects_beats   = {}   # subj_idx → (N_beats, N_samples) array
all_subjects_t_beat  = {}   # subj_idx → time axis
all_subjects_fs      = {}   # subj_idx → sample rate

for subj_idx in range(NUM_SUBJECTS):
    subj_num = subj_idx + 1
    rec_name = f'{DB_DIR}/Person_{subj_num:02d}/rec_1'
    print(f'\n→ Person {subj_num:02d}', end='  ')

    try:
        rec = wfdb.rdrecord(rec_name)
    except Exception as e:
        print(f'Could not read file: {e}')
        continue

    # Extract lead I signal — same try/except pattern as notebook 1
    try:
        sig = rec.p_signal[:, 0]
    except Exception:
        sig = np.asarray(rec.p_signal).flatten()

    # fs fallback identical to notebook 1
    fs = getattr(rec, 'fs', FS_DEFAULT) or FS_DEFAULT

    r_idx, cleaned = detect_rpeaks(sig, fs)
    beats, t_beat  = extract_beats(cleaned, r_idx, fs)
    print(f'{len(r_idx)} peaks found  →  {beats.shape[0]} beats saved'
          f'  (sample rate: {fs} Hz)')

    all_subjects_signals[subj_idx] = sig
    all_subjects_beats[subj_idx]   = beats
    all_subjects_t_beat[subj_idx]  = t_beat
    all_subjects_fs[subj_idx]      = fs

print('\n✓ All people loaded.')

# ── 5. Feature extraction  (matched to notebook 2 Step 3) ─────────────────────
#
# Raw feature set  : z-score per beat  (notebook 2: "(b - b.mean()) / (b.std() + 1e-8)")
# WTMM feature set : 26-d summary vector via wtmm_feature_vector above

print('\nExtracting features ...')

subject_feats_raw  = {}   # subj_idx → (N_beats, N_samples) float32, z-scored
subject_feats_wtmm = {}   # subj_idx → (N_beats, 26)       float32

for subj_idx in sorted(all_subjects_beats):
    beats = all_subjects_beats[subj_idx]
    fs    = all_subjects_fs[subj_idx]

    freqs  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
    scales = cf * fs / freqs
    _, f_out = pywt.cwt(beats[0], scales, WAVELET_NAME, sampling_period=1. / fs)

    # Raw waveform: z-score normalise each beat — identical to notebook 2
    subject_feats_raw[subj_idx] = np.vstack(
        [((b - b.mean()) / (b.std() + 1e-8)).astype(np.float32)
         for b in beats])

    # WTMM summary statistics — identical to notebook 2
    subject_feats_wtmm[subj_idx] = np.vstack(
        [wtmm_feature_vector(b, scales, f_out, fs)
         for b in tqdm(beats,
                       desc=f'  WTMM P{subj_idx+1:02d}',
                       leave=False)])

print('Features ready.')

# Global scalers for VISUALISATION only (PCA / t-SNE / UMAP projection figures)
all_raw_stack  = np.vstack([subject_feats_raw[i]  for i in sorted(subject_feats_raw)])
all_wtmm_stack = np.vstack([subject_feats_wtmm[i] for i in sorted(subject_feats_wtmm)])
all_labels     = np.concatenate(
    [np.full(subject_feats_raw[i].shape[0], i) for i in sorted(subject_feats_raw)])

scaler_vis_raw  = StandardScaler().fit(all_raw_stack)
scaler_vis_wtmm = StandardScaler().fit(all_wtmm_stack)

# ── 6. Model definitions  (matched to notebook 2 Step 5) ──────────────────────

class MLPAutoencoder(nn.Module):
    """Identical architecture to notebook 2 MLPAutoencoder (latent=16)."""
    def __init__(self, dim, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 64),  nn.ReLU(),
            nn.Linear(64, 128),    nn.ReLU(),
            nn.Linear(128, dim))

    def forward(self, x):
        return self.dec(self.enc(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(1).cpu().numpy()


def train_mlp(X, dim, epochs=NN_EPOCHS, lr=1e-3):
    """Identical to notebook 2 train_mlp."""
    model  = MLPAutoencoder(dim).to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=lr)
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
    """KAN edge: learnable B-spline + SiLU residual (notebook 2 BSplineEdge, G=5, k=3)."""
    def __init__(self, in_f, out_f, G=5, k=3):
        super().__init__()
        self.in_f, self.G, self.k = in_f, G, k
        n = G + k
        self.sp_w   = nn.Parameter(torch.randn(out_f, in_f, n) * 0.1)
        self.base_w = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        h    = 2.0 / G
        grid = torch.arange(-k, G + k + 1) * h - 1.0
        self.register_buffer('grid', grid)

    def bspline(self, x):
        x = x.unsqueeze(-1)
        g = self.grid.view(1, 1, -1)
        b = ((x >= g[:, :, :-1]) & (x < g[:, :, 1:])).float()
        for order in range(1, self.k + 1):
            gl, gr   = g[:, :, :-(order+1)], g[:, :, order:-1]
            gl2, gr2 = g[:, :, 1:-order],    g[:, :, (order+1):]
            d1, d2   = gr - gl, gr2 - gl2
            b1 = torch.where(d1 > 1e-8, (x - gl) / d1 * b[:, :, :-1],
                             torch.zeros_like(b[:, :, :-1]))
            b2 = torch.where(d2 > 1e-8, (gr2 - x) / d2 * b[:, :, 1:],
                             torch.zeros_like(b[:, :, 1:]))
            b  = b1 + b2
        return b

    def forward(self, x):
        sp = torch.einsum('oin,bin->bo', self.sp_w,
                          self.bspline(x.clamp(-1., 1.)))
        return sp + nn.functional.silu(x) @ self.base_w.T


class KANAutoencoder(nn.Module):
    """Identical architecture to notebook 2 KANAutoencoder (latent=12)."""
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
    """Identical to notebook 2 train_kan."""
    model  = KANAutoencoder(dim).to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
    """Identical to notebook 2 build_traditional_models."""
    lof_k = max(2, min(20, n_train - 1))
    return {
        'OC-SVM':            OneClassSVM(kernel='rbf', nu=0.1, gamma='scale'),
        'Isolation Forest':  IsolationForest(n_estimators=200,
                                              contamination=0.1, random_state=42),
        'LOF':               LocalOutlierFactor(n_neighbors=lof_k,
                                                 novelty=True, contamination=0.1),
        'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42),
    }


# ── 7. Evaluation helpers ─────────────────────────────────────────────────────

def eer_score(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i   = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2.


def safe_normalise(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


# ── 8. Per-subject benchmark  (matched to notebook 2 Step 6) ──────────────────
#
# Scaler is fitted on genuine training beats only (prevents data leakage).
# Notebook 2 uses a global scaler for all feature-space plots but fits the
# classifier on unscaled data — here we use a per-subject scaler throughout,
# which is strictly better for one-class learning and is the same approach
# used in the notebook's evaluate_subject function.

def evaluate_subject(subj_idx, feat_key='raw', pca_dim=40, verbose=False):
    """
    Returns (results_dict, roc_dict, recon_errors_dict).

    Matched to notebook 2 evaluate_subject:
      • positive class  = this subject's beats
      • negative class  = equal-sized sample from all other subjects
      • 70/30 train/test split on the positive class
      • PCA-40 applied to raw features (after per-subject scaling)
      • Threshold = median(score_samples(X_tr)) for classical models
      • Threshold = 90th-percentile recon error on X_tr for autoencoders
    """
    feat_dict = subject_feats_raw if feat_key == 'raw' else subject_feats_wtmm

    pos  = feat_dict[subj_idx]
    negs = np.vstack([feat_dict[i] for i in feat_dict if i != subj_idx])

    rng = np.random.default_rng(42 + subj_idx)
    neg = negs[rng.choice(len(negs), size=len(pos),
                           replace=len(negs) < len(pos))]

    n_pos = len(pos)
    if n_pos < 4:
        raise ValueError(f'Person {subj_idx+1:02d}: too few beats ({n_pos}).')

    perm    = rng.permutation(n_pos)
    n_train = min(max(3, int(n_pos * TRAIN_RATIO)), n_pos - 1)

    # Per-subject scaler: fit on training genuine beats only
    sc = StandardScaler().fit(pos[perm[:n_train]])
    X_tr  = sc.transform(pos[perm[:n_train]])
    X_pos = sc.transform(pos[perm[n_train:]])
    X_neg = sc.transform(neg)

    # PCA for raw waveform — matched to notebook 2 evaluate_subject
    if feat_key == 'raw' and pca_dim:
        n_comp = min(int(pca_dim), X_tr.shape[0], X_tr.shape[1])
        if n_comp >= 1 and X_tr.shape[1] > n_comp:
            pca_ = PCA(n_components=n_comp, random_state=42)
            X_tr  = pca_.fit_transform(X_tr)
            X_pos = pca_.transform(X_pos)
            X_neg = pca_.transform(X_neg)
            if verbose:
                print(f'PCA-{n_comp}', end=' ')

    X_test = np.vstack([X_pos, X_neg])
    y_test = np.array([1] * len(X_pos) + [0] * len(X_neg))
    dim    = X_tr.shape[1]

    results, roc_data, recon_errors = {}, {}, {}

    # Classical one-class classifiers
    for name, clf in build_traditional_models(n_train=len(X_tr)).items():
        try:
            clf.fit(X_tr)
            try:
                sc2   = clf.score_samples(X_test)
                sc_tr = clf.score_samples(X_tr)
            except Exception:
                sc2   = clf.decision_function(X_test)
                sc_tr = clf.decision_function(X_tr)
            norm = safe_normalise(sc2)
            thr  = np.median(sc_tr)
            auc  = roc_auc_score(y_test, norm)
            results[name] = dict(
                auroc=auc,
                f1=f1_score(y_test, (sc2 >= thr).astype(int), zero_division=0),
                eer=eer_score(y_test, norm))
            fp_, tp_, _ = roc_curve(y_test, norm)
            roc_data[name] = (fp_, tp_, auc)
        except Exception as e:
            if verbose:
                print(f'[{name} failed: {type(e).__name__}]', end=' ')
            results[name]  = dict(auroc=np.nan, f1=np.nan, eer=np.nan)
            roc_data[name] = (np.array([0, 1]), np.array([0, 1]), np.nan)

    Xtr_t   = torch.tensor(X_tr.astype(np.float32)).to(DEVICE)
    Xtest_t = torch.tensor(X_test.astype(np.float32)).to(DEVICE)

    # MLP Autoencoder
    try:
        mlp = train_mlp(X_tr.astype(np.float32), dim)
        err = mlp.recon_error(Xtest_t)
        norm = 1 - safe_normalise(err)
        thr  = np.percentile(mlp.recon_error(Xtr_t), 90)
        auc  = roc_auc_score(y_test, norm)
        results['MLP Autoencoder'] = dict(
            auroc=auc,
            f1=f1_score(y_test, (err <= thr).astype(int), zero_division=0),
            eer=eer_score(y_test, norm))
        fp_, tp_, _ = roc_curve(y_test, norm)
        roc_data['MLP Autoencoder'] = (fp_, tp_, auc)
        recon_errors['MLP Autoencoder'] = {
            'genuine':  err[y_test == 1],
            'impostor': err[y_test == 0]}
    except Exception as e:
        if verbose:
            print(f'[MLP failed: {type(e).__name__}]', end=' ')
        results['MLP Autoencoder']  = dict(auroc=np.nan, f1=np.nan, eer=np.nan)
        roc_data['MLP Autoencoder'] = (np.array([0, 1]), np.array([0, 1]), np.nan)

    # KAN Autoencoder
    try:
        kan = train_kan(X_tr.astype(np.float32), dim)
        err = kan.recon_error(Xtest_t)
        norm = 1 - safe_normalise(err)
        thr  = np.percentile(kan.recon_error(Xtr_t), 90)
        auc  = roc_auc_score(y_test, norm)
        results['KAN Autoencoder'] = dict(
            auroc=auc,
            f1=f1_score(y_test, (err <= thr).astype(int), zero_division=0),
            eer=eer_score(y_test, norm))
        fp_, tp_, _ = roc_curve(y_test, norm)
        roc_data['KAN Autoencoder'] = (fp_, tp_, auc)
        recon_errors['KAN Autoencoder'] = {
            'genuine':  err[y_test == 1],
            'impostor': err[y_test == 0]}
    except Exception as e:
        if verbose:
            print(f'[KAN failed: {type(e).__name__}]', end=' ')
        results['KAN Autoencoder']  = dict(auroc=np.nan, f1=np.nan, eer=np.nan)
        roc_data['KAN Autoencoder'] = (np.array([0, 1]), np.array([0, 1]), np.nan)

    return results, roc_data, recon_errors


# ── 9. Run benchmark ──────────────────────────────────────────────────────────

results_by_feat = {}   # feat_key → subj_idx → {model: metrics}
roc_by_feat     = {}   # feat_key → subj_idx → {model: (fpr, tpr, auc)}
recon_err_store = {}   # feat_key → subj_idx → {model: {genuine, impostor}}

for feat_key in ['raw', 'wtmm']:
    print(f'\n── Benchmark: {feat_key.upper()} ──')
    results_by_feat[feat_key] = {}
    roc_by_feat[feat_key]     = {}
    recon_err_store[feat_key] = {}
    pca = 40 if feat_key == 'raw' else None

    for idx in sorted(subject_feats_raw):
        print(f'  Person {idx+1:02d} ...', end=' ', flush=True)
        try:
            res, roc, re = evaluate_subject(idx, feat_key, pca, verbose=True)
            results_by_feat[feat_key][idx] = res
            roc_by_feat[feat_key][idx]     = roc
            recon_err_store[feat_key][idx] = re
            best = max(res, key=lambda m: res[m].get('auroc', -1)
                       if not np.isnan(res[m].get('auroc', np.nan)) else -1)
            print(f'→ {best}  AUROC={res[best].get("auroc", 0):.3f}')
        except Exception as e:
            print(f'SKIP ({e})')

print('\n✓ Benchmark complete.')

# ── 10. Plotting helpers ──────────────────────────────────────────────────────

MODEL_ORDER = ['OC-SVM', 'Isolation Forest', 'LOF', 'Elliptic Envelope',
               'MLP Autoencoder', 'KAN Autoencoder']

MODEL_COLORS = {
    'OC-SVM':            ('#1f77b4', 'solid'),
    'Isolation Forest':  ('#1f77b4', 'dashed'),
    'LOF':               ('#aec7e8', 'dotted'),
    'Elliptic Envelope': ('#aec7e8', 'solid'),
    'MLP Autoencoder':   ('#2ca02c', 'solid'),
    'KAN Autoencoder':   ('#d62728', 'solid'),
}

FEAT_LABELS = {'raw': 'Raw Beat Shape (PCA-40)', 'wtmm': 'WTMM Summary (26-d)'}


def to_df(res_dict, metric):
    rows = [{'Person': f'P{i+1:02d}',
             **{m: round(v.get(metric, np.nan), 4) for m, v in res.items()}}
            for i, res in sorted(res_dict.items())]
    df = pd.DataFrame(rows).set_index('Person')
    if len(df) > 0:
        df.loc['MEAN'] = df.mean(numeric_only=True).round(4)
    return df


def maybe_sub(X, y, n, seed=42):
    if n is None or len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]

# ── 11. Figure 2: Beat grid ───────────────────────────────────────────────────
print('\nPlotting fig2 ...')
n_show    = min(GRID_SUBJECTS, len(all_subjects_beats))
show_idxs = sorted(all_subjects_beats.keys())[:n_show]
fig, axes = plt.subplots(GRID_BEATS, n_show,
                          figsize=(3 * n_show, 2.6 * GRID_BEATS),
                          squeeze=False, sharex=True, sharey=True)
for col, si in enumerate(show_idxs):
    beats  = all_subjects_beats[si]
    t_beat = all_subjects_t_beat[si]
    for row in range(GRID_BEATS):
        ax = axes[row, col]
        if row < beats.shape[0]:
            ax.plot(t_beat, beats[row], lw=1.2)
            ax.axvline(0, ls='--', lw=0.8, color='red')
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f'Person {si+1:02d}')
        else:
            ax.axis('off')
plt.suptitle('Stage 2 — Individual Beats  (red line = heartbeat peak)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_extracted_beats.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig2_extracted_beats.png')

# ── 12. Figure 4: WTMM ridge lines ───────────────────────────────────────────
print('Plotting fig4 ...')
fig, axes = plt.subplots(GRID_BEATS, n_show,
                          figsize=(3 * n_show, 2.6 * GRID_BEATS),
                          squeeze=False, sharex=True, sharey=True)
for col, si in enumerate(show_idxs):
    beats = all_subjects_beats[si]
    fs    = all_subjects_fs[si]
    freqs  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
    scales = cf * fs / freqs
    for row in range(GRID_BEATS):
        ax = axes[row, col]
        if row < beats.shape[0]:
            coeffs, fo = pywt.cwt(beats[row], scales, WAVELET_NAME,
                                   sampling_period=1. / fs)
            power = np.abs(coeffs) ** 2
            lines = compute_wtmm_lines(power, scales)
            for line in lines:
                ax.plot([t / fs for _, t in line],
                        [fo[s] for s, _ in line],
                        '-', color='black', lw=0.8)
            ax.set_yscale('log')
            ax.set_ylim([CUT_FREQ, max(fo)])
            if row == 0:
                ax.set_title(f'Person {si+1:02d}')
        else:
            ax.axis('off')
plt.suptitle('Stage 4 — Line Images  (the heartbeat fingerprints)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_wtmm_lines.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig4_wtmm_lines.png')

# ── 13. Figure 5: Four processing stages ─────────────────────────────────────
print('Plotting fig5 ...')
si0 = sorted(all_subjects_signals.keys())[0]
sig = all_subjects_signals[si0]
fs  = all_subjects_fs[si0]
r_idx0, cleaned0 = detect_rpeaks(sig, fs)
beats0, t_beat0  = extract_beats(cleaned0,
                                  np.array([r_idx0[min(1, len(r_idx0)-1)]]), fs)
beat0   = beats0[0]
freqs0  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
scales0 = cf * fs / freqs0
coeff0, fo0 = pywt.cwt(beat0, scales0, WAVELET_NAME, sampling_period=1. / fs)
power0 = np.abs(coeff0) ** 2
lines0 = compute_wtmm_lines(power0, scales0)

center = r_idx0[min(1, len(r_idx0)-1)]
win    = int(1.0 * fs)
t_sig  = np.arange(len(sig)) / fs

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
sw, ew = max(0, center - win), min(len(sig), center + win)
ax.plot(t_sig[sw:ew], sig[sw:ew], lw=0.9)
ax.axvline(center / fs, ls='--', lw=1.2, color='red', label='R-peak')
ax.set_title(f'Stage 1 — Raw Signal  (Person {si0+1:02d})')
ax.set_xlabel('Time (seconds)'); ax.set_ylabel('Electrical amplitude')
ax.legend(); ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(t_beat0, beat0, lw=1.3)
ax.axvline(0, ls='--', lw=1.2, color='red', label='R-peak (t=0)')
ax.set_title('Stage 2 — Extracted Beat')
ax.set_xlabel('Time (seconds)'); ax.set_ylabel('Amplitude')
ax.legend(); ax.grid(alpha=0.25)

ax = axes[1, 0]
T0, F0 = np.meshgrid(np.arange(beat0.size) / fs, fo0)
pcm = ax.pcolormesh(T0, F0, power0, shading='auto', cmap='viridis')
ax.set_yscale('log'); ax.set_ylim([FREQ_MIN, FREQ_MAX])
ax.set_title('Stage 3 — Time-Frequency Image')
ax.set_xlabel('Time (seconds)'); ax.set_ylabel('Frequency (Hz)')
plt.colorbar(pcm, ax=ax, label='Energy |W|²')

ax = axes[1, 1]
for line in lines0:
    ax.plot([t / fs for _, t in line], [fo0[s] for s, _ in line],
            '-', color='navy', lw=0.9)
ax.set_yscale('log'); ax.set_ylim([CUT_FREQ, max(fo0)])
ax.set_title('Stage 4 — Line Image  (heartbeat fingerprint)')
ax.set_xlabel('Time (seconds)'); ax.set_ylabel('Frequency (Hz)')
ax.grid(alpha=0.2)

plt.suptitle(f'Four Processing Stages — Person {si0+1:02d}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_four_stages.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig5_four_stages.png')

# ── 14. Figure 6: PCA + t-SNE + UMAP ─────────────────────────────────────────
print('Plotting fig6 (PCA / t-SNE / UMAP) ...')
Xr_s, ys_r = maybe_sub(scaler_vis_raw.transform(all_raw_stack),
                         all_labels, PROJ_MAX_SAMPLES)
Xw_s, ys_w = maybe_sub(scaler_vis_wtmm.transform(all_wtmm_stack),
                         all_labels, PROJ_MAX_SAMPLES)

perp = min(TSNE_PERPLEXITY, len(Xr_s) - 1)
print(f'  t-SNE raw  ({len(Xr_s)} pts) ...')
tsne_raw  = TSNE(n_components=2, perplexity=perp,
                  random_state=42, n_jobs=-1).fit_transform(Xr_s)
print(f'  t-SNE WTMM ({len(Xw_s)} pts) ...')
tsne_wtmm = TSNE(n_components=2, perplexity=perp,
                  random_state=42, n_jobs=-1).fit_transform(Xw_s)
print('  UMAP raw  ...')
umap_raw  = umap.UMAP(n_components=2,
                       n_neighbors=min(UMAP_N_NEIGHBORS, len(Xr_s)-1),
                       min_dist=UMAP_MIN_DIST,
                       random_state=42).fit_transform(Xr_s)
print('  UMAP WTMM ...')
umap_wtmm = umap.UMAP(n_components=2,
                       n_neighbors=min(UMAP_N_NEIGHBORS, len(Xw_s)-1),
                       min_dist=UMAP_MIN_DIST,
                       random_state=42).fit_transform(Xw_s)

n_uniq  = len(np.unique(ys_r))
palette = plt.cm.get_cmap('tab10', max(10, n_uniq))
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
configs = [
    (axes[0, 0], tsne_raw,  ys_r, 't-SNE — Raw Beat Shape'),
    (axes[0, 1], tsne_wtmm, ys_w, 't-SNE — WTMM Summary'),
    (axes[1, 0], umap_raw,  ys_r, 'UMAP — Raw Beat Shape'),
    (axes[1, 1], umap_wtmm, ys_w, 'UMAP — WTMM Summary'),
]
for ax, proj, lbs, title in configs:
    for i in sorted(np.unique(lbs)):
        m = lbs == i
        ax.scatter(proj[m, 0], proj[m, 1],
                   color=palette(int(i) % 10),
                   label=f'Person {i+1:02d}',
                   s=20, alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
plt.suptitle(f'Feature Space — {n_uniq} subjects, {len(Xr_s)} beats shown',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_tsne_umap.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig6_tsne_umap.png')

# ── 15. Figure 7: AUROC heatmaps ─────────────────────────────────────────────
print('Plotting fig7 ...')
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, fk in zip(axes, ['raw', 'wtmm']):
    df = to_df(results_by_feat[fk], 'auroc')
    if 'MEAN' in df.index:
        df = df.drop(index='MEAN')
    df = df.astype(float)
    sns.heatmap(df, ax=ax, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.4, vmax=1.0, linewidths=0.4, linecolor='#ddd',
                cbar_kws={'label': 'AUROC'})
    ax.set_title(f'AUROC — {FEAT_LABELS[fk]}', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('AUROC per Person per Model  (green = good, red = poor)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_auroc_heatmaps.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig7_auroc_heatmaps.png')

# ── 16. Figure 8: Mean performance bars ──────────────────────────────────────
print('Plotting fig8 ...')
metric_info = [('auroc', 'AUROC  ↑ higher is better'),
               ('f1',    'F1 Score  ↑ higher is better'),
               ('eer',   'EER  ↓ lower is better')]
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for row, fk in enumerate(['raw', 'wtmm']):
    for col, (metric, mlabel) in enumerate(metric_info):
        ax    = axes[row, col]
        means = to_df(results_by_feat[fk], metric).loc['MEAN']
        colors = ['#DD8452' if 'KAN' in n else '#4C72B0' for n in means.index]
        bars   = ax.bar(means.index, means.values,
                        color=colors, edgecolor='white', lw=0.8)
        for b, v in zip(bars, means.values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.005,
                        f'{v:.3f}', ha='center', fontsize=8)
        ax.set_title(f'{FEAT_LABELS[fk]}\n{mlabel}', fontsize=10)
        ax.set_xticks(np.arange(len(means.index)))
        ax.set_xticklabels(means.index, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
fig.legend(handles=[
    mpatches.Patch(facecolor='#4C72B0', label='Traditional / MLP'),
    mpatches.Patch(facecolor='#DD8452', label='KAN'),
], loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.suptitle('Average Performance across All People', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig8_mean_performance.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig8_mean_performance.png')

# ── 17. Figure 9: Per-person AUROC comparison ─────────────────────────────────
print('Plotting fig9 ...')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, fk in zip(axes, ['raw', 'wtmm']):
    df   = to_df(results_by_feat[fk], 'auroc')
    if 'MEAN' in df.index:
        df = df.drop(index='MEAN')
    trad = [c for c in df.columns if 'KAN' not in c and 'MLP' not in c]
    best = df[trad].mean().idxmax() if trad else df.columns[0]
    x, w = np.arange(len(df)), 0.25
    ax.bar(x - w, df[best].values,
           width=w, label=best,              color='#4C72B0', edgecolor='white')
    ax.bar(x,     df['MLP Autoencoder'].values,
           width=w, label='MLP Autoencoder', color='#55A868', edgecolor='white')
    ax.bar(x + w, df['KAN Autoencoder'].values,
           width=w, label='KAN Autoencoder', color='#DD8452', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, ls=':', lw=1, color='gray', label='Random baseline (0.5)')
    ax.set_title(f'{FEAT_LABELS[fk]}\nPer-Person AUROC', fontsize=11)
    ax.set_ylabel('AUROC'); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Per-Person: Best Traditional vs MLP vs KAN', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig9_per_person.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig9_per_person.png')

# ── 18. Figure 10: ROC curves (WTMM) ─────────────────────────────────────────
print('Plotting fig10 ...')
wtmm_roc = roc_by_feat['wtmm']
n_subj   = len(wtmm_roc)
ncols    = min(5, n_subj)
nrows    = int(np.ceil(n_subj / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5 * ncols, 4 * nrows), squeeze=False)
af = axes.flatten()
for idx2, si in enumerate(sorted(wtmm_roc)):
    ax = af[idx2]
    for name, (fp, tp, auc) in wtmm_roc[si].items():
        c, ls = MODEL_COLORS.get(name, ('#888', 'solid'))
        label = f'{name} ({auc:.2f})' if not np.isnan(auc) else name
        ax.plot(fp, tp, color=c, linestyle=ls, lw=1.2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_title(f'Person {si+1:02d}', fontsize=9)
    ax.legend(fontsize=5); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('FPR', fontsize=8); ax.set_ylabel('TPR', fontsize=8)
for ax in af[n_subj:]:
    ax.axis('off')
plt.suptitle('ROC Curves per Subject — WTMM Features', fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_roc_curves.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig10_roc_curves.png')

# ── 19. Figure 11: KAN vs best traditional EER ───────────────────────────────
print('Plotting fig11 ...')
rd = results_by_feat['wtmm']
kan_eer, best_trad_eer, subj_ids = [], [], []
for si, res in sorted(rd.items()):
    k_eer     = res.get('KAN Autoencoder', {}).get('eer', np.nan)
    trad_eers = [res.get(m, {}).get('eer', np.nan)
                 for m in ['OC-SVM', 'Isolation Forest',
                            'LOF', 'Elliptic Envelope']]
    bt_eer = np.nanmin(trad_eers)
    if not (np.isnan(k_eer) or np.isnan(bt_eer)):
        kan_eer.append(k_eer); best_trad_eer.append(bt_eer)
        subj_ids.append(si)

kan_eer = np.array(kan_eer)
bt_eer  = np.array(best_trad_eer)
colors11 = ['#d62728' if k < b else '#1f77b4' for k, b in zip(kan_eer, bt_eer)]

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(bt_eer, kan_eer, c=colors11, s=90, edgecolors='white',
           linewidths=0.6, zorder=3)
lim = max(bt_eer.max(), kan_eer.max()) * 1.12
ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='Equal EER')
ax.fill_between([0, lim], [0, 0], [0, lim],  alpha=0.06, color='#d62728',
                label='KAN wins (lower EER)')
ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.06, color='#1f77b4',
                label='Traditional wins')
for i, si in enumerate(subj_ids):
    ax.annotate(f'P{si+1:02d}', (bt_eer[i], kan_eer[i]),
                textcoords='offset points', xytext=(4, 2), fontsize=7)
n_kan_wins = int(np.sum(kan_eer < bt_eer))
ax.set_xlabel('Best Traditional EER (WTMM)', fontsize=11)
ax.set_ylabel('KAN Autoencoder EER (WTMM)', fontsize=11)
ax.set_title(f'KAN vs Best Traditional — EER per Subject\n'
             f'KAN wins on {n_kan_wins}/{len(subj_ids)} subjects '
             f'({100*n_kan_wins/max(len(subj_ids),1):.0f}%)',
             fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig11_kan_advantage.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig11_kan_advantage.png')

# ── 20. Figure 12: Reconstruction error distributions ────────────────────────
print('Plotting fig12 ...')
re_store    = recon_err_store['wtmm']
n_show_re   = min(6, len(re_store))
show_re_idx = sorted(re_store.keys())[:n_show_re]
ae_models   = ['MLP Autoencoder', 'KAN Autoencoder']
fig, axes   = plt.subplots(len(ae_models), n_show_re,
                            figsize=(3.5 * n_show_re, 3.5 * len(ae_models)),
                            squeeze=False)
for row, model in enumerate(ae_models):
    for col, si in enumerate(show_re_idx):
        ax  = axes[row, col]
        re  = re_store.get(si, {}).get(model, {})
        gen = re.get('genuine', None)
        imp = re.get('impostor', None)
        if gen is not None and imp is not None:
            ax.hist(gen, bins=15, alpha=0.6, color='#2ca02c',
                    density=True, label='Genuine')
            ax.hist(imp, bins=15, alpha=0.6, color='#d62728',
                    density=True, label='Impostor')
            ax.set_title(f'{model[:6]}. P{si+1:02d}', fontsize=8)
            if col == 0:
                ax.set_ylabel('Density', fontsize=7)
            if row == len(ae_models) - 1:
                ax.set_xlabel('Recon. Error', fontsize=7)
            if col == n_show_re - 1:
                ax.legend(fontsize=6)
        else:
            ax.axis('off')
        ax.tick_params(labelsize=6)
plt.suptitle('Reconstruction Error Distributions (WTMM features)\n'
             'Green = genuine beats, Red = impostor beats', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig12_recon_errors.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('  → fig12_recon_errors.png')

# ── 21. CSV summary + console table ──────────────────────────────────────────
print('\nSaving benchmark_results.csv ...')
rows = []
for fk, res_fk in results_by_feat.items():
    for metric in ['auroc', 'f1', 'eer']:
        df = to_df(res_fk, metric)
        if 'MEAN' not in df.index:
            continue
        std_r = df.drop(index='MEAN').std(numeric_only=True)
        for model in df.columns:
            rows.append({
                'Feature set': FEAT_LABELS.get(fk, fk),
                'Model':       model,
                'Metric':      metric.upper(),
                'Mean':        round(float(df.loc['MEAN', model]), 4),
                'Std':         round(float(std_r.get(model, np.nan)), 4),
            })
csv_df = pd.DataFrame(rows)
csv_df.to_csv(f'{OUT_DIR}/benchmark_results.csv', index=False)
print(f'  → {OUT_DIR}/benchmark_results.csv')

print('\n' + '=' * 65)
print('FINAL SUMMARY — Average metrics across all people')
print('=' * 65)
for fk in ['raw', 'wtmm']:
    ad = to_df(results_by_feat[fk], 'auroc')
    fd = to_df(results_by_feat[fk], 'f1')
    ed = to_df(results_by_feat[fk], 'eer')
    if 'MEAN' not in ad.index:
        continue
    n_s = len(ad) - 1
    print(f'\n── {FEAT_LABELS[fk]}  ({n_s} subjects) ──')
    for m in MODEL_ORDER:
        if m not in ad.columns:
            continue
        a  = ad.loc['MEAN', m]; f  = fd.loc['MEAN', m]; e  = ed.loc['MEAN', m]
        as_= ad.drop(index='MEAN')[m].std()
        fs_= fd.drop(index='MEAN')[m].std()
        es_= ed.drop(index='MEAN')[m].std()
        print(f'  {m:<25s}  AUROC={a:.3f}±{as_:.3f}  '
              f'F1={f:.3f}±{fs_:.3f}  EER={e:.3f}±{es_:.3f}')

print(f'\nDone!  Figures + CSV → {OUT_DIR}/')