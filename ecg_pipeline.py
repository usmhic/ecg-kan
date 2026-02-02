"""
ecg_pipeline.py

All-in-one ECG processing + WTMM image generation + model training pipeline.

Features:
- Default datasets: ECG-ID (ecgiddb). Can switch to MIT-BIH (mitdb), MIT-NSR (nsrdb), PTB-XL (ptb-xl) via --dataset.
- Accepts a custom local dataset directory (--custom-dir) with structure:
    custom_dir/
      subject_01/
        img1.png img2.png ...   # or raw WFDB records
      subject_02/
        ...
- Steps available: download, preprocess (filter + R-peak + beat extraction), generate WTMM images, train models.
- Models supported: OneClassSVM, OneClassCNN_autoencoder, (optional) OneClassKAN if 'kan' installed.
- CLI interface so you can run pipelines on new data easily.

Usage examples:
  # Download ECG-ID and process (default) and generate images
  python ecg_pipeline.py --step all --dataset ecg-id

  # Process local WFDB directory and generate images
  python ecg_pipeline.py --step preprocess --custom-dir ./my_wfdb_data --out-root ./my_project

  # Train models on previously generated image folders
  python ecg_pipeline.py --step train --image-dir ./ecg_wtmm_images --models svm,cnn

  # Full run: download, preprocess, generate images, train
  python ecg_pipeline.py --step all --dataset ecg-id --models svm,cnn

Notes:
- This script organizes outputs by project folder (by default './ecg_project').
- Images are saved under ./ecg_project/images/subject_<n>/wtmm_beat_XXX.png
- Models and results are saved under ./ecg_project/models and ./ecg_project/results
"""

import argparse
import os
import glob
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# signal processing
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pywt

# wfdb for physionet datasets
import wfdb

# ML
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Optional imports
try:
    import neurokit2 as nk
    HAS_NK = True
except Exception:
    HAS_NK = False

try:
    from kan import KAN
    HAS_KAN = True
except Exception:
    HAS_KAN = False


# ---------------------------
# Configuration defaults
# ---------------------------
PROJECT_ROOT = os.path.abspath('./ecg_project')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMAGE_DIR_ROOT = os.path.join(PROJECT_ROOT, 'images')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Processing defaults
PRE_MS = 250
POST_MS = 450
WAVELET_NAME = 'morl'
FREQ_MIN = 1.0
FREQ_MAX = 60.0
N_FREQS = 80
CUT_FREQ = 5.0
MAX_BEATS_PER_SUBJECT = None  # None => keep all detected beats, or set e.g. 10

# Image defaults
IMAGE_DPI = 150
IMAGE_SIZE = 64

# Model defaults
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------
# Utility helpers
# ---------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def list_person_dirs_ecgiddb(base_dir):
    # ECG-ID stores Person_## directories
    dirs = sorted(glob.glob(os.path.join(base_dir, 'Person_*')))
    return dirs


# ---------------------------
# Download datasets
# ---------------------------
def download_dataset(name, dest=RAW_DIR):
    ensure_dir(dest)
    name_lookup = {
        'ecg-id': 'ecgiddb',
        'ecgiddb': 'ecgiddb',
        'mitdb': 'mitdb',
        'nsrdb': 'nsrdb',
        'ptb-xl': 'ptb-xl'
    }
    key = name_lookup.get(name.lower())
    if key is None:
        raise ValueError(f'Unknown dataset {name}')
    print(f"Downloading dataset {key} into {dest} (this may take a while)...")
    wfdb.dl_database(key, dest)
    print('Download finished.')


# ---------------------------
# Signal processing helpers
# ---------------------------
def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def detect_rpeaks(sig, fs):
    if HAS_NK:
        print('Using neurokit2 for R-peak detection')
        processed = nk.ecg_process(sig, sampling_rate=fs)
        signals = processed[0]
        info = processed[1]
        cleaned = signals['ECG_Clean'].values
        r_idx = np.array(info['ECG_R_Peaks'])
        return r_idx, 'neurokit2', cleaned
    else:
        cleaned = bandpass_filter(sig, fs)
        min_dist = int(0.35 * fs)
        peaks, _ = find_peaks(cleaned, distance=min_dist, prominence=0.35)
        return np.array(peaks), 'scipy_fallback', cleaned

def extract_beats(cleaned, r_idx, fs, pre_ms=PRE_MS, post_ms=POST_MS, max_beats=None):
    pre = int(pre_ms / 1000.0 * fs)
    post = int(post_ms / 1000.0 * fs)
    beats = []
    for idx in r_idx:
        s = int(idx) - pre
        e = int(idx) + post
        if s < 0 or e > len(cleaned):
            continue
        beats.append(cleaned[s:e])
        if max_beats is not None and len(beats) >= max_beats:
            break
    if len(beats) == 0:
        return np.empty((0, pre+post)), np.linspace(-pre/float(fs), post/float(fs), pre+post)
    return np.vstack(beats), np.linspace(-pre/float(fs), post/float(fs), beats.shape[1])

def compute_wtmm_lines(power, scales, min_len=2):
    n_scales, n_times = power.shape
    maxima_mask = np.zeros_like(power, dtype=bool)
    for s in range(n_scales):
        row = power[s]
        maxima_mask[s, 1:-1] = (row[1:-1] > row[0:-2]) & (row[1:-1] > row[2:])
        maxima_mask[s, 0] = row[0] > row[1]
        maxima_mask[s, -1] = row[-1] > row[-2]
    maxima_idx = [np.where(maxima_mask[s])[0].tolist() for s in range(n_scales)]

    used = [set() for _ in range(n_scales)]
    lines = []
    for s0 in range(n_scales):
        for t0 in maxima_idx[s0]:
            if t0 in used[s0]:
                continue
            line = [(s0, t0)]
            used[s0].add(t0)
            prev_t = t0
            for s in range(s0 + 1, n_scales):
                tol = max(1, int(round(scales[s] / 2.0)))
                cands = [tt for tt in maxima_idx[s] if tt not in used[s] and abs(int(tt) - int(prev_t)) <= tol]
                if not cands:
                    break
                nxt = min(cands, key=lambda x: abs(x - prev_t))
                line.append((s, int(nxt)))
                used[s].add(int(nxt))
                prev_t = int(nxt)
            if len(line) >= min_len:
                lines.append(line)
    return lines

def save_wtmm_image(lines, freqs_out, fs, out_path, cut_freq=CUT_FREQ, figsize=(1.5,1.5)):
    fig, ax = plt.subplots(figsize=figsize)
    for line in lines:
        xs = [t_idx / fs for s_idx, t_idx in line]
        ys = [freqs_out[s_idx] for s_idx, t_idx in line]
        ax.plot(xs, ys, '-', color='black', lw=0.8)
    ax.set_yscale('log')
    ax.set_ylim([cut_freq, max(freqs_out)])
    ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0, dpi=IMAGE_DPI)
    plt.close(fig)


# ---------------------------
# High-level processing pipeline
# ---------------------------
def process_ecg_id(root_raw_dir, image_root_dir, max_beats_per_subject=MAX_BEATS_PER_SUBJECT):
    """Process ECG-ID downloaded under root_raw_dir and save WTMM images under image_root_dir.
    """
    ensure_dir(image_root_dir)
    person_dirs = list_person_dirs_ecgiddb(root_raw_dir)
    if len(person_dirs) == 0:
        raise FileNotFoundError('No Person_* folders found in raw ECG-ID directory.')

    cf = pywt.central_frequency(WAVELET_NAME)

    for subj_idx, pd in enumerate(person_dirs):
        subj_out = os.path.join(image_root_dir, f'subject_{subj_idx+1:03d}')
        ensure_dir(subj_out)
        # prefer rec_1
        rec_files = sorted(glob.glob(os.path.join(pd, 'rec_*')))
        if len(rec_files) == 0:
            continue
        rec = wfdb.rdrecord(rec_files[0])
        sig = rec.p_signal[:,0] if rec.p_signal.ndim > 1 else rec.p_signal
        fs = getattr(rec, 'fs', 500)
        r_idx, method, cleaned = detect_rpeaks(sig, fs)
        beats, _ = extract_beats(cleaned, r_idx, fs, max_beats=max_beats_per_subject)
        freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
        scales = cf * fs / freqs
        for i, beat in enumerate(beats):
            coeffs, freqs_out = pywt.cwt(beat, scales, WAVELET_NAME, sampling_period=1.0/fs)
            power = np.abs(coeffs) ** 2
            lines = compute_wtmm_lines(power, scales, min_len=2)
            if len(lines) == 0:
                # save an empty placeholder (blank image) to keep counts consistent
                img_path = os.path.join(subj_out, f'wtmm_beat_{i+1:03d}.png')
                Image.new('L', (IMAGE_SIZE, IMAGE_SIZE)).save(img_path)
                continue
            img_path = os.path.join(subj_out, f'wtmm_beat_{i+1:03d}.png')
            save_wtmm_image(lines, freqs_out, fs, img_path)
        print(f'Processed subject {subj_idx+1}, saved {len(beats)} images to {subj_out}')


def process_custom_image_dir(custom_dir, image_root_dir):
    """If user already has image folders per-subject, simply copy or symlink them into our image_root_dir structure.
    Expect custom_dir/subject_xx/*.png
    """
    ensure_dir(image_root_dir)
    subj_dirs = sorted([d for d in glob.glob(os.path.join(custom_dir, '*')) if os.path.isdir(d)])
    for i, sd in enumerate(subj_dirs):
        dest = os.path.join(image_root_dir, f'subject_{i+1:03d}')
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(sd, dest)
    print('Copied custom images into project image folder')


# ---------------------------
# Dataset class for Image WTMMs
# ---------------------------
class WTMMImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------
# Models: CNN AE + training
# ---------------------------
class OneClassCNN(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = image_size // 8
        flatten_dim = 64 * conv_out * conv_out
        self.fc_enc = nn.Sequential(nn.Linear(flatten_dim, 128), nn.ReLU(), nn.Linear(128, 32))
        self.fc_dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, flatten_dim))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, conv_out, conv_out)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        lat = self.fc_enc(enc)
        dec = self.fc_dec(lat)
        out = self.decoder(dec)
        return out


class OneClassKANWrapper(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE):
        super().__init__()
        if not HAS_KAN:
            raise RuntimeError('KAN not installed')
        flat = image_size * image_size
        self.enc = KAN(width=[flat, 512, 128], grid=5, k=3)
        self.dec = KAN(width=[128, 512, flat], grid=5, k=3)

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        z = self.enc(flat)
        out_flat = self.dec(z)
        return out_flat.view(x.size())


# Train AE
def train_ae(model, dataloader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
            n += batch.size(0)
        if n > 0:
            print(f'Epoch {epoch+1}/{epochs} - loss={epoch_loss / n:.6f}')
    return model


def get_ae_scores(model, dataloader, device=DEVICE):
    model.to(device)
    model.eval()
    scores = []
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch).mean(dim=[1,2,3]).cpu().numpy()
            scores.extend(loss.tolist())
    return np.array(scores)


def dataloader_to_numpy_flat(loader):
    feats = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        arr = batch.view(batch.size(0), -1).cpu().numpy()
        feats.append(arr)
    if len(feats) == 0:
        return np.zeros((0,0))
    return np.concatenate(feats, axis=0)


# ---------------------------
# Training/evaluation driver
# ---------------------------
def evaluate_one_subject(genuine_idx, all_images, models_to_run=('svm','cnn','kan')):
    print(f"Evaluating subject {genuine_idx}")
    genuine_imgs = list(all_images[genuine_idx])
    if len(genuine_imgs) < 2:
        print('Not enough genuine samples, skipping')
        return None
    np.random.shuffle(genuine_imgs)
    split = max(1, int(0.8*len(genuine_imgs)))
    train_imgs = genuine_imgs[:split]
    test_genuine = genuine_imgs[split:]
    impostor_imgs = [img for sid, imgs in all_images.items() if sid!=genuine_idx for img in imgs]

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor()])
    train_loader = DataLoader(WTMMImageDataset(train_imgs, transform), batch_size=BATCH_SIZE, shuffle=True)
    test_genuine_loader = DataLoader(WTMMImageDataset(test_genuine, transform), batch_size=BATCH_SIZE)
    impostor_loader = DataLoader(WTMMImageDataset(impostor_imgs, transform), batch_size=BATCH_SIZE)

    results = {}

    if 'svm' in models_to_run:
        # train SVM on flattened images
        train_feats = dataloader_to_numpy_flat(train_loader)
        if train_feats.shape[0] > 0:
            clf = OneClassSVM(kernel='rbf', nu=0.1).fit(train_feats)
            train_scores = -clf.decision_function(train_feats)
            gen_feats = dataloader_to_numpy_flat(test_genuine_loader)
            imp_feats = dataloader_to_numpy_flat(impostor_loader)
            gen_scores = -clf.decision_function(gen_feats) if gen_feats.shape[0]>0 else np.array([])
            imp_scores = -clf.decision_function(imp_feats) if imp_feats.shape[0]>0 else np.array([])
            thresh = train_scores.mean() + 3*train_scores.std()
            results['OneClassSVM'] = compute_metrics(gen_scores, imp_scores, train_scores, thresh)

    if 'cnn' in models_to_run:
        model = OneClassCNN(image_size=IMAGE_SIZE)
        model = train_ae(model, train_loader)
        train_scores = get_ae_scores(model, train_loader)
        gen_scores = get_ae_scores(model, test_genuine_loader)
        imp_scores = get_ae_scores(model, impostor_loader)
        thresh = train_scores.mean() + 3*train_scores.std()
        results['OneClassCNN'] = compute_metrics(gen_scores, imp_scores, train_scores, thresh)

    if 'kan' in models_to_run and HAS_KAN:
        model = OneClassKANWrapper(image_size=IMAGE_SIZE)
        model = train_ae(model, train_loader)
        train_scores = get_ae_scores(model, train_loader)
        gen_scores = get_ae_scores(model, test_genuine_loader)
        imp_scores = get_ae_scores(model, impostor_loader)
        thresh = train_scores.mean() + 3*train_scores.std()
        results['OneClassKAN'] = compute_metrics(gen_scores, imp_scores, train_scores, thresh)

    return results


def compute_metrics(genuine_scores, impostor_scores, train_scores, threshold):
    def safe_auc(gs, iscores):
        if len(gs)==0 and len(iscores)==0:
            return np.nan, np.nan
        labels = np.concatenate([np.zeros(len(gs)), np.ones(len(iscores))])
        scores = np.concatenate([gs, iscores])
        if len(np.unique(labels))<2:
            return np.nan, np.nan
        fpr,tpr,_ = roc_curve(labels, scores)
        roc_auc = auc(fpr,tpr)
        prec, rec, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(rec, prec)
        return roc_auc, pr_auc

    roc_auc, pr_auc = safe_auc(genuine_scores, impostor_scores)
    tp = np.sum(np.array(genuine_scores) < threshold) if len(genuine_scores)>0 else 0
    tn = np.sum(np.array(impostor_scores) >= threshold) if len(impostor_scores)>0 else 0
    total = len(genuine_scores) + len(impostor_scores)
    accuracy = (tp + tn) / total if total>0 else np.nan
    return {'AUROC': float(roc_auc) if not np.isnan(roc_auc) else np.nan,
            'AUPRC': float(pr_auc) if not np.isnan(pr_auc) else np.nan,
            'Accuracy': float(accuracy),
            'Threshold': float(threshold)}


# ---------------------------
# CLI and orchestration
# ---------------------------
def gather_images_from_image_dir(image_dir):
    subj_dirs = sorted([d for d in glob.glob(os.path.join(image_dir, 'subject_*')) if os.path.isdir(d)])
    all_images = {}
    for i, sd in enumerate(subj_dirs):
        imgs = sorted([os.path.join(sd, f) for f in os.listdir(sd) if f.lower().endswith('.png')])
        all_images[i] = imgs
    return all_images

def main():
    parser = argparse.ArgumentParser(description='ECG pipeline: preprocess -> WTMM images -> train models')
    parser.add_argument('--step', choices=['download','preprocess','train','all'], default='all', help='pipeline step')
    parser.add_argument('--dataset', default='ecg-id', help='which default physionet dataset to use: ecg-id, mitdb, nsrdb, ptb-xl')
    parser.add_argument('--custom-dir', default=None, help='path to custom image or wfdb directory')
    parser.add_argument('--out-root', default=PROJECT_ROOT, help='project root output folder')
    parser.add_argument('--max-beats', type=int, default=None, help='max beats per subject (for preprocessing)')
    parser.add_argument('--image-dir', default=None, help='if training: path to images root (subject_*/...)')
    parser.add_argument('--models', default='svm,cnn', help='comma list: svm,cnn,kan')
    args = parser.parse_args()

    # override globals per run
    global PROJECT_ROOT, RAW_DIR, IMAGE_DIR_ROOT, MODEL_DIR, RESULTS_DIR, MAX_BEATS_PER_SUBJECT
    PROJECT_ROOT = os.path.abspath(args.out_root)
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    IMAGE_DIR_ROOT = os.path.join(PROJECT_ROOT, 'images')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    ensure_dir(PROJECT_ROOT); ensure_dir(RAW_DIR); ensure_dir(IMAGE_DIR_ROOT); ensure_dir(MODEL_DIR); ensure_dir(RESULTS_DIR)
    if args.max_beats is not None:
        MAX_BEATS_PER_SUBJECT = args.max_beats

    models_to_run = [m.strip().lower() for m in args.models.split(',') if m.strip()!='']

    if args.step in ('download','all'):
        print('Downloading dataset...')
        download_dataset(args.dataset, RAW_DIR)

    if args.step in ('preprocess','all'):
        print('Preprocessing: generating WTMM images...')
        if args.custom_dir:
            # if custom points to a folder with subject subfolders containing images, copy them
            if os.path.isdir(args.custom_dir):
                subdirs = sorted([d for d in glob.glob(os.path.join(args.custom_dir, '*')) if os.path.isdir(d)])
                if len(subdirs) > 0 and any(f.lower().endswith('.png') for s in subdirs for f in os.listdir(s)):
                    process_custom_image_dir(args.custom_dir, IMAGE_DIR_ROOT)
                else:
                    # try to treat custom_dir as WFDB-like layout (each folder may contain rec_*)
                    process_ecg_id(args.custom_dir, IMAGE_DIR_ROOT, max_beats_per_subject=MAX_BEATS_PER_SUBJECT)
            else:
                raise FileNotFoundError('Custom dir not found')
        else:
            if args.dataset.lower() in ('ecg-id','ecgiddb'):
                process_ecg_id(RAW_DIR, IMAGE_DIR_ROOT, max_beats_per_subject=MAX_BEATS_PER_SUBJECT)
            else:
                # generic wfdb datasets: try to find records at RAW_DIR and process first-level rec_* files per folder
                print('Attempting generic wfdb processing using records under', RAW_DIR)
                process_ecg_id(RAW_DIR, IMAGE_DIR_ROOT, max_beats_per_subject=MAX_BEATS_PER_SUBJECT)

    if args.step in ('train','all'):
        print('Training/evaluating models...')
        image_dir_for_training = args.image_dir if args.image_dir is not None else IMAGE_DIR_ROOT
        all_images = gather_images_from_image_dir(image_dir_for_training)
        if len(all_images)==0:
            raise FileNotFoundError('No subject_* folders with images found under ' + image_dir_for_training)
        summary = {}
        for subj_idx in sorted(all_images.keys()):
            res = evaluate_one_subject(subj_idx, all_images, models_to_run=models_to_run)
            summary[subj_idx] = res
        # save summary
        with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('Saved summary to', os.path.join(RESULTS_DIR, 'summary.json'))

if __name__ == '__main__':
    main()
