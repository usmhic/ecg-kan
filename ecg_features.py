"""ECG-ID loading, beat extraction and wavelet ridge features.

Shared by the command-line experiment, notebooks and figure code.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pywt
import wfdb
import neurokit2 as nk

if TYPE_CHECKING:
    from pipeline import PipelineConfig


def detect_rpeaks(sig, fs):
    """Clean the raw ECG with NeuroKit2 and return (R-peak indices, cleaned)."""
    signals, info = nk.ecg_process(sig, sampling_rate=fs)
    cleaned = signals["ECG_Clean"].values
    r_idx = np.asarray(info["ECG_R_Peaks"])
    return r_idx, cleaned


def extract_beats(cleaned, r_idx, fs, cfg: PipelineConfig):
    """Cut one window per R-peak and return a (n_beats, n_samples) array."""
    pre_s, post_s = cfg.pre_ms / 1000.0, cfg.post_ms / 1000.0
    epochs = nk.epochs_create(
        cleaned, events=r_idx, sampling_rate=fs,
        epochs_start=-pre_s, epochs_end=post_s,
    )
    keys = sorted(epochs.keys(), key=lambda k: int(k))
    if not keys:
        raise ValueError("No heartbeat windows were extracted")
    beats = np.array([epochs[k]["Signal"].values for k in keys])
    t_axis = np.linspace(-pre_s, post_s, beats.shape[1])
    # Incomplete boundary windows are NaN-padded by NeuroKit2.
    beats = beats[np.isfinite(beats).all(axis=1)]
    if not len(beats):
        raise ValueError("No complete, finite heartbeat windows remain")
    return beats, t_axis


# --------------------------------------------------------------------------
# Data loading: ECG-ID, or an explicitly requested synthetic demonstration.
# --------------------------------------------------------------------------
def load_ecg_id_from_physionet(cfg: PipelineConfig) -> bool:
    """Download ECG-ID if absent; report whether it is available."""
    try:
        if not os.path.isdir(cfg.db_dir) or not os.listdir(cfg.db_dir):
            print("Downloading ECG-ID from PhysioNet (first run only) ...")
            wfdb.dl_database("ecgiddb", cfg.db_dir)
        else:
            print("ECG-ID already present on disk.")
        return True
    except Exception as e:
        print(f"Could not reach PhysioNet ({type(e).__name__}: {e}).")
        return False


def simulate_synthetic_ecgid(cfg: PipelineConfig):
    """Explicit offline demonstration; never used as a download fallback.

    Every subject gets a different heart rate, noise level and amplitude.
    The run configuration and console banner identify synthetic_demo data.
    """
    print("\n" + "!" * 74)
    print("! Explicit SYNTHETIC demo ECG: these are NOT biometric results.")
    print("!" * 74 + "\n")

    rng = np.random.default_rng(cfg.seed)
    fs = cfg.fs_default
    duration_s = 40

    class _Rec:
        pass

    for idx in range(cfg.subject_start - 1,
                     cfg.subject_start - 1 + cfg.num_subjects):
        hr = float(rng.uniform(60, 95))
        noise = float(rng.uniform(0.01, 0.05))
        amp_scale = float(rng.uniform(0.8, 1.3))
        sig = nk.ecg_simulate(
            duration=duration_s, sampling_rate=fs, heart_rate=hr,
            noise=noise, random_state=int(cfg.seed + idx),
        )
        sig = sig * amp_scale
        rec = _Rec()
        rec.p_signal = sig.reshape(-1, 1)
        rec.fs = fs
        yield idx, rec


def load_subject_records(cfg: PipelineConfig):
    """Return an iterator of (idx, record) pairs and an explicit source tag."""
    if cfg.synthetic_demo:
        return simulate_synthetic_ecgid(cfg), "synthetic_demo"
    ok = load_ecg_id_from_physionet(cfg)
    if ok:
        def _gen():
            for idx in range(cfg.subject_start - 1,
                             cfg.subject_start - 1 + cfg.num_subjects):
                person = idx + 1
                rec_name = f"{cfg.db_dir}/Person_{person:02d}/rec_1"
                print(f"-> Person {person:02d}", end="  ")
                try:
                    rec = wfdb.rdrecord(rec_name)
                except Exception as e:
                    raise RuntimeError(f"Cannot read {rec_name}") from e
                yield idx, rec
        return _gen(), "physionet"
    else:
        raise RuntimeError("ECG-ID is unavailable. Supply --db-dir or download it "
                           "from PhysioNet. Use --synthetic-demo only for a demo.")


# --------------------------------------------------------------------------
# 5. WTMM helpers -- unchanged methodology
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


def wtmm_feature_vector(beat, scales, freqs_out, fs, cfg: PipelineConfig):
    """Turn one beat into a 26-D WTMM summary vector.

    Layout: [ total_lines,
              (count, mean_f, std_f, mean_p, std_p) x N_BANDS log-spaced bands ].
    Empty bands contribute five zeros.
    """
    coeffs, _ = pywt.cwt(beat, scales, cfg.wavelet_name, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    lines = compute_wtmm_lines(power, scales)

    edges = np.logspace(np.log10(cfg.freq_min), np.log10(cfg.freq_max), cfg.n_bands + 1)
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


def scales_for(fs, cf, cfg: PipelineConfig):
    freqs = np.logspace(np.log10(cfg.freq_min), np.log10(cfg.freq_max), cfg.n_freqs)
    return cf * fs / freqs


def feature_names(cfg: PipelineConfig):
    """Column names matching the layout of wtmm_feature_vector."""
    return ["total_lines"] + [
        f"band{band}_{stat}"
        for band in range(1, cfg.n_bands + 1)
        for stat in ("count", "mean_freq", "sd_freq", "mean_power", "sd_power")
    ]


def beat_features(beats, fs, cfg: PipelineConfig):
    """Build both input branches once, with identical arithmetic in every caller."""
    if len(beats) == 0:
        raise ValueError("Cannot extract features from an empty beat array")
    scales = scales_for(fs, pywt.central_frequency(cfg.wavelet_name), cfg)
    _, freqs = pywt.cwt(beats[0], scales, cfg.wavelet_name, sampling_period=1.0 / fs)
    return {
        "scales": scales, "freqs": freqs,
        "raw": np.vstack([((b - b.mean()) / (b.std() + 1e-8)).astype(np.float32)
                          for b in beats]),
        "wtmm": np.vstack([wtmm_feature_vector(b, scales, freqs, fs, cfg)
                           for b in beats]),
    }


def process_record(rec, cfg: PipelineConfig):
    """Clean a WFDB record and extract complete beats from its raw first channel."""
    signal = np.asarray(rec.p_signal)[:, 0]
    fs = int(getattr(rec, "fs", cfg.fs_default) or cfg.fs_default)
    r_peaks, cleaned = detect_rpeaks(signal, fs)
    beats, beat_time = extract_beats(cleaned, r_peaks, fs, cfg)
    return dict(signal=signal, fs=fs, cleaned=cleaned, r_peaks=r_peaks,
                beats=beats, beat_time=beat_time)


def load_person(person, cfg: PipelineConfig, record=1):
    """One person, one recording -> everything downstream needs.

    Runs exactly the chain main() runs -- read, clean, find R-peaks, cut beats,
    standardise, summarise each beat into the 26 WTMM values -- and hands back
    the intermediate products too, so the notebooks can show the steps without
    re-implementing any of them.

    Returns a dict with: signal, fs, cleaned, r_peaks, beats, beat_time,
    scales, freqs, raw (standardised waveforms) and wtmm (26 values per beat).
    """
    rec = wfdb.rdrecord(f"{cfg.db_dir}/Person_{person:02d}/rec_{record}")
    data = process_record(rec, cfg)
    return {**data, **beat_features(data["beats"], data["fs"], cfg)}


