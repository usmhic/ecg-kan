"""Figures and table formatting for the ECG-ID benchmark.

Two groups of figures are produced:

* signal figures (1-3), which need the ECG-ID recordings and show what the
  processing does to a beat;
* result figures (4-7), which need only saved metric tables and reconstruction
  errors, so they can be rebuilt from the published artifacts without
  retraining (see research/make_paper_figures.py).

Every result figure takes tidy inputs:

``metrics``  columns subject, feature_set ('raw'|'wtmm'), model, metric
             ('auroc'|'f1'|'eer'), value - one row per person.
``summary``  columns feature_set, model, metric, mean, ci_low, ci_high.
``recon``    {feature_set: {subject: {model: {'genuine': [...],
             'impostor': [...]}}}} reconstruction errors.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pywt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from ecg_features import extract_beats, scales_for, compute_wtmm_lines

if TYPE_CHECKING:
    from pipeline import PipelineConfig


MODEL_ORDER = ["OC-SVM", "Isolation Forest", "LOF", "Elliptic Envelope",
               "MLP Autoencoder", "KAN Autoencoder"]

# Labels written into the metric CSVs; the table builder parses them.
FEAT_LABELS = {"raw": "Raw Beat (before wavelet, PCA<=40)",
               "wtmm": "WTMM Summary (after wavelet, 26-d)"}

# Labels shown to readers.
BRANCH_NAMES = {"raw": "Waveform + PCA", "wtmm": "Wavelet summary (26 values)"}
MODEL_NAMES = {"MLP Autoencoder": "MLP autoencoder",
               "KAN Autoencoder": "KAN autoencoder"}

# Validated categorical slots (blue, orange, aqua) plus neutral ink.
INK, MUTED, GRID, NEUTRAL = "#1f1f1e", "#6b6a66", "#e4e3df", "#a3a29d"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLOURS = {"MLP Autoencoder": BLUE, "KAN Autoencoder": ORANGE,
                 "OC-SVM": AQUA}
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b"])

FULL_WIDTH = 7.0   # inches; scaled to \textwidth in the manuscript
DPI = 400


def to_df(res_dict, metric):
    rows = [{"Person": f"P{i + 1:02d}",
             **{m: round(v.get(metric, np.nan), 4) for m, v in res.items()}}
            for i, res in sorted(res_dict.items())]
    df = pd.DataFrame(rows, columns=["Person", *MODEL_ORDER]).set_index("Person")
    if len(df) > 0:
        df.loc["MEAN"] = df.mean(numeric_only=True).round(4)
    return df


# --------------------------------------------------------------------------
# Shared style and helpers
# --------------------------------------------------------------------------
def use_paper_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


def panel_title(ax, letter, title=""):
    """Left-aligned panel title that starts with a bold panel letter."""
    ax.set_title(rf"$\bf{{({letter})}}$  {title}", loc="left")


def hz_axis(ax, ticks=(1, 2, 5, 10, 20, 50)):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def wavelet_power(beat, fs, cf, cfg: PipelineConfig):
    scales = scales_for(fs, cf, cfg)
    coeffs, freqs = pywt.cwt(beat, scales, cfg.wavelet_name,
                             sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    return power, freqs, compute_wtmm_lines(power, scales)


def load_summary(path):
    """Read a benchmark_results.csv into the tidy ``summary`` layout."""
    df = pd.read_csv(path)
    return pd.DataFrame({
        "feature_set": np.where(df["Feature set"].str.startswith("Raw"),
                                "raw", "wtmm"),
        "model": df["Model"], "metric": df["Metric"].str.lower(),
        "mean": df["Mean"], "ci_low": df["CI95 low"],
        "ci_high": df["CI95 high"]})


def per_person(metrics, feature_set, metric):
    """Person x model table, rounded as in the exported summaries."""
    sub = metrics[(metrics.feature_set == feature_set) & (metrics.metric == metric)]
    table = sub.pivot(index="subject", columns="model", values="value")
    return table[[m for m in MODEL_ORDER if m in table.columns]].round(4)


# --------------------------------------------------------------------------
# Figure 1: from recording to wavelet ridges
# --------------------------------------------------------------------------
def plot_processing_steps(person, cf, cfg: PipelineConfig, fig_dir):
    """``person`` is the dict returned by ecg_features.load_person."""
    print("Plotting the processing steps ...")
    fs, raw, cleaned, peaks = (person["fs"], person["signal"],
                               person["cleaned"], person["r_peaks"])
    centre = peaks[min(1, len(peaks) - 1)]
    beat = extract_beats(cleaned, np.array([centre]), fs, cfg)[0][0]
    t_beat = np.arange(beat.size) / fs - cfg.pre_ms / 1000
    power, freqs, lines = wavelet_power(beat, fs, cf, cfg)

    fig = plt.figure(figsize=(FULL_WIDTH, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55,
                            wspace=0.28)

    ax = fig.add_subplot(grid[0, :])
    start = max(0, centre - int(1.0 * fs))
    stop = min(raw.size, centre + int(1.6 * fs))
    t = np.arange(start, stop) / fs
    ax.plot(t, raw[start:stop], color=NEUTRAL, lw=0.7, label="Recorded")
    ax.plot(t, cleaned[start:stop], color=BLUE, lw=0.9, label="Cleaned")
    shown = peaks[(peaks >= start) & (peaks < stop)]
    ax.plot(shown / fs, cleaned[shown], "o", ms=3.5, color=ORANGE,
            label="Detected R-peaks")
    ax.axvspan((centre - cfg.pre_ms * fs / 1000) / fs,
               (centre + cfg.post_ms * fs / 1000) / fs,
               color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(centre / fs, ax.get_ylim()[1], "beat window", ha="center",
            va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.22))
    panel_title(ax, "a", "Recording")

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(t_beat * 1000, beat, color=BLUE, lw=1.1)
    ax.axvline(0, ls="--", lw=0.7, color=MUTED)
    ax.text(12, beat.max(), "R-peak", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Amplitude (mV)")
    panel_title(ax, "b", "One 700 ms beat")

    ax = fig.add_subplot(grid[1, 1])
    mesh = ax.pcolormesh(t_beat * 1000, freqs, power, shading="auto",
                         cmap=BLUES, rasterized=True)
    for line in lines:
        ax.plot([t_beat[ti] * 1000 for _, ti in line],
                [freqs[si] for si, _ in line], color=ORANGE, lw=0.6)
    edges = np.logspace(np.log10(cfg.freq_min), np.log10(cfg.freq_max),
                        cfg.n_bands + 1)
    for e in edges[1:-1]:
        ax.axhline(e, color=INK, lw=0.5, ls=":")
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        ax.text(1.02, np.sqrt(lo * hi), f"B{b}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7, color=MUTED)
    hz_axis(ax)
    ax.set_ylim(cfg.freq_min, cfg.freq_max)
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Frequency (Hz)")
    panel_title(ax, "c", "Wavelet power and maxima ridges")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.08, fraction=0.05)
    cbar.set_label("Power (a.u.)")
    cbar.outline.set_visible(False)
    save(fig, fig_dir, "fig_stages.png")


# --------------------------------------------------------------------------
# Figure 2: beats and ridges for several people
# --------------------------------------------------------------------------
def plot_beat_examples(people, cf, cfg: PipelineConfig, fig_dir):
    """``people`` maps a 1-based person number to load_person output."""
    print("Plotting example beats ...")
    shown = sorted(people)[:cfg.grid_subjects]
    fig, axes = plt.subplots(2, len(shown), figsize=(FULL_WIDTH, 3.3),
                             squeeze=False, sharex=True)
    for col, pid in enumerate(shown):
        person = people[pid]
        fs, beats = person["fs"], person["beats"]
        t_ms = (np.arange(beats.shape[1]) / fs - cfg.pre_ms / 1000) * 1000

        ax = axes[0, col]
        for beat in beats[:cfg.grid_beats]:
            ax.plot(t_ms, (beat - beat.mean()) / (beat.std() + 1e-8),
                    color=BLUE, lw=0.7, alpha=0.7)
        ax.set_title(f"Person {pid:02d}")
        ax.set_ylim(-4.5, 6)
        if col:
            ax.tick_params(labelleft=False)

        ax = axes[1, col]
        power, freqs, lines = wavelet_power(beats[0], fs, cf, cfg)
        ax.pcolormesh(t_ms, freqs, power, shading="auto", cmap=BLUES,
                      rasterized=True)
        for line in lines:
            ax.plot([t_ms[ti] for _, ti in line], [freqs[si] for si, _ in line],
                    color=ORANGE, lw=0.45)
        hz_axis(ax, ticks=(1, 10, 60))
        ax.set_ylim(cfg.freq_min, cfg.freq_max)
        if col:
            ax.tick_params(labelleft=False)
    axes[0, 0].set_xticks([-200, 0, 200, 400])
    axes[0, 0].set_ylabel("Normalised\namplitude")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    fig.supxlabel("Time from R-peak (ms)", fontsize=8, y=0.0)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    save(fig, fig_dir, "fig_beats.png")


# --------------------------------------------------------------------------
# Figure 3: two-dimensional t-SNE view of both inputs
# --------------------------------------------------------------------------
def plot_tsne(feats_raw, feats_wtmm, cfg: PipelineConfig, fig_dir,
              highlight=(1, 14, 21)):
    """``feats_*`` map a 1-based person number to a (beats, features) array."""
    print("Plotting the t-SNE view ...")
    people = sorted(feats_raw)
    labels = np.concatenate([np.full(len(feats_raw[p]), p) for p in people])
    perplexity = min(cfg.tsne_perplexity, max(5, len(labels) - 1))
    views = []
    for feats in (feats_raw, feats_wtmm):
        X = StandardScaler().fit_transform(np.vstack([feats[p] for p in people]))
        views.append(TSNE(n_components=2, perplexity=perplexity, init="pca",
                          random_state=cfg.seed).fit_transform(X))

    colours = dict(zip(highlight, (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1))
    titles = ("Waveform (350 samples, standardised)",
              "Wavelet summary (26 values)")
    for k, (ax, view, title) in enumerate(zip(axes, views, titles)):
        rest = ~np.isin(labels, list(colours))
        ax.scatter(view[rest, 0], view[rest, 1], s=6, color=NEUTRAL,
                   alpha=0.45, lw=0)
        for pid, colour in colours.items():
            m = labels == pid
            ax.scatter(view[m, 0], view[m, 1], s=11, color=colour, lw=0.3,
                       edgecolor="white", label=f"Person {pid:02d}")
        panel_title(ax, "ab"[k], title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
                   handletextpad=0.2, markerscale=1.6)
    fig.text(0.5, -0.01, f"{len(labels)} beats from {len(people)} people; "
             "grey points are the other people", ha="center", fontsize=7,
             color=MUTED)
    fig.tight_layout()
    save(fig, fig_dir, "fig_tsne.png")


# --------------------------------------------------------------------------
# Figure 4: mean performance with 95% intervals
# --------------------------------------------------------------------------
def plot_mean_summary(summary, fig_dir):
    print("Plotting mean performance ...")
    metrics = [("auroc", "AUROC (higher is better)", (0.80, 1.0)),
               ("f1", "$F_1$ (higher is better)", (0.0, 1.0)),
               ("eer", "EER (lower is better)", (0.0, 0.25))]
    models = MODEL_ORDER[::-1]
    y = np.arange(len(models))
    style = {"raw": dict(offset=-0.17, face="white", edge=MUTED),
             "wtmm": dict(offset=0.17, face=BLUE, edge=BLUE)}

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7), sharey=True)
    for k, (ax, (metric, title, xlim)) in enumerate(zip(axes, metrics)):
        for fk, st in style.items():
            rows = (summary[(summary.feature_set == fk) & (summary.metric == metric)]
                    .set_index("model").reindex(models))
            yy = y + st["offset"]
            ax.hlines(yy, rows.ci_low, rows.ci_high, color=st["edge"], lw=1.1)
            ax.plot(rows["mean"], yy, "o", ms=4.8, mfc=st["face"],
                    mec=st["edge"], mew=1.1, label=BRANCH_NAMES[fk])
        ax.set_xlim(*xlim)
        panel_title(ax, "abc"[k], title)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_NAMES.get(m, m) for m in models])
    for label in axes[0].get_yticklabels():
        if label.get_text().startswith("KAN"):
            label.set_fontweight("bold")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=2, loc="lower center",
               bbox_to_anchor=(0.55, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, fig_dir, "fig_bars.png")


# --------------------------------------------------------------------------
# Figure 5: AUROC for every person and model
# --------------------------------------------------------------------------
def plot_auroc_heatmap(metrics, fig_dir, label_below=0.90):
    print("Plotting the AUROC heatmap ...")
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 3.5), sharex=True)
    image = None
    for k, (ax, fk) in enumerate(zip(axes, ("raw", "wtmm"))):
        table = per_person(metrics, fk, "auroc").T
        values = table.to_numpy(float)
        image = ax.imshow(values, cmap=BLUES, vmin=0.5, vmax=1.0,
                          aspect="auto", interpolation="nearest")
        for r, c in zip(*np.where(values < label_below)):
            ax.text(c, r, f"{values[r, c]:.2f}"[1:], ha="center", va="center",
                    fontsize=5.6, color="white" if values[r, c] >= 0.8 else INK)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in table.index])
        for label in ax.get_yticklabels():
            if label.get_text().startswith("KAN"):
                label.set_fontweight("bold")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{s:02d}" for s in table.columns], fontsize=6)
        ax.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
        panel_title(ax, "ab"[k], BRANCH_NAMES[fk])
    axes[1].set_xlabel("Person")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("AUROC")
    cbar.outline.set_visible(False)
    save(fig, fig_dir, "fig_heatmap.png")


# --------------------------------------------------------------------------
# Figure 6: KAN against MLP for each person
# --------------------------------------------------------------------------
def plot_kan_vs_mlp(metrics, fig_dir, feature_set="wtmm"):
    print("Plotting per-person KAN versus MLP ...")
    panels = [("auroc", "AUROC", (0.75, 1.01)), ("f1", "$F_1$", (-0.03, 1.03))]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    for k, (ax, (metric, name, lim)) in enumerate(zip(axes, panels)):
        table = per_person(metrics, feature_set, metric)
        mlp, kan = table["MLP Autoencoder"], table["KAN Autoencoder"]
        diff = (kan - mlp).round(4)
        ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--", zorder=0)
        ax.scatter(mlp, kan, s=18, color=ORANGE, alpha=0.75, lw=0.4,
                   edgecolor="white", zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"MLP autoencoder {name}")
        ax.set_ylabel(f"KAN autoencoder {name}")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        both_one = int(((mlp == 1) & (kan == 1)).sum())
        ax.text(0.04, 0.96, f"KAN higher: {(diff > 0).sum()}\n"
                f"Equal: {(diff == 0).sum()} ({both_one} at 1.00)\n"
                f"MLP higher: {(diff < 0).sum()}",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)
        panel_title(ax, "ab"[k], name)
    fig.tight_layout(w_pad=2.0)
    save(fig, fig_dir, "fig_perperson.png")


# --------------------------------------------------------------------------
# Figure 7: ROC curves of the two autoencoders
# --------------------------------------------------------------------------
def _person_roc(errors):
    genuine, impostor = (np.asarray(errors["genuine"], float),
                         np.asarray(errors["impostor"], float))
    labels = np.r_[np.ones(genuine.size), np.zeros(impostor.size)]
    # A smaller reconstruction error means a closer match.
    fpr, tpr, _ = roc_curve(labels, -np.r_[genuine, impostor])
    return fpr, tpr


def _mean_roc(curves, grid):
    """Vertical average: each person's best genuine-acceptance rate at or
    below every false-acceptance rate on the grid."""
    return np.mean([tpr[np.searchsorted(fpr, grid, side="right") - 1]
                    for fpr, tpr in curves], axis=0)


def plot_autoencoder_roc(recon, metrics, fig_dir):
    print("Plotting autoencoder ROC curves ...")
    grid = np.linspace(0, 1, 201)
    models = ["MLP Autoencoder", "KAN Autoencoder"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharey=True)
    for k, (ax, model) in enumerate(zip(axes, models)):
        colour = MODEL_COLOURS[model]
        curves = {fk: [_person_roc(d[model]) for _, d in sorted(
                       recon[fk].items(), key=lambda item: int(item[0]))
                       if model in d]
                  for fk in ("raw", "wtmm")}
        auroc = {fk: per_person(metrics, fk, "auroc")[model].mean()
                 for fk in ("raw", "wtmm")}
        for fpr, tpr in curves["wtmm"]:
            ax.plot(fpr, tpr, color=colour, lw=0.6, alpha=0.25)
        ax.plot(grid, _mean_roc(curves["raw"], grid), color=MUTED, lw=1.3,
                ls="--", label=f"Waveform + PCA ({auroc['raw']:.3f})")
        ax.plot(grid, _mean_roc(curves["wtmm"], grid), color=colour, lw=1.8,
                label=f"Wavelet summary ({auroc['wtmm']:.3f})")
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        panel_title(ax, "ab"[k], MODEL_NAMES[model])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False acceptance rate")
        ax.legend(loc="lower right", handlelength=1.6,
                  title="Mean curve (mean AUROC)", title_fontsize=7)
    axes[0].set_ylabel("Genuine acceptance rate")
    fig.tight_layout(w_pad=1.5)
    save(fig, fig_dir, "fig_roc.png")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def make_signal_figures(people, cf, cfg: PipelineConfig, fig_dir):
    """``people`` maps a 1-based person number to load_person output."""
    use_paper_style()
    first = sorted(people)[0]
    plot_processing_steps(people[first], cf, cfg, fig_dir)
    plot_beat_examples(people, cf, cfg, fig_dir)
    plot_tsne({p: d["raw"] for p, d in people.items()},
              {p: d["wtmm"] for p, d in people.items()}, cfg, fig_dir)


def make_result_figures(metrics, summary, recon, fig_dir):
    use_paper_style()
    plot_mean_summary(summary, fig_dir)
    plot_auroc_heatmap(metrics, fig_dir)
    plot_kan_vs_mlp(metrics, fig_dir)
    plot_autoencoder_roc(recon, metrics, fig_dir)
