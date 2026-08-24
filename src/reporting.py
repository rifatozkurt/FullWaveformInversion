"""
Shared reporting: aggregate CSVs and thesis-ready figures.

Every runner script calls into here so the figures look the same across
experiments and the numbers behind them are always written out as CSV. The rule
throughout: **every figure has a CSV next to it**, so a plot can be redrawn or
restyled later without re-running any GPU work.

Style choices are deliberate:
  * one consistent colour per model/method across every figure in the thesis
  * log scale for gamma MSE, which spans orders of magnitude
  * the trivial "no void anywhere" baseline drawn on convergence plots, because
    a curve that never crosses it has not reconstructed anything and that should
    be visible rather than inferred
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # scripts run headless in Colab
import matplotlib.pyplot as plt
import numpy as np

# One colour per method/model, used everywhere so figures are mutually readable.
PALETTE = {
    "unet": "#2f5aa8",
    "segformer": "#b64040",
    "segformer_imagenet": "#2f8f5b",
    "segformer_highres": "#8f6f2f",
    "inr_fwi": "#7f7f7f",
    "inr_siren_fwi": "#2f5aa8",
    "inr_siren_centered_fwi": "#7aa0d4",
    "inr_lr_fwi": "#b64040",
    "inr_mpe_fwi": "#2f8f5b",
    "inr_mpe_centered_fwi": "#78c39a",
    "inr_ig_fwi": "#6a3d9a",
    "inr_ig_centered_fwi": "#a684c9",
    "encoder": "#2f5aa8",
    "decoder": "#b64040",
    "random_encoder": "#2f8f5b",
    "none": "#4d4d4d",
}
FALLBACK = ["#2f5aa8", "#b64040", "#2f8f5b", "#8f6f2f", "#6a3d9a", "#3d8f8f",
            "#c1743a", "#7f7f7f"]

PRETTY = {
    "unet": "U-Net",
    "segformer": "SegFormer (random init)",
    "segformer_imagenet": "SegFormer (ImageNet init)",
    "segformer_highres": "SegFormer HighRes",
    "inr_fwi": "INR (tanh)",
    "inr_siren_fwi": "SIREN / IFWI",
    "inr_siren_centered_fwi": "SIREN (centred)",
    "inr_lr_fwi": "LR-FWI",
    "inr_mpe_fwi": "MPE-FWI",
    "inr_mpe_centered_fwi": "MPE-FWI (centred)",
    "inr_ig_fwi": "IG-FWI",
    "inr_ig_centered_fwi": "IG-FWI (centred)",
    "encoder": "freeze encoder",
    "decoder": "freeze decoder",
    "random_encoder": "frozen RANDOM encoder",
    "none": "full fine-tuning",
}


def label_of(key):
    return PRETTY.get(key, str(key).replace("_", " "))


def colour_of(key, index=0):
    return PALETTE.get(key, FALLBACK[index % len(FALLBACK)])


# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    """Write rows to CSV, unioning keys so ragged dicts do not silently drop columns."""
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_rows(rows, group_key, metric_keys):
    """Mean/std/min/max per group, plus n so the reader can judge the spread."""
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for name, items in groups.items():
        entry = {group_key: name, "n": len(items)}
        for metric in metric_keys:
            values = np.array([float(i[metric]) for i in items if metric in i], dtype=float)
            if not len(values):
                continue
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            entry[f"{metric}_min"] = float(values.min())
            entry[f"{metric}_max"] = float(values.max())
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
def plot_convergence_grid(path, histories, cases, ylabel="gamma MSE",
                          title="", key="mse", trivial=None):
    """
    One panel per case; one curve per method. The dashed line is the trivial
    "predict no void anywhere" solution -- a curve above it has reconstructed
    nothing, which is worth seeing directly.
    """
    cases = list(cases)
    if not histories or not cases:
        return None
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.5, 3.3 * len(cases)),
                             squeeze=False, constrained_layout=True)
    labels = list(dict.fromkeys(h["label"] for h in histories))
    for axis, case_id in zip(axes[:, 0], cases):
        for index, lab in enumerate(labels):
            match = [h for h in histories if h["case"] == case_id and h["label"] == lab]
            if not match:
                continue
            values = np.asarray(match[0][key], dtype=float)
            axis.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=3,
                      linewidth=1.9, color=colour_of(lab, index), label=label_of(lab))
        base = None
        if trivial and case_id in trivial:
            base = trivial[case_id]
        elif match and "trivial" in match[0]:
            base = match[0]["trivial"]
        if base:
            axis.axhline(base, color="k", linestyle="--", linewidth=1.1,
                         label="trivial (no void)")
        axis.set_yscale("log")
        axis.set_title(f"case {case_id}", fontsize=10)
        axis.set_xlabel("FWI epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7, ncol=2)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_reconstruction_gallery(path, reconstructions, cases, max_cases=4):
    """Target beside every method's final gamma -- the figure that goes in the results chapter."""
    cases = list(cases)[:max_cases]
    if not reconstructions or not cases:
        return None
    labels = list(dict.fromkeys(r["label"] for r in reconstructions))
    columns = 1 + len(labels)
    fig, axes = plt.subplots(len(cases), columns,
                             figsize=(2.6 * columns, 2.3 * len(cases)),
                             squeeze=False, constrained_layout=True)
    for row, case_id in enumerate(cases):
        entries = [r for r in reconstructions if r["case"] == case_id]
        if not entries:
            continue
        axes[row][0].imshow(np.transpose(entries[0]["target"]), vmin=0, vmax=1,
                            cmap="coolwarm", aspect="auto")
        axes[row][0].set_ylabel(f"case {case_id}", fontsize=9)
        if row == 0:
            axes[row][0].set_title("target", fontsize=9)
        for column, lab in enumerate(labels, start=1):
            match = [r for r in entries if r["label"] == lab]
            if match:
                axes[row][column].imshow(np.transpose(match[0]["final"]), vmin=0, vmax=1,
                                         cmap="coolwarm", aspect="auto")
            if row == 0:
                axes[row][column].set_title(label_of(lab), fontsize=8)
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_metric_bars(path, rows, group_key, metrics_to_plot):
    """Mean +- std per group, one panel per metric."""
    if not rows:
        return None
    groups = list(dict.fromkeys(r[group_key] for r in rows))
    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(4.6 * len(metrics_to_plot), 4.4),
                             squeeze=False, constrained_layout=True)
    positions = np.arange(len(groups))
    for axis, (metric, nice) in zip(axes[0], metrics_to_plot):
        means, errors = [], []
        for name in groups:
            values = np.array([float(r[metric]) for r in rows
                               if r[group_key] == name and metric in r], dtype=float)
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.bar(positions, means, yerr=errors, capsize=4,
                 color=[colour_of(g, i) for i, g in enumerate(groups)])
        axis.set_xticks(positions)
        axis.set_xticklabels([label_of(g) for g in groups], rotation=25,
                             ha="right", fontsize=8)
        axis.set_title(nice, fontsize=10)
        axis.grid(alpha=0.3, axis="y")
        if metric.startswith(("vs_", "gamma_mse", "mse")):
            axis.set_yscale("log")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_scaling_curve(path, rows, x_key, y_key, group_key,
                       xlabel, ylabel, title, logx=True, logy=False):
    """
    Metric vs pretraining-set size, one line per model family -- the figure the
    data-efficiency claim rests on. Error bars are the spread across eval cases.
    """
    if not rows:
        return None
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    groups = list(dict.fromkeys(r[group_key] for r in rows))
    for index, name in enumerate(groups):
        subset = [r for r in rows if r[group_key] == name]
        xs = sorted({float(r[x_key]) for r in subset})
        means, errors = [], []
        for x in xs:
            values = np.array([float(r[y_key]) for r in subset
                               if float(r[x_key]) == x], dtype=float)
            means.append(values.mean())
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.errorbar(xs, means, yerr=errors, marker="o", capsize=4, linewidth=2.1,
                      color=colour_of(name, index), label=label_of(name))
    if logx:
        axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_pretraining_curves(path, curves, ylabel, title, logy=True):
    """Training/validation curves for every pretrained model, grouped by family."""
    if not curves:
        return None
    families = list(dict.fromkeys(c["family"] for c in curves))
    fig, axes = plt.subplots(1, len(families), figsize=(6.0 * len(families), 4.4),
                             squeeze=False, constrained_layout=True)
    for axis, family in zip(axes[0], families):
        subset = [c for c in curves if c["family"] == family]
        sizes = sorted({c["samples"] for c in subset})
        cmap = plt.get_cmap("viridis")
        for index, size in enumerate(sizes):
            match = [c for c in subset if c["samples"] == size]
            if not match:
                continue
            values = np.asarray(match[0]["values"], dtype=float)
            axis.plot(np.arange(1, len(values) + 1), values, linewidth=1.8,
                      color=cmap(index / max(1, len(sizes) - 1)), label=f"{size}")
        axis.set_title(label_of(family), fontsize=10)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        if logy:
            axis.set_yscale("log")
        axis.grid(alpha=0.3, which="both")
        axis.legend(title="samples", fontsize=7, title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
