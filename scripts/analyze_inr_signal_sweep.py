import argparse
import csv
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np

from src.io import ensure_dir


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def final_epoch_rows(rows):
    latest = {}
    for row in rows:
        trial_id = row["trial_id"]
        epoch = int(float(row["epoch"]))
        if trial_id not in latest or epoch > int(float(latest[trial_id]["epoch"])):
            latest[trial_id] = row
    return list(latest.values())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows, group_key, metric_key):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(as_float(row, metric_key))
    return {key: float(np.nanmean(values)) for key, values in groups.items()}


def plot_scatter(path, rows, x_key, y_key, title, xlabel, ylabel):
    methods = sorted({row["method"] for row in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        items = [row for row in rows if row["method"] == method]
        x = [as_float(row, x_key) for row in items]
        y = [as_float(row, y_key) for row in items]
        ax.scatter(x, y, label=method, alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_top_bar(path, rows, key, title, ylabel, reverse=False):
    ranked = sorted(rows, key=lambda row: as_float(row, key), reverse=reverse)
    top = ranked[: min(15, len(ranked))]
    labels = [row["trial_id"] for row in top]
    values = [as_float(row, key) for row in top]
    fig_width = max(12, 0.7 * len(top))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(np.arange(len(top)), values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.45)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def format_trial(row):
    return (
        f"{row['trial_id']}: method={row['method']}, lr={as_float(row, 'lr'):.1e}, "
        f"costScaling={as_float(row, 'costScaling'):.1e}, final_bias={as_float(row, 'final_bias'):g}, "
        f"grid_init_std={as_float(row, 'grid_init_std'):g}, "
        f"mse_change={as_float(row, 'mse_change'):.3e}, cost_change={as_float(row, 'cost_change'):.3e}, "
        f"delta_mean_abs={as_float(row, 'delta_gamma_mean_abs'):.3e}, "
        f"delta_max_abs={as_float(row, 'delta_gamma_max_abs'):.3e}, "
        f"direct_max_abs={as_float(row, 'direct_update_max_abs'):.3e}, "
        f"delta/direct={as_float(row, 'delta_over_direct_norm'):.3e}, "
        f"cos={as_float(row, 'delta_direct_cosine'):.3f}, "
        f"sign={as_float(row, 'delta_direct_sign_agreement'):.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze debug_inr_signal_sweep metrics.")
    parser.add_argument(
        "--metrics",
        default="temp/inr_signal_sweep/case1/histories/inr_signal_sweep_metrics.csv",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else metrics_path.parents[1] / "analysis")

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final_rows = final_epoch_rows(rows)

    ranked_mse = sorted(final_rows, key=lambda row: as_float(row, "mse_change"))
    ranked_cost = sorted(final_rows, key=lambda row: as_float(row, "cost_change"))
    ranked_alignment = sorted(final_rows, key=lambda row: as_float(row, "delta_direct_cosine"), reverse=True)
    ranked_bad = sorted(final_rows, key=lambda row: as_float(row, "mse_change"), reverse=True)

    write_csv(output_dir / "final_epoch_ranked_by_mse.csv", ranked_mse)

    plot_scatter(
        output_dir / "mse_change_vs_update_alignment.png",
        final_rows,
        "delta_direct_cosine",
        "mse_change",
        "MSE change vs actual/direct update alignment",
        "cosine(actual delta gamma, direct adjoint update)",
        "MSE change after probe",
    )
    plot_scatter(
        output_dir / "mse_change_vs_update_size.png",
        final_rows,
        "delta_over_direct_norm",
        "mse_change",
        "MSE change vs update-size ratio",
        "||actual delta gamma|| / ||direct update||",
        "MSE change after probe",
    )
    plot_top_bar(
        output_dir / "best_mse_change.png",
        final_rows,
        "mse_change",
        "Best final MSE changes",
        "MSE change",
    )
    plot_top_bar(
        output_dir / "best_update_alignment.png",
        final_rows,
        "delta_direct_cosine",
        "Best actual/direct update alignment",
        "cosine similarity",
        reverse=True,
    )

    lines = [
        "INR signal sweep analysis",
        f"metrics: {metrics_path}",
        f"rows: {len(rows)}",
        f"trials: {len(final_rows)}",
        "",
        "Best MSE improvement:",
        *[format_trial(row) for row in ranked_mse[:10]],
        "",
        "Worst MSE change:",
        *[format_trial(row) for row in ranked_bad[:10]],
        "",
        "Best cost decrease:",
        *[format_trial(row) for row in ranked_cost[:10]],
        "",
        "Best actual/direct update alignment:",
        *[format_trial(row) for row in ranked_alignment[:10]],
        "",
        "Mean final MSE change by method:",
    ]
    for method, value in sorted(group_mean(final_rows, "method", "mse_change").items()):
        lines.append(f"{method}: {value:.3e}")
    lines.extend(["", "Mean update alignment by method:"])
    for method, value in sorted(group_mean(final_rows, "method", "delta_direct_cosine").items()):
        lines.append(f"{method}: {value:.3e}")
    lines.extend(["", "Mean update-size ratio by method:"])
    for method, value in sorted(group_mean(final_rows, "method", "delta_over_direct_norm").items()):
        lines.append(f"{method}: {value:.3e}")

    summary_path = output_dir / "analysis_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
