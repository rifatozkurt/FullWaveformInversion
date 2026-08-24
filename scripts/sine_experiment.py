import argparse
import csv
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np

from src import io
from src.config import load_config
from src.experiments.base import case_file_stem


def load_gamma(data_dir, case_id):
    stem = case_file_stem(case_id)
    return io.load_hdf(Path(data_dir) / f"material{stem}.h5").astype(np.float64)


def frequency_radius(shape):
    nx, ny = shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.rfftfreq(ny) * ny
    return np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)


def reconstruct_with_cutoff(coefficients, radius, cutoff, target_shape, min_frequency=0):
    mask = radius <= cutoff
    if min_frequency > 0:
        mask &= radius >= min_frequency
        mask[0, 0] = True
    masked = np.where(mask, coefficients, 0)
    reconstruction = np.fft.irfft2(masked, s=target_shape)
    return reconstruction, int(np.count_nonzero(mask) - 1)


def reconstruction_metrics(target, reconstruction):
    error = reconstruction - target
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    denom = float(np.ptp(target)) or 1.0
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": rmse / denom,
        "min": float(reconstruction.min()),
        "max": float(reconstruction.max()),
        "mean": float(reconstruction.mean()),
    }


def save_metrics(path, rows):
    io.ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_summary_figure(path, target, snapshots):
    columns = len(snapshots) + 1
    fig, axes = plt.subplots(2, columns, figsize=(3.1 * columns, 6.0), constrained_layout=True)
    axes[0, 0].imshow(target.T, vmin=0, vmax=1, cmap="coolwarm")
    axes[0, 0].set_title("target")
    axes[1, 0].axis("off")

    for column, snapshot in enumerate(snapshots, start=1):
        reconstruction = snapshot["reconstruction"]
        residual = reconstruction - target
        axes[0, column].imshow(reconstruction.T, vmin=0, vmax=1, cmap="coolwarm")
        axes[0, column].set_title(
            "k <= {cutoff}\n{waves} modes".format(
                cutoff=snapshot["cutoff"],
                waves=snapshot["waves"],
            ),
            fontsize=10,
        )
        vmax = max(abs(float(residual.min())), abs(float(residual.max())), 1e-12)
        axes[1, column].imshow(residual.T, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        axes[1, column].set_title("residual", fontsize=10)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_error_curve(path, rows):
    cutoffs = [row["cutoff"] for row in rows]
    modes = [row["waves"] for row in rows]
    mse = [row["mse"] for row in rows]

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(modes, mse, marker="o", linewidth=2)
    for cutoff, mode_count, value in zip(cutoffs, modes, mse):
        axis.annotate(str(cutoff), (mode_count, value), fontsize=8)
    axis.set_xlabel("retained non-DC Fourier modes")
    axis.set_ylabel("MSE")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.set_title("Reconstruction error by frequency cutoff")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_snapshot_indices(row_count, snapshot_count):
    if row_count <= snapshot_count:
        return list(range(row_count))
    return sorted(set(np.linspace(0, row_count - 1, snapshot_count, dtype=int).tolist()))


def run(args):
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    run_dir = io.create_run_dir(
        Path(config["paths"].get("runs", "runs")),
        prefix=f"sine_experiment_case{args.case}",
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "outputs", run_dir / "histories"])

    target = load_gamma(data_dir, args.case)
    coefficients = np.fft.rfft2(target)
    radius = frequency_radius(target.shape)
    max_available = int(np.floor(radius.max()))
    max_frequency = args.max_frequency or max_available
    cutoffs = list(range(args.start_frequency, max_frequency + 1, args.frequency_step))
    if not cutoffs or cutoffs[-1] != max_frequency:
        cutoffs.append(max_frequency)

    rows = []
    reconstructions = []
    for cutoff in cutoffs:
        reconstruction, waves = reconstruct_with_cutoff(
            coefficients,
            radius,
            cutoff,
            target.shape,
            min_frequency=args.min_frequency,
        )
        if args.clip:
            reconstruction = np.clip(reconstruction, 0.0, 1.0)
        metrics = reconstruction_metrics(target, reconstruction)
        row = {
            "case": args.case,
            "cutoff": cutoff,
            "waves": waves,
            **metrics,
        }
        rows.append(row)
        reconstructions.append(reconstruction)

    snapshot_indices = choose_snapshot_indices(len(rows), args.snapshots)
    snapshots = [
        {
            "cutoff": rows[index]["cutoff"],
            "waves": rows[index]["waves"],
            "reconstruction": reconstructions[index],
        }
        for index in snapshot_indices
    ]

    io.save_hdf(run_dir / "outputs" / f"sine_case{args.case}_target_gamma.h5", target, key="gamma")
    io.save_hdf(
        run_dir / "outputs" / f"sine_case{args.case}_final_reconstruction.h5",
        reconstructions[-1],
        key="gamma",
    )
    save_metrics(run_dir / "histories" / f"sine_case{args.case}_metrics.csv", rows)
    save_summary_figure(run_dir / "figures" / f"sine_case{args.case}_reconstructions.png", target, snapshots)
    save_error_curve(run_dir / "figures" / f"sine_case{args.case}_error_curve.png", rows)

    best = min(rows, key=lambda row: row["mse"])
    print(f"Saved sine experiment to {run_dir}")
    print(
        "Best cutoff: {cutoff}, modes: {waves}, MSE: {mse:.6e}, relative RMSE: {relative_rmse:.6e}".format(
            **best
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Approximate one gamma image with progressively richer 2D sine/Fourier bases."
    )
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--start-frequency", type=int, default=1)
    parser.add_argument("--max-frequency", type=int, default=100)
    parser.add_argument("--frequency-step", type=int, default=2)
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=0,
        help="Drop frequencies below this radius except the DC component.",
    )
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--clip", action="store_true", help="Clip reconstructions to [0, 1] before scoring.")
    args = parser.parse_args()

    if args.start_frequency < 0:
        raise ValueError("--start-frequency must be non-negative")
    if args.frequency_step <= 0:
        raise ValueError("--frequency-step must be positive")
    if args.snapshots <= 0:
        raise ValueError("--snapshots must be positive")

    run(args)


if __name__ == "__main__":
    main()
