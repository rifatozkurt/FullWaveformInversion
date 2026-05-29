import argparse
import csv
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import io
from src.config import load_config
from src.experiments.base import case_file_stem


def load_gamma(data_dir, case_id):
    stem = case_file_stem(case_id)
    return io.load_hdf(Path(data_dir) / f"material{stem}.h5").astype(np.float32)


def frequency_radius(shape):
    nx, ny = shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.rfftfreq(ny) * ny
    return np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)


def masked_fft_solution(target, mask):
    coefficients = np.fft.rfft2(target)
    reconstruction = np.fft.irfft2(np.where(mask, coefficients, 0), s=target.shape)
    return reconstruction.astype(np.float32)


def save_metrics(path, rows):
    io.ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_summary_figure(path, target, fft_reconstruction, snapshots):
    columns = len(snapshots) + 2
    fig, axes = plt.subplots(2, columns, figsize=(3.0 * columns, 6.0), constrained_layout=True)

    axes[0, 0].imshow(target.T, vmin=0, vmax=1, cmap="coolwarm")
    axes[0, 0].set_title("target")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(fft_reconstruction.T, vmin=0, vmax=1, cmap="coolwarm")
    axes[0, 1].set_title("FFT coefficients")
    residual = fft_reconstruction - target
    vmax = max(abs(float(residual.min())), abs(float(residual.max())), 1e-12)
    axes[1, 1].imshow(residual.T, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axes[1, 1].set_title("FFT residual", fontsize=10)

    for column, snapshot in enumerate(snapshots, start=2):
        reconstruction = snapshot["reconstruction"]
        residual = reconstruction - target
        axes[0, column].imshow(reconstruction.T, vmin=0, vmax=1, cmap="coolwarm")
        axes[0, column].set_title(f"epoch {snapshot['epoch']}", fontsize=10)
        vmax = max(abs(float(residual.min())), abs(float(residual.max())), 1e-12)
        axes[1, column].imshow(residual.T, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        axes[1, column].set_title("learned residual", fontsize=10)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_loss_curve(path, rows, fft_mse):
    epochs = [row["epoch"] for row in rows]
    mse = [row["mse"] for row in rows]

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(epochs, mse, linewidth=2, label="learned coefficients")
    axis.axhline(fft_mse, color="#b64040", linestyle="--", linewidth=1.5, label="FFT coefficients")
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title("Learnable sine coefficient fit")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    run_dir = io.create_run_dir(
        Path(config["paths"].get("runs", "runs")),
        prefix=f"learnable_sine_case{args.case}",
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])

    target_np = load_gamma(data_dir, args.case)
    radius = frequency_radius(target_np.shape)
    mask_np = radius <= args.frequency_cutoff
    if args.min_frequency > 0:
        mask_np &= radius >= args.min_frequency
        mask_np[0, 0] = True
    mode_count = int(np.count_nonzero(mask_np) - 1)

    fft_reconstruction = masked_fft_solution(target_np, mask_np)
    fft_mse = float(np.mean((fft_reconstruction - target_np) ** 2))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    coeff_real = torch.nn.Parameter(torch.zeros(mask_np.shape, dtype=torch.float32, device=device))
    coeff_imag = torch.nn.Parameter(torch.zeros(mask_np.shape, dtype=torch.float32, device=device))
    with torch.no_grad():
        coeff_real[0, 0] = float(target.mean() * target.numel())

    optimizer = torch.optim.Adam([coeff_real, coeff_imag], lr=args.lr)
    rows = []
    snapshots = []
    snapshot_epochs = set(np.linspace(0, args.epochs, args.snapshots, dtype=int).tolist())

    for epoch in range(args.epochs + 1):
        coefficients = torch.complex(coeff_real * mask, coeff_imag * mask)
        reconstruction = torch.fft.irfft2(coefficients, s=target.shape)
        loss = torch.mean((reconstruction - target) ** 2)

        if epoch in snapshot_epochs:
            snapshots.append(
                {
                    "epoch": epoch,
                    "reconstruction": reconstruction.detach().cpu().numpy().astype(np.float32),
                }
            )

        if epoch % args.log_every == 0 or epoch == args.epochs:
            rows.append(
                {
                    "epoch": epoch,
                    "frequency_cutoff": args.frequency_cutoff,
                    "modes": mode_count,
                    "mse": float(loss.detach().cpu()),
                    "fft_mse": fft_mse,
                }
            )
            print(
                "epoch {}/{} | learned MSE {:.6e} | FFT MSE {:.6e}".format(
                    epoch,
                    args.epochs,
                    float(loss.detach().cpu()),
                    fft_mse,
                ),
                flush=True,
            )

        if epoch == args.epochs:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_reconstruction = snapshots[-1]["reconstruction"]
    io.save_hdf(run_dir / "outputs" / f"learnable_sine_case{args.case}_target_gamma.h5", target_np, key="gamma")
    io.save_hdf(
        run_dir / "outputs" / f"learnable_sine_case{args.case}_final_reconstruction.h5",
        final_reconstruction,
        key="gamma",
    )
    io.save_hdf(
        run_dir / "outputs" / f"learnable_sine_case{args.case}_fft_reconstruction.h5",
        fft_reconstruction,
        key="gamma",
    )
    save_metrics(run_dir / "histories" / f"learnable_sine_case{args.case}_metrics.csv", rows)
    save_summary_figure(
        run_dir / "figures" / f"learnable_sine_case{args.case}_reconstructions.png",
        target_np,
        fft_reconstruction,
        snapshots,
    )
    save_loss_curve(
        run_dir / "figures" / f"learnable_sine_case{args.case}_loss_curve.png",
        rows,
        fft_mse,
    )

    print(f"Saved learnable sine experiment to {run_dir}")
    print(f"Cutoff {args.frequency_cutoff} keeps {mode_count} non-DC modes.")


def main():
    parser = argparse.ArgumentParser(
        description="Learn Fourier/sine coefficients for one gamma image with gradient descent."
    )
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--frequency-cutoff", type=int, default=5)
    parser.add_argument("--min-frequency", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.frequency_cutoff < 0:
        raise ValueError("--frequency-cutoff must be non-negative")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.snapshots <= 0:
        raise ValueError("--snapshots must be positive")

    run(args)


if __name__ == "__main__":
    main()
