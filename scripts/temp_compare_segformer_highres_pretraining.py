"""Quick matched pretraining comparison of SegFormer and SegFormer HighRes.

This is a disposable experiment runner. It trains both variants sequentially
with the same sample IDs, train/validation split, initialization seed, and
DataLoader seed, then plots their common validation metrics on shared axes.
"""

import argparse
import csv
import gc
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src import io
from src.config import load_config
from src.pretrain_segformer import pretrain_segformer, segformer_model_type


VARIANTS = (
    ("segformer", "SegFormer"),
    ("segformer_highres", "SegFormer HighRes"),
)


def read_metrics(run_dir):
    path = Path(run_dir) / "histories" / "segformer_pretraining_metrics.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metric rows were written to {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "epoch",
            "train_loss",
            "val_loss",
            "gamma_mse",
            "dice_score",
            "iou",
        )
    }


def load_ids(path):
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))


def verify_matched_splits(run_dirs):
    standard_outputs = Path(run_dirs["segformer"]) / "outputs"
    highres_outputs = Path(run_dirs["segformer_highres"]) / "outputs"
    pairs = (
        (
            standard_outputs / "segformer_pretraining_training_sample_ids.txt",
            highres_outputs
            / "segformer_pretraining_training_sample_ids.txt",
            "training",
        ),
        (
            standard_outputs / "segformer_pretraining_validation_sample_ids.txt",
            highres_outputs
            / "segformer_pretraining_validation_sample_ids.txt",
            "validation",
        ),
    )
    for standard_path, highres_path, split_name in pairs:
        if not np.array_equal(load_ids(standard_path), load_ids(highres_path)):
            raise RuntimeError(
                f"The {split_name} sample IDs differ between model variants"
            )


def write_summary(path, histories, model_paths, runtimes):
    fieldnames = [
        "model",
        "model_path",
        "runtime_seconds",
        "epochs_completed",
        "final_gamma_mse",
        "best_gamma_mse",
        "best_gamma_mse_epoch",
        "final_dice_score",
        "best_dice_score",
        "best_dice_epoch",
        "final_iou",
        "best_iou",
        "best_iou_epoch",
        "final_native_val_loss",
        "best_native_val_loss",
        "best_native_val_loss_epoch",
    ]
    rows = []
    for variant, label in VARIANTS:
        metrics = histories[variant]
        gamma_index = int(np.argmin(metrics["gamma_mse"]))
        dice_index = int(np.argmax(metrics["dice_score"]))
        iou_index = int(np.argmax(metrics["iou"]))
        loss_index = int(np.argmin(metrics["val_loss"]))
        rows.append(
            {
                "model": label,
                "model_path": str(model_paths[variant]),
                "runtime_seconds": runtimes[variant],
                "epochs_completed": len(metrics["epoch"]),
                "final_gamma_mse": metrics["gamma_mse"][-1],
                "best_gamma_mse": metrics["gamma_mse"][gamma_index],
                "best_gamma_mse_epoch": gamma_index + 1,
                "final_dice_score": metrics["dice_score"][-1],
                "best_dice_score": metrics["dice_score"][dice_index],
                "best_dice_epoch": dice_index + 1,
                "final_iou": metrics["iou"][-1],
                "best_iou": metrics["iou"][iou_index],
                "best_iou_epoch": iou_index + 1,
                "final_native_val_loss": metrics["val_loss"][-1],
                "best_native_val_loss": metrics["val_loss"][loss_index],
                "best_native_val_loss_epoch": loss_index + 1,
            }
        )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_comparison(path, histories, samples, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("gamma_mse", "Validation Gamma MSE", "Validation Gamma MSE"),
        ("dice_score", "Validation Dice Score", "Validation Dice Score"),
        ("iou", "Validation IoU", "Validation IoU"),
        (
            "val_loss",
            "Native Validation Loss",
            "Native Validation Loss (Weighted BCE + Dice)",
        ),
    )
    for variant, label in VARIANTS:
        metrics = histories[variant]
        for axis, (key, title, ylabel) in zip(axes.flat, panels):
            axis.plot(
                metrics["epoch"],
                metrics[key],
                linewidth=2,
                label=label,
            )
            axis.set_title(title)
            axis.set_xlabel("Epochs")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
    for axis in axes.flat:
        axis.legend()
    fig.suptitle(
        f"SegFormer Output-Resolution Comparison "
        f"({samples} Samples, Up to {epochs} Epochs)"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def print_summary(rows):
    print("\n=== Quick comparison result ===")
    for row in rows:
        print(
            "{}: best Gamma MSE={:.6E} (epoch {}), "
            "best Dice={:.4f} (epoch {}), best IoU={:.4f} (epoch {}), "
            "runtime={:.1f}s".format(
                row["model"],
                float(row["best_gamma_mse"]),
                row["best_gamma_mse_epoch"],
                float(row["best_dice_score"]),
                row["best_dice_epoch"],
                float(row["best_iou"]),
                row["best_iou_epoch"],
                float(row["runtime_seconds"]),
            )
        )

    standard = next(row for row in rows if row["model"] == "SegFormer")
    highres = next(row for row in rows if row["model"] == "SegFormer HighRes")
    gamma_change = (
        (float(highres["best_gamma_mse"]) - float(standard["best_gamma_mse"]))
        / max(float(standard["best_gamma_mse"]), 1e-12)
        * 100.0
    )
    dice_change = float(highres["best_dice_score"]) - float(
        standard["best_dice_score"]
    )
    print(
        "HighRes relative change: Gamma MSE={:+.2f}% "
        "(negative is better), Dice={:+.4f} (positive is better).".format(
            gamma_change,
            dice_change,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quickly compare standard and full-resolution SegFormer "
            "pretraining on exactly matched data."
        )
    )
    parser.add_argument("--config", default="configs/extended.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--available-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument(
        "--run-dir",
        default="runs/improve_transformer/quick_highres_250",
        help="Parent directory; each invocation creates a timestamped subrun.",
    )
    args = parser.parse_args()

    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.available_samples < args.samples:
        raise ValueError("--available-samples must be at least --samples")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    for sample_id in range(args.available_samples):
        for prefix in ("material", "gradient"):
            path = data_dir / f"{prefix}{sample_id}.h5"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Expected contiguous dataset file is missing: {path}"
                )

    base_config = load_config(args.config)
    root_run = io.create_run_dir(
        io.ensure_dir(args.run_dir),
        prefix=f"segformer_vs_highres_{args.samples}",
    )
    models_dir = io.ensure_dir(root_run / "models")
    io.ensure_dirs(
        [root_run / "figures", root_run / "histories", root_run / "outputs"]
    )
    shutil.copy2(args.config, root_run / "config.yaml")

    master_ids = list(range(args.available_samples))
    random.Random(args.seed).shuffle(master_ids)
    sample_ids = master_ids[: args.samples]
    np.savetxt(root_run / "outputs" / "sample_ids.txt", sample_ids, fmt="%d")

    histories = {}
    model_paths = {}
    runtimes = {}
    run_dirs = {}

    print(
        "Matched comparison: samples={}, epochs={}, batch_size={}, seed={}".format(
            args.samples,
            args.epochs,
            args.batch_size,
            args.seed,
        )
    )
    print(f"Results: {root_run}")

    for variant, label in VARIANTS:
        config = deepcopy(base_config)
        training_cfg = config["segformer_pretraining"]
        training_cfg.update(
            {
                "numberOfSamples": args.samples,
                "availableSamples": args.available_samples,
                "sample_ids": sample_ids,
                "seed": args.seed,
                "dataloader_seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "model_variant": variant,
                "resolved_model_type": segformer_model_type(
                    training_cfg,
                    variant,
                ),
                # Let every requested quick-run epoch complete.
                "minimum_epochs": args.epochs,
                "early_stopping_patience": None,
            }
        )
        run_dir = io.ensure_dir(root_run / variant)
        io.ensure_dirs(
            [run_dir / "figures", run_dir / "histories", run_dir / "outputs"]
        )
        with (run_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        print(f"\n=== Training {label} ===", flush=True)
        start = time.perf_counter()
        model_paths[variant] = pretrain_segformer(
            config,
            data_dir=data_dir,
            output_dir=models_dir,
            run_dir=run_dir,
            model_variant=variant,
        )
        runtimes[variant] = time.perf_counter() - start
        histories[variant] = read_metrics(run_dir)
        run_dirs[variant] = run_dir
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verify_matched_splits(run_dirs)
    summary_path = root_run / "histories" / "quick_comparison_summary.csv"
    rows = write_summary(summary_path, histories, model_paths, runtimes)
    plot_path = root_run / "figures" / "segformer_vs_highres.png"
    plot_comparison(
        plot_path,
        histories,
        samples=args.samples,
        epochs=args.epochs,
    )
    print_summary(rows)
    print(f"Verified identical training and validation sample IDs.")
    print(f"Summary: {summary_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
