import argparse
import csv
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np

from src import io
from src.config import load_config
from src.pretraining import pretrain_unet
from src.pretrain_segformer import pretrain_segformer


DEFAULT_SAMPLE_COUNTS = "100,250,500,1000"


def parse_csv_ints(text):
    return [int(item) for item in text.split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def unet_history(run_dir):
    histories = Path(run_dir) / "histories"
    train = np.loadtxt(histories / "pretraining_training_loss_history.txt", delimiter=",")
    val = np.loadtxt(histories / "pretraining_validation_loss_history.txt", delimiter=",")
    return np.atleast_1d(train), np.atleast_1d(val)


def segformer_history(run_dir):
    rows = []
    with (Path(run_dir) / "histories" / "segformer_pretraining_metrics.csv").open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    train = np.array([float(row["train_loss"]) for row in rows])
    val = np.array([float(row["val_loss"]) for row in rows])
    dice = np.array([float(row["dice_score"]) for row in rows])
    iou = np.array([float(row["iou"]) for row in rows])
    return train, val, dice, iou


def write_summary(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(path, histories):
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for item in histories:
        epochs = np.arange(1, len(item["val"]) + 1)
        label = f"{item['model']} {item['samples']}"
        ax = axes[0] if item["model"] == "unet" else axes[1]
        ax.plot(epochs, item["val"], label=label, linewidth=1.8)

    axes[0].set_title("U-Net validation gamma loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("0.5 * MSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("SegFormer validation segmentation loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("BCE + Dice")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_final_metrics(path, rows):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    models = sorted({row["model"] for row in rows})

    for model in models:
        items = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["samples"]),
        )
        samples = [int(row["samples"]) for row in items]
        final_val = [float(row["final_val_loss"]) for row in items]
        axes[0].plot(samples, final_val, marker="o", linewidth=2, label=model)

    axes[0].set_title("Final native validation loss")
    axes[0].set_xlabel("pretraining samples")
    axes[0].set_ylabel("native loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    segformer_items = sorted(
        [row for row in rows if row["model"] == "segformer"],
        key=lambda row: int(row["samples"]),
    )
    if segformer_items:
        samples = [int(row["samples"]) for row in segformer_items]
        dice = [float(row["final_dice_score"]) for row in segformer_items]
        iou = [float(row["final_iou"]) for row in segformer_items]
        axes[1].plot(samples, dice, marker="o", linewidth=2, label="Dice")
        axes[1].plot(samples, iou, marker="o", linewidth=2, label="IoU")
    axes[1].set_title("SegFormer validation mask metrics")
    axes[1].set_xlabel("pretraining samples")
    axes[1].set_ylabel("score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain U-Net and SegFormer on matched sample counts for comparison."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="models/comparative")
    parser.add_argument("--run-dir", default="runs/comparative_pretraining")
    parser.add_argument("--sample-counts", default=DEFAULT_SAMPLE_COUNTS)
    parser.add_argument("--available-samples", type=int, default=1000)
    parser.add_argument("--models", default="unet,segformer")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    sample_counts = parse_csv_ints(args.sample_counts)
    models = parse_csv_strings(args.models)
    data_dir = Path(args.data_dir)
    output_dir = io.ensure_dir(args.output_dir)
    root_run_dir = io.create_run_dir(io.ensure_dir(args.run_dir), prefix="comparative_pretraining")
    io.ensure_dirs([root_run_dir / "figures", root_run_dir / "histories", root_run_dir / "outputs"])
    shutil.copy2(args.config, root_run_dir / "config.yaml")

    summary_rows = []
    histories = []
    start_all = time.perf_counter()
    master_seed = int(args.seed if args.seed is not None else base_config["pretraining"]["seed"])
    rng = random.Random(master_seed)
    master_sample_ids = list(range(int(args.available_samples)))
    rng.shuffle(master_sample_ids)
    np.savetxt(root_run_dir / "outputs" / "master_sample_ids.txt", master_sample_ids, fmt="%d")

    for samples in sample_counts:
        sample_ids = master_sample_ids[:samples]
        if "unet" in models:
            config = deepcopy(base_config)
            config["pretraining"]["numberOfSamples"] = int(samples)
            config["pretraining"]["availableSamples"] = int(args.available_samples)
            config["pretraining"]["sample_ids"] = sample_ids
            if args.seed is not None:
                config["pretraining"]["seed"] = int(args.seed)

            run_dir = io.ensure_dir(root_run_dir / f"unet_{samples}")
            io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
            print(f"\n=== Pretraining U-Net with {samples} sample(s) ===", flush=True)
            start = time.perf_counter()
            model_path = pretrain_unet(config, data_dir=data_dir, output_dir=output_dir, run_dir=run_dir)
            runtime = time.perf_counter() - start
            train, val = unet_history(run_dir)
            histories.append({"model": "unet", "samples": samples, "train": train, "val": val})
            summary_rows.append(
                {
                    "model": "unet",
                    "samples": samples,
                    "model_path": str(model_path),
                    "run_dir": str(run_dir),
                    "runtime_seconds": runtime,
                    "final_train_loss": float(train[-1]),
                    "final_val_loss": float(val[-1]),
                    "best_val_loss": float(np.min(val)),
                    "best_epoch": int(np.argmin(val) + 1),
                    "final_dice_score": "",
                    "final_iou": "",
                }
            )

        if "segformer" in models:
            config = deepcopy(base_config)
            config["segformer_pretraining"]["numberOfSamples"] = int(samples)
            config["segformer_pretraining"]["availableSamples"] = int(args.available_samples)
            config["segformer_pretraining"]["sample_ids"] = sample_ids
            if args.seed is not None:
                config["segformer_pretraining"]["seed"] = int(args.seed)

            run_dir = io.ensure_dir(root_run_dir / f"segformer_{samples}")
            io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
            print(f"\n=== Pretraining SegFormer with {samples} sample(s) ===", flush=True)
            start = time.perf_counter()
            model_path = pretrain_segformer(config, data_dir=data_dir, output_dir=output_dir, run_dir=run_dir)
            runtime = time.perf_counter() - start
            train, val, dice, iou = segformer_history(run_dir)
            histories.append({"model": "segformer", "samples": samples, "train": train, "val": val})
            summary_rows.append(
                {
                    "model": "segformer",
                    "samples": samples,
                    "model_path": str(model_path),
                    "run_dir": str(run_dir),
                    "runtime_seconds": runtime,
                    "final_train_loss": float(train[-1]),
                    "final_val_loss": float(val[-1]),
                    "best_val_loss": float(np.min(val)),
                    "best_epoch": int(np.argmin(val) + 1),
                    "final_dice_score": float(dice[-1]),
                    "final_iou": float(iou[-1]),
                }
            )

        write_summary(root_run_dir / "histories" / "comparative_pretraining_summary.csv", summary_rows)
        plot_histories(root_run_dir / "figures" / "comparative_validation_histories.png", histories)
        plot_final_metrics(root_run_dir / "figures" / "comparative_final_metrics.png", summary_rows)

    elapsed_all = time.perf_counter() - start_all
    (root_run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: comparative_pretraining",
                f"config: {args.config}",
                f"data_dir: {data_dir}",
                f"output_dir: {output_dir}",
                f"sample_counts: {args.sample_counts}",
                f"available_samples: {args.available_samples}",
                f"models: {args.models}",
                f"runtime_seconds: {elapsed_all:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved comparative pretraining outputs to {root_run_dir}", flush=True)
    print(f"Saved comparative models to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
