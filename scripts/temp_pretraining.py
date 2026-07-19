"""Train U-Net and SegFormer scaling curves from one reproducible command.

The two models receive the same nested sample-ID prefixes at every requested
sample count. Runs are sequential so they remain safe on a single 4 GiB GPU.
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
from src.pretraining import pretrain_unet
from src.pretrain_segformer import (
    checkpoint_path_for_metric,
    pretrain_segformer,
)


def parse_csv_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def config_list_or_csv(cli_value, config_value, parser):
    if cli_value is not None:
        return parser(cli_value)
    if isinstance(config_value, str):
        return parser(config_value)
    return list(config_value)


def validate_inputs(data_dir, available_samples, sample_counts, models):
    if not models or any(model not in {"unet", "segformer"} for model in models):
        raise ValueError("--models must contain unet and/or segformer")
    if not sample_counts:
        raise ValueError("at least one sample count is required")
    if sample_counts != sorted(set(sample_counts)) or any(
        value <= 0 for value in sample_counts
    ):
        raise ValueError("sample counts must be unique, positive, and increasing")
    if available_samples <= 0:
        raise ValueError("available samples must be positive")
    if sample_counts[-1] > available_samples:
        raise ValueError(
            f"largest sample count {sample_counts[-1]} exceeds available samples "
            f"{available_samples}"
        )
    if not Path(data_dir).is_dir():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")


def unet_model_path(config, output_dir, samples):
    cfg = config["pretraining"]
    channels = config.get("models", {}).get("unet", {}).get(
        "channels", cfg["NNchannels"]
    )
    return Path(output_dir) / (
        f"model_{cfg['model_type']}_{int(cfg['epochs'])}_{cfg['trainingType']}_"
        f"{samples}_channel_{len(channels)}"
    )


def segformer_model_paths(config, output_dir, samples):
    cfg = config["segformer_pretraining"]
    primary = Path(output_dir) / (
        f"model_{cfg.get('model_type', 'SegFormer')}_{int(cfg['epochs'])}_"
        f"{cfg.get('trainingType', 'segmentation')}_{samples}.pt"
    )
    return [
        primary,
        checkpoint_path_for_metric(primary, "val_loss"),
        checkpoint_path_for_metric(primary, "dice_score"),
    ]


def validate_output_safety(config, output_dir, sample_counts, models, overwrite):
    expected_paths = []
    for samples in sample_counts:
        if "unet" in models:
            expected_paths.append(unet_model_path(config, output_dir, samples))
        if "segformer" in models:
            expected_paths.extend(segformer_model_paths(config, output_dir, samples))
    existing = [path for path in expected_paths if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(f"  {path}" for path in existing[:10])
        raise FileExistsError(
            "Refusing to overwrite existing comparative checkpoints. "
            "Use --overwrite only if replacement is intentional:\n" + preview
        )


def unet_history(run_dir):
    histories = Path(run_dir) / "histories"
    train = np.loadtxt(
        histories / "pretraining_training_loss_history.txt", delimiter=","
    )
    val = np.loadtxt(
        histories / "pretraining_validation_loss_history.txt", delimiter=","
    )
    return np.atleast_1d(train), np.atleast_1d(val)


def segformer_history(run_dir):
    with (
        Path(run_dir) / "histories" / "segformer_pretraining_metrics.csv"
    ).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = np.array([float(row["train_loss"]) for row in rows])
    val = np.array([float(row["val_loss"]) for row in rows])
    dice = np.array([float(row["dice_score"]) for row in rows])
    iou = np.array([float(row["iou"]) for row in rows])
    return train, val, dice, iou


def empty_summary_row(model, samples, model_path, run_dir, runtime):
    return {
        "model": model,
        "samples": samples,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "runtime_seconds": runtime,
        "epochs_completed": "",
        "final_train_loss": "",
        "final_val_loss": "",
        "best_val_loss": "",
        "best_val_epoch": "",
        "final_dice_score": "",
        "best_dice_score": "",
        "best_dice_epoch": "",
        "final_iou": "",
        "best_iou": "",
        "primary_metric": "",
        "primary_epoch": "",
    }


def write_summary(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(path, histories, log_scale=False):
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for item in histories:
        epochs = np.arange(1, len(item["val"]) + 1)
        label = f"{item['model']} {item['samples']}"
        axis = axes[0] if item["model"] == "unet" else axes[1]
        axis.plot(epochs, item["val"], label=label, linewidth=1.8)

    axes[0].set_title("U-Net validation gamma loss")
    axes[0].set_ylabel("0.5 × MSE")
    axes[1].set_title("SegFormer validation segmentation loss")
    axes[1].set_ylabel("weighted BCE + Dice")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)
        if log_scale:
            axis.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_scaling_metrics(path, rows):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    unet_rows = sorted(
        (row for row in rows if row["model"] == "unet"),
        key=lambda row: int(row["samples"]),
    )
    if unet_rows:
        axes[0].plot(
            [int(row["samples"]) for row in unet_rows],
            [float(row["best_val_loss"]) for row in unet_rows],
            marker="o",
            linewidth=2,
            label="Best validation MSE",
        )
    axes[0].set_title("U-Net scaling")
    axes[0].set_ylabel("best 0.5 × MSE")

    segformer_rows = sorted(
        (row for row in rows if row["model"] == "segformer"),
        key=lambda row: int(row["samples"]),
    )
    if segformer_rows:
        sample_values = [int(row["samples"]) for row in segformer_rows]
        axes[1].plot(
            sample_values,
            [float(row["best_dice_score"]) for row in segformer_rows],
            marker="o",
            linewidth=2,
            label="Best Dice",
        )
        axes[1].plot(
            sample_values,
            [float(row["best_iou"]) for row in segformer_rows],
            marker="o",
            linewidth=2,
            label="Best IoU",
        )
    axes[1].set_title("SegFormer scaling")
    axes[1].set_ylabel("validation score")

    for axis in axes:
        axis.set_xlabel("pretraining samples")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def cleanup_after_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_subrun_config(config, run_dir):
    with (Path(run_dir) / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def refresh_comparative_outputs(root_run_dir, summary_rows, histories):
    write_summary(
        Path(root_run_dir) / "histories" / "comparative_pretraining_summary.csv",
        summary_rows,
    )
    plot_histories(
        Path(root_run_dir) / "figures" / "comparative_validation_histories.png",
        histories,
    )
    plot_histories(
        Path(root_run_dir)
        / "figures"
        / "comparative_validation_histories_log.png",
        histories,
        log_scale=True,
    )
    plot_scaling_metrics(
        Path(root_run_dir) / "figures" / "comparative_scaling_metrics.png",
        summary_rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially pretrain U-Net and SegFormer on matched, nested "
            "sample-count prefixes."
        )
    )
    parser.add_argument("--config", default="configs/extended.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-counts", default=None)
    parser.add_argument("--available-samples", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of checkpoints with the same model/sample names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned runs without training.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    comparative_cfg = base_config.get("comparative_pretraining", {})
    sample_counts = config_list_or_csv(
        args.sample_counts,
        comparative_cfg.get("sample_counts", [1000, 5000, 10000, 15000]),
        parse_csv_ints,
    )
    models = config_list_or_csv(
        args.models,
        comparative_cfg.get("models", ["unet", "segformer"]),
        parse_csv_strings,
    )
    if not sample_counts:
        raise ValueError("at least one sample count is required")
    available_samples = int(
        args.available_samples
        if args.available_samples is not None
        else comparative_cfg.get("available_samples", max(sample_counts))
    )
    data_dir = Path(args.data_dir or base_config["paths"]["train_data"])
    output_dir = Path(
        args.output_dir
        or comparative_cfg.get(
            "output_dir", Path(base_config["paths"]["pretrained_models"]) / "comparative"
        )
    )
    run_root = Path(
        args.run_dir
        or comparative_cfg.get(
            "run_dir", Path(base_config["paths"]["runs"]) / "comparative_pretraining"
        )
    )

    validate_inputs(data_dir, available_samples, sample_counts, models)
    validate_output_safety(
        base_config,
        output_dir,
        sample_counts,
        models,
        overwrite=args.overwrite,
    )

    print("Comparative pretraining plan:")
    print(f"  config: {args.config}")
    print(f"  data: {data_dir}")
    print(f"  output: {output_dir}")
    print(f"  models: {models}")
    print(f"  sample counts: {sample_counts}")
    print(f"  available contiguous cases: 0-{available_samples - 1}")
    print("  execution: sequential (one model at a time)")
    if args.dry_run:
        print("Dry run complete; no training or files were created.")
        return

    output_dir = io.ensure_dir(output_dir)
    root_run_dir = io.create_run_dir(
        io.ensure_dir(run_root), prefix="comparative_pretraining"
    )
    io.ensure_dirs(
        [
            root_run_dir / "figures",
            root_run_dir / "histories",
            root_run_dir / "outputs",
        ]
    )
    shutil.copy2(args.config, root_run_dir / "config.yaml")
    with (root_run_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    master_seed = int(
        args.seed
        if args.seed is not None
        else comparative_cfg.get("seed", base_config["pretraining"]["seed"])
    )
    master_sample_ids = list(range(available_samples))
    random.Random(master_seed).shuffle(master_sample_ids)
    np.savetxt(
        root_run_dir / "outputs" / "master_sample_ids.txt",
        master_sample_ids,
        fmt="%d",
    )

    summary_rows = []
    histories = []
    start_all = time.perf_counter()
    for samples in sample_counts:
        sample_ids = master_sample_ids[:samples]

        if "unet" in models:
            config = deepcopy(base_config)
            config["pretraining"]["numberOfSamples"] = samples
            config["pretraining"]["availableSamples"] = available_samples
            config["pretraining"]["sample_ids"] = sample_ids
            config["pretraining"]["seed"] = master_seed
            run_dir = io.ensure_dir(root_run_dir / f"unet_{samples}")
            io.ensure_dirs(
                [run_dir / "figures", run_dir / "histories", run_dir / "outputs"]
            )
            save_subrun_config(config, run_dir)
            print(f"\n=== Pretraining U-Net with {samples} samples ===", flush=True)
            start = time.perf_counter()
            try:
                model_path = pretrain_unet(
                    config,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    run_dir=run_dir,
                )
                runtime = time.perf_counter() - start
                train, val = unet_history(run_dir)
                histories.append(
                    {"model": "unet", "samples": samples, "train": train, "val": val}
                )
                row = empty_summary_row(
                    "unet", samples, model_path, run_dir, runtime
                )
                row.update(
                    {
                        "epochs_completed": len(train),
                        "final_train_loss": float(train[-1]),
                        "final_val_loss": float(val[-1]),
                        "best_val_loss": float(np.min(val)),
                        "best_val_epoch": int(np.argmin(val) + 1),
                        # U-Net currently saves the final epoch, not best validation.
                        "primary_metric": "final_epoch",
                        "primary_epoch": len(train),
                    }
                )
                summary_rows.append(row)
                refresh_comparative_outputs(root_run_dir, summary_rows, histories)
            finally:
                cleanup_after_model()

        if "segformer" in models:
            config = deepcopy(base_config)
            config["segformer_pretraining"]["numberOfSamples"] = samples
            config["segformer_pretraining"]["availableSamples"] = available_samples
            config["segformer_pretraining"]["sample_ids"] = sample_ids
            config["segformer_pretraining"]["seed"] = master_seed
            run_dir = io.ensure_dir(root_run_dir / f"segformer_{samples}")
            io.ensure_dirs(
                [run_dir / "figures", run_dir / "histories", run_dir / "outputs"]
            )
            save_subrun_config(config, run_dir)
            print(
                f"\n=== Pretraining SegFormer with {samples} samples ===",
                flush=True,
            )
            start = time.perf_counter()
            try:
                model_path = pretrain_segformer(
                    config,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    run_dir=run_dir,
                )
                runtime = time.perf_counter() - start
                train, val, dice, iou = segformer_history(run_dir)
                histories.append(
                    {
                        "model": "segformer",
                        "samples": samples,
                        "train": train,
                        "val": val,
                    }
                )
                best_val_index = int(np.argmin(val))
                best_dice_index = int(np.argmax(dice))
                primary_metric = config["segformer_pretraining"].get(
                    "checkpoint_selection", "val_loss"
                )
                primary_epoch = (
                    best_dice_index + 1
                    if primary_metric == "dice_score"
                    else best_val_index + 1
                )
                row = empty_summary_row(
                    "segformer", samples, model_path, run_dir, runtime
                )
                row.update(
                    {
                        "epochs_completed": len(train),
                        "final_train_loss": float(train[-1]),
                        "final_val_loss": float(val[-1]),
                        "best_val_loss": float(val[best_val_index]),
                        "best_val_epoch": best_val_index + 1,
                        "final_dice_score": float(dice[-1]),
                        "best_dice_score": float(dice[best_dice_index]),
                        "best_dice_epoch": best_dice_index + 1,
                        "final_iou": float(iou[-1]),
                        "best_iou": float(np.max(iou)),
                        "primary_metric": primary_metric,
                        "primary_epoch": primary_epoch,
                    }
                )
                summary_rows.append(row)
                refresh_comparative_outputs(root_run_dir, summary_rows, histories)
            finally:
                cleanup_after_model()

    elapsed_all = time.perf_counter() - start_all
    (root_run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: comparative_pretraining",
                f"config: {args.config}",
                f"data_dir: {data_dir}",
                f"output_dir: {output_dir}",
                f"sample_counts: {','.join(map(str, sample_counts))}",
                f"available_samples: {available_samples}",
                f"models: {','.join(models)}",
                f"master_seed: {master_seed}",
                f"runtime_seconds: {elapsed_all:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved comparative runs to {root_run_dir}", flush=True)
    print(f"Saved comparative models to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
