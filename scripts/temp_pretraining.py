"""Train U-Net and SegFormer variants from one reproducible command.

Every selected model receives the same nested sample-ID prefixes at each
requested sample count. Runs are sequential.
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
from src.reporting import plot_pretraining_curves
from src.pretrain_segformer import (
    checkpoint_path_for_metric,
    parse_early_stopping_patience,
    pretrain_segformer,
    segformer_model_type,
)


SEGFORMER_MODELS = ("segformer", "segformer_imagenet", "segformer_highres")
SUPPORTED_MODELS = {"unet", *SEGFORMER_MODELS}
MODEL_LABELS = {
    "unet": "U-Net",
    "segformer": "SegFormer",
    "segformer_imagenet": "SegFormer (ImageNet init)",
    "segformer_highres": "SegFormer HighRes",
}


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
    if not models or any(model not in SUPPORTED_MODELS for model in models):
        raise ValueError(
            "--models must contain unet, segformer, and/or segformer_highres"
        )
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


def segformer_model_paths(config, output_dir, samples, model_variant="segformer"):
    cfg = config["segformer_pretraining"]
    model_type = segformer_model_type(cfg, model_variant)
    primary = Path(output_dir) / (
        f"model_{model_type}_{int(cfg['epochs'])}_"
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
        for model_variant in SEGFORMER_MODELS:
            if model_variant in models:
                expected_paths.extend(
                    segformer_model_paths(
                        config,
                        output_dir,
                        samples,
                        model_variant=model_variant,
                    )
                )
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
    gamma_mse = np.loadtxt(
        histories / "pretraining_validation_gamma_mse_history.txt", delimiter=","
    )
    dice = np.loadtxt(
        histories / "pretraining_validation_dice_history.txt", delimiter=","
    )
    iou = np.loadtxt(
        histories / "pretraining_validation_iou_history.txt", delimiter=","
    )
    return tuple(
        np.atleast_1d(values)
        for values in (train, val, gamma_mse, dice, iou)
    )


def segformer_history(run_dir):
    with (
        Path(run_dir) / "histories" / "segformer_pretraining_metrics.csv"
    ).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = np.array([float(row["train_loss"]) for row in rows])
    val = np.array([float(row["val_loss"]) for row in rows])
    gamma_mse = np.array([float(row["gamma_mse"]) for row in rows])
    dice = np.array([float(row["dice_score"]) for row in rows])
    iou = np.array([float(row["iou"]) for row in rows])
    return train, val, gamma_mse, dice, iou


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
        "final_gamma_mse": "",
        "best_gamma_mse": "",
        "best_gamma_mse_epoch": "",
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


def plot_native_histories(path, histories, log_scale=False):
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for item in histories:
        epochs = np.arange(1, len(item["val"]) + 1)
        label = f"{MODEL_LABELS[item['model']]} {item['samples']}"
        axis = axes[0] if item["model"] == "unet" else axes[1]
        axis.plot(epochs, item["val"], label=label, linewidth=1.8)

    scale_suffix = " (Log Scale)" if log_scale else ""
    fig.suptitle(f"Model-Native Validation Loss Histories{scale_suffix}")
    axes[0].set_title(f"U-Net Gamma Regression Loss{scale_suffix}")
    axes[0].set_ylabel("Native Loss (0.5 × Padded Gamma MSE)")
    axes[1].set_title(f"SegFormer Segmentation Loss{scale_suffix}")
    axes[1].set_ylabel("Native Loss (Weighted BCE + Dice)")
    for axis in axes:
        axis.set_xlabel("Epochs")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)
        if log_scale:
            axis.set_yscale("log")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_common_histories(
    path,
    histories,
    metric,
    ylabel,
    title,
    log_scale=False,
):
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for item in histories:
        values = np.asarray(item[metric])
        epochs = np.arange(1, len(values) + 1)
        ax.plot(
            epochs,
            values,
            label=f"{MODEL_LABELS[item['model']]} {item['samples']}",
            linewidth=1.8,
        )
    scale_suffix = " (Log Scale)" if log_scale else ""
    ax.set_title(f"{title}{scale_suffix}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_scaling_metrics(path, rows):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    fig.suptitle("Comparative Pretraining Scaling on Common Validation Metrics")

    for model_name, model_label in MODEL_LABELS.items():
        model_rows = sorted(
            (row for row in rows if row["model"] == model_name),
            key=lambda row: int(row["samples"]),
        )
        if not model_rows:
            continue
        sample_values = [int(row["samples"]) for row in model_rows]
        axes[0].plot(
            sample_values,
            [float(row["best_gamma_mse"]) for row in model_rows],
            marker="o",
            linewidth=2,
            label=model_label,
        )
        axes[1].plot(
            sample_values,
            [float(row["best_dice_score"]) for row in model_rows],
            marker="o",
            linewidth=2,
            label=model_label,
        )
    axes[0].set_title("Best Validation Gamma MSE")
    axes[0].set_ylabel("Best Validation Gamma MSE")
    axes[1].set_title("Best Validation Dice Score")
    axes[1].set_ylabel("Best Validation Dice Score")

    for axis in axes:
        axis.set_xlabel("Training Samples")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    plot_common_histories(
        Path(root_run_dir) / "figures" / "comparative_validation_histories.png",
        histories,
        metric="gamma_mse",
        ylabel="Validation Gamma MSE",
        title="Common Validation Gamma MSE Histories",
    )
    plot_common_histories(
        Path(root_run_dir)
        / "figures"
        / "comparative_validation_histories_log.png",
        histories,
        metric="gamma_mse",
        ylabel="Validation Gamma MSE",
        title="Common Validation Gamma MSE Histories",
        log_scale=True,
    )
    plot_common_histories(
        Path(root_run_dir)
        / "figures"
        / "comparative_validation_dice_histories.png",
        histories,
        metric="dice",
        ylabel="Validation Dice Score",
        title="Common Validation Dice Score Histories",
    )
    # One panel PER FAMILY with a colour gradient over sample sizes.
    # plot_common_histories puts every (model, size) pair on a single axis --
    # 3 families x 6 sizes is 18 lines and an 18-entry legend, which is not
    # readable. This shows how each family's curve shifts as data grows.
    for metric, ylabel, log in (("gamma_mse", "validation gamma MSE", True),
                                ("dice", "validation Dice", False)):
        plot_pretraining_curves(
            Path(root_run_dir) / "figures" / f"pretraining_{metric}_by_family.png",
            [{"family": h["model"], "samples": h["samples"], "values": h[metric]}
             for h in histories if metric in h],
            ylabel=ylabel,
            title=f"Pretraining {ylabel} by model family and training-set size",
            logy=log,
        )

    plot_native_histories(
        Path(root_run_dir)
        / "figures"
        / "comparative_native_validation_histories.png",
        histories,
    )
    plot_native_histories(
        Path(root_run_dir)
        / "figures"
        / "comparative_native_validation_histories_log.png",
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
            "Sequentially pretrain U-Net, SegFormer, and SegFormer HighRes "
            "on matched, nested sample-count prefixes."
        )
    )
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-counts", default=None)
    parser.add_argument("--available-samples", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--unet-epochs",
        type=int,
        default=None,
        help="Override the U-Net epoch count.",
    )
    parser.add_argument(
        "--segformer-epochs",
        type=int,
        default=None,
        help="Override the SegFormer epoch count for all selected variants.",
    )
    parser.add_argument(
        "--segformer-batch-size",
        type=int,
        default=None,
        help="Override the batch size for all selected SegFormer variants.",
    )
    parser.add_argument(
        "--segformer-minimum-epochs",
        type=int,
        default=None,
        help="Override how many epochs must finish before early stopping.",
    )
    parser.add_argument(
        "--segformer-early-stopping-patience",
        default=None,
        help=(
            "Override patience with a positive epoch count, or use 'none' "
            "to disable early stopping."
        ),
    )
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
    if args.unet_epochs is not None:
        if args.unet_epochs < 1:
            raise ValueError("--unet-epochs must be positive")
        base_config["pretraining"]["epochs"] = args.unet_epochs
    if args.segformer_epochs is not None:
        if args.segformer_epochs < 1:
            raise ValueError("--segformer-epochs must be positive")
        base_config["segformer_pretraining"]["epochs"] = args.segformer_epochs
    if args.segformer_batch_size is not None:
        if args.segformer_batch_size < 1:
            raise ValueError("--segformer-batch-size must be positive")
        base_config["segformer_pretraining"][
            "batch_size"
        ] = args.segformer_batch_size
    if args.segformer_minimum_epochs is not None:
        if args.segformer_minimum_epochs < 1:
            raise ValueError("--segformer-minimum-epochs must be positive")
        base_config["segformer_pretraining"][
            "minimum_epochs"
        ] = args.segformer_minimum_epochs
    if args.segformer_early_stopping_patience is not None:
        base_config["segformer_pretraining"][
            "early_stopping_patience"
        ] = parse_early_stopping_patience(
            args.segformer_early_stopping_patience
        )
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
    if "unet" in models:
        print(f"  U-Net epochs: {base_config['pretraining']['epochs']}")
    if any(model in models for model in SEGFORMER_MODELS):
        print(
            "  SegFormer epochs: {}; batch size: {}; minimum epochs: {}; "
            "early-stopping patience: {}".format(
                base_config["segformer_pretraining"]["epochs"],
                base_config["segformer_pretraining"]["batch_size"],
                base_config["segformer_pretraining"].get("minimum_epochs", 1),
                base_config["segformer_pretraining"].get(
                    "early_stopping_patience"
                ),
            )
        )
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
                train, val, gamma_mse, dice, iou = unet_history(run_dir)
                histories.append(
                    {
                        "model": "unet",
                        "samples": samples,
                        "train": train,
                        "val": val,
                        "gamma_mse": gamma_mse,
                        "dice": dice,
                        "iou": iou,
                    }
                )
                best_gamma_mse_index = int(np.argmin(gamma_mse))
                best_dice_index = int(np.argmax(dice))
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
                        "final_gamma_mse": float(gamma_mse[-1]),
                        "best_gamma_mse": float(
                            gamma_mse[best_gamma_mse_index]
                        ),
                        "best_gamma_mse_epoch": best_gamma_mse_index + 1,
                        "final_dice_score": float(dice[-1]),
                        "best_dice_score": float(dice[best_dice_index]),
                        "best_dice_epoch": best_dice_index + 1,
                        "final_iou": float(iou[-1]),
                        "best_iou": float(np.max(iou)),
                        # U-Net currently saves the final epoch, not best validation.
                        "primary_metric": "final_epoch",
                        "primary_epoch": len(train),
                    }
                )
                summary_rows.append(row)
                refresh_comparative_outputs(root_run_dir, summary_rows, histories)
            finally:
                cleanup_after_model()

        for model_variant in SEGFORMER_MODELS:
            if model_variant not in models:
                continue
            config = deepcopy(base_config)
            segformer_cfg = config["segformer_pretraining"]
            segformer_cfg["numberOfSamples"] = samples
            segformer_cfg["availableSamples"] = available_samples
            segformer_cfg["sample_ids"] = sample_ids
            segformer_cfg["seed"] = master_seed
            segformer_cfg["dataloader_seed"] = master_seed
            segformer_cfg["model_variant"] = model_variant
            segformer_cfg["resolved_model_type"] = segformer_model_type(
                segformer_cfg,
                model_variant,
            )
            run_dir = io.ensure_dir(
                root_run_dir / f"{model_variant}_{samples}"
            )
            io.ensure_dirs(
                [run_dir / "figures", run_dir / "histories", run_dir / "outputs"]
            )
            save_subrun_config(config, run_dir)
            display_name = MODEL_LABELS.get(model_variant, model_variant)
            print(
                f"\n=== Pretraining {display_name} with {samples} samples ===",
                flush=True,
            )
            start = time.perf_counter()
            try:
                model_path = pretrain_segformer(
                    config,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    run_dir=run_dir,
                    model_variant=model_variant,
                )
                runtime = time.perf_counter() - start
                train, val, gamma_mse, dice, iou = segformer_history(run_dir)
                histories.append(
                    {
                        "model": model_variant,
                        "samples": samples,
                        "train": train,
                        "val": val,
                        "gamma_mse": gamma_mse,
                        "dice": dice,
                        "iou": iou,
                    }
                )
                best_val_index = int(np.argmin(val))
                best_gamma_mse_index = int(np.argmin(gamma_mse))
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
                    model_variant,
                    samples,
                    model_path,
                    run_dir,
                    runtime,
                )
                row.update(
                    {
                        "epochs_completed": len(train),
                        "final_train_loss": float(train[-1]),
                        "final_val_loss": float(val[-1]),
                        "best_val_loss": float(val[best_val_index]),
                        "best_val_epoch": best_val_index + 1,
                        "final_gamma_mse": float(gamma_mse[-1]),
                        "best_gamma_mse": float(
                            gamma_mse[best_gamma_mse_index]
                        ),
                        "best_gamma_mse_epoch": best_gamma_mse_index + 1,
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
