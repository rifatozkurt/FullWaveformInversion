import argparse
import csv
import gc
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import metrics
from src.config import load_config, save_experiment_config
from src.experiments.transfer_learning_fwi import TransferLearningFWI
from src.experiments.transfer_segformer_fwi import TransferSegFormerFWI
from src.io import create_run_dir, ensure_dir, load_hdf
from src.pretrain_segformer import segformer_model_type


DEFAULT_SAMPLE_COUNTS = "100,250,500,1000,5000,10000"
# segformer_highres is a DISCARDED experiment. Its implementation is retained so
# old runs stay reproducible, but it is no longer part of the default comparison
# and must be requested explicitly with --models.
DEFAULT_MODELS = "unet,segformer"
SUPPORTED_MODELS = ("unet", "segformer", "segformer_highres")
MODEL_STYLES = {
    "unet": {"label": "U-Net", "color": "#2f5aa8"},
    "segformer": {"label": "SegFormer", "color": "#b64040"},
    "segformer_highres": {
        "label": "SegFormer HighRes",
        "color": "#2f8f5b",
    },
}
MODEL_METHODS = {
    "unet": "transfer_learning_fwi",
    "segformer": "transfer_segformer_fwi",
    "segformer_highres": "transfer_segformer_fwi",
}


def parse_csv_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def case_text(cases):
    return "_".join(f"case{case_id}" for case_id in cases)


def read_history(path):
    return np.atleast_1d(np.loadtxt(path, delimiter=","))


def unet_checkpoint_path(config, model_dir, samples):
    exp = config["experiments"]
    return (
        Path(model_dir)
        / f"model_{exp['modelType']}_{int(exp['epochs_pretrain'])}_supervised_{samples}_channel_{len(exp['NNchannels'])}"
    )


def segformer_checkpoint_path(
    config,
    model_dir,
    samples,
    model_variant="segformer",
    checkpoint_selection="primary",
):
    cfg = config["segformer_pretraining"]
    model_type = segformer_model_type(cfg, model_variant)
    path = (
        Path(model_dir)
        / f"model_{model_type}_{int(cfg['epochs'])}_"
        f"{cfg.get('trainingType', 'segmentation')}_{samples}.pt"
    )
    if checkpoint_selection == "primary":
        return path
    return path.with_name(
        f"{path.stem}_best_{checkpoint_selection}{path.suffix}"
    )


def checkpoint_path_for_model(
    config,
    model_dir,
    samples,
    model_name,
    segformer_checkpoint_selection,
):
    if model_name == "unet":
        return unet_checkpoint_path(config, model_dir, samples)
    return segformer_checkpoint_path(
        config,
        model_dir,
        samples,
        model_variant=model_name,
        checkpoint_selection=segformer_checkpoint_selection,
    )


def histories_for_existing_run(run_dir, method, case_id):
    histories_dir = Path(run_dir) / "histories"
    outputs_dir = Path(run_dir) / "outputs"
    cost_path = histories_dir / f"{method}_case{case_id}_cost_history.txt"
    mse_path = histories_dir / f"{method}_case{case_id}_mse_history.txt"
    final_gamma_path = outputs_dir / f"{method}_case{case_id}_final_gamma.h5"
    target_gamma_path = outputs_dir / f"{method}_case{case_id}_target_gamma.h5"
    if not all(
        path.exists()
        for path in (
            cost_path,
            mse_path,
            final_gamma_path,
            target_gamma_path,
        )
    ):
        return None
    return read_history(cost_path), read_history(mse_path)


def run_transfer(method, config, case_id, data_dir, run_dir, resume):
    existing = histories_for_existing_run(run_dir, method, case_id) if resume else None
    if existing is not None:
        cost_history, mse_history = existing
        return cost_history, mse_history, 0.0, True

    ensure_dir(run_dir)
    ensure_dir(Path(run_dir) / "figures")
    ensure_dir(Path(run_dir) / "histories")
    ensure_dir(Path(run_dir) / "outputs")
    save_experiment_config(config, method, case_id, Path(run_dir) / "config.yaml")

    experiment_classes = {
        "transfer_learning_fwi": TransferLearningFWI,
        "transfer_segformer_fwi": TransferSegFormerFWI,
    }
    if method not in experiment_classes:
        raise ValueError(f"Unknown transfer method: {method}")
    experiment_cls = experiment_classes[method]
    experiment = None
    result = None
    try:
        experiment = experiment_cls(config)
        start = time.perf_counter()
        result = experiment.run(case_id, data_dir, run_dir)
        runtime = time.perf_counter() - start
        cost_history = np.asarray(result.cost_history).copy()
        mse_history = np.asarray(result.mse_history).copy()
        (Path(run_dir) / "runtime.txt").write_text(
            "method: {}\ncase_id: {}\nruntime_seconds: {:.6f}\n".format(method, case_id, runtime),
            encoding="utf-8",
        )
        return cost_history, mse_history, runtime, False
    finally:
        del result, experiment
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_output_gammas(run_dir, method, case_id):
    outputs_dir = Path(run_dir) / "outputs"
    final_gamma = load_hdf(outputs_dir / f"{method}_case{case_id}_final_gamma.h5")
    target_gamma = load_hdf(outputs_dir / f"{method}_case{case_id}_target_gamma.h5")
    return np.asarray(final_gamma), np.asarray(target_gamma)


def gamma_metrics(final_gamma, target_gamma, void_threshold):
    """
    Continuous and binarized metrics side by side, from the shared definitions
    in src/metrics.py. Both arrays are already the physical interior here, so
    no ghost ring has to be stripped.
    """
    return metrics.all_metrics(
        np.asarray(final_gamma),
        np.asarray(target_gamma),
        ghost=0,
        threshold=float(void_threshold),
    )


def write_summary(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    groups = {}
    metric_keys = [
        "initial_mse",
        "final_mse",
        "best_mse",
        "final_cost",
        "gamma_mse",
        "gamma_mae",
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
        "void_fraction_pred",
        "void_fraction_target",
    ]
    for row in rows:
        key = (row["model"], int(row["samples"]))
        groups.setdefault(key, []).append(row)

    aggregate = []
    for (model, samples), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        out = {
            "model": model,
            "samples": samples,
            "cases": ",".join(str(row["case"]) for row in sorted(items, key=lambda row: int(row["case"]))),
            "num_cases": len(items),
        }
        for metric in metric_keys:
            values = np.array([float(row[metric]) for row in items], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = float(np.min(values))
            out[f"{metric}_max"] = float(np.max(values))
        aggregate.append(out)
    return aggregate


def plot_case_final_vs_samples(path, rows, metric_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, style in MODEL_STYLES.items():
        items = sorted(
            [row for row in rows if row["model"] == model_name],
            key=lambda row: int(row["samples"]),
        )
        if not items:
            continue
        samples = [int(row["samples"]) for row in items]
        values = [float(row[metric_key]) for row in items]
        ax.plot(
            samples,
            values,
            marker="o",
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )
    ax.set_title(title)
    ax.set_xlabel("Pretraining Samples")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_aggregate_vs_samples(path, rows, metric_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, style in MODEL_STYLES.items():
        items = sorted(
            [row for row in rows if row["model"] == model_name],
            key=lambda row: int(row["samples"]),
        )
        if not items:
            continue
        samples = [int(row["samples"]) for row in items]
        mean = [float(row[f"{metric_key}_mean"]) for row in items]
        std = [float(row[f"{metric_key}_std"]) for row in items]
        ax.errorbar(
            samples,
            mean,
            yerr=std,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=style["color"],
            label=style["label"],
        )
    ax.set_title(title)
    ax.set_xlabel("Pretraining Samples")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_mse_histories(path, histories):
    if not histories:
        return
    sample_counts = sorted({item["samples"] for item in histories})
    fig, axes = plt.subplots(len(sample_counts), 1, figsize=(8, 3.1 * len(sample_counts)), squeeze=False)
    for ax, samples in zip(axes[:, 0], sample_counts):
        for model_name, style in MODEL_STYLES.items():
            matches = [item for item in histories if item["samples"] == samples and item["model"] == model_name]
            if not matches:
                continue
            mse = matches[0]["mse"]
            ax.plot(
                np.arange(1, len(mse) + 1),
                mse,
                marker="o",
                linewidth=1.8,
                color=style["color"],
                label=style["label"],
            )
        ax.set_title(f"{samples} Pretraining Samples")
        ax.set_xlabel("FWI Epochs")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def validate_evaluation_inputs(
    config,
    data_dir,
    model_dir,
    cases,
    sample_counts,
    models,
    segformer_checkpoint_selection,
):
    if not cases:
        raise ValueError("At least one case ID is required")
    if len(cases) != len(set(cases)):
        raise ValueError("Case IDs must be unique")
    if not sample_counts or any(value <= 0 for value in sample_counts):
        raise ValueError("Sample counts must be positive")
    if len(sample_counts) != len(set(sample_counts)):
        raise ValueError("Sample counts must be unique")
    unknown_models = [model for model in models if model not in SUPPORTED_MODELS]
    if not models or unknown_models:
        raise ValueError(
            "--models must contain one or more of: "
            + ", ".join(SUPPORTED_MODELS)
        )
    if not Path(data_dir).is_dir():
        raise FileNotFoundError(f"Evaluation data directory not found: {data_dir}")
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    required_data = [Path(data_dir) / "source.h5"]
    for case_id in cases:
        required_data.extend(
            [
                Path(data_dir) / f"material{case_id}.h5",
                Path(data_dir) / f"measurement{case_id}.h5",
            ]
        )
        if "unet" in models:
            required_data.append(Path(data_dir) / f"gradient{case_id}.h5")
    missing_data = [path for path in required_data if not path.is_file()]

    checkpoints = {
        (model_name, samples): checkpoint_path_for_model(
            config,
            model_dir,
            samples,
            model_name,
            segformer_checkpoint_selection,
        )
        for samples in sample_counts
        for model_name in models
    }
    missing_checkpoints = [
        path for path in checkpoints.values() if not path.is_file()
    ]
    if missing_data or missing_checkpoints:
        messages = []
        if missing_data:
            preview = "\n".join(str(path) for path in missing_data[:20])
            messages.append(
                f"Missing evaluation data file(s) ({len(missing_data)}):\n"
                f"{preview}"
            )
        if missing_checkpoints:
            preview = "\n".join(
                str(path) for path in missing_checkpoints[:20]
            )
            messages.append(
                "Missing model checkpoint(s) "
                f"({len(missing_checkpoints)}):\n{preview}"
            )
        raise FileNotFoundError("\n\n".join(messages))
    return checkpoints


def refresh_comparison_outputs(
    root_run_dir,
    aggregate_dir,
    case_dir,
    case_id,
    rows,
    case_rows,
    case_histories,
):
    write_summary(case_dir / "histories" / "case_summary.csv", case_rows)
    write_summary(
        root_run_dir
        / "histories"
        / "compare_unet_transformer_all_cases.csv",
        rows,
    )
    aggregate = aggregate_rows(rows)
    write_summary(
        aggregate_dir / "histories" / "aggregate_summary.csv",
        aggregate,
    )
    plot_case_final_vs_samples(
        case_dir / "figures" / "final_mse_vs_pretraining_samples.png",
        case_rows,
        "final_mse",
        "Final MSE",
        f"Case {case_id}: Final MSE",
    )
    plot_case_final_vs_samples(
        case_dir / "figures" / "f1_vs_pretraining_samples.png",
        case_rows,
        "f1",
        "F1/Dice",
        f"Case {case_id}: Void-Mask F1",
    )
    plot_mse_histories(
        case_dir / "figures" / "mse_histories_by_sample_count.png",
        case_histories,
    )
    plot_aggregate_vs_samples(
        aggregate_dir
        / "figures"
        / "mean_final_mse_vs_pretraining_samples.png",
        aggregate,
        "final_mse",
        "Mean Final MSE",
        "Mean Final MSE Across Completed Cases",
    )
    plot_aggregate_vs_samples(
        aggregate_dir / "figures" / "mean_f1_vs_pretraining_samples.png",
        aggregate,
        "f1",
        "Mean F1/Dice",
        "Mean Void-Mask F1 Across Completed Cases",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare U-Net, SegFormer, and SegFormer HighRes transfer-FWI "
            "performance using matched pretraining sizes."
        )
    )
    parser.add_argument("--config", default="configs/experimental.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument(
        "--cases",
        default="1,2,3,4",
        help="Comma-separated case IDs. Overrides --case when set.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample-counts", default=DEFAULT_SAMPLE_COUNTS)
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated models: unet, segformer, "
            "segformer_highres."
        ),
    )
    parser.add_argument(
        "--segformer-checkpoint-selection",
        choices=("primary", "dice_score", "val_loss"),
        default="primary",
        help=(
            "Select the primary, best-Dice, or best-validation-loss "
            "checkpoint for both SegFormer variants."
        ),
    )
    parser.add_argument("--model-dir", default="models/improve_transformer")
    parser.add_argument("--output-dir", default="runs/compare_unet_transformer")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Exact reusable run directory. When omitted, a timestamped "
            "directory is created below --output-dir."
        ),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--void-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all case files and checkpoints without running FWI.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    data_dir = Path(args.data_dir or base_config["paths"]["casestudy_data"])
    sample_counts = parse_csv_ints(args.sample_counts)
    cases = parse_csv_ints(args.cases) if args.cases else [int(args.case)]
    models = parse_csv_strings(args.models)
    model_dir = Path(args.model_dir)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    checkpoints = validate_evaluation_inputs(
        base_config,
        data_dir,
        model_dir,
        cases,
        sample_counts,
        models,
        args.segformer_checkpoint_selection,
    )
    planned_runs = len(cases) * len(sample_counts) * len(models)
    print("Evaluation plan:")
    print(f"  Cases: {cases}")
    print(f"  Models: {models}")
    print(f"  Pretraining sample counts: {sample_counts}")
    print(f"  FWI epochs per run: {args.epochs}")
    print(f"  Total independent FWI runs: {planned_runs}")
    print(f"  Data directory: {data_dir}")
    print(f"  Model directory: {model_dir}")
    print(
        "  SegFormer checkpoint selection: "
        f"{args.segformer_checkpoint_selection}"
    )
    print("  Case execution: sequential")
    if args.dry_run:
        print("Dry run complete; all required data and checkpoints exist.")
        return

    root_run_dir = (
        ensure_dir(args.run_dir)
        if args.run_dir
        else create_run_dir(
            ensure_dir(args.output_dir),
            prefix=f"compare_{case_text(cases)}",
        )
    )
    ensure_dir(root_run_dir / "figures")
    ensure_dir(root_run_dir / "histories")
    ensure_dir(root_run_dir / "outputs")
    aggregate_dir = ensure_dir(root_run_dir / "aggregate")
    ensure_dir(aggregate_dir / "figures")
    ensure_dir(aggregate_dir / "histories")

    rows = []
    histories = []
    start_all = time.perf_counter()

    for case_id in cases:
        case_dir = ensure_dir(root_run_dir / f"case{case_id}")
        ensure_dir(case_dir / "figures")
        ensure_dir(case_dir / "histories")
        ensure_dir(case_dir / "outputs")
        case_rows = []
        case_histories = []

        for samples in sample_counts:
            for model_name in models:
                checkpoint = checkpoints[(model_name, samples)]
                method = MODEL_METHODS[model_name]
                config = deepcopy(base_config)
                config["paths"]["pretrained_models"] = str(model_dir)
                if model_name == "unet":
                    config["experiments"]["pretrain_samples"] = [int(samples)]
                    config["experiments"]["transfer_learning_fwi"][
                        "epochs"
                    ] = int(args.epochs)
                else:
                    transfer_cfg = config["experiments"][
                        "transfer_segformer_fwi"
                    ]
                    transfer_cfg["epochs"] = int(args.epochs)
                    transfer_cfg["pretrained_checkpoint"] = str(checkpoint)

                model_run_dir = case_dir / f"{model_name}_{samples}"
                print(
                    "\n=== Case {}: {} transfer FWI, {} pretraining "
                    "sample(s) ===".format(
                        case_id,
                        MODEL_STYLES[model_name]["label"],
                        samples,
                    ),
                    flush=True,
                )
                cost, mse, runtime, reused = run_transfer(
                    method,
                    config,
                    case_id,
                    data_dir,
                    model_run_dir,
                    args.resume,
                )
                final_gamma, target_gamma = load_output_gammas(
                    model_run_dir,
                    method,
                    case_id,
                )
                metrics = gamma_metrics(
                    final_gamma,
                    target_gamma,
                    args.void_threshold,
                )
                case_histories.append(
                    {
                        "model": model_name,
                        "samples": samples,
                        "cost": cost,
                        "mse": mse,
                    }
                )
                row = {
                    "model": model_name,
                    "samples": samples,
                    "case": case_id,
                    "epochs": len(mse),
                    "epochs_requested": args.epochs,
                    "checkpoint": str(checkpoint),
                    "run_dir": str(model_run_dir),
                    "reused_existing_run": reused,
                    "runtime_seconds": runtime,
                    "initial_mse": float(mse[0]),
                    "final_mse": float(mse[-1]),
                    "best_mse": float(np.min(mse)),
                    "final_cost": float(cost[-1]),
                    **metrics,
                }
                rows.append(row)
                case_rows.append(row)
                refresh_comparison_outputs(
                    root_run_dir,
                    aggregate_dir,
                    case_dir,
                    case_id,
                    rows,
                    case_rows,
                    case_histories,
                )

    runtime_all = time.perf_counter() - start_all
    (root_run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: compare_unet_transformer",
                f"config: {args.config}",
                f"cases: {','.join(str(case_id) for case_id in cases)}",
                f"epochs: {args.epochs}",
                f"sample_counts: {args.sample_counts}",
                f"models: {','.join(models)}",
                f"model_dir: {model_dir}",
                f"data_dir: {data_dir}",
                "segformer_checkpoint_selection: "
                f"{args.segformer_checkpoint_selection}",
                f"void_threshold: {args.void_threshold}",
                f"resume: {args.resume}",
                "case_execution: sequential",
                f"planned_runs: {planned_runs}",
                f"runtime_seconds: {runtime_all:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved comparison run to {root_run_dir}", flush=True)


if __name__ == "__main__":
    main()
