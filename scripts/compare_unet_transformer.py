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

from src.config import load_config, save_experiment_config
from src.experiments.transfer_learning_fwi import TransferLearningFWI
from src.experiments.transfer_segformer_fwi import TransferSegFormerFWI
from src.io import create_run_dir, ensure_dir, load_hdf


DEFAULT_SAMPLE_COUNTS = "100,250,500,1000"


def parse_csv_ints(text):
    return [int(item) for item in text.split(",") if item.strip()]


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


def segformer_checkpoint_path(config, model_dir, samples):
    cfg = config["segformer_pretraining"]
    return (
        Path(model_dir)
        / f"model_{cfg.get('model_type', 'SegFormer')}_{int(cfg['epochs'])}_{cfg.get('trainingType', 'segmentation')}_{samples}.pt"
    )


def histories_for_existing_run(run_dir, method, case_id):
    histories_dir = Path(run_dir) / "histories"
    cost_path = histories_dir / f"{method}_case{case_id}_cost_history.txt"
    mse_path = histories_dir / f"{method}_case{case_id}_mse_history.txt"
    if not cost_path.exists() or not mse_path.exists():
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

    experiment_cls = TransferLearningFWI if method == "transfer_learning_fwi" else TransferSegFormerFWI
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
    final_gamma = np.asarray(final_gamma)
    target_gamma = np.asarray(target_gamma)
    pred_void = final_gamma <= float(void_threshold)
    target_void = target_gamma <= float(void_threshold)
    tp = np.logical_and(pred_void, target_void).sum()
    fp = np.logical_and(pred_void, ~target_void).sum()
    fn = np.logical_and(~pred_void, target_void).sum()
    tn = np.logical_and(~pred_void, ~target_void).sum()
    eps = 1e-12
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)
    iou = tp / max(tp + fp + fn, eps)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, eps)
    mae = np.mean(np.abs(final_gamma - target_gamma))
    return {
        "void_threshold": float(void_threshold),
        "void_fraction_pred": float(pred_void.mean()),
        "void_fraction_target": float(target_void.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "mae": float(mae),
    }


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
        "mae",
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
    for model_name, color in (("unet", "#2f5aa8"), ("segformer", "#b64040")):
        items = sorted(
            [row for row in rows if row["model"] == model_name],
            key=lambda row: int(row["samples"]),
        )
        if not items:
            continue
        samples = [int(row["samples"]) for row in items]
        values = [float(row[metric_key]) for row in items]
        ax.plot(samples, values, marker="o", linewidth=2.2, color=color, label=model_name)
    ax.set_title(title)
    ax.set_xlabel("pretraining samples")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_aggregate_vs_samples(path, rows, metric_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in (("unet", "#2f5aa8"), ("segformer", "#b64040")):
        items = sorted(
            [row for row in rows if row["model"] == model_name],
            key=lambda row: int(row["samples"]),
        )
        if not items:
            continue
        samples = [int(row["samples"]) for row in items]
        mean = [float(row[f"{metric_key}_mean"]) for row in items]
        std = [float(row[f"{metric_key}_std"]) for row in items]
        ax.errorbar(samples, mean, yerr=std, marker="o", linewidth=2.2, capsize=4, color=color, label=model_name)
    ax.set_title(title)
    ax.set_xlabel("pretraining samples")
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
        for model_name, color in (("unet", "#2f5aa8"), ("segformer", "#b64040")):
            matches = [item for item in histories if item["samples"] == samples and item["model"] == model_name]
            if not matches:
                continue
            mse = matches[0]["mse"]
            ax.plot(np.arange(1, len(mse) + 1), mse, marker="o", linewidth=1.8, color=color, label=model_name)
        ax.set_title(f"{samples} pretraining samples")
        ax.set_xlabel("FWI epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare U-Net and SegFormer transfer-FWI performance using matched pretraining sizes."
    )
    parser.add_argument("--config", default="configs/experimental.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--cases", default="1,2,3,4", help="Comma-separated case ids. Overrides --case when set.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sample-counts", default=DEFAULT_SAMPLE_COUNTS)
    parser.add_argument("--model-dir", default="models/improve_transformer")
    parser.add_argument("--output-dir", default="runs/compare_unet_transformer")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--void-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    base_config = load_config(args.config)
    data_dir = Path(args.data_dir or base_config["paths"]["casestudy_data"])
    sample_counts = parse_csv_ints(args.sample_counts)
    cases = parse_csv_ints(args.cases) if args.cases else [int(args.case)]
    model_dir = Path(args.model_dir)

    root_run_dir = create_run_dir(ensure_dir(args.output_dir), prefix=f"compare_{case_text(cases)}")
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
            unet_ckpt = unet_checkpoint_path(base_config, model_dir, samples)
            segformer_ckpt = segformer_checkpoint_path(base_config, model_dir, samples)
            missing = [path for path in (unet_ckpt, segformer_ckpt) if not path.exists()]
            if missing:
                missing_text = "\n".join(str(path) for path in missing)
                raise FileNotFoundError(f"Missing comparative checkpoint(s):\n{missing_text}")

            unet_config = deepcopy(base_config)
            unet_config["paths"]["pretrained_models"] = str(model_dir)
            unet_config["experiments"]["pretrain_samples"] = [int(samples)]
            unet_config["experiments"]["transfer_learning_fwi"]["epochs"] = int(args.epochs)
            unet_run_dir = case_dir / f"unet_{samples}"
            print(f"\n=== Case {case_id}: U-Net transfer FWI, {samples} pretraining sample(s) ===", flush=True)
            cost, mse, runtime, reused = run_transfer(
                "transfer_learning_fwi",
                unet_config,
                case_id,
                data_dir,
                unet_run_dir,
                args.resume,
            )
            final_gamma, target_gamma = load_output_gammas(unet_run_dir, "transfer_learning_fwi", case_id)
            metrics = gamma_metrics(final_gamma, target_gamma, args.void_threshold)
            case_histories.append({"model": "unet", "samples": samples, "cost": cost, "mse": mse})
            row = {
                "model": "unet",
                "samples": samples,
                "case": case_id,
                "epochs": args.epochs,
                "checkpoint": str(unet_ckpt),
                "run_dir": str(unet_run_dir),
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

            write_summary(case_dir / "histories" / "case_summary.csv", case_rows)
            write_summary(root_run_dir / "histories" / "compare_unet_transformer_all_cases.csv", rows)
            aggregate = aggregate_rows(rows)
            write_summary(aggregate_dir / "histories" / "aggregate_summary.csv", aggregate)
            plot_case_final_vs_samples(case_dir / "figures" / "final_mse_vs_pretraining_samples.png", case_rows, "final_mse", "final MSE", f"Case {case_id}: final MSE")
            plot_case_final_vs_samples(case_dir / "figures" / "f1_vs_pretraining_samples.png", case_rows, "f1", "F1/Dice", f"Case {case_id}: void-mask F1")
            plot_mse_histories(case_dir / "figures" / "mse_histories_by_sample_count.png", case_histories)
            plot_aggregate_vs_samples(aggregate_dir / "figures" / "mean_final_mse_vs_pretraining_samples.png", aggregate, "final_mse", "mean final MSE", "Mean final MSE across completed cases")
            plot_aggregate_vs_samples(aggregate_dir / "figures" / "mean_f1_vs_pretraining_samples.png", aggregate, "f1", "mean F1/Dice", "Mean void-mask F1 across completed cases")

            seg_config = deepcopy(base_config)
            seg_config["paths"]["pretrained_models"] = str(model_dir)
            seg_config["experiments"]["transfer_segformer_fwi"]["epochs"] = int(args.epochs)
            seg_config["experiments"]["transfer_segformer_fwi"]["pretrained_checkpoint"] = str(segformer_ckpt)
            seg_run_dir = case_dir / f"segformer_{samples}"
            print(f"\n=== Case {case_id}: SegFormer transfer FWI, {samples} pretraining sample(s) ===", flush=True)
            cost, mse, runtime, reused = run_transfer(
                "transfer_segformer_fwi",
                seg_config,
                case_id,
                data_dir,
                seg_run_dir,
                args.resume,
            )
            final_gamma, target_gamma = load_output_gammas(seg_run_dir, "transfer_segformer_fwi", case_id)
            metrics = gamma_metrics(final_gamma, target_gamma, args.void_threshold)
            case_histories.append({"model": "segformer", "samples": samples, "cost": cost, "mse": mse})
            row = {
                "model": "segformer",
                "samples": samples,
                "case": case_id,
                "epochs": args.epochs,
                "checkpoint": str(segformer_ckpt),
                "run_dir": str(seg_run_dir),
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

            write_summary(case_dir / "histories" / "case_summary.csv", case_rows)
            write_summary(root_run_dir / "histories" / "compare_unet_transformer_all_cases.csv", rows)
            aggregate = aggregate_rows(rows)
            write_summary(aggregate_dir / "histories" / "aggregate_summary.csv", aggregate)
            plot_case_final_vs_samples(case_dir / "figures" / "final_mse_vs_pretraining_samples.png", case_rows, "final_mse", "final MSE", f"Case {case_id}: final MSE")
            plot_case_final_vs_samples(case_dir / "figures" / "f1_vs_pretraining_samples.png", case_rows, "f1", "F1/Dice", f"Case {case_id}: void-mask F1")
            plot_mse_histories(case_dir / "figures" / "mse_histories_by_sample_count.png", case_histories)
            plot_aggregate_vs_samples(aggregate_dir / "figures" / "mean_final_mse_vs_pretraining_samples.png", aggregate, "final_mse", "mean final MSE", "Mean final MSE across completed cases")
            plot_aggregate_vs_samples(aggregate_dir / "figures" / "mean_f1_vs_pretraining_samples.png", aggregate, "f1", "mean F1/Dice", "Mean void-mask F1 across completed cases")

    runtime_all = time.perf_counter() - start_all
    (root_run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: compare_unet_transformer",
                f"config: {args.config}",
                f"cases: {','.join(str(case_id) for case_id in cases)}",
                f"epochs: {args.epochs}",
                f"sample_counts: {args.sample_counts}",
                f"model_dir: {model_dir}",
                f"data_dir: {data_dir}",
                f"void_threshold: {args.void_threshold}",
                f"resume: {args.resume}",
                f"runtime_seconds: {runtime_all:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved comparison run to {root_run_dir}", flush=True)


if __name__ == "__main__":
    main()
