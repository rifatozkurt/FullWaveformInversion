"""
STANDALONE EVALUATION -- downstream transfer FWI for SegFormer checkpoints.
===========================================================================

Companion to `scripts/standalone_imagenet_b0_finetune.py`. This runs the REAL
evaluation: the pretrained network is used as the reparameterization ansatz for
the material field and driven through the full FWI loop

    gradient -> network -> gamma -> forward solve -> adjoint gradient
             -> backprop into the network weights -> optimizer step

exactly as `src/experiments/transfer_segformer_fwi.py` does for the thesis
experiments, on the same case studies. It is the checkpoint-keyed analogue of
`scripts/compare_unet_transformer.py`.

Two numbers are reported for every run, which is the whole point:

  * PRE-FWI  -- quality of gamma at epoch 0, i.e. what the pretrained weights
                predict before any FWI step. This is what the pretraining task
                bought you.
  * POST-FWI -- quality of gamma after the FWI optimization. This is the number
                the thesis actually reports.

A checkpoint can start worse and still finish better (or vice versa), so a
single forward pass is NOT a substitute for this.

SELF-CONTAINMENT
----------------
  * Writes ONLY under `runs/standalone_imagenet_b0_eval/` (override with
    `--output-dir`). Never touches `models/`, `runs/improve_transformer/`, or any
    directory the thesis experiments own.
  * Never modifies a config file. The config is deep-copied per run and only the
    in-memory copy gets the checkpoint path and epoch count.
  * All FWI settings (`lr`, `alpha`, `beta`, `clipGrad`, `costScaling`,
    `trainable_mode`, ...) are taken UNCHANGED from
    `experiments.transfer_segformer_fwi` in the config, so every checkpoint is
    evaluated under the identical downstream recipe.

NOTE: with the config's `trainable_mode: decoder_only`, the encoder is frozen
during FWI. ImageNet encoder features are therefore used as-is and never adapted
-- which is precisely what makes an ImageNet-vs-random comparison meaningful here.

Examples
--------
    # ImageNet run vs its random-init control vs the thesis baseline, 4 cases
    .venv/Scripts/python.exe scripts/standalone_imagenet_b0_eval.py \
        --checkpoints "models/standalone_imagenet_b0/model_SegFormerImageNetB0_100_segmentation_15000_imagenet_mit_b0.pt,models/standalone_imagenet_b0/model_SegFormerImageNetB0_100_segmentation_15000_random.pt,models/improve_transformer/model_SegFormer_100_segmentation_15000.pt" \
        --labels "ImageNet B0,Random B0,Baseline SegFormer"

    # check every file exists, run nothing
    .venv/Scripts/python.exe scripts/standalone_imagenet_b0_eval.py --dry-run
"""

import argparse
import csv
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the repo root on sys.path)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config, save_experiment_config
from src.experiments.base import simulation_parameters
from src.experiments.transfer_segformer_fwi import TransferSegFormerFWI
from src.io import create_run_dir, ensure_dir, load_hdf


DEFAULT_RUN_ROOT = "runs/standalone_imagenet_b0_eval"
DEFAULT_MODEL_DIR = "models/standalone_imagenet_b0"
METHOD = "transfer_segformer_fwi"
PALETTE = ["#2f5aa8", "#b64040", "#2f8f5b", "#8f6f2f", "#6a3d9a", "#3d8f8f"]


def parse_csv_ints(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def default_checkpoints(model_dir):
    found = sorted(Path(model_dir).glob("model_SegFormerImageNetB0_*.pt"))
    primary = [path for path in found if "_best_" not in path.name]
    if not primary:
        raise FileNotFoundError(
            f"No standalone checkpoint found in {model_dir}. Run "
            "scripts/standalone_imagenet_b0_finetune.py first, or pass "
            "--checkpoints explicitly."
        )
    return primary


def gamma_metrics(final_gamma, target_gamma, void_threshold):
    """Void-mask agreement between a reconstructed and a target gamma field."""
    final_gamma = np.asarray(final_gamma)
    target_gamma = np.asarray(target_gamma)
    predicted_void = final_gamma <= float(void_threshold)
    target_void = target_gamma <= float(void_threshold)
    tp = float(np.logical_and(predicted_void, target_void).sum())
    fp = float(np.logical_and(predicted_void, ~target_void).sum())
    fn = float(np.logical_and(~predicted_void, target_void).sum())
    tn = float(np.logical_and(~predicted_void, ~target_void).sum())
    eps = 1e-12
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / max(tp + fp + fn, eps),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, eps),
        "mse": float(np.mean((final_gamma - target_gamma) ** 2)),
        "mae": float(np.mean(np.abs(final_gamma - target_gamma))),
        "void_fraction_pred": float(predicted_void.mean()),
        "void_fraction_target": float(target_void.mean()),
    }


def load_gamma_history(run_dir, case_id, params):
    """gamma_history[0] is the pre-FWI prediction; [-1] is the final one."""
    path = Path(run_dir) / "histories" / f"{METHOD}_case{case_id}_gamma_history.h5"
    flat = np.asarray(load_hdf(path))
    return flat.reshape(flat.shape[0], params["Nx"] + 1, params["Ny"] + 1)


def already_complete(run_dir, case_id):
    histories = Path(run_dir) / "histories"
    outputs = Path(run_dir) / "outputs"
    required = [
        histories / f"{METHOD}_case{case_id}_cost_history.txt",
        histories / f"{METHOD}_case{case_id}_mse_history.txt",
        histories / f"{METHOD}_case{case_id}_gamma_history.h5",
        outputs / f"{METHOD}_case{case_id}_final_gamma.h5",
        outputs / f"{METHOD}_case{case_id}_target_gamma.h5",
    ]
    return all(path.exists() for path in required)


def run_one(config, case_id, data_dir, run_dir, resume):
    """Run (or reuse) one transfer-FWI inversion. Returns histories + runtime."""
    if resume and already_complete(run_dir, case_id):
        histories = Path(run_dir) / "histories"
        cost = np.atleast_1d(
            np.loadtxt(histories / f"{METHOD}_case{case_id}_cost_history.txt", delimiter=",")
        )
        mse = np.atleast_1d(
            np.loadtxt(histories / f"{METHOD}_case{case_id}_mse_history.txt", delimiter=",")
        )
        return cost, mse, 0.0, True

    ensure_dir(run_dir)
    for name in ("figures", "histories", "outputs"):
        ensure_dir(Path(run_dir) / name)
    save_experiment_config(config, METHOD, case_id, Path(run_dir) / "config.yaml")

    experiment = None
    result = None
    try:
        experiment = TransferSegFormerFWI(config)
        start = time.perf_counter()
        result = experiment.run(case_id, data_dir, run_dir)
        runtime = time.perf_counter() - start
        cost = np.asarray(result.cost_history).copy()
        mse = np.asarray(result.mse_history).copy()
        (Path(run_dir) / "runtime.txt").write_text(
            f"method: {METHOD}\ncase_id: {case_id}\nruntime_seconds: {runtime:.6f}\n",
            encoding="utf-8",
        )
        return cost, mse, runtime, False
    finally:
        del result, experiment
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    metric_keys = [
        "pre_fwi_f1", "pre_fwi_iou", "pre_fwi_mse",
        "post_fwi_f1", "post_fwi_iou", "post_fwi_mse",
        "final_cost", "best_mse", "epochs",
    ]
    groups = {}
    for row in rows:
        groups.setdefault(row["model"], []).append(row)
    out = []
    for model, items in groups.items():
        entry = {
            "model": model,
            "checkpoint": items[0]["checkpoint"],
            "num_cases": len(items),
            "cases": ",".join(str(i["case"]) for i in sorted(items, key=lambda r: r["case"])),
        }
        for key in metric_keys:
            values = np.array([float(i[key]) for i in items], dtype=float)
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(entry)
    return out


def plot_mse_histories(path, histories, cases, labels):
    if not histories:
        return
    fig, axes = plt.subplots(
        len(cases), 1, figsize=(8, 3.2 * len(cases)), squeeze=False
    )
    for axis, case_id in zip(axes[:, 0], cases):
        for index, label in enumerate(labels):
            match = [h for h in histories if h["case"] == case_id and h["model"] == label]
            if not match:
                continue
            mse = match[0]["mse"]
            axis.plot(
                np.arange(1, len(mse) + 1), mse, marker="o", linewidth=1.9,
                color=PALETTE[index % len(PALETTE)], label=label,
            )
        axis.set_title(f"Case {case_id}: FWI convergence")
        axis.set_xlabel("FWI epoch")
        axis.set_ylabel("gamma MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pre_post(path, aggregated):
    """The headline figure: what pretraining bought vs what FWI added."""
    labels = [entry["model"] for entry in aggregated]
    positions = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].bar(positions - width / 2, [e["pre_fwi_f1_mean"] for e in aggregated],
                width, yerr=[e["pre_fwi_f1_std"] for e in aggregated], capsize=4,
                label="Pre-FWI (pretrained weights)", color="#9ab4d8")
    axes[0].bar(positions + width / 2, [e["post_fwi_f1_mean"] for e in aggregated],
                width, yerr=[e["post_fwi_f1_std"] for e in aggregated], capsize=4,
                label="Post-FWI", color="#2f5aa8")
    axes[0].set_ylabel("Void-mask F1")
    axes[0].set_title("Reconstruction quality, before and after FWI")
    axes[0].set_ylim(0, 1)

    axes[1].bar(positions - width / 2, [e["pre_fwi_mse_mean"] for e in aggregated],
                width, label="Pre-FWI", color="#d8a9a9")
    axes[1].bar(positions + width / 2, [e["post_fwi_mse_mean"] for e in aggregated],
                width, label="Post-FWI", color="#b64040")
    axes[1].set_ylabel("gamma MSE")
    axes[1].set_yscale("log")
    axes[1].set_title("gamma MSE, before and after FWI")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_reconstructions(path, reconstructions, cases, labels):
    if not reconstructions:
        return
    columns = 1 + len(labels)
    fig, axes = plt.subplots(
        len(cases), columns, figsize=(3.1 * columns, 2.7 * len(cases)),
        squeeze=False, constrained_layout=True,
    )
    for row, case_id in enumerate(cases):
        entries = [r for r in reconstructions if r["case"] == case_id]
        if not entries:
            continue
        axes[row][0].imshow(np.transpose(entries[0]["target"]), vmin=0, vmax=1, aspect="auto")
        axes[row][0].set_ylabel(f"case {case_id}", fontsize=9)
        if row == 0:
            axes[row][0].set_title("Target", fontsize=10)
        for column, label in enumerate(labels, start=1):
            match = [r for r in entries if r["model"] == label]
            if not match:
                continue
            axes[row][column].imshow(
                np.transpose(match[0]["final"]), vmin=0, vmax=1, aspect="auto"
            )
            if row == 0:
                axes[row][column].set_title(label, fontsize=10)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone downstream transfer-FWI evaluation of SegFormer "
            "checkpoints on the case studies. Writes only to its own run directory."
        )
    )
    parser.add_argument("--config", default="configs/extended.yaml")
    parser.add_argument("--data-dir", default=None, help="Defaults to paths.casestudy_data")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated .pt paths. Defaults to the standalone checkpoint(s).")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated display names, one per checkpoint.")
    parser.add_argument("--cases", default="1,2,3,4")
    parser.add_argument("--epochs", type=int, default=None,
                        help="FWI epochs per run. Defaults to the config value.")
    parser.add_argument("--trainable-mode", default=None,
                        choices=("all", "decoder_only", "decoder_plus_last_stage"),
                        help="Override the config's downstream trainable mode.")
    parser.add_argument("--void-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", default=None,
                        help="Exact reusable run directory; pass the same one again to resume.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate cases and checkpoints without running FWI.")
    args = parser.parse_args()

    base_config = deepcopy(load_config(args.config))
    params = simulation_parameters(base_config)
    data_dir = Path(args.data_dir or base_config["paths"]["casestudy_data"])
    cases = parse_csv_ints(args.cases)

    if args.checkpoints:
        checkpoints = [Path(p) for p in parse_csv_strings(args.checkpoints)]
    else:
        checkpoints = default_checkpoints(args.model_dir)
    if args.labels:
        labels = parse_csv_strings(args.labels)
        if len(labels) != len(checkpoints):
            raise ValueError(
                f"--labels has {len(labels)} entries but {len(checkpoints)} checkpoints were given"
            )
    else:
        labels = [path.stem for path in checkpoints]

    # --- validate everything up front, before any expensive simulation ---------
    missing = [path for path in checkpoints if not path.is_file()]
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Case-study directory not found: {data_dir}")
    required = [data_dir / "source.h5"]
    for case_id in cases:
        required += [data_dir / f"material{case_id}.h5", data_dir / f"measurement{case_id}.h5"]
    missing += [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(str(p) for p in missing)
        )

    fwi_cfg = base_config["experiments"][METHOD]
    epochs = int(args.epochs if args.epochs is not None else fwi_cfg["epochs"])
    trainable_mode = args.trainable_mode or fwi_cfg.get("trainable_mode", "all")

    print("=" * 78)
    print("STANDALONE downstream transfer-FWI evaluation")
    print("=" * 78)
    print(f"  config         : {args.config} (read-only)")
    print(f"  case studies   : {cases}")
    print(f"  checkpoints    : {len(checkpoints)}")
    for label, path in zip(labels, checkpoints):
        print(f"      {label:<28} {path}")
    print(f"  FWI epochs     : {epochs}")
    print(f"  trainable mode : {trainable_mode}"
          f"{'  (encoder frozen during FWI)' if trainable_mode == 'decoder_only' else ''}")
    print(f"  total FWI runs : {len(cases) * len(checkpoints)}")
    print("=" * 78, flush=True)
    if args.dry_run:
        print("Dry run complete; all cases and checkpoints exist.")
        return

    root = (
        ensure_dir(args.run_dir)
        if args.run_dir
        else create_run_dir(ensure_dir(args.output_dir), prefix="standalone_b0_fwi")
    )
    for name in ("figures", "histories", "outputs"):
        ensure_dir(root / name)

    rows = []
    histories = []
    reconstructions = []
    start_all = time.perf_counter()

    for case_id in cases:
        for index, (checkpoint, label) in enumerate(zip(checkpoints, labels)):
            config = deepcopy(base_config)
            transfer_cfg = config["experiments"][METHOD]
            transfer_cfg["pretrained_checkpoint"] = str(checkpoint.resolve())
            transfer_cfg["epochs"] = epochs
            transfer_cfg["trainable_mode"] = trainable_mode

            run_dir = root / f"case{case_id}" / f"model{index}"
            print(f"\n=== Case {case_id}: {label} ===", flush=True)
            cost, mse, runtime, reused = run_one(
                config, case_id, data_dir, run_dir, args.resume
            )

            gamma_history = load_gamma_history(run_dir, case_id, params)
            target = np.asarray(
                load_hdf(run_dir / "outputs" / f"{METHOD}_case{case_id}_target_gamma.h5")
            )
            final = np.asarray(
                load_hdf(run_dir / "outputs" / f"{METHOD}_case{case_id}_final_gamma.h5")
            )
            pre = gamma_metrics(gamma_history[0], target, args.void_threshold)
            post = gamma_metrics(final, target, args.void_threshold)

            row = {
                "model": label,
                "case": case_id,
                "checkpoint": str(checkpoint),
                "run_dir": str(run_dir),
                "reused_existing_run": reused,
                "runtime_seconds": runtime,
                "epochs": len(mse),
                "final_cost": float(cost[-1]),
                "best_mse": float(np.min(mse)),
                **{f"pre_fwi_{k}": v for k, v in pre.items()},
                **{f"post_fwi_{k}": v for k, v in post.items()},
            }
            rows.append(row)
            histories.append({"model": label, "case": case_id, "mse": mse, "cost": cost})
            reconstructions.append({"model": label, "case": case_id,
                                    "final": final, "target": target})

            print(
                f"  pre-FWI : F1={pre['f1']:.4f}  MSE={pre['mse']:.6E}\n"
                f"  post-FWI: F1={post['f1']:.4f}  MSE={post['mse']:.6E}  "
                f"({'reused' if reused else f'{runtime:.1f}s'})",
                flush=True,
            )

            # refresh outputs after every run, so a long job is never lost
            aggregated = aggregate(rows)
            write_csv(root / "histories" / "all_runs.csv", rows)
            write_csv(root / "histories" / "aggregate.csv", aggregated)
            plot_mse_histories(root / "figures" / "fwi_convergence.png",
                               histories, cases, labels)
            plot_pre_post(root / "figures" / "pre_vs_post_fwi.png", aggregated)
            plot_reconstructions(root / "figures" / "reconstructions.png",
                                 reconstructions, cases, labels)

    aggregated = aggregate(rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "standalone_imagenet_b0_transfer_fwi",
                "config": str(args.config),
                "cases": cases,
                "epochs": epochs,
                "trainable_mode": trainable_mode,
                "data_dir": str(data_dir),
                "checkpoints": {l: str(c) for l, c in zip(labels, checkpoints)},
                "aggregate": aggregated,
                "runtime_seconds": time.perf_counter() - start_all,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print(f"{'model':<26}{'pre-FWI F1':>12}{'post-FWI F1':>13}{'post-FWI MSE':>16}")
    print("-" * 84)
    for entry in aggregated:
        print(
            f"{entry['model'][:25]:<26}"
            f"{entry['pre_fwi_f1_mean']:>12.4f}"
            f"{entry['post_fwi_f1_mean']:>13.4f}"
            f"{entry['post_fwi_mse_mean']:>16.6E}"
        )
    print("=" * 84)
    print(f"\nSaved evaluation to {root}")


if __name__ == "__main__":
    main()
