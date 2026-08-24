"""
STANDALONE SIDE EXPERIMENT -- ImageNet-pretrained SegFormer-B0 fine-tuning.
===========================================================================

This script is DELIBERATELY SELF-CONTAINED and does not participate in, or
interfere with, any of the thesis experiments.

  * It NEVER writes into `models/improve_transformer/`, `runs/improve_transformer/`,
    or any other directory used by the main experiments. Its defaults are
    `models/standalone_imagenet_b0/` and `runs/standalone_imagenet_b0/`.
  * It NEVER modifies a config file. The config is read once, deep-copied, and
    only the in-memory copy is touched.
  * It NEVER changes `configs/*.yaml: models.segformer`. The main SegFormer keeps
    its narrowed widths [24, 48, 96, 192]; this script builds a separate model at
    true MiT-B0 widths [32, 64, 160, 256] because ImageNet weights only fit those.
  * Checkpoints are written with a distinct model_type (`SegFormerImageNetB0`), so
    the filename cannot collide with an existing checkpoint.
  * It refuses to overwrite an existing checkpoint unless `--overwrite` is passed.

What it does
------------
Loads the ImageNet-1k pretrained MiT-B0 encoder from `nvidia/mit-b0`, transplants
it into the project's own `GradientSegFormer`, and fine-tunes it on the same
gradient -> void-mask segmentation task, with the same recipe, seeds, sample
selection and train/val split as `segformer_pretraining` in the config. The only
intended differences from the baseline SegFormer run are:

    (1) encoder widths are MiT-B0's (2.01M -> 3.71M parameters), and
    (2) the encoder starts from ImageNet weights instead of random.

Because (1) and (2) are confounded, run the control too:

    # ImageNet-initialised (the actual experiment)
    py -3 scripts/standalone_imagenet_b0_finetune.py --config configs/extended.yaml

    # same B0 widths, random init -- isolates the effect of the weights
    py -3 scripts/standalone_imagenet_b0_finetune.py --config configs/extended.yaml --random-init

Evaluate either with `scripts/standalone_imagenet_b0_eval.py`.
"""

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401  (puts the repo root on sys.path)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import io
from src import networks as NN
from src.config import load_config
from src.experiments.base import get_device, simulation_parameters
from src.pretrain_segformer import (
    build_pretraining_scheduler,
    checkpoint_path_for_metric,
    dice_loss_from_logits,
    evaluate_model,
    parse_early_stopping_patience,
)


# --- Everything this standalone experiment is allowed to write ------------------
DEFAULT_MODEL_DIR = "models/standalone_imagenet_b0"
DEFAULT_RUN_ROOT = "runs/standalone_imagenet_b0"
MODEL_TYPE = "SegFormerImageNetB0"

# --- The one architecture that ImageNet MiT-B0 weights actually fit -------------
# Verified against https://huggingface.co/nvidia/mit-b0/raw/main/config.json
MIT_B0_HIDDEN_SIZES = (32, 64, 160, 256)
MIT_B0_NUM_ATTENTION_HEADS = (1, 2, 5, 8)
MIT_B0_STRUCTURE = {
    "depths": (2, 2, 2, 2),
    "sr_ratios": (8, 4, 2, 1),
    "patch_sizes": (7, 3, 3, 3),
    "strides": (4, 2, 2, 2),
    "mlp_ratios": (4, 4, 4, 4),
}
DEFAULT_HF_REPO = "nvidia/mit-b0"


def build_b0_spec(config, drop_path_rate=None):
    """
    MiT-B0 widths, but every non-width setting inherited from the project's own
    SegFormer config so that the recipe stays comparable to the baseline run.

    Fails loudly if the config's structural hyperparameters have drifted away
    from MiT-B0, because the pretrained weights would then not fit.
    """
    base = dict(config.get("models", {}).get("segformer", {}))

    for key, expected in MIT_B0_STRUCTURE.items():
        found = tuple(base.get(key, expected))
        if found != expected:
            raise ValueError(
                f"models.segformer.{key} is {list(found)}, but ImageNet MiT-B0 "
                f"requires {list(expected)}. The pretrained weights cannot be "
                "loaded into this architecture."
            )

    spec = dict(base)
    spec["hidden_sizes"] = list(MIT_B0_HIDDEN_SIZES)
    spec["num_attention_heads"] = list(MIT_B0_NUM_ATTENTION_HEADS)
    spec["in_channels"] = int(base.get("in_channels", 1))
    if drop_path_rate is not None:
        spec["drop_path_rate"] = float(drop_path_rate)
    return spec


def load_imagenet_encoder(model, repo_id=DEFAULT_HF_REPO, verbose=True):
    """
    Transplant the ImageNet-1k MiT-B0 encoder into this model's encoder.

    `SegformerModel` is the encoder alone, so nothing from the classification
    head is pulled in and the decode head keeps exactly the initialization
    `GradientSegFormer.__init__` gave it.

    No state-dict key is hardcoded: `transformers` 4.x and 5.x use different
    module paths (`encoder.patch_embeddings.0` vs `stages.0.patch_embeddings`),
    but `from_pretrained` normalises the checkpoint to whichever layout the
    installed version uses, so the source and target key sets match exactly. The
    only tensor that then differs is the RGB stem, which is identified by shape
    and collapsed to one channel by summing over the colour axis -- preserving
    the filters' response magnitude for a single-channel input.
    """
    try:
        from transformers import SegformerModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Loading ImageNet weights requires the 'transformers' package."
        ) from exc

    source_state = SegformerModel.from_pretrained(repo_id).state_dict()
    target_encoder = model.segformer.segformer
    target_state = target_encoder.state_dict()

    unknown = sorted(set(source_state) - set(target_state))
    if unknown:
        raise RuntimeError(
            f"{repo_id} has {len(unknown)} tensor(s) this architecture does not: "
            f"{unknown[:5]}"
        )

    adapted = {}
    stem_keys = []
    for key, value in source_state.items():
        expected = target_state[key]
        if value.shape == expected.shape:
            adapted[key] = value
        elif (
            value.ndim == 4
            and expected.ndim == 4
            and expected.shape[1] == 1
            and value.shape[0] == expected.shape[0]
            and value.shape[2:] == expected.shape[2:]
        ):
            adapted[key] = value.sum(dim=1, keepdim=True)
            stem_keys.append(key)
        else:
            raise ValueError(
                f"Cannot adapt tensor {key}: checkpoint has {tuple(value.shape)}, "
                f"this architecture expects {tuple(expected.shape)}. The widths "
                "must match MiT-B0 exactly."
            )

    missing, unexpected = target_encoder.load_state_dict(adapted, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ImageNet encoder transfer was incomplete -- refusing to continue, "
            "because a silently half-initialised encoder would look exactly like "
            "'pretraining did not help'.\n"
            f"  missing tensors:    {len(missing)} {list(missing)[:5]}\n"
            f"  unexpected tensors: {len(unexpected)} {list(unexpected)[:5]}"
        )

    transferred = sum(value.numel() for value in adapted.values())
    if verbose:
        print(
            f"ImageNet transfer OK: {len(adapted)} encoder tensors "
            f"({transferred:,} parameters) from {repo_id}; "
            f"RGB stem summed to 1 channel ({stem_keys or 'not needed'}); "
            "decode head left at its own initialization.",
            flush=True,
        )
    return {
        "repo_id": repo_id,
        "tensors_transferred": len(adapted),
        "parameters_transferred": int(transferred),
        "stem_adaptation": "sum_rgb_to_1ch" if stem_keys else "none",
        "stem_keys": stem_keys,
    }


def load_dataset(config, cfg, params, data_dir, verbose=True):
    """Replicates `pretrain_segformer`'s sample selection exactly (same seeds)."""
    Nx = params["Nx"]
    Ny = params["Ny"]
    destination = Path(data_dir or config["paths"]["train_data"])

    number_of_samples = int(cfg["numberOfSamples"])
    available_samples = int(cfg["availableSamples"])
    if "sample_ids" in cfg:
        sample_ids = [int(item) for item in cfg["sample_ids"][:number_of_samples]]
        if len(sample_ids) != number_of_samples:
            raise ValueError(
                "segformer_pretraining.sample_ids must contain at least "
                "numberOfSamples entries"
            )
    else:
        sample_ids = random.sample(range(available_samples), number_of_samples)

    gradients = torch.zeros(
        (number_of_samples, 1, Nx + 1, Ny + 1), dtype=torch.float32
    )
    masks = torch.zeros((number_of_samples, 1, Nx + 1, Ny + 1), dtype=torch.float32)
    void_threshold = float(cfg["void_gamma_threshold"])

    print(f"Loading {number_of_samples} sample(s) from {destination}", flush=True)
    step = max(1, number_of_samples // 20)
    for row, sample_id in enumerate(sample_ids):
        if verbose and (row == 0 or (row + 1) % step == 0 or row + 1 == number_of_samples):
            print(f"  sample {row + 1}/{number_of_samples}", flush=True)
        gradients[row, 0] = torch.tensor(
            io.load_hdf(destination / f"gradient{sample_id}.h5"), dtype=torch.float32
        )
        gamma = torch.tensor(
            io.load_hdf(destination / f"material{sample_id}.h5"), dtype=torch.float32
        )
        masks[row, 0] = (gamma <= void_threshold).float()

    norm_cfg = dict(cfg.get("gradient_normalization", {}))
    gradients = NN.normalize_gradient_for_transformer(gradients, **norm_cfg)
    return gradients, masks, sample_ids, norm_cfg


def save_history(run_dir, rows):
    run_dir = Path(run_dir)
    histories_dir = io.ensure_dir(run_dir / "histories")
    figures_dir = io.ensure_dir(run_dir / "figures")

    csv_path = histories_dir / "standalone_imagenet_b0_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Train", color="#2f5aa8", linewidth=2)
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="Validation", color="#b64040", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss (weighted BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, [r["dice_score"] for r in rows], label="Dice", color="#2f8f5b", linewidth=2)
    axes[1].plot(epochs, [r["iou"] for r in rows], label="IoU", color="#8f6f2f", linewidth=2)
    axes[1].set_title("Validation segmentation quality")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plot_path = figures_dir / "standalone_imagenet_b0_history.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return csv_path, plot_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone: fine-tune an ImageNet-pretrained SegFormer-B0 on the "
            "gradient -> void-mask task. Writes only to its own directories."
        )
    )
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--data-dir", default=None, help="Defaults to paths.train_data")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--samples", type=int, default=None, help="Override numberOfSamples")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None,
                        help="Config default (0.02) is used when omitted; MiT-B0's own value is 0.1.")
    parser.add_argument("--minimum-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", default=None,
                        help="Positive epoch count, or 'none' to disable.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--random-init", action="store_true",
                        help="CONTROL RUN: same B0 widths, no ImageNet weights.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Train the decode head only.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing checkpoint of the same name.")
    args = parser.parse_args()

    # Read-only: the on-disk config is never modified.
    config = deepcopy(load_config(args.config))
    cfg = dict(config["segformer_pretraining"])
    if args.samples is not None:
        cfg["numberOfSamples"] = int(args.samples)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.minimum_epochs is not None:
        cfg["minimum_epochs"] = int(args.minimum_epochs)
    if args.early_stopping_patience is not None:
        cfg["early_stopping_patience"] = parse_early_stopping_patience(
            args.early_stopping_patience
        )

    params = simulation_parameters(config)
    device = get_device()
    initialization = "random" if args.random_init else "imagenet_mit_b0"

    torch.manual_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))

    number_of_samples = int(cfg["numberOfSamples"])
    epochs = int(cfg["epochs"])
    model_dir = io.ensure_dir(args.model_dir)
    primary_path = Path(model_dir) / (
        f"model_{MODEL_TYPE}_{epochs}_{cfg.get('trainingType', 'segmentation')}"
        f"_{number_of_samples}_{initialization}.pt"
    )
    if primary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {primary_path}. Pass --overwrite to replace it."
        )

    run_dir = io.create_run_dir(
        io.ensure_dir(args.output_dir), prefix=f"standalone_b0_{initialization}"
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])

    print("=" * 78)
    print("STANDALONE ImageNet SegFormer-B0 experiment (isolated from all others)")
    print("=" * 78)
    print(f"  config          : {args.config} (read-only)")
    print(f"  initialization  : {initialization}")
    print(f"  samples         : {number_of_samples}")
    print(f"  epochs          : {epochs}")
    print(f"  checkpoint      : {primary_path}")
    print(f"  run directory   : {run_dir}")
    print("=" * 78, flush=True)

    gradients, masks, sample_ids, norm_cfg = load_dataset(
        config, cfg, params, args.data_dir
    )

    dataset = NN.FWIDataset(gradients, masks, device)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(int(cfg["split_seed"])),
    )
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=max(1, min(len(val_set), batch_size)))

    outputs_dir = io.ensure_dir(run_dir / "outputs")
    np.savetxt(outputs_dir / "sample_ids.txt", sample_ids, fmt="%d")
    np.savetxt(outputs_dir / "training_subset_indices.txt", train_set.indices, fmt="%d")
    np.savetxt(outputs_dir / "validation_subset_indices.txt", val_set.indices, fmt="%d")

    spec = build_b0_spec(config, drop_path_rate=args.drop_path_rate)
    model = NN.GradientSegFormer(
        spec=spec,
        gamma_min=params["gamma0"],
        void_prior=float(cfg["void_prior"]),
    )

    transfer_info = None
    if not args.random_init:
        transfer_info = load_imagenet_encoder(model, repo_id=args.hf_repo)
        # The head keeps its own init; restore the void prior on the bias.
        model.reset_classifier_bias()
    else:
        print("CONTROL RUN: random initialization, no ImageNet weights.", flush=True)

    if args.freeze_encoder:
        for parameter in model.segformer.segformer.parameters():
            parameter.requires_grad_(False)
        print("Encoder frozen: training the decode head only.", flush=True)

    model = model.to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {MODEL_TYPE} widths={spec['hidden_sizes']} "
        f"heads={spec['num_attention_heads']} "
        f"parameters={total_parameters:,} trainable={trainable_parameters:,}",
        flush=True,
    )

    pos_weight_cfg = cfg.get("bce_pos_weight", "auto")
    if pos_weight_cfg == "auto":
        positive = masks[train_set.indices].sum()
        negative = masks[train_set.indices].numel() - positive
        pos_weight = (negative / torch.clamp(positive, min=1.0)).to(device)
    elif pos_weight_cfg is None:
        pos_weight = None
    else:
        pos_weight = torch.tensor(float(pos_weight_cfg), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler, scheduler_per_step = build_pretraining_scheduler(
        optimizer, cfg, steps_per_epoch=len(train_loader)
    )

    dice_weight = float(cfg["dice_weight"])
    eval_threshold = float(cfg["eval_threshold"])
    metric_paths = {
        "val_loss": checkpoint_path_for_metric(primary_path, "val_loss"),
        "dice_score": checkpoint_path_for_metric(primary_path, "dice_score"),
    }
    best_metrics = {"val_loss": float("inf"), "dice_score": -float("inf")}
    checkpoint_selection = str(cfg.get("checkpoint_selection", "val_loss"))
    if checkpoint_selection not in best_metrics:
        raise ValueError("checkpoint_selection must be val_loss or dice_score")
    early_stopping_patience = cfg.get("early_stopping_patience")
    early_stopping_patience = (
        None if early_stopping_patience is None else int(early_stopping_patience)
    )
    minimum_epochs = int(cfg.get("minimum_epochs", 1))
    epochs_without_improvement = 0
    print_every = max(1, int(cfg.get("print_every_batches", 100)))

    rows = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_loss = train_bce = train_dice = train_grad_norm = 0.0
        for batch, (gradient, target) in enumerate(train_loader):
            gradient = gradient.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(gradient)
            bce_loss = criterion(logits, target)
            dice_loss = dice_loss_from_logits(logits, target)
            loss = bce_loss + dice_weight * dice_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["clipGrad"])
            )
            optimizer.step()
            if scheduler is not None and scheduler_per_step:
                scheduler.step()
            train_loss += float(loss.detach().cpu())
            train_bce += float(bce_loss.detach().cpu())
            train_dice += float(dice_loss.detach().cpu())
            train_grad_norm += float(grad_norm.detach().cpu())
            if batch == 0 or batch + 1 == len(train_loader) or (batch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} batch {batch + 1}/{len(train_loader)} "
                    f"train_loss={float(loss.detach().cpu()):.6E}",
                    flush=True,
                )
        if scheduler is not None and not scheduler_per_step:
            scheduler.step()

        batches = max(1, len(train_loader))
        val_metrics = evaluate_model(
            model, val_loader, criterion, dice_weight, eval_threshold, device
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / batches,
            "train_bce_loss": train_bce / batches,
            "train_dice_loss": train_dice / batches,
            "train_grad_norm": train_grad_norm / batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **val_metrics,
        }
        rows.append(row)

        improved = []
        for metric in best_metrics:
            better = (
                row[metric] < best_metrics[metric]
                if metric == "val_loss"
                else row[metric] > best_metrics[metric]
            )
            if not better:
                continue
            best_metrics[metric] = row[metric]
            improved.append(metric)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": model.architecture_dict(),
                "model_variant": "segformer",
                "gamma_min": float(params["gamma0"]),
                "void_prior": float(cfg["void_prior"]),
                "gradient_normalization": norm_cfg,
                "void_gamma_threshold": float(cfg["void_gamma_threshold"]),
                "training_config": {
                    **cfg,
                    "model_variant": "segformer",
                    "model_type": MODEL_TYPE,
                },
                "epoch": epoch + 1,
                "validation_metrics": val_metrics,
                "checkpoint_metric": metric,
                "checkpoint_metric_value": float(row[metric]),
                # provenance for this standalone experiment
                "standalone_experiment": "imagenet_b0",
                "initialization": initialization,
                "imagenet_transfer": transfer_info,
                "encoder_frozen": bool(args.freeze_encoder),
                "source_config": str(args.config),
            }
            torch.save(checkpoint, metric_paths[metric])
            if metric == checkpoint_selection:
                torch.save(checkpoint, primary_path)

        epochs_without_improvement = (
            0 if checkpoint_selection in improved else epochs_without_improvement + 1
        )
        elapsed = time.perf_counter() - start
        print(
            f"Epoch: {epoch}/{epochs - 1}\tTrain loss: {row['train_loss']:.6E}\t"
            f"Val loss: {row['val_loss']:.6E}\tDice: {row['dice_score']:.4f}\t"
            f"IoU: {row['iou']:.4f}\tElapsed: {elapsed:.2f}s",
            flush=True,
        )
        start = time.perf_counter()

        if (
            early_stopping_patience is not None
            and epoch + 1 >= minimum_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping after epoch {epoch + 1}: {checkpoint_selection} "
                f"did not improve for {epochs_without_improvement} epoch(s).",
                flush=True,
            )
            break

    if rows:
        csv_path, plot_path = save_history(run_dir, rows)
        print(f"metrics_csv: {csv_path}", flush=True)
        print(f"plot_path:   {plot_path}", flush=True)

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "standalone_imagenet_b0",
                "initialization": initialization,
                "config": str(args.config),
                "samples": number_of_samples,
                "epochs_run": len(rows),
                "epochs_requested": epochs,
                "hidden_sizes": list(spec["hidden_sizes"]),
                "num_attention_heads": list(spec["num_attention_heads"]),
                "total_parameters": int(total_parameters),
                "trainable_parameters": int(trainable_parameters),
                "encoder_frozen": bool(args.freeze_encoder),
                "imagenet_transfer": transfer_info,
                "best_val_loss": best_metrics["val_loss"],
                "best_dice_score": best_metrics["dice_score"],
                "checkpoint": str(primary_path),
                "checkpoints_by_metric": {k: str(v) for k, v in metric_paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nCheckpoint: {primary_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
