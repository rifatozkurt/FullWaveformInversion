import csv
import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src import io
from src import networks as NN
from src.config import load_config
from src.experiments.base import get_device, simulation_parameters



# this is a very useful loss for our case
def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denominator = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_metrics(logits, target, threshold=0.5, eps=1e-8):
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "dice_score": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
    }


def save_segformer_pretraining_outputs(run_dir, rows):
    run_dir = Path(run_dir)
    histories_dir = io.ensure_dir(run_dir / "histories")
    figures_dir = io.ensure_dir(run_dir / "figures")
    outputs_dir = io.ensure_dir(run_dir / "outputs")

    csv_path = histories_dir / "segformer_pretraining_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    val_loss = [row["val_loss"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, train_loss, label="Training loss", color="#2f5aa8", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation loss", color="#b64040", linewidth=2)
    ax.set_title("SegFormer pretraining loss history")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = figures_dir / "segformer_pretraining_loss_history.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, train_loss, label="Training loss", color="#2f5aa8", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation loss", color="#b64040", linewidth=2)
    ax.set_title("SegFormer pretraining loss history (log scale)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    log_plot_path = figures_dir / "segformer_pretraining_loss_history_log.png"
    fig.savefig(log_plot_path, dpi=160)
    plt.close(fig)

    np.savez(outputs_dir / "segformer_pretraining_metrics.npz", rows=np.array(rows, dtype=object))
    return {
        "metrics_csv": csv_path,
        "plot_path": plot_path,
        "log_plot_path": log_plot_path,
    }


def build_pretraining_scheduler(optimizer, cfg, steps_per_epoch):
    """Build a scheduler and report whether it advances per optimizer step."""
    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    epochs = max(1, int(cfg["epochs"]))
    if scheduler_name in ("none", "constant"):
        return None, False
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unknown SegFormer pretraining scheduler: {scheduler_name}")

    total_steps = max(1, epochs * max(1, int(steps_per_epoch)))
    warmup_steps = max(0, int(cfg.get("warmup_epochs", 0)) * max(1, int(steps_per_epoch)))
    warmup_start_factor = float(cfg.get("warmup_start_factor", 0.1))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
    if not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def lr_factor(step):
        step = min(max(0, int(step)), total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / max(1, warmup_steps)
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor), True


def checkpoint_path_for_metric(primary_path, metric):
    return primary_path.with_name(primary_path.stem + f"_best_{metric}" + primary_path.suffix)


def evaluate_model(model, dataloader, criterion, dice_weight, threshold, device):
    model.eval()
    totals = {
        "val_loss": 0.0,
        "val_bce_loss": 0.0,
        "val_dice_loss": 0.0,
        "dice_score": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for gradient, target in dataloader:
            gradient = gradient.to(device)
            target = target.to(device)
            logits = model.forward_logits(gradient)
            bce_loss = criterion(logits, target)
            dice_loss = dice_loss_from_logits(logits, target)
            loss = bce_loss + float(dice_weight) * dice_loss
            metrics = segmentation_metrics(logits, target, threshold)
            totals["val_loss"] += float(loss.detach().cpu())
            totals["val_bce_loss"] += float(bce_loss.detach().cpu())
            totals["val_dice_loss"] += float(dice_loss.detach().cpu())
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def load_segformer_checkpoint(path, device):
    checkpoint = torch.load(Path(path), map_location=device)
    model = NN.GradientSegFormer(
        spec=checkpoint["architecture"],
        gamma_min=checkpoint["gamma_min"],
        void_prior=checkpoint["void_prior"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def pretrain_segformer(config, data_dir=None, output_dir=None, progress_callback=None, run_dir=None):
    params = simulation_parameters(config)
    cfg = config["segformer_pretraining"]
    model_cfg = config.get("models", {}).get("segformer", {})
    device = get_device()

    torch.manual_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))

    Nx = params["Nx"]
    Ny = params["Ny"]
    destination = Path(data_dir or config["paths"]["train_data"])
    output_dir = io.ensure_dir(output_dir or config["paths"]["pretrained_models"])

    number_of_samples = int(cfg["numberOfSamples"])
    available_samples = int(cfg["availableSamples"])
    if "sample_ids" in cfg:
        sample_ids = [int(item) for item in cfg["sample_ids"][:number_of_samples]]
        if len(sample_ids) != number_of_samples:
            raise ValueError("segformer_pretraining.sample_ids must contain at least numberOfSamples entries")
    else:
        sample_ids = random.sample(range(available_samples), number_of_samples)
    gradients = torch.zeros((number_of_samples, 1, Nx + 1, Ny + 1), dtype=torch.float32)
    masks = torch.zeros((number_of_samples, 1, Nx + 1, Ny + 1), dtype=torch.float32)

    print(f"Loading {number_of_samples} SegFormer pretraining sample(s) from {destination}", flush=True)
    for row, sample_id in enumerate(sample_ids):
        if row == 0 or (row + 1) % max(1, number_of_samples // 20) == 0 or row + 1 == number_of_samples:
            print(f"Loading sample {row + 1}/{number_of_samples}: material{sample_id}.h5, gradient{sample_id}.h5", flush=True)
        gradients[row, 0] = torch.tensor(io.load_hdf(destination / f"gradient{sample_id}.h5"), dtype=torch.float32)
        gamma = torch.tensor(io.load_hdf(destination / f"material{sample_id}.h5"), dtype=torch.float32)
        masks[row, 0] = (gamma <= float(cfg["void_gamma_threshold"])).float()

    norm_cfg = dict(cfg.get("gradient_normalization", {}))
    gradients = NN.normalize_gradient_for_transformer(gradients, **norm_cfg)

    dataset = NN.FWIDataset(gradients, masks, device)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(int(cfg["split_seed"])),
    )
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=max(1, min(len(val_set), batch_size)))

    if run_dir is not None:
        outputs_dir = io.ensure_dir(Path(run_dir) / "outputs")
        np.savetxt(outputs_dir / "segformer_pretraining_sample_ids.txt", sample_ids, fmt="%d")
        np.savetxt(outputs_dir / "segformer_pretraining_training_subset_indices.txt", train_set.indices, fmt="%d")
        np.savetxt(outputs_dir / "segformer_pretraining_validation_subset_indices.txt", val_set.indices, fmt="%d")

    model = NN.GradientSegFormer(
        spec=model_cfg,
        gamma_min=params["gamma0"],
        void_prior=float(cfg["void_prior"]),
    ).to(device)

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
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    epochs = int(cfg["epochs"])
    scheduler, scheduler_per_step = build_pretraining_scheduler(
        optimizer,
        cfg,
        steps_per_epoch=len(train_loader),
    )
    dice_weight = float(cfg["dice_weight"])
    eval_threshold = float(cfg["eval_threshold"])
    rows = []
    primary_path = Path(output_dir) / (
        "model_"
        + cfg.get("model_type", "SegFormer")
        + "_"
        + str(epochs)
        + "_"
        + cfg.get("trainingType", "segmentation")
        + "_"
        + str(number_of_samples)
        + ".pt"
    )
    metric_paths = {
        "val_loss": checkpoint_path_for_metric(primary_path, "val_loss"),
        "dice_score": checkpoint_path_for_metric(primary_path, "dice_score"),
    }
    best_metrics = {"val_loss": float("inf"), "dice_score": -float("inf")}
    checkpoint_selection = str(cfg.get("checkpoint_selection", "val_loss"))
    if checkpoint_selection not in best_metrics:
        raise ValueError(
            "segformer_pretraining.checkpoint_selection must be val_loss or dice_score"
        )
    early_stopping_patience = cfg.get("early_stopping_patience")
    early_stopping_patience = (
        None if early_stopping_patience is None else int(early_stopping_patience)
    )
    minimum_epochs = int(cfg.get("minimum_epochs", 1))
    selection_epochs_without_improvement = 0

    start = time.perf_counter()
    print_every_batches = max(1, int(cfg.get("print_every_batches", 1)))
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_dice = 0.0
        train_grad_norm = 0.0
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
            if (
                batch == 0
                or batch + 1 == len(train_loader)
                or (batch + 1) % print_every_batches == 0
            ):
                print(
                    "Epoch {}/{} batch {}/{} train_loss={:.6E}".format(
                        epoch + 1,
                        epochs,
                        batch + 1,
                        len(train_loader),
                        float(loss.detach().cpu()),
                    ),
                    flush=True,
                )
        if scheduler is not None and not scheduler_per_step:
            scheduler.step()

        train_batches = max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, criterion, dice_weight, eval_threshold, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_batches,
            "train_bce_loss": train_bce / train_batches,
            "train_dice_loss": train_dice / train_batches,
            "train_grad_norm": train_grad_norm / train_batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **val_metrics,
        }
        rows.append(row)

        improved_metrics = []
        for metric in best_metrics:
            improved = (
                row[metric] < best_metrics[metric]
                if metric == "val_loss"
                else row[metric] > best_metrics[metric]
            )
            if not improved:
                continue
            best_metrics[metric] = row[metric]
            improved_metrics.append(metric)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": model.architecture_dict(),
                "gamma_min": float(params["gamma0"]),
                "void_prior": float(cfg["void_prior"]),
                "gradient_normalization": norm_cfg,
                "void_gamma_threshold": float(cfg["void_gamma_threshold"]),
                "training_config": dict(cfg),
                "epoch": epoch + 1,
                "validation_metrics": val_metrics,
                "checkpoint_metric": metric,
                "checkpoint_metric_value": float(row[metric]),
            }
            torch.save(checkpoint, metric_paths[metric])
            if metric == checkpoint_selection:
                torch.save(checkpoint, primary_path)

        if checkpoint_selection in improved_metrics:
            selection_epochs_without_improvement = 0
        else:
            selection_epochs_without_improvement += 1

        elapsed = time.perf_counter() - start
        print(
            "Epoch: {}/{}\tTrain loss: {:.6E}\tVal loss: {:.6E}\tDice: {:.4f}\tIoU: {:.4f}\tElapsed: {:.2f}s".format(
                epoch,
                epochs - 1,
                row["train_loss"],
                row["val_loss"],
                row["dice_score"],
                row["iou"],
                elapsed,
            ),
            flush=True,
        )
        start = time.perf_counter()

        if progress_callback is not None:
            progress_callback(epoch, epochs, row["train_loss"], row["val_loss"])

        if (
            early_stopping_patience is not None
            and epoch + 1 >= minimum_epochs
            and selection_epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping after epoch {}: {} did not improve for {} epoch(s).".format(
                    epoch + 1,
                    checkpoint_selection,
                    selection_epochs_without_improvement,
                ),
                flush=True,
            )
            break

    if run_dir is not None and rows:
        paths = save_segformer_pretraining_outputs(run_dir, rows)
        for key, value in paths.items():
            print(f"{key}: {value}", flush=True)
    return primary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = io.create_run_dir(
        io.ensure_dir(config["paths"].get("runs", "runs")) / "pretraining_segformer",
        prefix="pretraining_segformer",
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
    shutil.copy2(args.config, run_dir / "config.yaml")

    start = time.perf_counter()
    model_path = pretrain_segformer(config, run_dir=run_dir)
    elapsed = time.perf_counter() - start
    (run_dir / "runtime.txt").write_text(
        "run_type: pretraining_segformer\nmodel_path: {}\nruntime_seconds: {:.6f}\n".format(
            model_path,
            elapsed,
        ),
        encoding="utf-8",
    )
    print("Saved model to {}".format(model_path))
    print("Saved SegFormer pretraining run to {}".format(run_dir))


if __name__ == "__main__":
    main()
