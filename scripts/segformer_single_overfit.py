"""Overfit plain SegFormer on one material/gradient pair.

This is a capacity sanity check, not a generalization experiment. The same
sample is used for training and evaluation, with no train/validation split.
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import _bootstrap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src import io
from src.config import load_config
from src.experiments.base import get_device, simulation_parameters
from src.networks import normalize_gradient_for_transformer
from src.pretrain_segformer import (
    build_pretraining_scheduler,
    build_segformer_model,
    dice_loss_from_logits,
    segmentation_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train plain SegFormer from scratch on one sample."
    )
    parser.add_argument("--config", default="configs/extended.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder containing matching material<ID>.h5 and gradient<ID>.h5 files.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Use a specific sample instead of choosing one randomly.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=7500,
        help="Number of optimizer updates; one sample means one update per epoch.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="For example: cuda or cpu.")
    parser.add_argument(
        "--run-dir",
        default="runs/improve_transformer/single_overfit",
        help="Parent folder for the timestamped run.",
    )
    parser.add_argument("--print-every", type=int, default=None)
    return parser.parse_args()


def available_sample_ids(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    sample_ids = []
    for material_path in data_dir.glob("material*.h5"):
        suffix = material_path.stem.removeprefix("material")
        if suffix.isdigit() and (data_dir / f"gradient{suffix}.h5").exists():
            sample_ids.append(int(suffix))
    if not sample_ids:
        raise FileNotFoundError(
            f"No matching material<ID>.h5 and gradient<ID>.h5 pairs in {data_dir}"
        )
    return sorted(sample_ids)


def make_run_dir(parent):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(parent) / f"segformer_single_overfit_{stamp}"
    for name in ("figures", "histories", "outputs", "checkpoints"):
        io.ensure_dir(run_dir / name)
    return run_dir


def make_criterion(target, cfg, device):
    configured = cfg.get("bce_pos_weight", "auto")
    if configured == "auto":
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / torch.clamp(positive, min=1.0)).to(device)
    elif configured is None:
        pos_weight = None
    else:
        pos_weight = torch.tensor(float(configured), device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight), pos_weight


def evaluate(model, gradient, target_mask, target_gamma, criterion, dice_weight, threshold):
    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(gradient)
        bce = criterion(logits, target_mask)
        dice_loss = dice_loss_from_logits(logits, target_mask)
        loss = bce + dice_weight * dice_loss
        gamma = model(gradient)
        gamma_mse = torch.mean((gamma - target_gamma) ** 2)
        metrics = segmentation_metrics(logits, target_mask, threshold=threshold)
    return {
        "loss": float(loss.cpu()),
        "bce_loss": float(bce.cpu()),
        "dice_loss": float(dice_loss.cpu()),
        "gamma_mse": float(gamma_mse.cpu()),
        **metrics,
    }, logits.detach(), gamma.detach()


def rectangle_contrast(gamma, edge_width=8, interior_width=20):
    image = gamma[0, 0]
    height, width = image.shape
    rows = torch.arange(height, device=image.device)[:, None]
    columns = torch.arange(width, device=image.device)[None, :]
    distance = torch.minimum(
        torch.minimum(rows, height - 1 - rows),
        torch.minimum(columns, width - 1 - columns),
    )
    return float(
        (image[distance < edge_width].mean() - image[distance >= interior_width].mean())
        .detach()
        .cpu()
    )


def plot_loss(rows, path):
    epochs = [row["epoch"] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, [row["loss"] for row in rows], label="Total Loss", linewidth=2)
    axis.plot(epochs, [row["bce_loss"] for row in rows], label="Weighted BCE", alpha=0.85)
    axis.plot(epochs, [row["dice_loss"] for row in rows], label="Dice Loss", alpha=0.85)
    axis.set_title("Single-Sample SegFormer Training Loss")
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_progress(snapshots, target_gamma, path):
    items = [(f"Epoch {epoch}", gamma) for epoch, gamma in sorted(snapshots.items())]
    items.append(("Target", target_gamma))
    columns = 3
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    image = None
    for axis, (title, gamma) in zip(axes, items):
        image = axis.imshow(np.transpose(gamma), vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(items) :]:
        axis.axis("off")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.85)
        colorbar.set_label("Gamma")
    fig.suptitle("Single-Sample Gamma Reconstruction Progress")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_evaluation(normalized_gradient, target_gamma, predicted_gamma, predicted_voidness, zero_gamma, path):
    arrays = [
        normalized_gradient,
        target_gamma,
        predicted_gamma,
        np.abs(predicted_gamma - target_gamma),
        predicted_voidness,
        zero_gamma,
    ]
    titles = [
        "Normalized Gradient",
        "Target Gamma",
        "Predicted Gamma",
        "Absolute Gamma Error",
        "Predicted Void Probability",
        "Zero-Input Gamma",
    ]
    limits = [None, (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, array, title, limit in zip(axes.ravel(), arrays, titles, limits):
        kwargs = {} if limit is None else {"vmin": limit[0], "vmax": limit[1]}
        image = axis.imshow(np.transpose(array), **kwargs)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Single-Sample SegFormer Evaluation")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    config = load_config(args.config)
    cfg = dict(config["segformer_pretraining"])
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    learning_rate = float(cfg["lr"] if args.learning_rate is None else args.learning_rate)
    data_dir = Path(args.data_dir or config["paths"]["train_data"])
    candidates = available_sample_ids(data_dir)
    sample_id = args.sample_id
    if sample_id is None:
        sample_id = random.Random(seed).choice(candidates)
    if sample_id not in candidates:
        raise FileNotFoundError(
            f"Sample {sample_id} has no matching material/gradient pair in {data_dir}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(args.device)
    run_dir = make_run_dir(args.run_dir)
    print(f"Device: {device}")
    print(f"Selected sample: {sample_id}")
    print(f"Data directory: {data_dir}")
    print(f"Run directory: {run_dir}")

    params = simulation_parameters(config)
    target_gamma = torch.tensor(
        io.load_hdf(data_dir / f"material{sample_id}.h5"),
        dtype=torch.float32,
        device=device,
    )[None, None]
    gradient = torch.tensor(
        io.load_hdf(data_dir / f"gradient{sample_id}.h5"),
        dtype=torch.float32,
        device=device,
    )[None, None]
    target_mask = (target_gamma <= float(cfg["void_gamma_threshold"])).float()
    norm_cfg = dict(cfg.get("gradient_normalization", {}))
    normalized_gradient = normalize_gradient_for_transformer(gradient, **norm_cfg)

    expected_shape = (params["Nx"] + 1, params["Ny"] + 1)
    if tuple(target_gamma.shape[-2:]) != expected_shape:
        raise ValueError(
            f"Expected material shape {expected_shape}, got {tuple(target_gamma.shape[-2:])}"
        )

    model, variant = build_segformer_model(
        config,
        params,
        model_variant="segformer",
    )
    if variant != "segformer":
        raise RuntimeError(f"Expected plain SegFormer, got {variant}")
    model = model.to(device)

    criterion, pos_weight = make_criterion(target_mask, cfg, device)
    dice_weight = float(cfg["dice_weight"])
    threshold = float(cfg["eval_threshold"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler_cfg = {**cfg, "epochs": int(args.epochs)}
    scheduler, scheduler_per_step = build_pretraining_scheduler(
        optimizer,
        scheduler_cfg,
        steps_per_epoch=1,
    )

    snapshot_epochs = {
        0,
        max(1, args.epochs // 10),
        max(1, args.epochs // 4),
        max(1, args.epochs // 2),
        max(1, 3 * args.epochs // 4),
        args.epochs,
    }
    snapshots = {}
    initial_metrics, _, initial_gamma = evaluate(
        model,
        normalized_gradient,
        target_mask,
        target_gamma,
        criterion,
        dice_weight,
        threshold,
    )
    snapshots[0] = initial_gamma[0, 0].cpu().numpy()

    rows = []
    print_every = args.print_every or max(1, args.epochs // 20)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(normalized_gradient)
        bce_loss = criterion(logits, target_mask)
        dice_loss = dice_loss_from_logits(logits, target_mask)
        loss = bce_loss + dice_weight * dice_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(cfg["clipGrad"]),
        )
        optimizer.step()
        if scheduler is not None and scheduler_per_step:
            scheduler.step()

        metrics, _, gamma = evaluate(
            model,
            normalized_gradient,
            target_mask,
            target_gamma,
            criterion,
            dice_weight,
            threshold,
        )
        rows.append(
            {
                "epoch": epoch,
                **metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": float(grad_norm.detach().cpu()),
            }
        )
        if epoch in snapshot_epochs:
            snapshots[epoch] = gamma[0, 0].cpu().numpy()
        if epoch == 1 or epoch % print_every == 0 or epoch == args.epochs:
            print(
                "Epoch {}/{}: loss={:.6E}, gamma_mse={:.6E}, "
                "dice={:.4f}, iou={:.4f}".format(
                    epoch,
                    args.epochs,
                    metrics["loss"],
                    metrics["gamma_mse"],
                    metrics["dice_score"],
                    metrics["iou"],
                ),
                flush=True,
            )
        if scheduler is not None and not scheduler_per_step:
            scheduler.step()

    elapsed = time.perf_counter() - start
    final_metrics, final_logits, final_gamma = evaluate(
        model,
        normalized_gradient,
        target_mask,
        target_gamma,
        criterion,
        dice_weight,
        threshold,
    )
    with torch.no_grad():
        zero_gamma = model(torch.zeros_like(normalized_gradient))
    predicted_voidness = torch.sigmoid(final_logits)

    initial_contrast = rectangle_contrast(initial_gamma)
    final_contrast = rectangle_contrast(final_gamma)
    target_contrast = rectangle_contrast(target_gamma)
    zero_contrast = rectangle_contrast(zero_gamma)

    with (run_dir / "histories" / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_loss(rows, run_dir / "figures" / "training_loss.png")
    plot_progress(
        snapshots,
        target_gamma[0, 0].detach().cpu().numpy(),
        run_dir / "figures" / "gamma_progress.png",
    )
    plot_evaluation(
        normalized_gradient[0, 0].detach().cpu().numpy(),
        target_gamma[0, 0].detach().cpu().numpy(),
        final_gamma[0, 0].cpu().numpy(),
        predicted_voidness[0, 0].cpu().numpy(),
        zero_gamma[0, 0].cpu().numpy(),
        run_dir / "figures" / "same_sample_evaluation.png",
    )

    io.save_hdf(
        run_dir / "outputs" / "target_gamma.h5",
        target_gamma[0, 0].detach().cpu().numpy(),
        key="gamma",
    )
    io.save_hdf(
        run_dir / "outputs" / "predicted_gamma.h5",
        final_gamma[0, 0].cpu().numpy(),
        key="gamma",
    )
    io.save_hdf(
        run_dir / "outputs" / "predicted_void_probability.h5",
        predicted_voidness[0, 0].cpu().numpy(),
        key="void_probability",
    )
    io.save_hdf(
        run_dir / "outputs" / "zero_input_gamma.h5",
        zero_gamma[0, 0].cpu().numpy(),
        key="gamma",
    )

    checkpoint_path = run_dir / "checkpoints" / "segformer_single_overfit_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture_dict(),
            "model_variant": "segformer",
            "gamma_min": float(params["gamma0"]),
            "void_prior": float(cfg["void_prior"]),
            "gradient_normalization": norm_cfg,
            "sample_id": sample_id,
            "epochs": args.epochs,
            "training_config": scheduler_cfg,
            "final_metrics": final_metrics,
        },
        checkpoint_path,
    )

    summary = {
        "sample_id": sample_id,
        "data_dir": str(data_dir),
        "device": str(device),
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "pos_weight": None if pos_weight is None else float(pos_weight.cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "runtime_seconds": elapsed,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "rectangle_contrast": {
            "target": target_contrast,
            "initial": initial_contrast,
            "final": final_contrast,
            "zero_input": zero_contrast,
        },
        "checkpoint": str(checkpoint_path),
    }
    with (run_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print("\nFinished single-sample overfit test")
    print(f"Runtime: {elapsed:.2f}s")
    print(f"Final loss: {final_metrics['loss']:.6E}")
    print(f"Final gamma MSE: {final_metrics['gamma_mse']:.6E}")
    print(f"Final Dice: {final_metrics['dice_score']:.4f}")
    print(f"Final IoU: {final_metrics['iou']:.4f}")
    print(
        "Rectangle contrast (target / initial / final / zero): "
        f"{target_contrast:.5f} / {initial_contrast:.5f} / "
        f"{final_contrast:.5f} / {zero_contrast:.5f}"
    )
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
