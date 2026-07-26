"""Quickly test the mask capacity of different SegFormer output grids.

This is an oracle representation test, not a network-training experiment. For
each requested resolution it directly optimizes a free logit grid, bilinearly
upsamples that grid to the material-mask size, and measures how closely the
result can reproduce the target mask.

Example:
    python scripts/temp_segformer_output_resolution_oracle.py \
        --data-dir data/test --sample-count 32
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the representational ceiling of coarse SegFormer output grids."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument(
        "--resolutions",
        default="64x32,128x64,256x128",
        help="Comma-separated HxW logit-grid resolutions.",
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--void-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def numeric_case_id(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else path.stem


def parse_resolutions(value):
    resolutions = []
    for item in value.split(","):
        height, width = item.strip().lower().split("x", maxsplit=1)
        resolution = (int(height), int(width))
        if resolution[0] < 1 or resolution[1] < 1:
            raise ValueError(f"Invalid resolution: {item}")
        resolutions.append(resolution)
    if not resolutions:
        raise ValueError("At least one output resolution is required")
    return resolutions


def load_pandas_hdf_matrix(path):
    """Read a 2D matrix from the simple pandas HDF files used by this project."""
    with h5py.File(path, "r") as handle:
        datasets = []

        def collect(name, value):
            if isinstance(value, h5py.Dataset) and value.ndim == 2:
                datasets.append((name, value[()]))

        handle.visititems(collect)

    if not datasets:
        raise ValueError(f"No 2D dataset found in {path}")

    block_values = [array for name, array in datasets if name.endswith("block0_values")]
    return np.asarray(block_values[0] if block_values else datasets[0][1])


def load_masks(data_dir, sample_count, seed, void_threshold):
    material_paths = sorted(data_dir.glob("material*.h5"), key=numeric_case_id)
    if not material_paths:
        raise FileNotFoundError(f"No material*.h5 files found in {data_dir}")

    rng = random.Random(seed)
    selected = rng.sample(material_paths, min(sample_count, len(material_paths)))
    selected.sort(key=numeric_case_id)

    masks = []
    expected_shape = None
    for path in selected:
        gamma = load_pandas_hdf_matrix(path)
        if expected_shape is None:
            expected_shape = gamma.shape
        elif gamma.shape != expected_shape:
            raise ValueError(
                f"Inconsistent mask shapes: expected {expected_shape}, "
                f"got {gamma.shape} in {path}"
            )
        masks.append(gamma <= void_threshold)

    target = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()
    return target, [numeric_case_id(path) for path in selected]


def soft_dice_loss(logits, target, eps=1.0e-6):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def calculate_metrics(logits, target, threshold, eps=1.0e-8):
    probability = torch.sigmoid(logits)
    prediction = probability >= threshold
    target_bool = target >= 0.5

    intersection = (prediction & target_bool).sum(dim=(1, 2, 3)).float()
    pred_count = prediction.sum(dim=(1, 2, 3)).float()
    target_count = target_bool.sum(dim=(1, 2, 3)).float()
    union = pred_count + target_count - intersection

    dice = (2.0 * intersection + eps) / (pred_count + target_count + eps)
    iou = (intersection + eps) / (union + eps)
    mse = ((probability - target) ** 2).mean(dim=(1, 2, 3))
    return {
        "dice": dice.mean().item(),
        "dice_min": dice.min().item(),
        "iou": iou.mean().item(),
        "mse": mse.mean().item(),
    }


def optimize_resolution(
    target,
    resolution,
    steps,
    learning_rate,
    dice_weight,
    prediction_threshold,
):
    target_size = target.shape[-2:]
    initial_probability = F.interpolate(target, size=resolution, mode="area")
    initial_probability = initial_probability.clamp(0.01, 0.99)
    coarse_logits = torch.nn.Parameter(torch.logit(initial_probability))

    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).detach()
    optimizer = torch.optim.Adam([coarse_logits], lr=learning_rate)

    start = time.perf_counter()
    best_loss = float("inf")
    best_logits = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        full_logits = F.interpolate(
            coarse_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        bce = F.binary_cross_entropy_with_logits(
            full_logits,
            target,
            pos_weight=pos_weight,
        )
        loss = bce + dice_weight * soft_dice_loss(full_logits, target)
        loss.backward()
        optimizer.step()

        current_loss = loss.detach().item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_logits = coarse_logits.detach().clone()

    if target.is_cuda:
        torch.cuda.synchronize(target.device)
    elapsed = time.perf_counter() - start

    full_logits = F.interpolate(
        best_logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    metrics = calculate_metrics(full_logits, target, prediction_threshold)
    metrics.update(loss=best_loss, seconds=elapsed)
    return metrics


def main():
    args = parse_args()
    if args.sample_count < 1 or args.steps < 1:
        raise ValueError("sample-count and steps must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    resolutions = parse_resolutions(args.resolutions)
    target, case_ids = load_masks(
        args.data_dir,
        args.sample_count,
        args.seed,
        args.void_threshold,
    )
    target = target.to(device)

    print(
        f"Loaded {len(case_ids)} masks with shape {tuple(target.shape[-2:])}; "
        f"device={device}; steps={args.steps}"
    )
    print(f"Case IDs: {case_ids}")
    print()
    print("Grid       Dice mean  Dice min   IoU mean   Prob. MSE    Loss       Time")
    print("---------  ---------  ---------  ---------  -----------  ---------  -------")

    results = {}
    for resolution in resolutions:
        metrics = optimize_resolution(
            target,
            resolution,
            args.steps,
            args.learning_rate,
            args.dice_weight,
            args.prediction_threshold,
        )
        results[resolution] = metrics
        name = f"{resolution[0]}x{resolution[1]}"
        print(
            f"{name:<9}  {metrics['dice']:<9.6f}  {metrics['dice_min']:<9.6f}  "
            f"{metrics['iou']:<9.6f}  {metrics['mse']:<11.8f}  "
            f"{metrics['loss']:<9.6f}  {metrics['seconds']:.2f}s"
        )

    baseline = results[resolutions[0]]
    best_resolution = max(resolutions, key=lambda item: results[item]["dice"])
    best = results[best_resolution]
    print()
    print(
        "Best Dice improvement over the first grid: "
        f"{best['dice'] - baseline['dice']:+.6f} "
        f"({best_resolution[0]}x{best_resolution[1]})."
    )
    print(
        "Interpretation: a material improvement at higher resolution indicates "
        "that the current output grid imposes a representational ceiling."
    )


if __name__ == "__main__":
    main()
