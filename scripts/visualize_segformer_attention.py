"""
Visualize what a trained SegFormer attends to when it reads an adjoint gradient.

READ THIS BEFORE USING THE FIGURES IN THE THESIS
------------------------------------------------
SegFormer does NOT use dense self-attention. Following PVT, each stage reduces
the key/value resolution by its SR ratio (8, 4, 2, 1 here), so every block
attends from a full query grid to only ~32 spatially-reduced key locations.
These are therefore NOT the per-pixel attention maps usually shown for plain
ViT, and describing them that way would be wrong. What they legitimately show:

  * WHERE each output location looks (query-side maps, at that stage's
    resolution), and
  * HOW BROADLY it looks (attention entropy per query -- low entropy means the
    query concentrates on a few key regions, high entropy means it spreads).

Three products, chosen to say something the loss curves cannot:

  1. entropy maps per stage -- does the model become more local or more global
     with depth?
  2. query-focused maps -- for a query placed ON a void and one placed in intact
     material, which key regions does each attend to?
  3. an optional A/B between two checkpoints (e.g. ImageNet-initialized vs
     randomly-initialized MiT-B0), showing whether pretraining changes where the
     network looks, not merely how well it scores.

This script is READ-ONLY with respect to the experiments: it loads checkpoints
and writes figures under its own output directory.

Example
-------
    .venv/Scripts/python.exe scripts/visualize_segformer_attention.py \
        --checkpoint models/improve_transformer/model_SegFormer_100_segmentation_15000.pt \
        --case 1
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import networks as NN
from src.config import load_config
from src.experiments.base import (get_device, load_case_data, normalize_input_data,
                                  simulation_parameters)
from src.io import create_run_dir, ensure_dir
from src.pretrain_segformer import load_segformer_checkpoint


def load_with_eager_attention(checkpoint_path, device):
    """Reload a checkpoint into an eager-attention copy so maps can be extracted."""
    model, checkpoint = load_segformer_checkpoint(checkpoint_path, device)
    spec = dict(checkpoint["architecture"])
    spec["attn_implementation"] = "eager"
    eager = NN.GradientSegFormer(
        spec=spec,
        gamma_min=checkpoint["gamma_min"],
        void_prior=checkpoint["void_prior"],
    ).to(device)
    eager.load_state_dict(model.state_dict())
    eager.eval()
    del model
    return eager, checkpoint


def attention_entropy(attention):
    """Normalized entropy per query, in [0, 1]. 0 = one key, 1 = uniform."""
    probabilities = attention.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(-1)      # [B, heads, queries]
    return (entropy / np.log(attention.shape[-1])).mean(1)[0]      # mean over heads


def plot_entropy_maps(path, attentions, shapes, title):
    blocks = len(attentions)
    fig, axes = plt.subplots(1, blocks, figsize=(2.5 * blocks, 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, attention, (h, w), index in zip(axes, attentions, shapes, range(blocks)):
        entropy = attention_entropy(attention).reshape(h, w).detach().cpu().numpy()
        image = ax.imshow(entropy.T, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"block {index}\n{h}x{w}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=axes.tolist(), shrink=0.8, label="attention entropy (0=focused, 1=uniform)")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_query_maps(path, attentions, shapes, gradient, target_gamma, title):
    """For a query on a void and one in intact material: which keys does it use?"""
    void = target_gamma <= 0.5
    if not void.any():
        raise ValueError("case has no void cells; cannot place a void query")
    vy, vx = np.argwhere(void)[len(np.argwhere(void)) // 2]
    iy, ix = np.argwhere(~void)[len(np.argwhere(~void)) // 2]

    blocks = len(attentions)
    fig, axes = plt.subplots(2, blocks + 1, figsize=(2.4 * (blocks + 1), 5.2),
                             constrained_layout=True)
    for row, (py, px, label) in enumerate([(vy, vx, "query ON void"),
                                           (iy, ix, "query in intact material")]):
        limit = np.abs(gradient).max() or 1.0
        axes[row][0].imshow(gradient.T, cmap="seismic", vmin=-limit, vmax=limit, aspect="auto")
        axes[row][0].plot(py, px, "o", mfc="none", mec="lime", ms=11, mew=2)
        axes[row][0].set_ylabel(label, fontsize=9)
        if row == 0:
            axes[row][0].set_title("input gradient", fontsize=9)
        axes[row][0].set_xticks([]); axes[row][0].set_yticks([])

        for column, (attention, (h, w)) in enumerate(zip(attentions, shapes), start=1):
            qy = min(int(py * h / gradient.shape[0]), h - 1)
            qx = min(int(px * w / gradient.shape[1]), w - 1)
            weights = attention[0, :, qy * w + qx, :].mean(0).detach().cpu().numpy()
            axes[row][column].bar(np.arange(len(weights)), weights, color="#2f5aa8")
            axes[row][column].set_ylim(0, max(weights.max() * 1.1, 1e-6))
            if row == 0:
                axes[row][column].set_title(f"block {column-1}\n{len(weights)} keys", fontsize=9)
            axes[row][column].set_xticks([])
            axes[row][column].tick_params(labelsize=6)
    fig.suptitle(title + "\n(keys are spatially reduced by the SR ratio, so the key axis is coarse)",
                 fontsize=10)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def mean_entropy_by_block(attentions):
    return [float(attention_entropy(a).mean()) for a in attentions]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-checkpoint", default=None,
                        help="Optional second checkpoint for an A/B (e.g. ImageNet vs random init).")
    parser.add_argument("--labels", default=None, help="Comma-separated names for the two checkpoints.")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="runs/segformer_attention")
    args = parser.parse_args()

    config = load_config(args.config)
    params = simulation_parameters(config)
    device = get_device()
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])

    gamma, initial_gradient, _, _ = load_case_data(
        args.case, data_dir, params, device, load_gradient=True)
    inputs = normalize_input_data(initial_gradient, config).to(device)
    gradient_np = initial_gradient[0, 0].detach().cpu().numpy()
    target_np = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()

    checkpoints = [args.checkpoint] + ([args.compare_checkpoint] if args.compare_checkpoint else [])
    labels = (args.labels.split(",") if args.labels
              else [Path(c).stem[:34] for c in checkpoints])
    if len(labels) != len(checkpoints):
        raise ValueError("--labels count must match the number of checkpoints")

    run_dir = create_run_dir(ensure_dir(args.output_dir), prefix=f"attention_case{args.case}")
    ensure_dir(run_dir / "figures")

    summary = {}
    for path, label in zip(checkpoints, labels):
        model, checkpoint = load_with_eager_attention(path, device)
        with torch.no_grad():
            _, attentions = model.forward_attentions(inputs)
        shapes = model.attention_query_shapes(inputs.shape[-2], inputs.shape[-1])
        safe = label.replace(" ", "_").replace("/", "_")
        plot_entropy_maps(run_dir / "figures" / f"entropy_{safe}.png",
                          attentions, shapes, f"{label}: attention entropy per query")
        plot_query_maps(run_dir / "figures" / f"queries_{safe}.png",
                        attentions, shapes, gradient_np, target_np,
                        f"{label}: where a single query attends")
        summary[label] = mean_entropy_by_block(attentions)
        print(f"{label}: trained {checkpoint.get('epoch','?')} epochs, "
              f"init={checkpoint.get('initialization','n/a')}")
        print("  mean entropy per block: " +
              ", ".join(f"{v:.3f}" for v in summary[label]), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for (label, values), colour in zip(summary.items(), ["#2f5aa8", "#b64040"]):
        ax.plot(range(len(values)), values, marker="o", linewidth=2, color=colour, label=label)
    ax.set_xlabel("transformer block (shallow to deep)")
    ax.set_ylabel("mean attention entropy")
    ax.set_title("Does attention become more focused with depth?")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(run_dir / "figures" / "entropy_by_depth.png", dpi=170)
    plt.close(fig)

    print(f"\nSaved figures to {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
