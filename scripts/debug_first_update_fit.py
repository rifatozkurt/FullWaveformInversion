import argparse
import csv
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.experiments.base import (
    load_case_data,
    model_path,
    normalize_input_data,
    simulation_parameters,
    tensor_norm,
    tensor_stats,
)
from src.io import ensure_dir, save_history
from src.networks import (
    INR,
    INR_IG,
    INR_IG_CENTERED,
    INR_LR,
    INR_MPE,
    INR_MPE_CENTERED,
    INRSIREN,
    INRSIREN_CENTERED,
    GradientSegFormer,
    Unet,
    initWeights,
    normalize_gradient_for_transformer,
)
from src.pretrain_segformer import load_segformer_checkpoint


ALIASES = {
    "unet": "transfer_learning_fwi",
    "segformer": "transfer_segformer_fwi",
    "transformer": "transfer_segformer_fwi",
    "inr": "inr_fwi",
    "siren": "inr_siren_fwi",
    "siren_centered": "inr_siren_centered_fwi",
    "lr": "inr_lr_fwi",
    "mpe": "inr_mpe_fwi",
    "mpe_centered": "inr_mpe_centered_fwi",
    "ig": "inr_ig_fwi",
    "ig_centered": "inr_ig_centered_fwi",
}
METHODS = tuple(ALIASES) + tuple(ALIASES.values())
DEFAULT_SAVED_EPOCHS = "0,1,5,10,25,50,100"


def make_coords(params, device):
    x = torch.linspace(-1, 1, params["Nx"] + 1, device=device)
    y = torch.linspace(-1, 1, params["Ny"] + 1, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def build_model(method, cfg, gamma0, device):
    if method == "transfer_learning_fwi":
        channels = cfg["NNchannels"]
        blocks = int(cfg["numberOfConvolutionsPerBlock"])
        model = Unet(channels, blocks, gamma0, bnorm=True)
        initWeights(model)
        torch.nn.init.normal_(model.convolutionsUp[-1].weight, std=0.01, mean=0.7)
        model.convolutionsUp[-1].bias.data.fill_(3)
        return model.to(device)
    if method == "transfer_segformer_fwi":
        model = GradientSegFormer(
            spec=cfg["model_spec"],
            gamma_min=gamma0,
            void_prior=float(cfg.get("void_prior", 0.01)),
        )
        return model.to(device)

    common = {
        "gamma0": gamma0,
        "output_mode": cfg.get("output_mode", "voidness"),
        "final_bias": float(cfg.get("final_bias", -5.0)),
    }

    if method == "inr_fwi":
        model = INR(int(cfg["hidden_features"]), int(cfg["hidden_layers"]), **common)
    elif method == "inr_siren_fwi":
        model = INRSIREN(
            int(cfg["hidden_features"]),
            int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            **common,
        )
    elif method == "inr_siren_centered_fwi":
        model = INRSIREN_CENTERED(
            int(cfg["hidden_features"]),
            int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            **common,
        )
    elif method == "inr_lr_fwi":
        model = INR_LR(
            rank_x=int(cfg.get("rank_x", cfg.get("rank", 128))),
            rank_y=int(cfg.get("rank_y", cfg.get("rank", 64))),
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            core_init_std=float(cfg.get("core_init_std", 1e-3)),
            **common,
        )
    elif method in ("inr_mpe_fwi", "inr_mpe_centered_fwi"):
        cls = INR_MPE_CENTERED if method == "inr_mpe_centered_fwi" else INR_MPE
        model = cls(
            num_levels=int(cfg["num_levels"]),
            base_resolution=int(cfg["base_resolution"]),
            per_level_scale=float(cfg["per_level_scale"]),
            features_per_level=int(cfg["features_per_level"]),
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            grid_init_std=float(cfg.get("grid_init_std", 1e-2 if "centered" in method else 1e-4)),
            align_corners=bool(cfg.get("align_corners", True)),
            swap_grid_coords=bool(cfg.get("swap_grid_coords", False)),
            **common,
        )
    elif method in ("inr_ig_fwi", "inr_ig_centered_fwi"):
        cls = INR_IG_CENTERED if method == "inr_ig_centered_fwi" else INR_IG
        model = cls(
            alpha=float(cfg.get("fusion_alpha", 0.5)),
            num_levels=int(cfg["num_levels"]),
            base_resolution=int(cfg["base_resolution"]),
            per_level_scale=float(cfg["per_level_scale"]),
            features_per_level=int(cfg["features_per_level"]),
            grid_init_std=float(cfg.get("grid_init_std", 1e-2 if "centered" in method else 1e-4)),
            align_corners=bool(cfg.get("align_corners", True)),
            swap_grid_coords=bool(cfg.get("swap_grid_coords", False)),
            siren_hidden_features=int(cfg["siren_hidden_features"]),
            siren_hidden_layers=int(cfg["siren_hidden_layers"]),
            siren_out_features=int(cfg["siren_out_features"]),
            omega0=float(cfg.get("omega0", 30)),
            fusion_hidden_features=int(cfg["fusion_hidden_features"]),
            fusion_hidden_layers=int(cfg["fusion_hidden_layers"]),
            **common,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device)


def vlim(image, percentile=99):
    value = float(np.percentile(np.abs(image), percentile))
    return value if value > 0 else 1.0


def plot_image_grid(path, method, case_id, images):
    items = [
        ("gamma initial", images["gamma_initial"], "coolwarm", None),
        ("direct gamma target", images["gamma_direct_target"], "coolwarm", None),
        ("raw adjoint gradient", images["raw_gradient"], "seismic", vlim(images["raw_gradient"])),
        ("effective adjoint gradient", images["effective_gradient"], "seismic", vlim(images["effective_gradient"])),
        ("direct gamma update", images["target_update"], "seismic", vlim(images["target_update"])),
        ("final predicted gamma", images["final_gamma"], "coolwarm", None),
        ("final predicted update", images["final_update"], "seismic", vlim(images["final_update"])),
        ("update error", images["update_error"], "seismic", vlim(images["update_error"])),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    for ax, (title, image, cmap, scale) in zip(axes.ravel(), items):
        kwargs = {"vmin": image.min(), "vmax": 1.0} if scale is None else {"vmin": -scale, "vmax": scale}
        plot = ax.imshow(image.T, origin="lower", cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{method}, case {case_id}: first-update supervised fit")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_history(path, rows):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    plots = [
        (axes[0, 0], "supervised_update_loss", "Supervised update loss", True),
        (axes[0, 1], "gamma_mse", "Gamma MSE to direct target", True),
        (axes[1, 0], "update_mse", "Update MSE", True),
        (axes[1, 1], "pred_update_std", "Update std", True),
    ]
    for ax, key, title, logscale in plots:
        ax.plot(epochs, [row[key] for row in rows], linewidth=2, label=key)
        ax.set_title(title)
        if logscale:
            ax.set_yscale("log")

    axes[1, 1].plot(epochs, [row["target_update_std"] for row in rows], label="target_update_std", linewidth=2)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_update_snapshots(path, snapshots, target_update):
    epochs = sorted(snapshots)
    fig, axes = plt.subplots(3, len(epochs), figsize=(3.2 * len(epochs), 9), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, len(epochs))
    scale = vlim(target_update)

    for col, epoch in enumerate(epochs):
        pred_update = snapshots[epoch]
        for row, title, image in (
            (0, f"pred update e{epoch}", pred_update),
            (1, "target update", target_update),
            (2, "update error", pred_update - target_update),
        ):
            plot = axes[row, col].imshow(image.T, origin="lower", cmap="seismic", vmin=-scale, vmax=scale)
            axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(plot, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.suptitle("Saved supervised update snapshots")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fit an INR to the first direct gamma update implied by one FWI adjoint gradient."
    )
    parser.add_argument("--config", default="configs/experimental.yaml")
    parser.add_argument("--method", default="transfer_segformer_fwi", choices=METHODS)
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="temp/first_update_fit")
    parser.add_argument("--unet-init", default="legacy_random", choices=("pretrained", "legacy_random"))
    parser.add_argument("--segformer-init", default="random", choices=("pretrained", "random"))
    parser.add_argument("--segformer-checkpoint", default=None)
    parser.add_argument(
        "--unet-default-lr",
        type=float,
        default=1e-8,
        help="Used for U-Net supervised fitting when --lr is omitted.",
    )
    parser.add_argument(
        "--segformer-default-lr",
        type=float,
        default=1e-6,
        help="Used for SegFormer supervised fitting when --lr is omitted.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=None, help="Supervised optimizer lr. Defaults to the method config lr.")
    parser.add_argument("--step-mode", default="auto", choices=("auto", "manual"))
    parser.add_argument("--direct-step-scale", type=float, default=1.0)
    parser.add_argument("--target-max-delta-gamma", type=float, default=1e-3)
    parser.add_argument("--loss-normalization", default="target_rms", choices=("target_rms", "target_max", "none"))
    parser.add_argument("--min-loss-scale", type=float, default=1e-8)
    parser.add_argument("--saved-epochs", default=DEFAULT_SAVED_EPOCHS)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    method = ALIASES.get(args.method, args.method)
    cfg = config["experiments"][method]
    exp_cfg = config["experiments"]
    params = simulation_parameters(config)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])

    out = ensure_dir(Path(args.output_dir) / f"{method}_case{args.case}")
    figures = ensure_dir(out / "figures")
    histories = ensure_dir(out / "histories")
    outputs = ensure_dir(out / "outputs")
    saved_epochs = {int(item) for item in args.saved_epochs.split(",") if item.strip()} | {0, args.epochs}

    torch.manual_seed(int(cfg.get("seed", 50)))
    if method == "transfer_learning_fwi":
        unet_cfg = {
            **cfg,
            "NNchannels": exp_cfg["NNchannels"],
            "numberOfConvolutionsPerBlock": exp_cfg["numberOfConvolutionsPerBlock"],
        }
        model = build_model(method, unet_cfg, params["gamma0"], device)
        if args.unet_init == "pretrained":
            sample = int(exp_cfg["pretrain_samples"][0])
            path = model_path(
                config,
                exp_cfg["modelType"],
                int(exp_cfg["epochs_pretrain"]),
                "supervised",
                sample,
                exp_cfg["NNchannels"],
            )
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded pretrained U-Net from {path}")
    elif method == "transfer_segformer_fwi":
        if args.segformer_init == "pretrained":
            checkpoint_path = Path(args.segformer_checkpoint or cfg["pretrained_checkpoint"])
            if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
                checkpoint_path = Path(config["paths"]["pretrained_models"]) / checkpoint_path
            model, segformer_checkpoint = load_segformer_checkpoint(checkpoint_path, device)
            segformer_norm_cfg = dict(segformer_checkpoint["gradient_normalization"])
            print(f"Loaded pretrained SegFormer from {checkpoint_path}")
        else:
            segformer_cfg = {
                **cfg,
                "model_spec": config.get("models", {}).get("segformer", {}),
                "void_prior": config.get("segformer_pretraining", {}).get("void_prior", 0.01),
            }
            model = build_model(method, segformer_cfg, params["gamma0"], device)
            segformer_norm_cfg = dict(
                config.get("segformer_pretraining", {}).get(
                    "gradient_normalization",
                    {
                        "mode": "robust_abs",
                        "quantile": 0.99,
                        "eps": 1e-8,
                        "clamp": 1.0,
                    },
                )
            )
    else:
        model = build_model(method, cfg, params["gamma0"], device)
    coords = make_coords(params, device)
    gamma_case, initial_gradient, _um, _source = load_case_data(args.case, data_dir, params, device, load_gradient=True)
    target_physical_gamma = gamma_case[0, 0, 1:-1, 1:-1]

    unet_input = None
    segformer_input = None
    if method == "transfer_learning_fwi":
        unet_input = normalize_input_data(initial_gradient).to(device)
        with torch.no_grad():
            gamma_initial = model(unet_input)[0, 0].detach()
    elif method == "transfer_segformer_fwi":
        segformer_input = normalize_gradient_for_transformer(
            initial_gradient.to(device),
            **segformer_norm_cfg,
        ).detach()
        with torch.no_grad():
            gamma_initial = model(segformer_input)[0, 0].detach()
    else:
        with torch.no_grad():
            gamma_initial = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1).detach()

    # Use the saved first conventional gradient so every architecture fits the
    # same case-specific first-gradient update direction. This avoids
    # recomputing the adjoint while keeping each architecture's own initial
    # gamma as the baseline it must update from.
    gradient = initial_gradient[0, 0].to(device=device, dtype=torch.float32)
    effective_gradient = gradient * float(cfg["costScaling"])

    if args.step_mode == "auto":
        max_abs_grad = float(effective_gradient.detach().abs().max().cpu())
        direct_step_scale = 0.0 if max_abs_grad == 0.0 else args.target_max_delta_gamma / max_abs_grad
    else:
        direct_step_scale = args.direct_step_scale

    gamma_direct_target = torch.clamp(
        gamma_initial - direct_step_scale * effective_gradient,
        min=float(params["gamma0"]),
        max=1.0,
    ).detach()
    update_target = gamma_direct_target - gamma_initial
    if args.loss_normalization == "target_rms":
        loss_scale = torch.sqrt(torch.mean(update_target**2)).detach()
    elif args.loss_normalization == "target_max":
        loss_scale = update_target.detach().abs().max()
    else:
        loss_scale = torch.tensor(1.0, device=device)
    loss_scale = torch.clamp(loss_scale, min=float(args.min_loss_scale))

    supervised_lr = float(args.lr if args.lr is not None else cfg["lr"])
    if method == "transfer_learning_fwi" and args.lr is None:
        supervised_lr = float(args.unet_default_lr)
    if method == "transfer_segformer_fwi" and args.lr is None:
        supervised_lr = float(args.segformer_default_lr)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=supervised_lr,
        weight_decay=float(cfg.get("l2", 0.0)),
    )
    rows = []
    snapshots = {}

    
    print(
        f"method={method} case={args.case} saved_first_gradient=true "
        f"direct_step_scale={direct_step_scale:.6e} supervised_lr={supervised_lr:.6e}"
    )
    print(
        f"loss_normalization={args.loss_normalization} loss_scale={float(loss_scale.cpu()):.6e} "
        f"target_update_std={float(update_target.std().cpu()):.6e}"
    )

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            if method == "transfer_learning_fwi":
                gamma_pred = model(unet_input)[0, 0]
            elif method == "transfer_segformer_fwi":
                gamma_pred = model(segformer_input)[0, 0]
            else:
                gamma_pred = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
            update_residual = gamma_pred - gamma_initial - update_target
            loss = torch.mean((update_residual / loss_scale) ** 2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if method == "transfer_learning_fwi":
                gamma_eval = model(unet_input)[0, 0]
            elif method == "transfer_segformer_fwi":
                gamma_eval = model(segformer_input)[0, 0]
            else:
                gamma_eval = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
            update_pred = gamma_eval - gamma_initial
            update_error = update_pred - update_target
            update_mse = torch.mean(update_error**2)
            row = {
                "epoch": epoch,
                "supervised_update_loss": float(torch.mean((update_error / loss_scale) ** 2).cpu()),
                "gamma_mse": float(torch.mean((gamma_eval - gamma_direct_target) ** 2).cpu()),
                "update_mse": float(update_mse.cpu()),
                "physical_gamma_mse": float(torch.mean((gamma_eval - target_physical_gamma) ** 2).cpu()),
                **tensor_stats(update_pred, "pred_update"),
                **tensor_stats(update_target, "target_update"),
                "direct_step_scale": float(direct_step_scale),
                "loss_scale": float(loss_scale.cpu()),
                "initial_fwi_cost": float("nan"),
                "raw_adjoint_grad_norm": tensor_norm(gradient),
                "effective_adjoint_grad_norm": tensor_norm(effective_gradient),
            }
            rows.append(row)
            if epoch in saved_epochs:
                snapshots[epoch] = update_pred.detach().cpu().numpy()

        if epoch in (0, 1, args.epochs) or (epoch > 0 and epoch % args.print_every == 0):
            print(
                f"epoch {epoch:04d} | update_loss {row['supervised_update_loss']:.6e} | "
                f"pred_update_std {row['pred_update_std']:.3e}"
            )

    final_gamma = gamma_eval.detach()
    final_update = final_gamma - gamma_initial
    update_error = final_update - update_target

    metrics_path = histories / f"{method}_case{args.case}_first_update_fit_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_history(histories / f"{method}_case{args.case}_update_loss_history.txt", [r["supervised_update_loss"] for r in rows])
    save_history(histories / f"{method}_case{args.case}_gamma_mse_history.txt", [r["gamma_mse"] for r in rows])

    arrays = {
        "gamma_initial": gamma_initial.detach().cpu().numpy(),
        "gamma_direct_target": gamma_direct_target.detach().cpu().numpy(),
        "raw_gradient": gradient.detach().cpu().numpy(),
        "effective_gradient": effective_gradient.detach().cpu().numpy(),
        "target_update": update_target.detach().cpu().numpy(),
        "final_gamma": final_gamma.detach().cpu().numpy(),
        "final_update": final_update.detach().cpu().numpy(),
        "update_error": update_error.detach().cpu().numpy(),
    }
    np.savez(outputs / f"{method}_case{args.case}_first_update_fit_outputs.npz", **arrays)
    plot_image_grid(figures / f"{method}_case{args.case}_first_update_fit_summary.png", method, args.case, arrays)
    plot_history(figures / f"{method}_case{args.case}_first_update_fit_history.png", rows)
    plot_update_snapshots(figures / f"{method}_case{args.case}_first_update_fit_saved_updates.png", snapshots, arrays["target_update"])

    (out / "runtime.txt").write_text(
        "\n".join(
            [
                "First-update fit diagnostic",
                "",
                "This diagnostic does not test whether the network can eventually fit the final physical void target.",
                "It tests whether the network can quickly reproduce the first material-field update implied by the adjoint gradient.",
                "If an INR needs many supervised epochs merely to reproduce this first update, it may be an inefficient ansatz for adjoint-driven FWI.",
                "",
                f"method: {method}",
                f"case_id: {args.case}",
                f"config: {args.config}",
                f"device: {args.device}",
                f"epochs: {args.epochs}",
                f"unet_init: {args.unet_init if method == 'transfer_learning_fwi' else 'n/a'}",
                f"segformer_init: {args.segformer_init if method == 'transfer_segformer_fwi' else 'n/a'}",
                f"supervised_lr: {supervised_lr}",
                f"costScaling: {cfg['costScaling']}",
                f"step_mode: {args.step_mode}",
                f"direct_step_scale: {direct_step_scale:.12e}",
                f"target_max_delta_gamma: {args.target_max_delta_gamma:.12e}",
                f"loss_normalization: {args.loss_normalization}",
                f"loss_scale: {float(loss_scale.cpu()):.12e}",
                "initial_fwi_cost: n/a_saved_gradient_used",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved first-update fit diagnostics to {out}")


if __name__ == "__main__":
    main()
