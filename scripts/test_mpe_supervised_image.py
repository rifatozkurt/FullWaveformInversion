import argparse
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.io import load_hdf
from src.networks import INR, INR_IG, INR_LR, INR_MPE, INRSIREN, INRSIREN_CENTERED


selected_method = "inr_ig_fwi"

METHODS = (
    "inr_fwi",
    "inr_siren_fwi",
    "inr_siren_centered_fwi",
    "inr_lr_fwi",
    "inr_mpe_fwi",
    "inr_ig_fwi",
)



def make_coords(nx, ny, device):
    x = torch.linspace(-1.0, 1.0, nx, device=device)
    y = torch.linspace(-1.0, 1.0, ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def synthetic_gamma(nx, ny, gamma0):
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gamma = np.ones((nx, ny), dtype=np.float32)
    crack = (np.abs(xx + 0.45 * yy) < 0.045) & (xx > -0.55) & (xx < 0.6)
    void = (xx - 0.35) ** 2 + (yy + 0.15) ** 2 < 0.11 ** 2
    gamma[crack | void] = gamma0
    return gamma


def load_target_gamma(config, material_path):
    gamma0 = float(config["simulation"]["gamma0"])
    nx = int(config["simulation"]["Nx"]) + 1
    ny = int(config["simulation"]["Ny"]) + 1

    if material_path and Path(material_path).exists():
        gamma = load_hdf(material_path).astype(np.float32)
        return gamma, f"Loaded {material_path}"

    default_path = Path("data/casestudy/material1.h5")
    if default_path.exists():
        gamma = load_hdf(default_path).astype(np.float32)
        return gamma, f"Loaded {default_path}"

    return synthetic_gamma(nx, ny, gamma0), "Using synthetic target because no material file was found."


def build_model(config, method, device):
    sim = config["simulation"]
    cfg = config["experiments"][method]
    gamma0 = float(sim["gamma0"])

    common = {
        "gamma0": gamma0,
        "output_mode": cfg.get("output_mode", "voidness"),
        "final_bias": float(cfg.get("final_bias", -5.0)),
    }

    if method == "inr_fwi":
        model = INR(
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            **common,
        )
    elif method == "inr_siren_fwi":
        model = INRSIREN(
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            **common,
        )
    elif method == "inr_siren_centered_fwi":
        model = INRSIREN_CENTERED(
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            **common,
        )
    elif method == "inr_lr_fwi":
        model = INR_LR(
            rank=int(cfg["rank"]),
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            omega0=float(cfg.get("omega0", 30)),
            core_init_std=float(cfg.get("core_init_std", 1e-3)),
            **common,
        )
    elif method == "inr_mpe_fwi":
        model = INR_MPE(
            num_levels=int(cfg["num_levels"]),
            base_resolution=int(cfg["base_resolution"]),
            per_level_scale=float(cfg["per_level_scale"]),
            features_per_level=int(cfg["features_per_level"]),
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            grid_init_std=float(cfg.get("grid_init_std", 1e-4)),
            align_corners=bool(cfg.get("align_corners", True)),
            swap_grid_coords=bool(cfg.get("swap_grid_coords", False)),
            **common,
        )
    elif method == "inr_ig_fwi":
        model = INR_IG(
            alpha=float(cfg.get("fusion_alpha", 0.5)),
            num_levels=int(cfg["num_levels"]),
            base_resolution=int(cfg["base_resolution"]),
            per_level_scale=float(cfg["per_level_scale"]),
            features_per_level=int(cfg["features_per_level"]),
            grid_init_std=float(cfg.get("grid_init_std", 1e-4)),
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


def plot_progress(target, prediction, losses, title, method):
    error = np.abs(prediction - target)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    images = [
        axes[0].imshow(target.T, origin="lower", cmap="coolwarm", vmin=target.min(), vmax=1.0),
        axes[1].imshow(prediction.T, origin="lower", cmap="coolwarm", vmin=target.min(), vmax=1.0),
        axes[2].imshow(error.T, origin="lower", cmap="magma"),
    ]
    axes[0].set_title("Target gamma")
    axes[1].set_title(f"{method} prediction")
    axes[2].set_title("|error|")
    for ax, image in zip(axes[:3], images):
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[3].plot(losses, linewidth=2)
    axes[3].set_title("MSE loss")
    axes[3].set_xlabel("epoch")
    axes[3].set_yscale("log")
    axes[3].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Quick supervised sanity check: can an INR model learn one gamma image without FWI?"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default=selected_method, choices=METHODS)
    parser.add_argument("--material-path", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(int(config["experiments"][args.method].get("seed", 50)))

    target_np, target_message = load_target_gamma(config, args.material_path)
    target = torch.as_tensor(target_np.reshape(-1, 1), dtype=torch.float32, device=device)
    coords = make_coords(target_np.shape[0], target_np.shape[1], device)

    model = build_model(config, args.method, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    print(target_message)
    print(f"Target shape: {target_np.shape}")
    print(f"Coordinates: {tuple(coords.shape)}")
    print(f"Device: {device}")
    print(f"Training {args.method} directly on one image with MSE.")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(coords)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                pred_stats = pred.detach()
                print(
                    f"epoch {epoch:04d} | loss {losses[-1]:.6e} | "
                    f"pred min/mean/max "
                    f"{pred_stats.min().item():.4f}/"
                    f"{pred_stats.mean().item():.4f}/"
                    f"{pred_stats.max().item():.4f}"
                )

    with torch.no_grad():
        prediction = model(coords).reshape(target_np.shape).detach().cpu().numpy()

    title = f"{args.method} supervised one-image fit, final MSE={losses[-1]:.3e}"
    plot_progress(target_np, prediction, losses, title, args.method)


if __name__ == "__main__":
    main()
