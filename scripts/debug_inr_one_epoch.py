import argparse
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import adjoint
from src.config import load_config
from src.experiments.base import (
    create_forward_solver,
    create_initial_conditions,
    load_case_data,
    model_path,
    normalize_input_data,
    parameter_grad_norm,
    simulation_parameters,
    tensor_norm,
    tensor_stats,
)
from src.io import ensure_dir
from src.networks import INR, INR_IG, INR_IG_CENTERED, INR_LR, INR_MPE, INR_MPE_CENTERED, INRSIREN, Unet


METHODS = (
    "inr_fwi",
    "inr_siren_fwi",
    "inr_lr_fwi",
    "inr_mpe_fwi",
    "inr_mpe_centered_fwi",
    "inr_ig_fwi",
    "inr_ig_centered_fwi",
    "transfer_learning_fwi",
)


def method_config(config, method):
    if method == "inr_mpe_centered_fwi":
        return config["experiments"].get(
            "inr_mpe_centered_fwi",
            config["experiments"]["inr_mpe_fwi"],
        )
    if method == "inr_ig_centered_fwi":
        return config["experiments"].get(
            "inr_ig_centered_fwi",
            config["experiments"]["inr_ig_fwi"],
        )
    return config["experiments"][method]


def build_model(config, method, params, device):
    cfg = method_config(config, method)
    common = {
        "gamma0": params["gamma0"],
        "output_mode": cfg.get("output_mode", "voidness"),
        "final_bias": float(cfg.get("final_bias", -5.0)),
    }

    if method == "inr_fwi":
        model = INR(
            int(cfg["hidden_features"]),
            int(cfg["hidden_layers"]),
            **common,
        )
    elif method == "inr_siren_fwi":
        model = INRSIREN(
            int(cfg["hidden_features"]),
            int(cfg["hidden_layers"]),
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
    elif method == "inr_mpe_centered_fwi":
        model = INR_MPE_CENTERED(
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
    elif method == "inr_ig_centered_fwi":
        model = INR_IG_CENTERED(
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
        raise ValueError(f"Unknown INR method: {method}")

    return model.to(device)


def build_transfer_model(config, params, device):
    exp_cfg = config["experiments"]
    cfg = exp_cfg["transfer_learning_fwi"]
    sample = int(exp_cfg["pretrain_samples"][0])
    channels = exp_cfg["NNchannels"]
    convolutions_per_block = int(exp_cfg["numberOfConvolutionsPerBlock"])
    model_type = exp_cfg["modelType"]
    epochs_pretrain = int(exp_cfg["epochs_pretrain"])
    path = model_path(
        config,
        model_type,
        epochs_pretrain,
        "supervised",
        sample,
        channels,
    )

    torch.manual_seed(int(cfg.get("seed", 99)))
    model = Unet(channels, convolutions_per_block, params["gamma0"], bnorm=True)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device), path


def make_coords(params, device):
    x = torch.linspace(-1, 1, params["Nx"] + 1, device=device)
    y = torch.linspace(-1, 1, params["Ny"] + 1, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def robust_vlim(array, percentile=99):
    value = float(np.percentile(np.abs(array), percentile))
    return value if value > 0 else 1.0


def final_bias_parameter(model, method):
    if method == "inr_fwi":
        return model.final_layer.bias
    if method == "inr_siren_fwi":
        return model.layers[-1].bias
    if method == "inr_lr_fwi":
        return model.raw_bias
    if method == "inr_mpe_fwi":
        for module in reversed(model.mlp):
            if isinstance(module, torch.nn.Linear):
                return module.bias
    if method in ("inr_mpe_centered_fwi", "inr_ig_centered_fwi"):
        return None
    if method == "inr_ig_fwi":
        for module in reversed(model.fusion_mlp):
            if isinstance(module, torch.nn.Linear):
                return module.bias
    return None


def set_final_bias(model, method, value):
    if method in ("inr_mpe_centered_fwi", "inr_ig_centered_fwi"):
        with torch.no_grad():
            model.final_bias.fill_(float(value))
        return
    bias = final_bias_parameter(model, method)
    if bias is None:
        raise ValueError(f"No final/output bias parameter found for {method}")
    with torch.no_grad():
        bias.fill_(float(value))


def grad_norm(param):
    if param is None or param.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(param.grad.detach()).cpu())


def named_grad_report(model, top_k=12):
    rows = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        rows.append(
            (
                float(torch.linalg.vector_norm(param.grad.detach()).cpu()),
                name,
                tuple(param.shape),
            )
        )
    rows.sort(reverse=True, key=lambda item: item[0])
    return rows[:top_k]


def plot_debug_images(output_path, method, case_id, images):
    items = [
        ("target gamma", images["target"], "coolwarm", None),
        ("gamma before", images["gamma_before"], "coolwarm", None),
        ("gamma after", images["gamma_after"], "coolwarm", None),
        ("adjoint gradient", images["gradient"], "seismic", robust_vlim(images["gradient"])),
        ("scaled gradient", images["scaled_gradient"], "seismic", robust_vlim(images["scaled_gradient"])),
        ("gamma update", images["delta_gamma"], "seismic", robust_vlim(images["delta_gamma"])),
    ]
    if "input_gradient" in images:
        items.insert(1, ("fixed input gradient", images["input_gradient"], "seismic", robust_vlim(images["input_gradient"])))

    ncols = 4 if len(items) > 6 else 3
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, image, cmap, vlim) in zip(axes, items):
        if vlim is None:
            plot = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=image.min(), vmax=max(1.0, image.max()))
        else:
            plot = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(f"{method}, case {case_id}: first adjoint gradient debug")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run one debug INR/transfer-learning epoch and plot the first adjoint gradient."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="inr_ig_centered_fwi", choices=METHODS)
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="temp/inr_one_epoch_debug")
    parser.add_argument("--no-update", action="store_true", help="Only compute the first gradient; do not take the optimizer step.")
    parser.add_argument("--final-bias", type=float, default=None, help="Override INR final/output bias for this debug run only.")
    parser.add_argument("--freeze-final-bias", action="store_true", help="Set final/output bias requires_grad=False for this debug run only.")
    args = parser.parse_args()

    config = load_config(args.config)
    params = simulation_parameters(config)
    cfg = method_config(config, args.method)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["paths"]["casestudy_data"])
    output_dir = ensure_dir(args.output_dir)

    torch.manual_seed(int(cfg.get("seed", 50)))
    coords = None
    input_gradient_image = None

    if args.method == "transfer_learning_fwi":
        model, loaded_model_path = build_transfer_model(config, params, device)
        gamma_target, initial_gradient, um, F = load_case_data(
            args.case, data_dir, params, device, load_gradient=True
        )
        input_data = normalize_input_data(initial_gradient).to(device)
        input_gradient_image = initial_gradient[0, 0].detach().cpu().numpy()
        print(f"Loaded pretrained model: {loaded_model_path}")
    else:
        model = build_model(config, args.method, params, device)
        coords = make_coords(params, device)
        gamma_target, um, F = load_case_data(args.case, data_dir, params, device)
        input_data = None

    output_bias = final_bias_parameter(model, args.method)
    if args.final_bias is not None:
        set_final_bias(model, args.method, args.final_bias)
        print(f"Overrode final/output bias to {args.final_bias}")
    if args.freeze_final_bias:
        if output_bias is None:
            print("No trainable final/output bias to freeze for this method.")
        else:
            output_bias.requires_grad_(False)
            print("Frozen final/output bias for this debug run.")
    output_bias_before = None
    if output_bias is not None:
        output_bias_before = output_bias.detach().clone()

    u0, u1 = create_initial_conditions(params, device)
    forward_solver = create_forward_solver(params, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("l2", 0.0)),
    )

    optimizer.zero_grad(set_to_none=True)
    if args.method == "transfer_learning_fwi":
        model.train()
        gamma_model_output = model(input_data)
        gamma_inner = gamma_model_output[0, 0]
    else:
        gamma_inner = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
    gamma_before = gamma_inner.detach().clone()
    target_inner = gamma_target[0, 0, 1:-1, 1:-1]

    gamma_pred = torch.ones(
        (1, 1, params["Nx"] + 3, params["Ny"] + 3),
        device=device,
        dtype=torch.float32,
    )
    gamma_pred[:, :, 1:-1, 1:-1] = gamma_inner

    cost, gradient = adjoint.getAdjointGradient(
        forward_solver,
        u0,
        u1,
        params["c"],
        params["rho"],
        gamma_pred.detach(),
        F,
        params["Nx"],
        params["dx"],
        params["Ny"],
        params["dy"],
        params["N"],
        params["dt"],
        params["numberOfSources"],
        um.to(device),
        params["selx"],
        params["sely"],
        device,
    )
    gradient = gradient.to(device=device, dtype=torch.float32)
    scaled_gradient = gradient * float(cfg["costScaling"])

    external_grad = torch.zeros_like(gamma_pred)
    external_grad[0, 0, 1:-1, 1:-1] = scaled_gradient
    gamma_pred.backward(external_grad)

    param_norm = parameter_grad_norm(model)
    output_bias_grad_norm = grad_norm(output_bias)
    output_bias_grad_mean = 0.0
    if output_bias is not None and output_bias.grad is not None:
        output_bias_grad_mean = float(output_bias.grad.detach().mean().cpu())

    if args.no_update:
        gamma_after = gamma_before
    else:
        if args.method == "transfer_learning_fwi" and "clipGrad" in cfg:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clipGrad"]))
        optimizer.step()
        with torch.no_grad():
            if args.method == "transfer_learning_fwi":
                gamma_after = model(input_data)[0, 0]
            else:
                gamma_after = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
    output_bias_after = None
    if output_bias is not None:
        output_bias_after = output_bias.detach().clone()

    delta_gamma = gamma_after - gamma_before
    mse_after = 0.5 * torch.mean((gamma_after - target_inner) ** 2)

    print(f"method: {args.method}")
    print(f"case: {args.case}")
    print(f"device: {device}")
    print(f"cost: {float(cost.detach().cpu()):.6e}")
    print(f"mse after debug step: {float(mse_after.detach().cpu()):.6e}")
    print(f"param_grad_norm: {param_norm:.6e}")
    if output_bias is not None:
        bias_before_value = float(output_bias_before.detach().mean().cpu())
        bias_after_value = float(output_bias_after.detach().mean().cpu())
        print(f"final/output bias before: {bias_before_value:.6e}")
        print(f"final/output bias after:  {bias_after_value:.6e}")
        print(f"final/output bias update: {bias_after_value - bias_before_value:.6e}")
        print(f"final/output bias grad mean: {output_bias_grad_mean:.6e}")
        print(f"final/output bias grad norm: {output_bias_grad_norm:.6e}")
        if param_norm > 0:
            print(f"bias_grad_norm / total_param_grad_norm: {output_bias_grad_norm / param_norm:.6e}")
    print(f"adjoint_grad_norm: {tensor_norm(gradient):.6e}")
    print(f"scaled_grad_norm: {tensor_norm(scaled_gradient):.6e}")
    print(f"gamma mean before: {float(gamma_before.mean().detach().cpu()):.6e}")
    print(f"gamma mean after:  {float(gamma_after.mean().detach().cpu()):.6e}")
    print(tensor_stats(gamma_before, "gamma_before"))
    print(tensor_stats(gradient, "adjoint_grad"))
    print(tensor_stats(delta_gamma, "delta_gamma"))
    print("largest parameter-gradient norms:")
    for norm, name, shape in named_grad_report(model):
        print(f"  {name:45s} {norm:.6e} shape={shape}")

    images = {
        "target": target_inner.detach().cpu().numpy(),
        "gamma_before": gamma_before.detach().cpu().numpy(),
        "gamma_after": gamma_after.detach().cpu().numpy(),
        "gradient": gradient.detach().cpu().numpy(),
        "scaled_gradient": scaled_gradient.detach().cpu().numpy(),
        "delta_gamma": delta_gamma.detach().cpu().numpy(),
    }
    if input_gradient_image is not None:
        images["input_gradient"] = input_gradient_image
    output_path = output_dir / f"{args.method}_case{args.case}_first_gradient.png"
    plot_debug_images(output_path, args.method, args.case, images)
    print(f"Saved debug plot to {output_path}")


if __name__ == "__main__":
    main()
