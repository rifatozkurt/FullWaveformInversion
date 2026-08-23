import argparse
import csv
import itertools
import sys
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import adjoint
from src import metrics
from src.config import load_config
from src.experiments.base import (
    create_forward_solver,
    create_initial_conditions,
    load_case_data,
    parameter_grad_norm,
    simulation_parameters,
    tensor_norm,
    tensor_stats,
    total_variation_loss,
)
from src.io import ensure_dir
from scripts.debug_first_update_fit import ALIASES, build_model, make_coords


DEFAULT_METHODS = "mpe_centered"
DEFAULT_LRS = "1e-2,2e-2,3e-2"
DEFAULT_COST_SCALINGS = "1e11,3e11,1e12"
DEFAULT_FINAL_BIASES = "-4.0,-3.75"
DEFAULT_GRID_INIT_STDS = "1e-2"
DEFAULT_OUTPUT_DIR = "temp/inr_signal_sweep_mpe_high_scale"


def parse_csv_floats(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_csv_strings(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_negative_csv_args(argv):
    normalized = []
    index = 0
    while index < len(argv):
        if argv[index] == "--final-biases" and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and not value.startswith("--"):
                normalized.append(f"--final-biases={value}")
                index += 2
                continue
        normalized.append(argv[index])
        index += 1
    return normalized


def safe_cosine(a, b, eps=1e-30):
    numerator = torch.sum(a * b)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((numerator / torch.clamp(denominator, min=eps)).detach().cpu())


def sign_agreement(a, b, threshold=0.0):
    mask = (a.abs() > threshold) | (b.abs() > threshold)
    if not bool(mask.any()):
        return float("nan")
    return float((torch.sign(a[mask]) == torch.sign(b[mask])).float().mean().detach().cpu())


def direct_update_from_scaled_gradient(gamma_before, scaled_gradient, gamma0, target_max_delta):
    max_abs = torch.clamp(scaled_gradient.detach().abs().max(), min=1e-30)
    direct_step_scale = float(target_max_delta) / float(max_abs.cpu())
    target_gamma = torch.clamp(
        gamma_before - direct_step_scale * scaled_gradient,
        min=float(gamma0),
        max=1.0,
    )
    return target_gamma - gamma_before, direct_step_scale


def plot_trial_images(path, title, images):
    items = [
        ("gamma before", images["gamma_before"], "coolwarm", None),
        ("target gamma", images["target_gamma"], "coolwarm", None),
        ("scaled adjoint grad", images["scaled_gradient"], "seismic", None),
        ("direct update", images["direct_update"], "seismic", None),
        ("actual delta gamma", images["delta_gamma"], "seismic", None),
        ("delta - direct", images["delta_error"], "seismic", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, (name, image, cmap, _scale) in zip(axes.ravel(), items):
        if name in ("gamma before", "target gamma"):
            kwargs = {"vmin": image.min(), "vmax": 1.0}
        else:
            scale = float(np.percentile(np.abs(image), 99))
            if scale <= 0:
                scale = 1.0
            kwargs = {"vmin": -scale, "vmax": scale}
        plot = ax.imshow(image.T, origin="lower", cmap=cmap, **kwargs)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_metric_summary(path, rows):
    if not rows:
        return
    final_rows = {}
    for row in rows:
        final_rows[row["trial_id"]] = row
    final_rows = list(final_rows.values())
    final_rows.sort(key=lambda row: float(row["mse_change"]))
    top = final_rows[: min(20, len(final_rows))]

    fig_width = max(14, 0.65 * len(top))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 10))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.34, hspace=0.5, wspace=0.25)
    x = np.arange(len(top))
    labels = [row["trial_id"] for row in top]
    metrics = [
        ("mse_change", "MSE change"),
        ("cost_change", "Cost change"),
        ("delta_direct_cosine", "Actual vs direct-update cosine"),
        ("delta_gamma_max_abs", "Max |delta gamma|"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [float(row[key]) for row in top]
        ax.bar(x, values)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def read_existing_metrics(path):
    if not path.exists():
        return [], set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs_by_trial = {}
    for row in rows:
        trial_id = row.get("trial_id", "")
        try:
            epoch = int(float(row.get("epoch", -1)))
        except ValueError:
            continue
        epochs_by_trial[trial_id] = max(epoch, epochs_by_trial.get(trial_id, -1))
    return rows, epochs_by_trial


def write_metrics(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_trial(
    trial_id,
    method,
    cfg,
    params,
    device,
    gamma_true,
    um,
    source,
    coords,
    forward_solver,
    u0,
    u1,
    probe_epochs,
    target_max_delta,
    save_images,
    figures_dir,
):
    torch.manual_seed(int(cfg.get("seed", 50)))
    model = build_model(method, cfg, params["gamma0"], device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("l2", 0.0)),
    )
    target_inner = gamma_true[0, 0, 1:-1, 1:-1]
    tv_weight = float(cfg.get("tv_weight", 0.0))
    tv_type = str(cfg.get("tv_type", "anisotropic"))
    use_tv = tv_weight > 0 and tv_type.lower() not in ("none", "null", "false", "off")

    rows = []
    first_cost = None
    first_mse = None
    final_images = None

    for epoch in range(int(probe_epochs)):
        optimizer.zero_grad(set_to_none=True)
        gamma_before = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
        gamma_before_detached = gamma_before.detach().clone()
        gamma_pred = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        gamma_pred[:, :, 1:-1, 1:-1] = gamma_before

        cost, gradient = adjoint.getAdjointGradient(
            forward_solver,
            u0,
            u1,
            params["c"],
            params["rho"],
            gamma_pred.detach(),
            source,
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
        direct_update, direct_step_scale = direct_update_from_scaled_gradient(
            gamma_before_detached,
            scaled_gradient,
            params["gamma0"],
            target_max_delta,
        )

        external_grad = torch.zeros_like(gamma_pred)
        external_grad[0, 0, 1:-1, 1:-1] = scaled_gradient
        gamma_pred.backward(external_grad, retain_graph=use_tv)

        tv_raw_value = 0.0
        tv_loss_value = 0.0
        if use_tv:
            tv_raw = total_variation_loss(gamma_before, tv_type=tv_type)
            tv_loss = tv_weight * tv_raw
            tv_loss.backward()
            tv_raw_value = float(tv_raw.detach().cpu())
            tv_loss_value = float(tv_loss.detach().cpu())

        param_norm = parameter_grad_norm(model)
        optimizer.step()

        with torch.no_grad():
            gamma_after = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
        delta_gamma = gamma_after - gamma_before_detached
        mse_before = metrics.gamma_mse(gamma_before_detached, target_inner)
        mse_after = metrics.gamma_mse(gamma_after, target_inner)
        cost_value = float(cost.detach().cpu())
        if first_cost is None:
            first_cost = cost_value
            first_mse = float(mse_before.detach().cpu())

        row = {
            "trial_id": trial_id,
            "method": method,
            "epoch": epoch,
            "lr": float(cfg["lr"]),
            "costScaling": float(cfg["costScaling"]),
            "final_bias": float(cfg.get("final_bias", float("nan"))),
            "grid_init_std": float(cfg.get("grid_init_std", float("nan"))),
            "omega0": float(cfg.get("omega0", float("nan"))),
            "fusion_alpha": float(cfg.get("fusion_alpha", cfg.get("alpha", float("nan")))),
            "tv_weight": tv_weight,
            "tv_type": tv_type,
            "cost": cost_value,
            "cost_change": cost_value - first_cost,
            "mse_before": float(mse_before.detach().cpu()),
            "mse_after": float(mse_after.detach().cpu()),
            "mse_change": float(mse_after.detach().cpu()) - first_mse,
            **tensor_stats(gamma_before_detached, "gamma_before"),
            **tensor_stats(gamma_after, "gamma_after"),
            **tensor_stats(gradient, "adjoint_grad"),
            "adjoint_grad_norm": tensor_norm(gradient),
            "scaled_grad_norm": tensor_norm(scaled_gradient),
            "scaled_grad_min": float(scaled_gradient.detach().min().cpu()),
            "scaled_grad_max": float(scaled_gradient.detach().max().cpu()),
            "param_grad_norm": param_norm,
            "delta_gamma_mean_abs": float(delta_gamma.detach().abs().mean().cpu()),
            "delta_gamma_max_abs": float(delta_gamma.detach().abs().max().cpu()),
            "delta_gamma_std": float(delta_gamma.detach().std().cpu()),
            "direct_update_mean_abs": float(direct_update.detach().abs().mean().cpu()),
            "direct_update_max_abs": float(direct_update.detach().abs().max().cpu()),
            "direct_update_std": float(direct_update.detach().std().cpu()),
            "direct_step_scale": direct_step_scale,
            "delta_direct_cosine": safe_cosine(delta_gamma, direct_update),
            "delta_direct_sign_agreement": sign_agreement(delta_gamma, direct_update),
            "delta_direct_error_norm": tensor_norm(delta_gamma - direct_update),
            "delta_over_direct_norm": tensor_norm(delta_gamma) / max(tensor_norm(direct_update), 1e-30),
            "tv_raw": tv_raw_value,
            "tv_loss": tv_loss_value,
        }
        rows.append(row)
        final_images = {
            "gamma_before": gamma_before_detached.detach().cpu().numpy(),
            "target_gamma": target_inner.detach().cpu().numpy(),
            "scaled_gradient": scaled_gradient.detach().cpu().numpy(),
            "direct_update": direct_update.detach().cpu().numpy(),
            "delta_gamma": delta_gamma.detach().cpu().numpy(),
            "delta_error": (delta_gamma - direct_update).detach().cpu().numpy(),
        }

    if save_images and final_images is not None:
        plot_trial_images(
            figures_dir / f"{trial_id}_images.png",
            f"{trial_id}: {method}",
            final_images,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Sweep INR signal-propagation settings and log detailed adjoint-to-gamma diagnostics."
    )
    parser.add_argument("--config", default="configs/experimental.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--lrs", default=DEFAULT_LRS)
    parser.add_argument("--cost-scalings", default=DEFAULT_COST_SCALINGS)
    parser.add_argument("--final-biases", default=DEFAULT_FINAL_BIASES)
    parser.add_argument("--grid-init-stds", default=DEFAULT_GRID_INIT_STDS)
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--target-max-delta-gamma", type=float, default=1e-3)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means run all generated trials.")
    parser.add_argument("--save-trial-images", default=True, action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from the existing metrics CSV. Use --no-resume to start from scratch.",
    )
    args = parser.parse_args(normalize_negative_csv_args(sys.argv[1:]))

    config = load_config(args.config)
    params = simulation_parameters(config)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])

    run_dir = ensure_dir(Path(args.output_dir) / f"case{args.case}")
    figures_dir = ensure_dir(run_dir / "figures")
    histories_dir = ensure_dir(run_dir / "histories")
    metrics_path = histories_dir / "inr_signal_sweep_metrics.csv"
    if args.resume:
        all_rows, epochs_by_trial = read_existing_metrics(metrics_path)
    else:
        all_rows, epochs_by_trial = [], {}
    complete_epoch = int(args.probe_epochs) - 1

    gamma_true, um, source = load_case_data(args.case, data_dir, params, device)
    u0, u1 = create_initial_conditions(params, device)
    forward_solver = create_forward_solver(params, device)
    coords = make_coords(params, device)

    methods = [ALIASES.get(method, method) for method in parse_csv_strings(args.methods)]
    lrs = parse_csv_floats(args.lrs)
    cost_scalings = parse_csv_floats(args.cost_scalings)
    final_biases = parse_csv_floats(args.final_biases)
    grid_init_stds = parse_csv_floats(args.grid_init_stds)

    trial_count = 0
    skipped_count = 0
    run_count = 0
    for method in methods:
        base_cfg = dict(config["experiments"][method])
        method_grid_stds = grid_init_stds if "mpe" in method or "ig" in method else [None]
        for lr, cost_scaling, final_bias, grid_init_std in itertools.product(
            lrs,
            cost_scalings,
            final_biases,
            method_grid_stds,
        ):
            if args.max_trials and trial_count >= args.max_trials:
                break
            cfg = dict(base_cfg)
            cfg["lr"] = lr
            cfg["costScaling"] = cost_scaling
            cfg["final_bias"] = final_bias
            if grid_init_std is not None:
                cfg["grid_init_std"] = grid_init_std
            trial_count += 1
            trial_id = (
                f"trial{trial_count:03d}_{method}"
                f"_lr{lr:.0e}_cs{cost_scaling:.0e}_fb{final_bias:g}"
            )
            if grid_init_std is not None:
                trial_id += f"_grid{grid_init_std:.0e}"
            if epochs_by_trial.get(trial_id, -1) >= complete_epoch:
                skipped_count += 1
                print(f"\nSkipping completed {trial_id}")
                continue
            print(f"\nRunning {trial_id}")
            rows = run_trial(
                trial_id,
                method,
                cfg,
                params,
                device,
                gamma_true,
                um,
                source,
                coords,
                forward_solver,
                u0,
                u1,
                args.probe_epochs,
                args.target_max_delta_gamma,
                args.save_trial_images,
                figures_dir,
            )
            all_rows.extend(rows)
            run_count += 1
            epochs_by_trial[trial_id] = max(int(row["epoch"]) for row in rows)
            write_metrics(metrics_path, all_rows)
            plot_metric_summary(figures_dir / "inr_signal_sweep_summary.png", all_rows)
        if args.max_trials and trial_count >= args.max_trials:
            break

    if all_rows:
        write_metrics(metrics_path, all_rows)
        plot_metric_summary(figures_dir / "inr_signal_sweep_summary.png", all_rows)

    (run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "INR signal propagation sweep",
                f"config: {args.config}",
                f"case_id: {args.case}",
                f"methods: {args.methods}",
                f"lrs: {args.lrs}",
                f"cost_scalings: {args.cost_scalings}",
                f"final_biases: {args.final_biases}",
                f"grid_init_stds: {args.grid_init_stds}",
                f"probe_epochs: {args.probe_epochs}",
                f"target_max_delta_gamma: {args.target_max_delta_gamma}",
                f"resume: {args.resume}",
                f"trials_generated: {trial_count}",
                f"trials_skipped_complete: {skipped_count}",
                f"trials_run_this_session: {run_count}",
                f"metrics: {metrics_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved INR signal sweep diagnostics to {run_dir}")


if __name__ == "__main__":
    main()
