import copy
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import adjoint
from src import io
from src import metrics
from src import networks as NN
from src.experiments.base import (
    ExperimentResult,
    create_forward_solver,
    create_initial_conditions,
    get_device,
    load_case_data,
    plot_reconstruction_history,
    save_histories,
    save_outputs,
    simulation_parameters,
)
from src.pretrain_segformer import load_segformer_checkpoint


def set_segformer_trainable_mode(model, mode):
    if mode not in ("all", "decoder_only", "decoder_plus_last_stage"):
        raise ValueError(f"Unknown SegFormer trainable_mode: {mode}")
    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return sum(parameter.numel() for parameter in model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad = False

    selected_modules = [model.segformer.decode_head]
    # The HighRes refinement path is part of the output decoder. Include it in
    # decoder-only transfer so the defining residual correction is not frozen.
    high_resolution_module_names = (
        "high_resolution_input",
        "high_resolution_fusion",
        "high_resolution_correction",
    )
    selected_modules.extend(
        getattr(model, name)
        for name in high_resolution_module_names
        if hasattr(model, name)
    )
    if mode == "decoder_plus_last_stage":
        backbone = model.segformer.segformer
        if hasattr(backbone, "stages"):
            selected_modules.append(backbone.stages[-1])
        elif hasattr(backbone, "encoder"):
            encoder = backbone.encoder
            selected_modules.extend(
                [
                    encoder.patch_embeddings[-1],
                    encoder.block[-1],
                    encoder.layer_norm[-1],
                ]
            )
        else:
            raise ValueError("Could not locate the final SegFormer encoder stage")

    for module in selected_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    selected = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    if selected == 0:
        raise ValueError(f"Trainable mode selected no SegFormer parameters: {mode}")
    return selected


def global_tensor_norm(tensors):
    values = [tensor.detach().float().norm().square() for tensor in tensors if tensor is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu())


def set_optimization_model_mode(model, mode):
    if mode == "train":
        model.train()
    elif mode == "eval":
        # eval() disables decoder dropout and running-stat updates but leaves
        # autograd enabled, which is preferable for deterministic one-case FWI.
        model.eval()
    else:
        raise ValueError("optimization_model_mode must be train or eval")


def save_transfer_diagnostics(run_dir, method_name, case_id, rows):
    path = Path(run_dir) / "histories" / f"{method_name}_case{case_id}_diagnostics.csv"
    io.ensure_dir(path.parent)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def save_segformer_layout_plot(run_dir, method_name, case_id, first_gradient, voidness, gamma_image, gamma_solver):
    path = Path(run_dir) / "figures" / f"{method_name}_case{case_id}_layout_check.png"
    io.ensure_dir(path.parent)
    arrays = [
        first_gradient[0, 0].detach().cpu().numpy(),
        voidness[0, 0].detach().cpu().numpy(),
        gamma_image[0, 0].detach().cpu().numpy(),
        gamma_solver[0, 0, 1:-1, 1:-1].detach().cpu().numpy(),
    ]
    titles = ["first adjoint gradient", "transformer voidness", "transformer gamma", "solver-layout gamma"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
    for axis, array, title in zip(axes, arrays, titles):
        image = axis.imshow(np.transpose(array))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


class TransferSegFormerFWI:
    name = "transfer_segformer_fwi"

    def __init__(self, config, device=None):
        self.config = config
        self.device = get_device(device)
        self.params = simulation_parameters(config)

    def run(self, case_id: int, data_dir: Path, run_dir: Path):
        params = self.params
        cfg = self.config["experiments"][self.name]
        device = self.device
        torch.manual_seed(int(cfg.get("seed", 99)))
        print("Using device: {}".format(device))
        print("Number of sensors: {:d}".format(len(params["selx"])))
        print(case_id)

        checkpoint_path = Path(cfg["pretrained_checkpoint"])
        if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
            checkpoint_path = Path(self.config["paths"]["pretrained_models"]) / checkpoint_path

        gamma, um, F = load_case_data(case_id, data_dir, params, device, load_gradient=False)
        u0, u1 = create_initial_conditions(params, device)
        forward_solver = create_forward_solver(params, device)

        gamma_homogeneous = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        initial_cost, first_gradient_2d = adjoint.getAdjointGradient(
            forward_solver,
            u0,
            u1,
            params["c"],
            params["rho"],
            gamma_homogeneous.detach(),
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

        model, checkpoint = load_segformer_checkpoint(checkpoint_path, device)
        model.to(device)
        trainable_mode = cfg.get("trainable_mode", "all")
        trainable_parameter_count = set_segformer_trainable_mode(model, trainable_mode)
        norm_cfg = dict(checkpoint["gradient_normalization"])
        # See src/networks.py: `eps` used to act as an absolute floor on the
        # normalization denominator, which for gradients of magnitude ~1e-14
        # rescaled every input to ~1e-7. A checkpoint fit to that scaling is
        # incompatible with the corrected normalization -- the input it now
        # receives differs by ~6 orders of magnitude.
        if int(checkpoint.get("normalization_version", 1)) < 2:
            raise RuntimeError(
                "SegFormer checkpoint %s predates the gradient-normalization fix "
                "(normalization_version < 2). It was trained on inputs scaled to "
                "~1e-7 and cannot be used with the corrected normalization. "
                "Re-run SegFormer pretraining." % checkpoint_path
            )
        first_gradient = first_gradient_2d.unsqueeze(0).unsqueeze(0)
        input_data = NN.normalize_gradient_for_transformer(first_gradient, **norm_cfg).detach()

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            "SegFormer trainable mode: {}; parameters: {}/{} ({:.2f}%)".format(
                trainable_mode,
                trainable_parameter_count,
                total_parameter_count,
                100.0 * trainable_parameter_count / max(1, total_parameter_count),
            )
        )

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        epochs = int(cfg["epochs"])
        cost_history = np.zeros(epochs)
        mse_history = np.zeros(epochs)
        gamma_history = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))
        diagnostics = []
        optimization_model_mode = cfg.get("optimization_model_mode", "train")
        record_post_step = bool(cfg.get("record_post_step", False))
        restore_best_observed = bool(cfg.get("restore_best_observed", False))
        early_stopping_patience = cfg.get("early_stopping_patience")
        early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        best_observed_cost = float("inf")
        best_observed_epoch = -1
        best_model_state = None
        epochs_without_improvement = 0
        completed_epochs = 0

        with torch.no_grad():
            model.eval()
            initial_gamma_image = model(input_data)
            initial_gamma_pred = torch.ones_like(gamma_homogeneous)
            initial_gamma_pred[:, :, 1:-1, 1:-1] = initial_gamma_image
            gamma_history[0] = initial_gamma_image[0, 0].detach().cpu().numpy()
            initial_voidness = model.forward_voidness(input_data)
            save_segformer_layout_plot(
                run_dir,
                self.name,
                case_id,
                input_data,
                initial_voidness,
                initial_gamma_image,
                initial_gamma_pred,
            )

        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            set_optimization_model_mode(model, optimization_model_mode)

            gamma_image = model(input_data)
            gamma_pred = torch.ones_like(gamma_homogeneous)
            gamma_pred[:, :, 1:-1, 1:-1] = gamma_image
            gamma_pred.grad = torch.zeros_like(gamma_pred.detach(), device=device)

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
            pre_step_mse = metrics.gamma_mse(gamma_pred, gamma, ghost=1)
            observed_cost = float(cost.detach().cpu())
            if observed_cost < best_observed_cost:
                best_observed_cost = observed_cost
                best_observed_epoch = epoch
                epochs_without_improvement = 0
                if restore_best_observed:
                    best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            raw_gamma_grad_norm = float(gradient.detach().float().norm().cpu())
            scaled_gradient = gradient * float(cfg["costScaling"])
            scaled_gamma_grad_norm = float(scaled_gradient.detach().float().norm().cpu())
            gamma_pred.grad[0, 0, 1:-1, 1:-1] = scaled_gradient
            gamma_pred.backward(gamma_pred.grad)
            parameter_grad_norm_before_clip = global_tensor_norm(
                [parameter.grad for parameter in trainable_parameters]
            )
            clip_grad = cfg.get("clipGrad")
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(clip_grad))
            parameter_grad_norm_after_clip = global_tensor_norm(
                [parameter.grad for parameter in trainable_parameters]
            )
            parameter_before = [parameter.detach().clone() for parameter in trainable_parameters]
            optimizer.step()
            scheduler.step()

            parameter_update_norm = global_tensor_norm(
                [
                    parameter.detach() - before
                    for parameter, before in zip(trainable_parameters, parameter_before)
                ]
            )

            with torch.no_grad():
                model.eval()
                post_step_gamma_image = model(input_data)
                post_step_gamma_pred = torch.ones_like(gamma_homogeneous)
                post_step_gamma_pred[:, :, 1:-1, 1:-1] = post_step_gamma_image
                post_step_mse = metrics.gamma_mse(
                    post_step_gamma_pred, gamma, ghost=1
                )
                gamma_update = post_step_gamma_image - gamma_image.detach()

            cost_history[epoch] = observed_cost
            mse_history[epoch] = post_step_mse if record_post_step else pre_step_mse
            stored_gamma = post_step_gamma_image if record_post_step else gamma_image
            gamma_history[epoch + 1] = stored_gamma[0, 0].detach().cpu().numpy()
            diagnostics.append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "cost_pre_step": observed_cost,
                    "mse_pre_step": pre_step_mse,
                    "mse_post_step": post_step_mse,
                    "raw_gamma_grad_norm": raw_gamma_grad_norm,
                    "scaled_gamma_grad_norm": scaled_gamma_grad_norm,
                    "parameter_grad_norm_before_clip": parameter_grad_norm_before_clip,
                    "parameter_grad_norm_after_clip": parameter_grad_norm_after_clip,
                    "parameter_update_norm": parameter_update_norm,
                    "gamma_update_rms": float(torch.sqrt(torch.mean(gamma_update**2)).cpu()),
                    "gamma_update_max_abs": float(gamma_update.abs().max().cpu()),
                    "best_observed_cost": best_observed_cost,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )
            completed_epochs = epoch + 1

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                print(
                    "Epoch: {}/{}\t\tCost function: {:.7E}\t\tMSE: {:.7E}\t\tElapsed time: {:2f}".format(
                        epoch,
                        epochs - 1,
                        cost_history[epoch],
                        mse_history[epoch],
                        elapsed_time,
                    )
                )
            start = time.perf_counter()

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    "Early stopping: FWI cost did not improve for {} epoch(s).".format(
                        epochs_without_improvement
                    )
                )
                break

        cost_history = cost_history[:completed_epochs]
        mse_history = mse_history[:completed_epochs]
        gamma_history = gamma_history[: completed_epochs + 1]

        if restore_best_observed and best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                restored_gamma = model(input_data)
                restored_gamma_pred = torch.ones_like(gamma_homogeneous)
                restored_gamma_pred[:, :, 1:-1, 1:-1] = restored_gamma
            gamma_history[-1] = restored_gamma[0, 0].detach().cpu().numpy()
            mse_history[-1] = metrics.gamma_mse(
                restored_gamma_pred, gamma, ghost=1
            )
            cost_history[-1] = best_observed_cost

        save_transfer_diagnostics(run_dir, self.name, case_id, diagnostics)

        plot_reconstruction_history(
            gamma_history,
            gamma,
            completed_epochs,
            Path(run_dir) / "figures" / f"{self.name}_case{case_id}_figure.svg",
        )
        save_histories(
            run_dir,
            f"{self.name}_case{case_id}_gamma_history",
            f"{self.name}_case{case_id}_cost_history",
            f"{self.name}_case{case_id}_mse_history",
            gamma_history,
            cost_history,
            mse_history,
        )
        target_gamma = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()
        save_outputs(run_dir, self.name, case_id, gamma_history[-1], target_gamma)

        checkpoint_dir = io.ensure_dir(Path(run_dir) / "checkpoints")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "source_checkpoint": str(checkpoint_path),
                "architecture": model.architecture_dict(),
                "gamma_min": float(model.gamma_min.detach().cpu()),
                "void_prior": float(model.void_prior),
                "gradient_normalization": norm_cfg,
                "initial_fwi_cost": float(initial_cost.detach().cpu()),
                "epochs_requested": epochs,
                "epochs_completed": completed_epochs,
                "trainable_mode": trainable_mode,
                "trainable_parameter_count": trainable_parameter_count,
                "best_observed_cost": best_observed_cost,
                "best_observed_epoch": best_observed_epoch + 1,
                "optimization_model_mode": optimization_model_mode,
                "restore_best_observed": restore_best_observed,
            },
            checkpoint_dir / f"{self.name}_case{case_id}_final.pt",
        )

        return ExperimentResult(
            method_name=self.name,
            case_id=case_id,
            gamma_history=gamma_history,
            cost_history=cost_history,
            mse_history=mse_history,
            final_gamma=gamma_history[-1],
            target_gamma=target_gamma,
            run_dir=Path(run_dir),
            metadata={
                "epochs": completed_epochs,
                "pretrained_checkpoint": str(checkpoint_path),
                "initial_fwi_cost": float(initial_cost.detach().cpu()),
                "trainable_mode": trainable_mode,
                "trainable_parameter_count": trainable_parameter_count,
                "best_observed_cost": best_observed_cost,
                "best_observed_epoch": best_observed_epoch + 1,
            },
        )
