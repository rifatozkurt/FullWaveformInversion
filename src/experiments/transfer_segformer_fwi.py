import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import adjoint
from src import io
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
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    selected = 0
    for name, parameter in model.named_parameters():
        use_parameter = "decode_head" in name
        if mode == "decoder_plus_last_stage":
            use_parameter = use_parameter or "encoder.block.3" in name or "encoder.patch_embeddings.3" in name
        if use_parameter:
            parameter.requires_grad = True
            selected += parameter.numel()

    if selected == 0:
        raise ValueError(f"Trainable mode selected no SegFormer parameters: {mode}")


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
        set_segformer_trainable_mode(model, cfg.get("trainable_mode", "all"))
        norm_cfg = dict(checkpoint["gradient_normalization"])
        first_gradient = first_gradient_2d.unsqueeze(0).unsqueeze(0)
        input_data = NN.normalize_gradient_for_transformer(first_gradient, **norm_cfg).detach()

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        epochs = int(cfg["epochs"])
        cost_history = np.zeros(epochs)
        mse_history = np.zeros(epochs)
        gamma_history = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))

        with torch.no_grad():
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
            model.train()

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
            gamma_pred.grad[0, 0, 1:-1, 1:-1] = gradient * float(cfg["costScaling"])
            gamma_pred.backward(gamma_pred.grad)
            if "clipGrad" in cfg:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clipGrad"]))
            optimizer.step()
            scheduler.step()

            cost_history[epoch] = float(cost.detach().cpu())
            mse_history[epoch] = float(0.5 * torch.mean((gamma_pred[0] - gamma) ** 2).detach().cpu())
            gamma_history[epoch + 1] = gamma_pred[0, 0, 1:-1, 1:-1].detach().cpu().numpy()

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

        plot_reconstruction_history(
            gamma_history,
            gamma,
            epochs,
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
                "epochs": epochs,
                "trainable_mode": cfg.get("trainable_mode", "all"),
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
                "epochs": epochs,
                "pretrained_checkpoint": str(checkpoint_path),
                "initial_fwi_cost": float(initial_cost.detach().cpu()),
                "trainable_mode": cfg.get("trainable_mode", "all"),
            },
        )
