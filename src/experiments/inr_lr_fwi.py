import time
from pathlib import Path

import numpy as np
import torch

from src import adjoint
from src import metrics
from src import networks as NN
from src.experiments.base import (
    ExperimentResult,
    create_forward_solver,
    create_initial_conditions,
    get_device,
    load_case_data,
    parameter_grad_norm,
    plot_reconstruction_history,
    save_histories,
    save_inr_diagnostics,
    save_outputs,
    simulation_parameters,
    tensor_norm,
    tensor_stats,
    total_variation_loss,
)


class INRLrFWI:
    name = "inr_lr_fwi"

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

        gamma, um, F = load_case_data(case_id, data_dir, params, device)
        u0, u1 = create_initial_conditions(params, device)
        forwardSolver = create_forward_solver(params, device)

        torch.manual_seed(int(cfg["seed"]))
        model = NN.INR_LR(
            rank_x=int(cfg.get("rank_x", cfg.get("rank", 128))),
            rank_y=int(cfg.get("rank_y", cfg.get("rank", 64))),
            hidden_features=int(cfg["hidden_features"]),
            hidden_layers=int(cfg["hidden_layers"]),
            gamma0=params["gamma0"],
            omega0=float(cfg["omega0"]),
            output_mode=cfg.get("output_mode", "direct_gamma"),
            final_bias=float(cfg.get("final_bias", 3.0)),
            core_init_std=float(cfg.get("core_init_std", 1e-3)),
        ).to(device)

        x = torch.linspace(-1, 1, params["Nx"] + 1, device=device)
        y = torch.linspace(-1, 1, params["Ny"] + 1, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coords = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

        epochs = int(cfg["epochs"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        tv_weight = float(cfg.get("tv_weight", 0.0))
        tv_type = str(cfg.get("tv_type", "anisotropic"))
        use_tv = tv_weight > 0 and tv_type.lower() not in ("none", "null", "false", "off")

        costHistory = np.zeros(epochs)
        mseHistory = np.zeros(epochs)
        gammaHistory = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))
        gammaPred = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        gammaPred[:, :, 1:-1, 1:-1] = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
        gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()
        initial_stats = tensor_stats(gammaPred[0, 0, 1:-1, 1:-1], "initial_gamma")
        print(
            "Initial gamma stats: min={initial_gamma_min:.6f}, max={initial_gamma_max:.6f}, "
            "mean={initial_gamma_mean:.6f}, std={initial_gamma_std:.6f}".format(**initial_stats)
        )

        diagnostics = []
        target_inner = gamma[0, 0, 1:-1, 1:-1]
        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)

            gamma_inner = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
            gamma_before = gamma_inner.detach().clone()
            gammaPred = torch.ones(
                (1, 1, params["Nx"] + 3, params["Ny"] + 3),
                device=device,
                dtype=torch.float32,
            )
            gammaPred[:, :, 1:-1, 1:-1] = gamma_inner

            cost, gradient = adjoint.getAdjointGradient(
                forwardSolver,
                u0,
                u1,
                params["c"],
                params["rho"],
                gammaPred.detach(),
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
            scaled_gradient = gradient * cfg["costScaling"]
            external_grad = torch.zeros_like(gammaPred, device=device)
            # The adjoint routine gives dL/dgamma. PyTorch supplies dgamma/dtheta
            # through the low-rank INR so this external gradient updates weights.
            external_grad[0, 0, 1:-1, 1:-1] = scaled_gradient
            gammaPred.backward(external_grad, retain_graph=use_tv)

            tv_raw_value = 0.0
            tv_loss_value = 0.0
            if use_tv:
                tv_raw = total_variation_loss(gamma_inner, tv_type=tv_type)
                tv_loss = tv_weight * tv_raw
                tv_loss.backward()
                tv_raw_value = float(tv_raw.detach().cpu())
                tv_loss_value = float(tv_loss.detach().cpu())

            param_norm = parameter_grad_norm(model)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                gamma_after = model(coords).reshape(params["Nx"] + 1, params["Ny"] + 1)
            delta_gamma = gamma_after - gamma_before

            costHistory[epoch] = cost.detach().cpu()
            mseHistory[epoch] = metrics.gamma_mse(gamma_after, target_inner)
            gammaHistory[epoch + 1] = gamma_after.detach().cpu()

            row = {
                "epoch": epoch,
                "cost": float(cost.detach().cpu()),
                "mse": float(mseHistory[epoch]),
                **tensor_stats(gamma_inner, "gamma"),
                **tensor_stats(gradient, "adjoint_grad"),
                "adjoint_grad_norm": tensor_norm(gradient),
                "scaled_grad_min": float(scaled_gradient.detach().min().cpu()),
                "scaled_grad_max": float(scaled_gradient.detach().max().cpu()),
                "scaled_grad_norm": tensor_norm(scaled_gradient),
                "param_grad_norm": param_norm,
                "delta_gamma_mean_abs": float(delta_gamma.detach().abs().mean().cpu()),
                "delta_gamma_max_abs": float(delta_gamma.detach().abs().max().cpu()),
                "delta_gamma_std": float(delta_gamma.detach().std().cpu()),
                "tv_raw": tv_raw_value,
                "tv_loss": tv_loss_value,
                "tv_weight": tv_weight,
                "tv_type": tv_type,
            }
            diagnostics.append(row)

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = (
                    "Epoch: {}/{}\t\tCost function: {:.3E}\t\tMSE: {:.3E}\t\t"
                    "dGamma mean/max: {:.3E}/{:.3E}\t\tParam grad: {:.3E}\t\tElapsed time: {:2f}"
                )
                print(
                    string.format(
                        epoch,
                        epochs - 1,
                        costHistory[epoch],
                        mseHistory[epoch],
                        row["delta_gamma_mean_abs"],
                        row["delta_gamma_max_abs"],
                        row["param_grad_norm"],
                        elapsed_time,
                    )
                )
            start = time.perf_counter()

        plot_reconstruction_history(
            gammaHistory,
            gamma,
            epochs,
            Path(run_dir) / "figures" / f"{self.name}_case{case_id}_figure.svg",
        )
        save_histories(
            run_dir,
            f"{self.name}_case{case_id}_gamma_history",
            f"{self.name}_case{case_id}_cost_history",
            f"{self.name}_case{case_id}_mse_history",
            gammaHistory,
            costHistory,
            mseHistory,
        )
        save_inr_diagnostics(run_dir, self.name, case_id, diagnostics)
        target_gamma = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()
        save_outputs(run_dir, self.name, case_id, gammaHistory[-1], target_gamma)

        return ExperimentResult(
            method_name=self.name,
            case_id=case_id,
            gamma_history=gammaHistory,
            cost_history=costHistory,
            mse_history=mseHistory,
            final_gamma=gammaHistory[-1],
            target_gamma=target_gamma,
            run_dir=Path(run_dir),
            metadata={
                "epochs": epochs,
                "rank_x": int(cfg.get("rank_x", cfg.get("rank", 128))),
                "rank_y": int(cfg.get("rank_y", cfg.get("rank", 64))),
                "omega0": float(cfg["omega0"]),
                "core_init_std": float(cfg.get("core_init_std", 1e-3)),
                **initial_stats,
                "tv_weight": tv_weight,
                "tv_type": tv_type,
            },
        )
