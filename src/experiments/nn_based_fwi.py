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
    plot_reconstruction_history,
    save_histories,
    save_outputs,
    simulation_parameters,
)


class NNBasedFWI:
    name = "nn_based_fwi"

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
        model = NN.CNNGamma(params["gamma0"])
        NN.initWeights(model)
        torch.nn.init.normal_(model.convOut.weight, std=0.01)
        model.convOut.bias.data.fill_(3)
        model.to(device)

        inputData = torch.randn(tuple(cfg["input_shape"]), device=device)
        torch.nn.init.trunc_normal_(
            inputData,
            mean=cfg["trunc_normal_mean"],
            std=cfg["trunc_normal_std"],
            a=cfg["trunc_normal_a"],
            b=cfg["trunc_normal_b"],
        )
        inputData = (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
            torch.amax(inputData, (2, 3), keepdim=True)
            - torch.amin(inputData, (2, 3), keepdim=True)
        ) * 2 - 1

        epochs = int(cfg["epochs"])
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=cfg["l2"])
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        costHistory = np.zeros(epochs)
        mseHistory = np.zeros(epochs)
        gammaHistory = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))
        gammaPred = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        gammaPred[:, :, 2:-2, 2:-2] = model(inputData)
        gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

        start = time.perf_counter()
        for epoch in range(epochs):
            gammaPred = torch.ones(
                (1, 1, params["Nx"] + 3, params["Ny"] + 3),
                device=device,
                dtype=torch.float32,
            )
            gammaPred[:, :, 2:-2, 2:-2] = model(inputData)
            gammaPred.grad = torch.zeros_like(gammaPred.detach(), device=device)

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
            gammaPred.grad[0, 0, 1:-1, 1:-1] = gradient * cfg["costScaling"]
            gammaPred.backward(gammaPred.grad)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clipGrad"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            costHistory[epoch] = cost.detach().cpu()
            # Memory/transfer patch: compute MSE on-device instead of moving
            # the reference gamma to the device every epoch.
            mseHistory[epoch] = metrics.gamma_mse(gammaPred, gamma, ghost=1)
            gammaHistory[epoch + 1] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = "Epoch: {}/{}\t\tCost function: {:.3E}\t\tMSE: {:.3E}\t\tElapsed time: {:2f}"
                print(string.format(epoch, epochs - 1, costHistory[epoch], mseHistory[epoch], elapsed_time))
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
            metadata={"epochs": epochs},
        )
