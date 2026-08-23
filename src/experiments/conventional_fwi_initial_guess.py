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
    model_path,
    normalize_input_data,
    plot_reconstruction_history,
    save_histories,
    save_outputs,
    simulation_parameters,
)


class ConventionalFWIWithInitialGuess:
    name = "conventional_fwi_initial_guess"

    def __init__(self, config, device=None):
        self.config = config
        self.device = get_device(device)
        self.params = simulation_parameters(config)

    def run(self, case_id: int, data_dir: Path, run_dir: Path):
        params = self.params
        cfg = self.config["experiments"][self.name]
        exp_cfg = self.config["experiments"]
        device = self.device
        print("Using device: {}".format(device))
        print("Number of sensors: {:d}".format(len(params["selx"])))
        print(case_id)

        gamma, initialGradient, um, F = load_case_data(
            case_id, data_dir, params, device, load_gradient=True
        )

        NNchannels = exp_cfg["NNchannels"]
        numberOfConvolutionsPerBlock = int(exp_cfg["numberOfConvolutionsPerBlock"])
        modelType = exp_cfg["modelType"]
        epochs_pretrain = int(exp_cfg["epochs_pretrain"])
        pretrain_samples = int(cfg["pretrain_samples"])
        path = model_path(
            self.config,
            modelType,
            epochs_pretrain,
            "supervised",
            pretrain_samples,
            NNchannels,
        )

        torch.manual_seed(int(cfg["seed"]))
        model = NN.Unet(
            NNchannels, numberOfConvolutionsPerBlock, params["gamma0"], bnorm=True
        )
        model.load_state_dict(torch.load(path, map_location=device))

        forwardSolver = create_forward_solver(params, device)
        inputData = normalize_input_data(initialGradient, self.config).to(device)

        epochs = int(cfg["epochs"])
        u0, u1 = create_initial_conditions(params, device)
        gammaPred = torch.nn.Parameter(
            torch.ones((1, 1, params["Nx"] + 3, params["Ny"] + 3), device=device)
        )
        gammaPred.data[0, 0, 1:-1, 1:-1] = model(inputData)
        gammaPred.grad = torch.zeros_like(gammaPred, device=device)

        optimizer = torch.optim.Adam((gammaPred,), lr=cfg["lr"])
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        historyCost = np.zeros(epochs)
        historyMSE = np.zeros(epochs)
        historyGamma = np.zeros((epochs + 1, params["Nx"] + 3, params["Ny"] + 3))
        start = time.perf_counter()
        historyGamma[0] = gammaPred.detach().cpu()

        for epoch in range(epochs):
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
            gammaPred.grad[0, 0, params["selx"], params["sely"]] = 0
            gammaPred.grad[0, 0, params["sourceLocationsx"], params["sourceLocationsy"]] = 0
            gammaPred.backward(gammaPred.grad)

            torch.nn.utils.clip_grad_norm_(gammaPred, cfg["clipGrad"])
            optimizer.step()
            scheduler.step()
            gammaPred.data = gammaPred.data.clamp(params["gamma0"], 1)

            historyCost[epoch] = cost
            # Memory/transfer patch: compute MSE on-device instead of copying
            # the full predicted gamma to CPU each epoch.
            historyMSE[epoch] = metrics.gamma_mse(gammaPred, gamma, ghost=1)
            historyGamma[epoch + 1] = gammaPred.detach().cpu()
            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = "Epoch: {}/{}\tCost function: {:.3E}\tMSE: {:.3E}\tElapsed time: {:2f}"
                print(string.format(epoch, epochs - 1, historyCost[epoch], historyMSE[epoch], elapsed_time))
            start = time.perf_counter()

        plot_reconstruction_history(
            historyGamma,
            gamma,
            epochs,
            Path(run_dir) / "figures" / f"{self.name}_case{case_id}_figure.svg",
        )
        save_histories(
            run_dir,
            f"{self.name}_case{case_id}_gamma_history",
            f"{self.name}_case{case_id}_cost_history",
            f"{self.name}_case{case_id}_mse_history",
            historyGamma,
            historyCost,
            historyMSE,
        )
        target_gamma = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()
        save_outputs(run_dir, self.name, case_id, historyGamma[-1], target_gamma)

        return ExperimentResult(
            method_name=self.name,
            case_id=case_id,
            gamma_history=historyGamma,
            cost_history=historyCost,
            mse_history=historyMSE,
            final_gamma=historyGamma[-1],
            target_gamma=target_gamma,
            run_dir=Path(run_dir),
            metadata={"epochs": epochs, "pretrain_samples": pretrain_samples},
        )
