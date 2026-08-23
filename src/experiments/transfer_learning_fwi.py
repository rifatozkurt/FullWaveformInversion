import copy
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


class TransferLearningFWI:
    name = "transfer_learning_fwi"

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

        sample = int(exp_cfg["pretrain_samples"][0])
        NNchannels = exp_cfg["NNchannels"]
        numberOfConvolutionsPerBlock = int(exp_cfg["numberOfConvolutionsPerBlock"])
        modelType = exp_cfg["modelType"]
        epochs_pretrain = int(exp_cfg["epochs_pretrain"])
        path = model_path(
            self.config,
            modelType,
            epochs_pretrain,
            "supervised",
            sample,
            NNchannels,
        )

        gamma, initialGradient, um, F = load_case_data(
            case_id, data_dir, params, device, load_gradient=True
        )
        u0, u1 = create_initial_conditions(params, device)

        torch.manual_seed(int(cfg["seed"]))
        model = NN.Unet(
            NNchannels, numberOfConvolutionsPerBlock, params["gamma0"], bnorm=True
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        forwardSolver = create_forward_solver(params, device)
        inputData = normalize_input_data(initialGradient, self.config).to(device)

        gammaPred = torch.ones(
            (1, 1, params["Nx"] + 3, params["Ny"] + 3),
            device=device,
            dtype=torch.float32,
        )
        gammaPred[:, :, 1:-1, 1:-1] = model(inputData)

        epochs = int(cfg["epochs"])
        optimizer = torch.optim.Adam(model.parameters(), cfg["lr"])
        lr_lambda = lambda epoch: (cfg["beta"] * epoch + 1) ** cfg["alpha"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        costHistory = np.zeros(epochs)
        mseHistory = np.zeros(epochs)
        gammaHistory = np.zeros((epochs + 1, params["Nx"] + 1, params["Ny"] + 1))
        gammaHistory[0] = gammaPred[0, 0, 1:-1, 1:-1].cpu().detach().numpy()

        # Restore-best-observed, matching transfer_segformer_fwi so the two
        # methods are selected by the same rule. The FWI cost is not monotone,
        # so reporting whatever the final iterate happens to be adds noise.
        restore_best_observed = bool(cfg.get("restore_best_observed", False))
        best_observed_cost = float("inf")
        best_observed_epoch = -1
        best_model_state = None

        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            model.train()

            gammaPred = torch.ones(
                (1, 1, params["Nx"] + 3, params["Ny"] + 3),
                device=device,
                dtype=torch.float32,
            )
            gammaPred[:, :, 1:-1, 1:-1] = model(inputData)
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

            observed_cost = float(cost.detach().cpu())
            costHistory[epoch] = observed_cost
            # Memory/transfer patch: compute MSE on-device instead of copying
            # the full predicted gamma to CPU each epoch.
            mseHistory[epoch] = metrics.gamma_mse(gammaPred, gamma, ghost=1)
            gammaHistory[epoch + 1] = gammaPred[0, 0, 1:-1, 1:-1].detach().cpu()

            if observed_cost < best_observed_cost:
                best_observed_cost = observed_cost
                best_observed_epoch = epoch
                if restore_best_observed:
                    best_model_state = copy.deepcopy(model.state_dict())

            elapsed_time = time.perf_counter() - start
            if epoch % 2 == 0:
                string = "Epoch: {}/{}\t\tCost function: {:.7E}\t\tMSE: {:.7E}\t\tElapsed time: {:2f}"
                print(string.format(epoch, epochs - 1, costHistory[epoch], mseHistory[epoch], elapsed_time))
            start = time.perf_counter()

        if restore_best_observed and best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                restored = torch.ones(
                    (1, 1, params["Nx"] + 3, params["Ny"] + 3),
                    device=device,
                    dtype=torch.float32,
                )
                restored[:, :, 1:-1, 1:-1] = model(inputData)
            gammaHistory[-1] = restored[0, 0, 1:-1, 1:-1].detach().cpu()
            mseHistory[-1] = metrics.gamma_mse(restored, gamma, ghost=1)
            costHistory[-1] = best_observed_cost

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
            metadata={
                "epochs": epochs,
                "restore_best_observed": restore_best_observed,
                "best_observed_cost": best_observed_cost,
                "best_observed_epoch": best_observed_epoch + 1,
                "pretrain_samples": sample,
                "epochs_pretrain": epochs_pretrain,
            },
        )
