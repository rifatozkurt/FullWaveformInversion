from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import finite_difference as FiniteDifference
from src import io
from src import utils

mpl.rcParams.update({"font.size": 14})
mpl.rc("image", cmap="coolwarm")


def case_file_stem(case_id):
    case_id = str(case_id)
    if case_id.isdigit():
        return case_id
    return "_" + case_id


@dataclass
class ExperimentResult:
    method_name: str
    case_id: int
    gamma_history: np.ndarray
    cost_history: np.ndarray
    mse_history: np.ndarray
    final_gamma: np.ndarray
    target_gamma: np.ndarray
    run_dir: Path
    metadata: dict


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulation_parameters(config):
    sim = config["simulation"]
    Lx = sim["Lx"]
    Ly = sim["Ly"]
    Nx = int(sim["Nx"])
    Ny = int(sim["Ny"])
    dx = Lx / Nx
    dy = Ly / Ny
    dt = sim["dt"]
    N = int(sim["N"])
    gamma0 = sim["gamma0"]
    rho = sim["rho"]
    c = sim["c"]
    numberOfSources = int(sim["numberOfSources"])
    distanceBetweenSources = int(sim["distanceBetweenSources"])
    distanceBetweenSensors = int(sim["distanceBetweenSensors"])
    sourceLocationsx, sourceLocationsy = utils.getSourceLocations(
        Nx, Ny, distanceBetweenSources, numberOfSources
    )
    selx, sely = utils.getSensorLocations(
        Nx, Ny, distanceBetweenSensors, sourceLocationsx
    )
    return {
        "Lx": Lx,
        "Ly": Ly,
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "N": N,
        "gamma0": gamma0,
        "rho": rho,
        "c": c,
        "numberOfSources": numberOfSources,
        "distanceBetweenSources": distanceBetweenSources,
        "distanceBetweenSensors": distanceBetweenSensors,
        "sourceLocationsx": sourceLocationsx,
        "sourceLocationsy": sourceLocationsy,
        "selx": selx,
        "sely": sely,
        "numberOfSensors": len(selx),
    }


def create_initial_conditions(params, device):
    return (
        torch.zeros(
            (params["numberOfSources"], 1, params["Nx"] + 3, params["Ny"] + 3),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(
            (params["numberOfSources"], 1, params["Nx"] + 3, params["Ny"] + 3),
            dtype=torch.float32,
            device=device,
        ),
    )


def create_forward_solver(params, device):
    return FiniteDifference.FiniteDifference(
        params["dt"], params["dx"], params["dy"], params["c"], params["rho"], device=device
    )


def load_case_data(case_id, data_dir, params, device, load_gradient=False):
    data_dir = Path(data_dir)
    Nx = params["Nx"]
    Ny = params["Ny"]
    N = params["N"]
    numberOfSources = params["numberOfSources"]
    numberOfSensors = params["numberOfSensors"]

    # Memory/transfer patch: load case tensors directly on the selected device.
    # The legacy scripts often built CPU tensors and moved/cast during assignment;
    # direct device allocation avoids extra copies and prevents CPU/GPU assignment
    # problems. Tensor shapes and saved file formats are unchanged.
    gamma = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
    stem = case_file_stem(case_id)

    gamma[0, 0, 1:-1, 1:-1] = torch.tensor(
        io.load_hdf(data_dir / f"material{stem}.h5")
    ).to(device=device, dtype=torch.float32)

    um = torch.zeros((1, numberOfSources, numberOfSensors, N), dtype=torch.float32, device=device)
    um[0, :, :, :] = torch.tensor(
        io.load_hdf(data_dir / f"measurement{stem}.h5")
    ).view(1, numberOfSources, numberOfSensors, N).to(device=device, dtype=torch.float32)

    source = torch.tensor(io.load_hdf(data_dir / "source.h5")).reshape(
        (numberOfSources, 1, Nx + 3, Ny + 3, N)
    ).to(device=device, dtype=torch.float32)

    if not load_gradient:
        return gamma, um, source

    initialGradient = torch.ones((1, 1, Nx + 1, Ny + 1), dtype=torch.float32, device=device)
    initialGradient[0, 0, :, :] = torch.tensor(
        io.load_hdf(data_dir / f"gradient{stem}.h5")
    ).to(device=device, dtype=torch.float32)
    return gamma, initialGradient, um, source


def normalize_input_data(inputData):
    return (inputData - torch.amin(inputData, (2, 3), keepdim=True)) / (
        torch.amax(inputData, (2, 3), keepdim=True)
        - torch.amin(inputData, (2, 3), keepdim=True)
    ) * 2 - 1


def model_path(config, model_type, epochs, training_type, samples, channels):
    return (
        Path(config["paths"]["pretrained_models"])
        / f"model_{model_type}_{epochs}_{training_type}_{samples}_channel_{len(channels)}"
    )


def save_histories(
    run_dir,
    gamma_name,
    cost_name,
    mse_name,
    gamma_history,
    cost_history,
    mse_history,
):
    run_dir = Path(run_dir)
    io.save_hdf(
        run_dir / "histories" / f"{gamma_name}.h5",
        gamma_history.reshape(gamma_history.shape[0], -1),
        key="f",
    )
    io.save_history(run_dir / "histories" / f"{cost_name}.txt", cost_history)
    io.save_history(run_dir / "histories" / f"{mse_name}.txt", mse_history)


def save_outputs(run_dir, method_name, case_id, final_gamma, target_gamma):
    run_dir = Path(run_dir)
    io.save_hdf(
        run_dir / "outputs" / f"{method_name}_case{case_id}_final_gamma.h5",
        final_gamma,
        key="gamma",
    )
    io.save_hdf(
        run_dir / "outputs" / f"{method_name}_case{case_id}_target_gamma.h5",
        target_gamma,
        key="gamma",
    )


def plot_reconstruction_history(history_gamma, gamma, epochs, path):
    path = Path(path)
    io.ensure_dir(path.parent)

    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    sample_count = len(axes) - 2
    indices = np.linspace(0, max(len(history_gamma) - 1, 0), sample_count, dtype=int)

    image = None
    for axis, history_index in zip(axes[:sample_count], indices):
        image = axis.imshow(
            np.transpose(history_gamma[history_index, 1:-1, 1:-1]),
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"epoch {history_index}", fontsize=10)
        axis.axis("off")

    image = axes[-2].imshow(np.transpose(history_gamma[-1, 1:-1, 1:-1]), vmin=0, vmax=1)
    axes[-2].set_title("final", fontsize=10)
    axes[-2].axis("off")
    image = axes[-1].imshow(
        np.transpose(gamma[0, 0, 1:-1, 1:-1].detach().cpu()),
        vmin=0,
        vmax=1,
    )
    axes[-1].set_title("target", fontsize=10)
    axes[-1].axis("off")

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
        colorbar.set_label("gamma")

    plt.savefig(path)
    plt.close()
