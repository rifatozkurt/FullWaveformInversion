from dataclasses import dataclass
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch

from src import finite_difference as FiniteDifference
from src import io
from src import networks as NN
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


def gradient_normalization_config(config):
    """
    The single normalization setting shared by every model.

    Read from the top-level `gradient_normalization:` block so that pretraining
    and downstream FWI cannot drift apart, and so that the U-Net and the
    SegFormer are fed identically preprocessed inputs.
    """
    return dict((config or {}).get("gradient_normalization", {}) or {})


def normalize_input_data(inputData, config=None, **overrides):
    """
    Normalize an adjoint gradient for network input.

    Delegates to `networks.normalize_gradient`, which every architecture now
    shares. Pass `config` to pick up the run's `gradient_normalization` block;
    without it the default (`robust_abs`) applies.
    """
    settings = gradient_normalization_config(config)
    settings.update(overrides)
    return NN.normalize_gradient(inputData, **settings)


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
    plot_cost_mse_history(
        cost_history,
        mse_history,
        run_dir / "figures" / f"{mse_name}.png",
        title=mse_name.removesuffix("_mse_history").replace("_", " "),
    )


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
    target_gamma = gamma[0, 0, 1:-1, 1:-1].detach().cpu().numpy()

    def physical_frame(frame):
        if frame.shape == target_gamma.shape:
            return frame
        if frame.ndim == 2 and frame[1:-1, 1:-1].shape == target_gamma.shape:
            return frame[1:-1, 1:-1]
        return frame

    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    sample_count = len(axes) - 2
    indices = np.linspace(0, max(len(history_gamma) - 1, 0), sample_count, dtype=int)

    image = None
    for axis, history_index in zip(axes[:sample_count], indices):
        image = axis.imshow(
            np.transpose(physical_frame(history_gamma[history_index])),
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"epoch {history_index}", fontsize=10)
        axis.axis("off")

    image = axes[-2].imshow(np.transpose(physical_frame(history_gamma[-1])), vmin=0, vmax=1)
    axes[-2].set_title("final", fontsize=10)
    axes[-2].axis("off")
    image = axes[-1].imshow(
        np.transpose(target_gamma),
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


def plot_cost_mse_history(cost_history, mse_history, path, title=None):
    path = Path(path)
    io.ensure_dir(path.parent)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    axes[0].plot(np.atleast_1d(cost_history), color="#2f5aa8", linewidth=2)
    axes[0].set_title("Cost history", fontsize=11)
    axes[0].set_ylabel("cost")
    axes[0].grid(True, alpha=0.3)
    cost_formatter = ScalarFormatter(useMathText=True)
    cost_formatter.set_powerlimits((-3, 3))
    cost_formatter.set_useOffset(False)
    axes[0].yaxis.set_major_formatter(cost_formatter)

    axes[1].plot(np.atleast_1d(mse_history), color="#b64040", linewidth=2)
    axes[1].set_title("MSE history", fontsize=11)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    mse_formatter = ScalarFormatter(useMathText=True)
    mse_formatter.set_powerlimits((-5, 5))
    mse_formatter.set_useOffset(False)
    axes[1].yaxis.set_major_formatter(mse_formatter)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def tensor_stats(tensor, prefix):
    values = tensor.detach()
    return {
        f"{prefix}_min": float(values.min().cpu()),
        f"{prefix}_max": float(values.max().cpu()),
        f"{prefix}_mean": float(values.mean().cpu()),
        f"{prefix}_std": float(values.std().cpu()),
    }


def tensor_norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach()).cpu())


def parameter_grad_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def total_variation_loss(gamma_2d, tv_type="anisotropic", eps=1e-8):
    # TV is applied directly to the generated gamma image and backpropagated
    # through PyTorch; the FWI data gradient is still supplied by the adjoint.
    dx = gamma_2d[1:, :] - gamma_2d[:-1, :]
    dy = gamma_2d[:, 1:] - gamma_2d[:, :-1]

    if tv_type == "anisotropic":
        return dx.abs().mean() + dy.abs().mean()
    if tv_type == "isotropic":
        dx_c = dx[:, :-1]
        dy_c = dy[:-1, :]
        return torch.sqrt(dx_c**2 + dy_c**2 + eps).mean()
    raise ValueError(f"Unknown tv_type: {tv_type}")


def save_inr_diagnostics(run_dir, method_name, case_id, rows):
    if not rows:
        return None
    path = Path(run_dir) / "histories" / f"{method_name}_case{case_id}_diagnostics.csv"
    io.ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
