"""Experimental multi-case GPU batching for Colab and large-memory GPUs.

This module deliberately leaves :mod:`src.data_generation` unchanged.  It
batches independent material cases through both the fine forward solve and the
coarse forward/adjoint solves.  Peak GPU memory grows approximately linearly
with ``case_batch_size`` because the adjoint calculation retains two complete
coarse-grid wavefields per case.
"""

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import adjoint, io, utils
from src.data_generation import _prepare_generation


def _forward_receiver_batch(solver, gamma, context):
    """Run fine-grid cases together and retain receiver traces only."""
    fine = context["fine"]
    params = context["params"]
    device = context["device"]
    batch_size = gamma.shape[0]
    sources = params["numberOfSources"]
    simulations = batch_size * sources

    u0 = torch.zeros(
        (simulations, 1, fine["Nx"] + 3, fine["Ny"] + 3),
        dtype=torch.float32,
        device=device,
    )
    u1 = torch.zeros_like(u0)
    gamma_flat = gamma.repeat_interleave(sources, dim=0)
    f_t = torch.zeros_like(u0)
    measurements = torch.empty(
        (simulations, len(fine["selx"]), fine["N"]),
        dtype=torch.float32,
        device=device,
    )

    rows = torch.arange(simulations, device=device)
    source_x = torch.as_tensor(
        fine["sourceLocationsx"] * batch_size, dtype=torch.long, device=device
    )
    source_y = torch.as_tensor(
        fine["sourceLocationsy"] * batch_size, dtype=torch.long, device=device
    )

    for step in range(fine["N"]):
        f_t.zero_()
        f_t[rows, 0, source_x, source_y] = fine["source_time"][step]
        u2 = solver.forward(u0, u1, gamma_flat, f_t)
        u0, u1 = u1, u2
        measurements[:, :, step] = u2[:, 0, fine["selx"], fine["sely"]]

    return measurements.view(batch_size, sources, len(fine["selx"]), fine["N"])


def _forward_full_batch(solver, gamma, source, time_steps, batch_size):
    """Forward solve with a shared per-source signal and a full saved wavefield."""
    sources, _, height, width, _ = source.shape
    simulations = batch_size * sources
    device = gamma.device
    u0 = torch.zeros((simulations, 1, height, width), dtype=torch.float32, device=device)
    u1 = torch.zeros_like(u0)
    wavefield = torch.empty(
        (simulations, height, width, time_steps + 1),
        dtype=torch.float32,
        device=device,
    )
    wavefield[..., 0] = 0

    for step in range(time_steps):
        forcing = source[..., step].repeat(batch_size, 1, 1, 1)
        u2 = solver.forward(u0, u1, gamma, forcing)
        u0, u1 = u1, u2
        wavefield[..., step + 1] = u2[:, 0]
    return wavefield


def _forward_full_dense_batch(solver, gamma, source):
    """Forward solve for an already batched dense time-dependent source."""
    simulations, _, height, width, time_steps = source.shape
    device = gamma.device
    u0 = torch.zeros((simulations, 1, height, width), dtype=torch.float32, device=device)
    u1 = torch.zeros_like(u0)
    wavefield = torch.empty(
        (simulations, height, width, time_steps + 1),
        dtype=torch.float32,
        device=device,
    )
    wavefield[..., 0] = 0

    for step in range(time_steps):
        u2 = solver.forward(u0, u1, gamma, source[..., step])
        u0, u1 = u1, u2
        wavefield[..., step + 1] = u2[:, 0]
    return wavefield


def _adjoint_gradient_batch(context, measurements):
    """Return one homogeneous-model adjoint gradient per material case."""
    params = context["params"]
    coarse = context["coarse"]
    device = context["device"]
    batch_size, sources, sensors, time_steps = measurements.shape
    simulations = batch_size * sources
    height = params["Nx"] + 3
    width = params["Ny"] + 3
    gamma = torch.ones((simulations, 1, height, width), dtype=torch.float32, device=device)

    forward = _forward_full_batch(
        coarse["solver"], gamma, coarse["F"], time_steps, batch_size
    )
    observed = measurements.reshape(simulations, sensors, time_steps)
    residual = forward[:, params["selx"], params["sely"], 1:] - observed

    adjoint_source = torch.zeros(
        (simulations, 1, height, width, time_steps),
        dtype=torch.float32,
        device=device,
    )
    adjoint_source[:, 0, params["selx"], params["sely"], :] = -residual
    adjoint_source = torch.flip(adjoint_source, dims=(4,))
    adjoint_wavefield = _forward_full_dense_batch(
        coarse["solver"], gamma, adjoint_source
    )
    adjoint_wavefield = torch.flip(adjoint_wavefield, dims=(3,))

    integrand = -params["rho"] * (
        adjoint.getDerivativeInT(adjoint_wavefield, params["dt"])
        * adjoint.getDerivativeInT(forward, params["dt"])
    )[:, 1:-1, 1:-1, :] + params["rho"] * params["c"] ** 2 * (
        (
            adjoint.getDerivativeInX(adjoint_wavefield, params["dx"])
            * adjoint.getDerivativeInX(forward, params["dx"])
        )[:, :, 1:-1, 1:-1]
        + (
            adjoint.getDerivativeInY(adjoint_wavefield, params["dy"])
            * adjoint.getDerivativeInY(forward, params["dy"])
        )[:, 1:-1, :, 1:-1]
    )

    per_source = torch.trapz(integrand, dx=params["dt"], axis=3)
    gradients = per_source.view(
        batch_size, sources, params["Nx"] + 1, params["Ny"] + 1
    ).mean(dim=1)
    return gradients * params["dx"] * params["dy"] * 2


def _preflight(output_dir, case_ids, source_values, overwrite):
    if overwrite:
        return
    collisions = [
        output_dir / f"{prefix}{case_id}.h5"
        for case_id in case_ids
        for prefix in ("material", "parametersMaterial", "measurement", "gradient")
        if (output_dir / f"{prefix}{case_id}.h5").exists()
    ]
    if collisions:
        preview = ", ".join(str(path) for path in collisions[:10])
        suffix = " ..." if len(collisions) > 10 else ""
        raise FileExistsError(
            f"Refusing to overwrite {len(collisions)} existing case file(s): {preview}{suffix}"
        )
    source_path = output_dir / "source.h5"
    if source_path.exists():
        existing = io.load_hdf(source_path)
        if existing.shape != source_values.shape or not np.array_equal(existing, source_values):
            raise FileExistsError(f"Existing {source_path} is incompatible with this config.")


def generate_dataset_colab(
    config,
    split="train",
    output_dir=None,
    start_case_id=0,
    number_of_cases=None,
    case_batch_size=2,
    overwrite=True,
):
    """Generate independent cases in GPU batches for large-memory accelerators."""
    data_cfg = config["data_generation"]
    split_cfg = data_cfg[split]
    torch.manual_seed(int(data_cfg["seed"]))
    np.random.seed(int(data_cfg["seed"]))

    if output_dir is None:
        path_key = "train_data" if split == "train" else "test_data"
        output_dir = config["paths"][path_key]
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)

    start_case_id = int(start_case_id)
    number_of_cases = int(
        split_cfg["number_of_cases"] if number_of_cases is None else number_of_cases
    )
    case_batch_size = int(case_batch_size)
    if start_case_id < 0 or number_of_cases < 0:
        raise ValueError("start_case_id and number_of_cases must be non-negative.")
    if case_batch_size < 1:
        raise ValueError("case_batch_size must be at least 1.")

    context = _prepare_generation(config, split_cfg["factor_to_avoid_inverse_crime"])
    params = context["params"]
    fine = context["fine"]
    factor = context["factor"]
    device = context["device"]
    source_values = context["coarse"]["F"].reshape(-1, params["N"]).cpu().numpy()
    all_case_ids = list(range(start_case_id, start_case_id + number_of_cases))
    _preflight(output_dir, all_case_ids, source_values, overwrite)

    print(f"Device: {device}; case batch size: {case_batch_size}")
    started = time.perf_counter()
    for offset in tqdm(range(0, number_of_cases, case_batch_size), desc=f"{split} batches"):
        batch_ids = all_case_ids[offset : offset + case_batch_size]
        materials = []
        parameters = []
        for _ in batch_ids:
            gamma_case = torch.ones(
                (1, 1, fine["Nx"] + 3, fine["Ny"] + 3),
                dtype=torch.float32,
                device=device,
            )
            last_parameters = None
            for _ in range(int(split_cfg["number_of_damages"])):
                gamma_damage, x0, y0, a, b, theta = utils.generateGamma(
                    fine["Nx"], fine["Ny"], fine["dx"], fine["dy"],
                    params["Lx"], params["Ly"], params["gamma0"]
                )
                gamma_case *= gamma_damage.to(device)
                last_parameters = [x0, y0, a, b, theta]
            materials.append(gamma_case)
            parameters.append(last_parameters)

        gamma_fine = torch.cat(materials, dim=0)
        measurements = _forward_receiver_batch(fine["solver"], gamma_fine, context)
        measurements = measurements[..., (factor - 1)::factor]
        gradients = _adjoint_gradient_batch(context, measurements)
        gamma_coarse = gamma_fine[:, 0, 1:-1:factor, 1:-1:factor]

        for index, case_id in enumerate(batch_ids):
            io.save_hdf(output_dir / f"material{case_id}.h5", gamma_coarse[index].cpu().numpy(), key="gamma")
            io.save_hdf(output_dir / f"parametersMaterial{case_id}.h5", parameters[index], key="parameters")
            io.save_hdf(
                output_dir / f"measurement{case_id}.h5",
                measurements[index].reshape(-1, params["N"]).cpu().numpy(),
                key="U",
            )
            io.save_hdf(output_dir / f"gradient{case_id}.h5", gradients[index].cpu().numpy(), key="U")

        del gamma_fine, measurements, gradients, gamma_coarse, materials
        if device.type == "cuda":
            torch.cuda.empty_cache()

    source_path = output_dir / "source.h5"
    if overwrite or not source_path.exists():
        io.save_hdf(source_path, source_values, key="f")
    print(f"Generated {number_of_cases} cases in {time.perf_counter() - started:.2f}s")

