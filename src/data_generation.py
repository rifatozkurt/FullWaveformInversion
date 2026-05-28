import time
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
from tqdm import tqdm

from src import adjoint
from src import finite_difference as FiniteDifference
from src import io
from src import utils
from src.experiments.base import get_device, simulation_parameters
from src.experiments.base import case_file_stem


def _forward_sensor_measurements(
    solver,
    u0,
    u1,
    gamma,
    source_time,
    source_locations_x,
    source_locations_y,
    sensor_locations_x,
    sensor_locations_y,
    Nx,
    Ny,
    N,
    number_of_sources,
    device,
):
    # Memory patch: the legacy forwardNSteps(...) stores the full fine-grid
    # wavefield U with shape (sources, Nx+3, Ny+3, N+1). For full data
    # generation this can be ~8 GiB on the fine grid. Synthetic data only needs
    # receiver traces, so this uses the same finite-difference update but stores
    # only Um[:, sensors, time]. To revert, replace this call with
    # solver.forwardNSteps(... )[:, selx, sely, 1:].
    f_t = torch.zeros((number_of_sources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
    Um = torch.zeros(
        (number_of_sources, len(sensor_locations_x), N),
        dtype=torch.float32,
        device=device,
    )

    for i in range(N):
        f_t.zero_()
        for i_source in range(number_of_sources):
            f_t[
                i_source,
                0,
                source_locations_x[i_source],
                source_locations_y[i_source],
            ] = source_time[i]
        u2 = solver.forward(u0.clone(), u1.clone(), gamma, f_t)
        u0[:], u1[:] = u1, u2
        Um[:, :, i] = u2[:, 0, sensor_locations_x, sensor_locations_y]

    return Um


def _prepare_generation(config, factor_to_avoid_inverse_crime):
    source = config["source"]
    params = simulation_parameters(config)
    factor = int(factor_to_avoid_inverse_crime)
    device = get_device()

    Lx = params["Lx"]
    Ly = params["Ly"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    dx = params["dx"]
    dy = params["dy"]
    dt = params["dt"]
    N = params["N"]
    rho = params["rho"]
    c = params["c"]
    numberOfSources = params["numberOfSources"]
    sourceLocationsx = params["sourceLocationsx"]
    sourceLocationsy = params["sourceLocationsy"]
    selx = params["selx"]
    sely = params["sely"]

    Nx_ = Nx * factor
    Ny_ = Ny * factor
    dx_ = Lx / Nx_
    dy_ = Ly / Ny_
    dt_ = dt / factor
    N_ = N * factor
    distanceBetweenSources_ = params["distanceBetweenSources"] * factor
    distanceBetweenSensors_ = params["distanceBetweenSensors"] * factor

    sourceLocationsx_, sourceLocationsy_ = utils.getSourceLocations(
        Nx_, Ny_, distanceBetweenSources_, numberOfSources
    )
    selx_, sely_ = utils.getSensorLocations(
        Nx_, Ny_, distanceBetweenSensors_, sourceLocationsx_
    )

    x = np.linspace(0 - dx, Lx + dx, Nx + 3)
    x_ = np.linspace(0 - dx_, Lx + dx_, Nx_ + 3)
    if np.sum((x[selx] - x_[selx_]) ** 2) > 1e-20:
        print("Error: sensor locations are unequal")

    print("Fine source mesh: ({}, {}, {})".format(Nx_ + 3, Ny_ + 3, N_))
    one_legacy_mesh_gib = (Nx_ + 3) * (Ny_ + 3) * N_ * 8 / 1024**3
    fine_source_tensor_gib = numberOfSources * (Nx_ + 3) * (Ny_ + 3) * N_ * 4 / 1024**3
    fine_wavefield_gib = numberOfSources * (Nx_ + 3) * (Ny_ + 3) * (N_ + 1) * 4 / 1024**3
    print("Approx. one legacy float64 mesh array: {:.2f} GiB".format(one_legacy_mesh_gib))
    print(
        "Approx. fine source tensor: {:.2f} GiB".format(
            fine_source_tensor_gib
        )
    )
    print(
        "Fine full wavefield if stored: {:.2f} GiB".format(
            fine_wavefield_gib
        )
    )

    setup_start = time.perf_counter()
    print("Creating fine-grid source time function...")
    # Uses a time vector for the fine-grid source instead of storing the full
    # fine-grid source tensor. The source is inserted into f_t each time step in
    # _forward_sensor_measurements(...).
    source_time_ = utils.getSourceTime(
        source["frequency"], source["cycles"], source["amplitude"],
        N_, dt_, dx_, dy_, device
    )

    print("Creating inversion-grid source...")
    # Same source values as utils.getSource(...), but without dense meshgrid
    # temporaries. The resulting F tensor shape is unchanged.
    F = utils.getSourceDirect(
        source["frequency"], source["cycles"], source["amplitude"],
        sourceLocationsx, sourceLocationsy,
        Nx, Ny, dx, dy, N, dt
    ).to(device)

    return {
        "params": params,
        "device": device,
        "factor": factor,
        "fine": {
            "Nx": Nx_,
            "Ny": Ny_,
            "dx": dx_,
            "dy": dy_,
            "dt": dt_,
            "N": N_,
            "u0": torch.zeros((numberOfSources, 1, Nx_ + 3, Ny_ + 3), dtype=torch.float32, device=device),
            "u1": torch.zeros((numberOfSources, 1, Nx_ + 3, Ny_ + 3), dtype=torch.float32, device=device),
            "solver": FiniteDifference.FiniteDifference(dt_, dx_, dy_, c, rho, device=device),
            "source_time": source_time_,
            "sourceLocationsx": sourceLocationsx_,
            "sourceLocationsy": sourceLocationsy_,
            "selx": selx_,
            "sely": sely_,
        },
        "coarse": {
            "u0": torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device),
            "u1": torch.zeros((numberOfSources, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device),
            "solver": FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device=device),
            "F": F,
        },
        "setup_seconds": time.perf_counter() - setup_start,
    }


def _generate_case_from_context(context, case_id, output_dir, n_damages=1, show_progress=True):
    params = context["params"]
    device = context["device"]
    factor = context["factor"]
    fine = context["fine"]
    coarse = context["coarse"]
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)

    Nx = params["Nx"]
    Ny = params["Ny"]
    dx = params["dx"]
    dy = params["dy"]
    dt = params["dt"]
    N = params["N"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    gamma0 = params["gamma0"]
    rho = params["rho"]
    c = params["c"]
    numberOfSources = params["numberOfSources"]
    selx = params["selx"]
    sely = params["sely"]

    progress = tqdm(total=4, desc=f"case {case_id}", leave=False) if show_progress else None
    try:
        case_start = time.perf_counter()
        gamma_ = torch.ones((1, 1, fine["Nx"] + 3, fine["Ny"] + 3)).to(device)
        for i in range(n_damages):
            gamma_t, x0, y0, a, b, theta = utils.generateGamma(
                fine["Nx"], fine["Ny"], fine["dx"], fine["dy"], Lx, Ly, gamma0
            )
            gamma_ *= gamma_t.to(device)
        if progress is not None:
            progress.update(1)

        fine_start = time.perf_counter()
        Um = _forward_sensor_measurements(
            fine["solver"],
            fine["u0"].clone(),
            fine["u1"].clone(),
            gamma_,
            fine["source_time"],
            fine["sourceLocationsx"],
            fine["sourceLocationsy"],
            fine["selx"],
            fine["sely"],
            fine["Nx"],
            fine["Ny"],
            fine["N"],
            numberOfSources,
            device,
        )
        fine_seconds = time.perf_counter() - fine_start
        if progress is not None:
            progress.update(1)

        gamma = torch.ones((1, 1, Nx + 3, Ny + 3)).to(device)
        gamma[:, :, 1:-1, 1:-1] = gamma_[:, :, 1:-1:factor, 1:-1:factor]
        Um = Um[:, :, (factor - 1)::factor]

        adjoint_start = time.perf_counter()
        _, gradient = adjoint.getAdjointGradient(
            coarse["solver"],
            coarse["u0"].clone(),
            coarse["u1"].clone(),
            c,
            rho,
            gamma * 0 + 1,
            coarse["F"],
            Nx,
            dx,
            Ny,
            dy,
            N,
            dt,
            numberOfSources,
            Um,
            selx,
            sely,
            device,
        )
        adjoint_seconds = time.perf_counter() - adjoint_start
        if progress is not None:
            progress.update(1)

        gamma_np = gamma[0, 0, 1:-1, 1:-1].cpu().numpy()
        io.save_hdf(output_dir / f"material{case_id}.h5", gamma_np, key="gamma")
        io.save_hdf(output_dir / f"parametersMaterial{case_id}.h5", [x0, y0, a, b, theta], key="parameters")
        io.save_hdf(output_dir / f"measurement{case_id}.h5", Um.reshape(-1, N).cpu().numpy(), key="U")
        io.save_hdf(output_dir / f"gradient{case_id}.h5", gradient.cpu().numpy(), key="U")
        if progress is not None:
            progress.update(1)

        total_seconds = time.perf_counter() - case_start
        if progress is not None:
            progress.set_postfix(total_s=f"{total_seconds:.1f}")
    finally:
        if progress is not None:
            progress.close()

    return {
        "case_id": case_id,
        "fine_forward_seconds": fine_seconds,
        "adjoint_gradient_seconds": adjoint_seconds,
        "total_seconds": total_seconds,
    }


def generate_single_case(
    config,
    case_id: int,
    output_dir,
    n_damages: int = 1,
    factor_to_avoid_inverse_crime: int = 2,
):
    """Generate one synthetic case using the same logic as the legacy scripts."""
    context = _prepare_generation(config, factor_to_avoid_inverse_crime)
    result = _generate_case_from_context(context, case_id, output_dir, n_damages=n_damages)
    params = context["params"]
    io.save_hdf(
        Path(output_dir) / "source.h5",
        context["coarse"]["F"].reshape(-1, params["N"]).cpu().numpy(),
        key="f",
    )
    print(
        "case {}: fine_forward={:.2f}s, adjoint_gradient={:.2f}s, total={:.2f}s".format(
            case_id,
            result["fine_forward_seconds"],
            result["adjoint_gradient_seconds"],
            result["total_seconds"],
        )
    )
    return context["coarse"]["F"]


def generate_dataset(config, split: str = "train", output_dir=None):
    """
    Generate synthetic material fields, measurements, initial adjoint gradients,
    and source.h5 for either train or test split.
    """
    data_cfg = config["data_generation"]
    split_cfg = data_cfg[split]
    torch.manual_seed(int(data_cfg["seed"]))

    if output_dir is None:
        path_key = "train_data" if split == "train" else "test_data"
        output_dir = config["paths"][path_key]
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)

    context = _prepare_generation(config, split_cfg["factor_to_avoid_inverse_crime"])
    params = context["params"]
    print("Number of sensors: {:d}".format(len(params["selx"])))
    print("Shared setup time: {:.2f}s".format(context["setup_seconds"]))

    number_of_cases = int(split_cfg["number_of_cases"])
    n_damages = int(split_cfg["number_of_damages"])
    timings = []

    for case in tqdm(range(number_of_cases), desc=f"{split} cases"):
        timings.append(
            _generate_case_from_context(
                context,
                case,
                output_dir,
                n_damages=n_damages,
                show_progress=False,
            )
        )
        last = timings[-1]
        tqdm.write(
            "case {}: fine_forward={:.2f}s, adjoint_gradient={:.2f}s, total={:.2f}s".format(
                case,
                last["fine_forward_seconds"],
                last["adjoint_gradient_seconds"],
                last["total_seconds"],
            )
        )

    io.save_hdf(
        output_dir / "source.h5",
        context["coarse"]["F"].reshape(-1, params["N"]).cpu().numpy(),
        key="f",
    )

    if timings:
        total = sum(item["total_seconds"] for item in timings)
        print("Average case time: {:.2f}s".format(total / len(timings)))
        print("Total case time: {:.2f}s".format(total))


def _upsample_gamma_to_fine_grid(gamma, Nx, Ny, factor):
    """Nearest-neighbor upsample that preserves legacy fine-to-coarse slicing."""
    Nx_ = Nx * factor
    Ny_ = Ny * factor
    ix = np.rint(np.arange(Nx_ + 1) / factor).astype(int)
    iy = np.rint(np.arange(Ny_ + 1) / factor).astype(int)
    ix = np.clip(ix, 0, Nx)
    iy = np.clip(iy, 0, Ny)
    return gamma[ix[:, None], iy[None, :]]


def generate_case_from_gamma(config, gamma, case_name, output_dir):
    """
    Generate a custom case from a gamma field on the inversion/material grid.

    The drawn material is lifted to the finer reference grid, measurements are
    generated there, and receiver traces are downsampled to the inversion grid,
    matching the inverse-crime-avoidance convention used by train/test data.
    """
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)
    stem = case_file_stem(case_name)

    split_cfg = config.get("data_generation", {}).get("test", {})
    factor = int(split_cfg.get("factor_to_avoid_inverse_crime", 2))
    context = _prepare_generation(config, factor)
    params = context["params"]
    device = context["device"]
    fine = context["fine"]
    coarse = context["coarse"]

    Nx = params["Nx"]
    Ny = params["Ny"]
    dt = params["dt"]
    N = params["N"]
    gamma = np.asarray(gamma, dtype=np.float32)
    if gamma.shape != (Nx + 1, Ny + 1):
        raise ValueError(f"Expected gamma shape {(Nx + 1, Ny + 1)}, got {gamma.shape}")

    gamma_fine = _upsample_gamma_to_fine_grid(gamma, Nx, Ny, factor)
    gamma_fine_full = torch.ones(
        (1, 1, fine["Nx"] + 3, fine["Ny"] + 3),
        dtype=torch.float32,
        device=device,
    )
    gamma_fine_full[0, 0, 1:-1, 1:-1] = torch.as_tensor(
        gamma_fine,
        dtype=torch.float32,
        device=device,
    )

    Um = _forward_sensor_measurements(
        fine["solver"],
        fine["u0"].clone(),
        fine["u1"].clone(),
        gamma_fine_full,
        fine["source_time"],
        fine["sourceLocationsx"],
        fine["sourceLocationsy"],
        fine["selx"],
        fine["sely"],
        fine["Nx"],
        fine["Ny"],
        fine["N"],
        params["numberOfSources"],
        device,
    )
    Um = Um[:, :, (factor - 1)::factor]

    gamma_full = torch.ones((1, 1, Nx + 3, Ny + 3), dtype=torch.float32, device=device)
    gamma_full[0, 0, 1:-1, 1:-1] = torch.as_tensor(gamma, dtype=torch.float32, device=device)

    _, gradient = adjoint.getAdjointGradient(
        coarse["solver"],
        coarse["u0"].clone(),
        coarse["u1"].clone(),
        params["c"],
        params["rho"],
        gamma_full * 0 + 1,
        coarse["F"],
        Nx,
        params["dx"],
        Ny,
        params["dy"],
        N,
        dt,
        params["numberOfSources"],
        Um,
        params["selx"],
        params["sely"],
        device,
    )

    material_path = output_dir / f"material{stem}.h5"
    measurement_path = output_dir / f"measurement{stem}.h5"
    gradient_path = output_dir / f"gradient{stem}.h5"
    parameters_path = output_dir / f"parametersMaterial{stem}.h5"
    source_path = output_dir / "source.h5"

    io.save_hdf(material_path, gamma, key="gamma")
    io.save_hdf(measurement_path, Um.reshape(-1, N).cpu().numpy(), key="U")
    io.save_hdf(gradient_path, gradient.cpu().numpy(), key="U")
    io.save_hdf(parameters_path, [np.nan, np.nan, np.nan, np.nan, np.nan], key="parameters")
    io.save_hdf(source_path, coarse["F"].reshape(-1, N).cpu().numpy(), key="f")

    return {
        "material_path": material_path,
        "measurement_path": measurement_path,
        "gradient_path": gradient_path,
        "parameters_path": parameters_path,
        "source_path": source_path,
        "reference_grid_factor": factor,
    }


def _raw_case_ids(raw_dir):
    pattern = re.compile(r"material(\d+)\.h5$")
    case_ids = []
    for path in Path(raw_dir).glob("material*.h5"):
        match = pattern.match(path.name)
        if match:
            case_ids.append(int(match.group(1)))
    return sorted(case_ids)


def _copy_case_files(raw_dir, output_dir, old_case_id, new_case_id, overwrite=False):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)
    copied = []
    for prefix in ("material", "parametersMaterial", "measurement", "gradient"):
        source = raw_dir / f"{prefix}{old_case_id}.h5"
        if not source.exists():
            continue
        target = output_dir / f"{prefix}{new_case_id}.h5"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists. Enable overwrite or choose another output folder.")
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def split_raw_dataset(
    raw_dir="data/raw",
    train_dir="data/train",
    test_dir="data/test",
    train_ratio=0.8,
    seed=2,
    overwrite=False,
):
    """
    Split raw generated cases into contiguous train/test folders.

    Source case IDs are shuffled deterministically, then copied and reindexed
    from zero in each split so pretraining can sample range(availableSamples)
    exactly as the legacy script expects.
    """
    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    train_ratio = float(train_ratio)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    case_ids = _raw_case_ids(raw_dir)
    if not case_ids:
        raise FileNotFoundError(f"No numeric material*.h5 files found in {raw_dir}")

    missing_gradients = [case_id for case_id in case_ids if not (raw_dir / f"gradient{case_id}.h5").exists()]
    if missing_gradients:
        preview = ", ".join(str(item) for item in missing_gradients[:10])
        raise FileNotFoundError(f"Missing gradient files for raw case(s): {preview}")

    shuffled = list(case_ids)
    random.Random(int(seed)).shuffle(shuffled)
    number_train = int(round(len(shuffled) * train_ratio))
    number_train = min(max(number_train, 1), len(shuffled) - 1)
    train_ids = shuffled[:number_train]
    test_ids = shuffled[number_train:]

    train_copied = []
    test_copied = []
    for new_case_id, old_case_id in enumerate(train_ids):
        train_copied.extend(_copy_case_files(raw_dir, train_dir, old_case_id, new_case_id, overwrite=overwrite))
    for new_case_id, old_case_id in enumerate(test_ids):
        test_copied.extend(_copy_case_files(raw_dir, test_dir, old_case_id, new_case_id, overwrite=overwrite))

    source_path = raw_dir / "source.h5"
    source_targets = []
    if source_path.exists():
        for target_dir in (train_dir, test_dir):
            target = target_dir / "source.h5"
            if target.exists() and not overwrite:
                raise FileExistsError(f"{target} already exists. Enable overwrite or choose another output folder.")
            io.ensure_dir(target_dir)
            shutil.copy2(source_path, target)
            source_targets.append(target)

    return {
        "raw_dir": raw_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "total_cases": len(case_ids),
        "train_cases": len(train_ids),
        "test_cases": len(test_ids),
        "train_source_ids": train_ids,
        "test_source_ids": test_ids,
        "train_files_copied": len(train_copied),
        "test_files_copied": len(test_copied),
        "source_targets": source_targets,
    }
