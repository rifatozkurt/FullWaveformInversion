from pathlib import Path
import contextlib
import io as text_io
import queue
import shutil
import threading
import time
import traceback

import numpy as np

from src import io
from src.config import load_config, save_experiment_config
from src.data_generation import split_raw_dataset
from src.experiments.base import plot_cost_mse_history
from src.registry import EXPERIMENTS
from src.ui import case_viewer
from src.pretraining import pretrain_unet
from src.ui import canvas
from src.ui.config_forms import save_custom_config, save_pretrain_config
from src.ui.formatting import format_run_status


def refresh_cases():
    choices = case_viewer.case_choices()
    return choices, choices


def toggle_config_mode(mode):
    use_yaml = mode == "Use existing YAML path"
    return {
        "yaml_visible": use_yaml,
        "custom_visible": not use_yaml,
    }


def _history_figure_path(run_dir, method_name, case_id):
    path = Path(run_dir) / "figures" / f"{method_name}_case{case_id}_figure.svg"
    return path if path.exists() else None


def _gamma_image(path):
    if path is None or not Path(path).exists():
        return None
    return case_viewer.material_to_image(io.load_hdf(path))


def _cost_mse_plot(run_dir, method_name, case_id):
    run_dir = Path(run_dir)
    saved_plot_path = run_dir / "figures" / f"{method_name}_case{case_id}_mse_history.png"
    if saved_plot_path.exists():
        return saved_plot_path

    cost_path = run_dir / "histories" / f"{method_name}_case{case_id}_cost_history.txt"
    mse_path = run_dir / "histories" / f"{method_name}_case{case_id}_mse_history.txt"
    if not cost_path.exists() or not mse_path.exists():
        return None

    cost = np.loadtxt(cost_path, delimiter=",")
    mse = np.loadtxt(mse_path, delimiter=",")
    return plot_cost_mse_history(
        cost,
        mse,
        saved_plot_path,
        title=f"{method_name} case {case_id}",
    )


class _QueueWriter:
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def write(self, text):
        if text:
            self.output_queue.put(text)

    def flush(self):
        pass


def _drain_output(output_queue, console, last_status_line):
    while True:
        try:
            text = output_queue.get_nowait()
        except queue.Empty:
            break
        console.write(text)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                last_status_line = stripped
    return last_status_line


def _run_method(config_path, method_name, case_id, data_dir):
    config_path = Path(config_path)
    config = load_config(config_path)
    run_dir = io.create_run_dir(
        config["paths"]["runs"],
        prefix=f"{method_name}_case{case_id}",
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
    save_experiment_config(config, method_name, case_id, run_dir / "config.yaml")

    experiment = EXPERIMENTS[method_name](config)
    start = time.perf_counter()
    result = experiment.run(case_id, data_dir, run_dir)
    elapsed = time.perf_counter() - start

    (run_dir / "runtime.txt").write_text(
        "method: {}\ncase_id: {}\nruntime_seconds: {:.6f}\n".format(
            method_name, case_id, elapsed
        ),
        encoding="utf-8",
    )
    return {
        "method_name": method_name,
        "case_id": case_id,
        "run_dir": run_dir,
        "elapsed_seconds": elapsed,
        "result": result,
    }


def run_experiments_from_ui(
    method_name,
    config_mode,
    yaml_path,
    custom_config_name,
    selected_cases,
    Lx,
    Ly,
    Nx,
    Ny,
    dt,
    N,
    gamma0,
    rho,
    c,
    frequency,
    cycles,
    amplitude,
    number_of_sources,
    distance_between_sources,
    distance_between_sensors,
    epochs,
    learning_rate,
    clip_grad,
    cost_scaling,
    l2,
    alpha,
    beta,
    pretrained_model_path,
    progress=None,
):
    if not selected_cases:
        yield "Select at least one case.", "", "", [], ""
        return

    if config_mode == "Create custom YAML from UI parameters":
        yaml_path = save_custom_config(
            "configs/default.yaml",
            method_name,
            custom_config_name,
            {
                "Lx": Lx,
                "Ly": Ly,
                "Nx": Nx,
                "Ny": Ny,
                "dt": dt,
                "N": N,
                "gamma0": gamma0,
                "rho": rho,
                "c": c,
                "frequency": frequency,
                "cycles": cycles,
                "amplitude": amplitude,
                "number_of_sources": number_of_sources,
                "distance_between_sources": distance_between_sources,
                "distance_between_sensors": distance_between_sensors,
            },
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "clip_grad": clip_grad,
                "cost_scaling": cost_scaling,
                "l2": l2,
                "alpha": alpha,
                "beta": beta,
                "pretrained_model_path": pretrained_model_path,
            },
        )

    cases = case_viewer.labels_to_cases(selected_cases)
    results = []
    gallery = []
    total_start = time.perf_counter()
    console = text_io.StringIO()
    yield (
        f"Starting {method_name} for {len(cases)} case(s)...",
        "0.00 s",
        "",
        [],
        "",
    )

    for index, case in enumerate(cases):
        if progress is not None:
            progress((index, len(cases)), desc=f"{method_name} case {case['case_id']}")

        output_queue = queue.Queue()
        worker_result = {}

        def worker():
            try:
                writer = _QueueWriter(output_queue)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    worker_result["result"] = _run_method(
                        yaml_path,
                        method_name,
                        case["case_id"],
                        case["data_dir"],
                    )
            except Exception:
                output_queue.put(traceback.format_exc())
                worker_result["error"] = True

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        last_status_line = ""

        while thread.is_alive():
            last_status_line = _drain_output(output_queue, console, last_status_line)
            status_line = "Running {} case {} ({}/{}).".format(
                method_name, case["case_id"], index + 1, len(cases)
            )
            if last_status_line:
                status_line += "\n" + last_status_line
            yield (
                status_line,
                f"{time.perf_counter() - total_start:.2f} s",
                "\n".join(str(item["run_dir"]) for item in results),
                gallery,
                console.getvalue(),
            )
            time.sleep(1.0)

        thread.join()
        last_status_line = _drain_output(output_queue, console, last_status_line)
        if worker_result.get("error"):
            status = "Run failed. See console output."
            elapsed = time.perf_counter() - total_start
            yield status, f"{elapsed:.2f} s", "\n".join(str(item["run_dir"]) for item in results), gallery, console.getvalue()
            return

        result = worker_result["result"]
        results.append(result)

        run_dir = Path(result["run_dir"])
        final_path = run_dir / "outputs" / f"{method_name}_case{case['case_id']}_final_gamma.h5"
        final_image = _gamma_image(final_path)
        if final_image is not None:
            gallery.append((final_image, f"{method_name} case {case['case_id']} final gamma"))

        history_path = _history_figure_path(run_dir, method_name, case["case_id"])
        if history_path is not None:
            gallery.append((str(history_path), f"{method_name} case {case['case_id']} history"))

        cost_mse_path = _cost_mse_plot(run_dir, method_name, case["case_id"])
        if cost_mse_path is not None:
            gallery.append((str(cost_mse_path), f"{method_name} case {case['case_id']} cost/MSE"))

    elapsed = time.perf_counter() - total_start
    status = format_run_status(results)
    run_dirs = "\n".join(str(item["run_dir"]) for item in results)
    yield status, f"{elapsed:.2f} s", run_dirs, gallery, console.getvalue()


def initialize_canvas(config_path):
    try:
        image = canvas.create_blank_canvas(config_path)
        return image, "Canvas initialized."
    except Exception:
        return None, traceback.format_exc()


def preview_canvas_gamma(canvas_data, config_path):
    try:
        gamma = canvas.canvas_to_gamma(canvas_data, config_path)
        return canvas.gamma_to_preview_image(gamma), f"Preview generated. Gamma shape: {gamma.shape}"
    except Exception:
        return None, traceback.format_exc()


def generate_custom_case_from_canvas(
    canvas_data,
    case_name,
    output_dir,
    config_path,
    generate_measurement_gradient,
):
    try:
        gamma = canvas.canvas_to_gamma(canvas_data, config_path)
        result = canvas.save_custom_gamma_case(
            gamma,
            case_name,
            output_dir,
            config_path,
            generate_measurement_gradient=generate_measurement_gradient,
        )
        lines = ["Custom case saved:"]
        for key, value in result.items():
            lines.append(f"{key}: {value}")
        choices = case_viewer.case_choices()
        return canvas.gamma_to_preview_image(gamma), "\n".join(lines), gr_update_choices(choices)
    except Exception:
        return None, traceback.format_exc(), gr_update_choices(case_viewer.case_choices())


def split_raw_dataset_from_ui(split_mode, raw_dir, train_dir, test_dir, train_ratio, seed, overwrite):
    if split_mode != "Split data/raw into train/test":
        choices = case_viewer.case_choices()
        return "Automatic split is disabled. Select split mode first.", gr_update_choices(choices), gr_update_choices(choices)
    try:
        result = split_raw_dataset(
            raw_dir=raw_dir,
            train_dir=train_dir,
            test_dir=test_dir,
            train_ratio=train_ratio,
            seed=seed,
            overwrite=overwrite,
        )
        lines = [
            "Raw dataset split complete.",
            f"raw_dir: {result['raw_dir']}",
            f"train_dir: {result['train_dir']}",
            f"test_dir: {result['test_dir']}",
            f"total_cases: {result['total_cases']}",
            f"train_cases: {result['train_cases']}",
            f"test_cases: {result['test_cases']}",
            f"train_files_copied: {result['train_files_copied']}",
            f"test_files_copied: {result['test_files_copied']}",
            "Use number_of_samples <= train_cases when pretraining from this split.",
        ]
        if result["source_targets"]:
            lines.append("source.h5 copied to:")
            lines.extend(str(path) for path in result["source_targets"])
        choices = case_viewer.case_choices()
        return "\n".join(lines), gr_update_choices(choices), gr_update_choices(choices)
    except Exception:
        choices = case_viewer.case_choices()
        return traceback.format_exc(), gr_update_choices(choices), gr_update_choices(choices)


def gr_update_choices(choices):
    import gradio as gr

    return gr.update(choices=choices)


def start_pretraining_from_ui(
    config_path,
    data_dir,
    output_dir,
    model_name,
    number_of_samples,
    epochs,
    batch_size,
    learning_rate,
    clip_grad,
    l2,
    alpha,
    beta,
    channels,
    number_of_convolutions_per_block,
    batch_norm,
    progress=None,
):
    start = time.perf_counter()
    try:
        generated_config = save_pretrain_config(
            config_path,
            data_dir,
            output_dir,
            model_name,
            number_of_samples,
            epochs,
            batch_size,
            learning_rate,
            clip_grad,
            l2,
            alpha,
            beta,
            channels,
            number_of_convolutions_per_block,
            batch_norm,
        )
        config = load_config(generated_config)
        run_dir = io.create_run_dir(
            Path(config["paths"].get("runs", "runs")) / "pretraining",
            prefix="pretraining",
        )
        io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
        shutil.copy2(generated_config, run_dir / "config.yaml")
    except Exception:
        elapsed = time.perf_counter() - start
        yield traceback.format_exc(), "", f"{elapsed:.2f} s", None
        return

    output_queue = queue.Queue()
    console = text_io.StringIO()
    worker_result = {}

    def progress_callback(epoch, total_epochs, train_loss, val_loss):
        if progress is not None:
            progress((epoch + 1, total_epochs), desc=f"epoch {epoch + 1}/{total_epochs}")
        print(
            "Epoch {}/{} complete: train={:.6E}, validation={:.6E}".format(
                epoch + 1,
                total_epochs,
                train_loss,
                val_loss,
            ),
            flush=True,
        )

    def worker():
        try:
            writer = _QueueWriter(output_queue)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                print(f"Generated config: {generated_config}", flush=True)
                print(f"Run directory: {run_dir}", flush=True)
                worker_result["model_path"] = pretrain_unet(
                    config,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    progress_callback=progress_callback,
                    run_dir=run_dir,
                )
        except Exception:
            output_queue.put(traceback.format_exc())
            worker_result["error"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_status_line = "Pretraining started..."
    yield last_status_line, "", "0.00 s", None

    while thread.is_alive():
        last_status_line = _drain_output(output_queue, console, last_status_line)
        status = "Pretraining running."
        if last_status_line:
            status += "\n" + last_status_line
        status += "\n\nRecent output:\n" + "\n".join(console.getvalue().splitlines()[-12:])
        yield status, "", f"{time.perf_counter() - start:.2f} s", None
        time.sleep(1.0)

    thread.join()
    last_status_line = _drain_output(output_queue, console, last_status_line)
    elapsed = time.perf_counter() - start
    if worker_result.get("error"):
        yield "Pretraining failed.\n\n" + console.getvalue(), "", f"{elapsed:.2f} s", None
        return

    model_path = worker_result["model_path"]
    (run_dir / "runtime.txt").write_text(
        "run_type: pretraining\nmodel_path: {}\nruntime_seconds: {:.6f}\n".format(
            model_path,
            elapsed,
        ),
        encoding="utf-8",
    )
    plot_path = run_dir / "figures" / "pretraining_loss_history.png"
    status = (
        "Pretraining finished.\n"
        f"Model: {model_path}\n"
        f"Run directory: {run_dir}\n"
        f"Config: {run_dir / 'config.yaml'}\n"
        f"Elapsed: {elapsed:.2f} s\n\n"
        "Recent output:\n"
        + "\n".join(console.getvalue().splitlines()[-20:])
    )
    yield status, str(model_path), f"{elapsed:.2f} s", str(plot_path) if plot_path.exists() else None
