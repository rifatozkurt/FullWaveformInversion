from pathlib import Path

import numpy as np
from PIL import Image

from src import io
from src.config import load_config
from src.data_generation import generate_case_from_gamma
from src.experiments.base import case_file_stem
from src.ui.case_viewer import material_to_image


def create_blank_canvas(config_path):
    config = load_config(config_path)
    nx = int(config["simulation"]["Nx"]) + 1
    ny = int(config["simulation"]["Ny"]) + 1
    canvas = np.zeros((ny, nx, 3), dtype=np.uint8)
    canvas[:, :, 0] = 180
    canvas[:, :, 1] = 40
    canvas[:, :, 2] = 40
    return canvas


def _extract_canvas_array(canvas_data):
    if canvas_data is None:
        return None
    if isinstance(canvas_data, dict):
        for key in ("composite", "layers", "background"):
            value = canvas_data.get(key)
            if key == "layers" and value:
                return np.asarray(value[-1])
            if value is not None:
                return np.asarray(value)
        return None
    return np.asarray(canvas_data)


def canvas_to_gamma(canvas_data, config_path):
    config = load_config(config_path)
    nx = int(config["simulation"]["Nx"]) + 1
    ny = int(config["simulation"]["Ny"]) + 1
    gamma0 = float(config["simulation"]["gamma0"])
    array = _extract_canvas_array(canvas_data)
    if array is None:
        raise ValueError("Canvas is empty. Initialize or draw on the canvas first.")

    image = Image.fromarray(array.astype(np.uint8)).convert("RGBA").resize((nx, ny))
    rgba = np.asarray(image)
    red = rgba[:, :, 0].astype(float)
    green = rgba[:, :, 1].astype(float)
    blue = rgba[:, :, 2].astype(float)
    alpha = rgba[:, :, 3].astype(float)

    blue_damage = (blue > red + 20) & (blue > green + 20)
    non_red_damage = (alpha > 0) & ~((red > 120) & (green < 100) & (blue < 100))
    damage = blue_damage | non_red_damage

    gamma = np.ones((nx, ny), dtype=np.float32)
    gamma[damage.T] = gamma0
    return gamma


def gamma_to_preview_image(gamma):
    return material_to_image(gamma)


def save_custom_gamma_case(gamma, case_name, output_dir, config_path, generate_measurement_gradient=True):
    config = load_config(config_path)
    output_dir = Path(output_dir)
    io.ensure_dir(output_dir)
    stem = case_file_stem(case_name)
    material_path = output_dir / f"material{stem}.h5"
    io.save_hdf(material_path, gamma, key="gamma")

    result = {"material_path": material_path}
    if generate_measurement_gradient:
        result.update(generate_case_from_gamma(config, gamma, case_name, output_dir))
    return result
