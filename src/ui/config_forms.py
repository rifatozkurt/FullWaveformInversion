from pathlib import Path
from datetime import datetime

import yaml

from src.config import load_config
from src.io import ensure_dir


SHARED_FIELD_MAP = {
    "Lx": ("simulation", "Lx", float),
    "Ly": ("simulation", "Ly", float),
    "Nx": ("simulation", "Nx", int),
    "Ny": ("simulation", "Ny", int),
    "dt": ("simulation", "dt", float),
    "N": ("simulation", "N", int),
    "gamma0": ("simulation", "gamma0", float),
    "rho": ("simulation", "rho", float),
    "c": ("simulation", "c", float),
    "frequency": ("source", "frequency", float),
    "cycles": ("source", "cycles", int),
    "amplitude": ("source", "amplitude", float),
    "number_of_sources": ("simulation", "numberOfSources", int),
    "distance_between_sources": ("simulation", "distanceBetweenSources", int),
    "distance_between_sensors": ("simulation", "distanceBetweenSensors", int),
}


EXPERIMENT_FIELD_MAP = {
    "epochs": ("epochs", int),
    "learning_rate": ("lr", float),
    "clip_grad": ("clipGrad", float),
    "cost_scaling": ("costScaling", float),
    "l2": ("l2", float),
    "alpha": ("alpha", float),
    "beta": ("beta", float),
    "output_mode": ("output_mode", str),
    "final_bias": ("final_bias", float),
    "tv_weight": ("tv_weight", float),
    "tv_type": ("tv_type", str),
    "rank": ("rank", int),
    "core_init_std": ("core_init_std", float),
    "num_levels": ("num_levels", int),
    "base_resolution": ("base_resolution", int),
    "per_level_scale": ("per_level_scale", float),
    "features_per_level": ("features_per_level", int),
    "grid_init_std": ("grid_init_std", float),
    "align_corners": ("align_corners", bool),
    "swap_grid_coords": ("swap_grid_coords", bool),
    "fusion_alpha": ("fusion_alpha", float),
    "siren_hidden_features": ("siren_hidden_features", int),
    "siren_hidden_layers": ("siren_hidden_layers", int),
    "siren_out_features": ("siren_out_features", int),
    "omega0": ("omega0", float),
    "fusion_hidden_features": ("fusion_hidden_features", int),
    "fusion_hidden_layers": ("fusion_hidden_layers", int),
}


def default_form_values(config_path="configs/default.yaml", method_name="conventional_fwi"):
    config = load_config(config_path)
    exp = config["experiments"].get(method_name, {})
    values = {
        form_name: config[section][key]
        for form_name, (section, key, _) in SHARED_FIELD_MAP.items()
    }
    values.update(
        {
            "epochs": exp.get("epochs", 1),
            "learning_rate": exp.get("lr", 0.0),
            "clip_grad": exp.get("clipGrad", 0.0),
            "cost_scaling": exp.get("costScaling", 0.0),
            "l2": exp.get("l2", 0.0),
            "alpha": exp.get("alpha", 0.0),
            "beta": exp.get("beta", 0.0),
            "output_mode": exp.get("output_mode", "voidness"),
            "final_bias": exp.get("final_bias", -5.0),
            "tv_weight": exp.get("tv_weight", 0.0),
            "tv_type": exp.get("tv_type", "anisotropic"),
            "rank": exp.get("rank", 32),
            "core_init_std": exp.get("core_init_std", 1e-3),
            "num_levels": exp.get("num_levels", 16),
            "base_resolution": exp.get("base_resolution", 50),
            "per_level_scale": exp.get("per_level_scale", 1.05),
            "features_per_level": exp.get("features_per_level", 2),
            "grid_init_std": exp.get("grid_init_std", 1e-2),
            "align_corners": exp.get("align_corners", True),
            "swap_grid_coords": exp.get("swap_grid_coords", False),
            "fusion_alpha": exp.get("fusion_alpha", 0.5),
            "siren_hidden_features": exp.get("siren_hidden_features", 128),
            "siren_hidden_layers": exp.get("siren_hidden_layers", 2),
            "siren_out_features": exp.get("siren_out_features", 128),
            "omega0": exp.get("omega0", 30),
            "fusion_hidden_features": exp.get("fusion_hidden_features", 64),
            "fusion_hidden_layers": exp.get("fusion_hidden_layers", 2),
            "pretrained_model_path": config["paths"].get("pretrained_models", "models/pretrained"),
        }
    )
    return values


def experiment_form_values(config_path="configs/default.yaml", method_name="conventional_fwi"):
    values = default_form_values(config_path=config_path, method_name=method_name)
    return {
        key: values[key]
        for key in (
            "epochs",
            "learning_rate",
            "clip_grad",
            "cost_scaling",
            "l2",
            "alpha",
            "beta",
            "output_mode",
            "final_bias",
            "tv_weight",
            "tv_type",
            "rank",
            "core_init_std",
            "num_levels",
            "base_resolution",
            "per_level_scale",
            "features_per_level",
            "grid_init_std",
            "align_corners",
            "swap_grid_coords",
            "fusion_alpha",
            "siren_hidden_features",
            "siren_hidden_layers",
            "siren_out_features",
            "omega0",
            "fusion_hidden_features",
            "fusion_hidden_layers",
            "pretrained_model_path",
        )
    }


def save_custom_config(base_config_path, method_name, config_name, shared_values, experiment_values):
    config = load_config(base_config_path)

    for form_name, value in shared_values.items():
        section, key, cast = SHARED_FIELD_MAP[form_name]
        config[section][key] = cast(value)

    exp = config["experiments"][method_name]
    for form_name, value in experiment_values.items():
        if form_name == "pretrained_model_path":
            config["paths"]["pretrained_models"] = str(value)
            continue
        key, cast = EXPERIMENT_FIELD_MAP[form_name]
        if key in exp:
            exp[key] = cast(value)

    clean_name = config_name.strip() or f"{method_name}_custom"
    if not clean_name.endswith(".yaml"):
        clean_name += ".yaml"
    path = Path("configs/custom") / clean_name
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def parse_channels(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def save_pretrain_config(
    base_config_path,
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
):
    config = load_config(base_config_path)
    channels = parse_channels(channels)
    config["paths"]["train_data"] = str(data_dir)
    config["paths"]["pretrained_models"] = str(output_dir)
    config["pretraining"]["numberOfSamples"] = int(number_of_samples)
    config["pretraining"]["availableSamples"] = int(number_of_samples)
    config["pretraining"]["epochs"] = int(epochs)
    config["pretraining"]["batchDivisor"] = max(1, int(number_of_samples) // max(1, int(batch_size)))
    config["pretraining"]["lr"] = float(learning_rate)
    config["pretraining"]["clipGrad"] = float(clip_grad)
    config["pretraining"]["l2"] = float(l2)
    config["pretraining"]["alpha"] = float(alpha)
    config["pretraining"]["beta"] = float(beta)
    config["pretraining"]["model_type"] = model_name or config["pretraining"].get("model_type", "Unet")
    config["pretraining"]["NNchannels"] = channels
    config["pretraining"]["numberOfConvolutionsPerBlock"] = int(number_of_convolutions_per_block)
    config.setdefault("models", {}).setdefault("unet", {})
    config["models"]["unet"]["channels"] = channels
    config["models"]["unet"]["number_of_convolutions_per_block"] = int(number_of_convolutions_per_block)
    config["models"]["unet"]["batch_norm"] = bool(batch_norm)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("configs/custom") / f"pretrain_{stamp}.yaml"
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path
