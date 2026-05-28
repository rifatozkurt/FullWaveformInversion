from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    """Loads a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def experiment_config(config, method_name, case_id):
    """Return only the config entries used to reproduce one experiment run."""
    experiments = config["experiments"]
    common_experiment_keys = [
        "modelType",
        "NNchannels",
        "numberOfConvolutionsPerBlock",
        "pretrain_samples",
        "epochs_pretrain",
    ]
    common_experiment_config = {
        key: experiments[key]
        for key in common_experiment_keys
        if key in experiments
    }

    return {
        "run": {
            "method": method_name,
            "case_id": case_id,
        },
        "paths": {
            "casestudy_data": config["paths"]["casestudy_data"],
            "pretrained_models": config["paths"]["pretrained_models"],
            "runs": config["paths"]["runs"],
        },
        "simulation": config["simulation"],
        "source": config.get("source", {}),
        "experiments": {
            **common_experiment_config,
            method_name: experiments[method_name],
        },
    }


def save_experiment_config(config, method_name, case_id, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            experiment_config(config, method_name, case_id),
            handle,
            sort_keys=False,
        )
