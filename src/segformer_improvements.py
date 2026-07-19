"""Helpers shared by the opt-in SegFormer improvement scripts.

The improvement profile is intentionally a small overlay on an existing full
repository config. This keeps the experiment isolated and makes every changed
parameter visible in one YAML file.
"""

from copy import deepcopy
from pathlib import Path

import yaml

from src.config import load_config


def deep_merge(base, overrides):
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_improvement_profile(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    if "base_config" not in profile:
        raise ValueError(f"Improvement profile {path} must define base_config")

    base_path = Path(profile["base_config"])
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    with base_path.open("r", encoding="utf-8") as handle:
        base_document = yaml.safe_load(handle) or {}
    if "base_config" in base_document:
        config, _ = load_improvement_profile(base_path)
    else:
        config = load_config(base_path)
    config = deep_merge(config, profile.get("overrides", {}))
    return config, base_path
