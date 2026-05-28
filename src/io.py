from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths):
    return [ensure_dir(path) for path in paths]


def save_hdf(path, data, key="U"):
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(data).to_hdf(path, key=key, index=False, mode="w", complevel=1)


def load_hdf(path):
    return pd.read_hdf(Path(path)).values


def save_history(path, values):
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(path, values, delimiter=", ")


def create_run_dir(root="runs", prefix="run"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(root) / f"{prefix}_{stamp}")
