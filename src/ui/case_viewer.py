from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from src import io


CASE_DIRS = [Path("data/casestudy"), Path("data/test"), Path("data/custom")]


def scan_cases(case_dirs=None):
    case_dirs = case_dirs or CASE_DIRS
    cases = []
    for directory in case_dirs:
        if not directory.exists():
            continue
        for material_path in sorted(directory.glob("material*.h5")):
            match = re.match(r"material(_?[A-Za-z0-9-]+)\.h5$", material_path.name)
            if not match:
                continue
            raw_id = match.group(1)
            case_id = int(raw_id) if raw_id.isdigit() else raw_id.lstrip("_")
            stem = raw_id
            cases.append(
                {
                    "label": f"{directory} / case {case_id}",
                    "case_id": case_id,
                    "stem": stem,
                    "data_dir": directory,
                    "material_path": material_path,
                    "measurement_path": directory / f"measurement{stem}.h5",
                    "gradient_path": directory / f"gradient{stem}.h5",
                }
            )
    return cases


def case_choices():
    return [case["label"] for case in scan_cases()]


def get_case(label):
    for case in scan_cases():
        if case["label"] == label:
            return case
    return None


def labels_to_cases(labels):
    return [case for case in scan_cases() if case["label"] in set(labels or [])]


def material_to_image(material):
    material = np.asarray(material, dtype=float)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    rgba = plt.get_cmap("coolwarm")(norm(material.T))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def view_case(label):
    case = get_case(label)
    if case is None:
        return None, "No case selected."

    material = io.load_hdf(case["material_path"])
    image = material_to_image(material)
    info = [
        f"case path: {case['data_dir']}",
        f"case id: {case['case_id']}",
        f"measurement exists: {case['measurement_path'].exists()}",
        f"gradient exists: {case['gradient_path'].exists()}",
        f"material shape: {material.shape}",
    ]
    return image, "\n".join(info)
