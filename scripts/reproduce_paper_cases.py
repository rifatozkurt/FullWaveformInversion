import argparse
import time
from pathlib import Path

import _bootstrap
from src.config import load_config, save_experiment_config
from src.io import create_run_dir
from src.registry import EXPERIMENTS


PAPER_EXPERIMENTS = [
    "conventional_fwi",
    "nn_based_fwi",
    "transfer_learning_fwi",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=int)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    data_dir = Path(config["paths"]["casestudy_data"])
    cases = args.case or config["experiments"]["cases"]

    for case_id in cases:
        for method_name in PAPER_EXPERIMENTS:
            run_dir = create_run_dir(
                config["paths"]["runs"],
                prefix=f"{method_name}_case{case_id}",
            )
            save_experiment_config(config, method_name, case_id, run_dir / "config.yaml")
            print("Running {} case {}".format(method_name, case_id))
            start = time.perf_counter()
            result = EXPERIMENTS[method_name](config).run(case_id, data_dir, run_dir)
            elapsed = time.perf_counter() - start
            (run_dir / "runtime.txt").write_text(
                "method: {}\ncase_id: {}\nruntime_seconds: {:.6f}\n".format(
                    method_name, case_id, elapsed
                ),
                encoding="utf-8",
            )
            print("Runtime: {:.2f} seconds".format(elapsed))
            print("Saved run to {}".format(result.run_dir))


if __name__ == "__main__":
    main()
