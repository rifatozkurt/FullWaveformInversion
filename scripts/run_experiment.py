import argparse
import time
from pathlib import Path

import _bootstrap
from src.config import load_config, save_experiment_config
from src.io import create_run_dir
from src.registry import EXPERIMENTS, get_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="inr_siren_fwi", required=False, choices=EXPERIMENTS.keys())
    parser.add_argument("--case", default=1, required=False, type=int)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    data_dir = Path(config["paths"]["casestudy_data"])
    run_dir = create_run_dir(
        config["paths"]["runs"],
        prefix=f"{args.method}_case{args.case}",
    )
    save_experiment_config(config, args.method, args.case, run_dir / "config.yaml")

    experiment = get_experiment(args.method)(config)
    start = time.perf_counter()
    result = experiment.run(args.case, data_dir, run_dir)
    elapsed = time.perf_counter() - start
    (run_dir / "runtime.txt").write_text(
        "method: {}\ncase_id: {}\nruntime_seconds: {:.6f}\n".format(
            args.method, args.case, elapsed
        ),
        encoding="utf-8",
    )
    print("Runtime: {:.2f} seconds".format(elapsed))
    print("Saved run to {}".format(result.run_dir))


if __name__ == "__main__":
    main()
