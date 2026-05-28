import argparse
import shutil
import time

import _bootstrap
from src.config import load_config
from src import io
from src.pretraining import pretrain_unet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = io.create_run_dir(
        io.ensure_dir(config["paths"].get("runs", "runs")) / "pretraining",
        prefix="pretraining",
    )
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
    shutil.copy2(args.config, run_dir / "config.yaml")
    start = time.perf_counter()
    model_path = pretrain_unet(config, run_dir=run_dir)
    elapsed = time.perf_counter() - start
    (run_dir / "runtime.txt").write_text(
        "run_type: pretraining\nmodel_path: {}\nruntime_seconds: {:.6f}\n".format(
            model_path,
            elapsed,
        ),
        encoding="utf-8",
    )
    print("Saved model to {}".format(model_path))
    print("Saved pretraining run to {}".format(run_dir))


if __name__ == "__main__":
    main()
