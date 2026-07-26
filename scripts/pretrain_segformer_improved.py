import argparse
import shutil
import time
from pathlib import Path

import _bootstrap
import yaml

from src import io
from src.pretrain_segformer import (
    parse_early_stopping_patience,
    pretrain_segformer,
)
from src.segformer_improvements import load_improvement_profile


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain the opt-in stabilized SegFormer profile."
    )
    parser.add_argument("--profile", default="configs/segformer_improved.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--number-of-samples", type=int, default=None)
    parser.add_argument("--available-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--minimum-epochs", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        default=None,
        help="Positive epoch count, or 'none' to disable early stopping.",
    )
    args = parser.parse_args()

    config, base_path = load_improvement_profile(args.profile)
    cfg = config["segformer_pretraining"]
    if args.number_of_samples is not None:
        cfg["numberOfSamples"] = int(args.number_of_samples)
    if args.available_samples is not None:
        cfg["availableSamples"] = int(args.available_samples)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.minimum_epochs is not None:
        if args.minimum_epochs < 1:
            raise ValueError("--minimum-epochs must be positive")
        cfg["minimum_epochs"] = int(args.minimum_epochs)
    if args.early_stopping_patience is not None:
        cfg["early_stopping_patience"] = parse_early_stopping_patience(
            args.early_stopping_patience
        )

    run_root = Path(args.run_root or config["paths"].get("runs", "runs"))
    run_dir = io.create_run_dir(run_root, prefix="pretraining_segformer_improved")
    io.ensure_dirs([run_dir / "figures", run_dir / "histories", run_dir / "outputs"])
    shutil.copy2(args.profile, run_dir / "profile.yaml")
    shutil.copy2(base_path, run_dir / "base_config.yaml")
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    start = time.perf_counter()
    model_path = pretrain_segformer(
        config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_dir=run_dir,
    )
    elapsed = time.perf_counter() - start
    (run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: pretraining_segformer_improved",
                f"profile: {args.profile}",
                f"resolved_base_config: {base_path}",
                f"model_path: {model_path}",
                f"runtime_seconds: {elapsed:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved model to {model_path}")
    print(f"Saved improved pretraining run to {run_dir}")


if __name__ == "__main__":
    main()
