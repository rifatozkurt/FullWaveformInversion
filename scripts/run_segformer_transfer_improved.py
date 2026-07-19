import argparse
import shutil
import time
from pathlib import Path

import _bootstrap
import yaml

from src import io
from src.experiments.transfer_segformer_fwi import TransferSegFormerFWI
from src.segformer_improvements import load_improvement_profile


def main():
    parser = argparse.ArgumentParser(
        description="Run the stabilized, opt-in SegFormer transfer-FWI experiment."
    )
    parser.add_argument("--profile", default="configs/segformer_improved.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--trainable-mode",
        choices=("all", "decoder_only", "decoder_plus_last_stage"),
        default=None,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cost-scaling", type=float, default=None)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--no-restore-best", action="store_true")
    args = parser.parse_args()

    config, base_path = load_improvement_profile(args.profile)
    cfg = config["experiments"]["transfer_segformer_fwi"]
    if args.checkpoint is not None:
        cfg["pretrained_checkpoint"] = args.checkpoint
    if args.trainable_mode is not None:
        cfg["trainable_mode"] = args.trainable_mode
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.cost_scaling is not None:
        cfg["costScaling"] = float(args.cost_scaling)
    if args.clip_grad is not None:
        cfg["clipGrad"] = float(args.clip_grad)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.early_stopping_patience is not None:
        cfg["early_stopping_patience"] = int(args.early_stopping_patience)
    if args.no_restore_best:
        cfg["restore_best_observed"] = False

    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    output_root = Path(args.output_root or config["paths"].get("runs", "runs"))
    prefix = "transfer_segformer_improved_case{}_{}".format(
        args.case, cfg.get("trainable_mode", "all")
    )
    run_dir = io.create_run_dir(output_root, prefix=prefix)
    io.ensure_dirs(
        [
            run_dir / "figures",
            run_dir / "histories",
            run_dir / "outputs",
            run_dir / "checkpoints",
        ]
    )
    shutil.copy2(args.profile, run_dir / "profile.yaml")
    shutil.copy2(base_path, run_dir / "base_config.yaml")
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    start = time.perf_counter()
    result = TransferSegFormerFWI(config, device=args.device).run(
        args.case,
        data_dir,
        run_dir,
    )
    elapsed = time.perf_counter() - start
    (run_dir / "runtime.txt").write_text(
        "\n".join(
            [
                "run_type: transfer_segformer_improved",
                f"profile: {args.profile}",
                f"resolved_base_config: {base_path}",
                f"case: {args.case}",
                f"checkpoint: {cfg['pretrained_checkpoint']}",
                f"trainable_mode: {cfg.get('trainable_mode')}",
                f"runtime_seconds: {elapsed:.6f}",
                f"final_mse: {float(result.mse_history[-1]):.12e}",
                f"best_mse: {float(result.mse_history.min()):.12e}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved improved transfer run to {run_dir}")


if __name__ == "__main__":
    main()
