"""
Experiment 1: what actually transfers when a pretrained U-Net is fine-tuned by FWI?

One notebook cell:

    !python scripts/run_freezing_ablation.py \
        --config configs/config_final.yaml \
        --data-dir /content/eval \
        --run-dir runs/final/freezing

WHAT THIS ADJUDICATES
---------------------
The literature disagrees, and the disagreement is confounded by domain gap.
Amiri et al. find that freezing the encoder and fine-tuning the decoder is often
the WORST choice for ultrasound U-Nets -- but they pretrain on natural images,
so their early layers face a large distribution shift. Yosinski et al.'s
general-to-specific result is usually read as implying the opposite. Raghu et al.
and Karimi et al. argue that much of the benefit of transfer is initialization
scale rather than learned features at all.

This thesis transfers IN-DOMAIN: pretraining and downstream FWI both operate on
adjoint gradients of the same physics, produced by the same generator. That
removes the domain-gap confound, so the question becomes sharp -- does the
freeze-early-is-harmful finding survive when the domain gap is gone?

Four modes, and the third is the one that makes this an adjudication rather than
an illustration:

    encoder         freeze down path + bottleneck, train decoder
    decoder         the reverse -- Amiri et al.'s better-performing direction
    random_encoder  CONTROL: randomly RE-INITIALIZE the encoder, freeze it, and
                    train only the decoder. If this matches a frozen PRETRAINED
                    encoder, then what transfers is not the learned features,
                    which is exactly the Raghu/Karimi claim. No amount of
                    freeze-vs-finetune comparison can settle that on its own.
    none            full fine-tuning baseline

Requires the pretrained U-Net from scripts/pretrain.py to exist first.
Everything lands under --run-dir, with figures and CSVs in <run-dir>/report/.
"""

import argparse
import gc
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src import metrics
from src.config import load_config, save_experiment_config
from src.experiments.base import model_path
from src.experiments.transfer_learning_fwi_frozen_encoder import (
    FREEZE_MODES, TransferLearningFWIFrozenEncoder)
from src.io import ensure_dir, load_hdf
from src.reporting import (aggregate_rows, plot_convergence_grid, plot_metric_bars,
                           plot_reconstruction_gallery, write_csv)

METHOD = "transfer_learning_fwi_frozen_encoder"


def parse_list(text, cast=str):
    return [cast(i.strip()) for i in str(text).split(",") if i.strip()]


def is_complete(run_dir, case_id):
    outputs, histories = Path(run_dir) / "outputs", Path(run_dir) / "histories"
    return all(p.exists() for p in (
        outputs / f"{METHOD}_case{case_id}_final_gamma.h5",
        outputs / f"{METHOD}_case{case_id}_target_gamma.h5",
        histories / f"{METHOD}_case{case_id}_mse_history.txt"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--modes", default=",".join(FREEZE_MODES),
                        help=f"Comma-separated subset of {FREEZE_MODES}.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case ids. Default: experiments.cases.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override FWI epochs. Omitted -> config value.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    modes = parse_list(args.modes)
    bad = [m for m in modes if m not in FREEZE_MODES]
    if bad:
        raise SystemExit(f"unknown freeze mode(s) {bad}; valid: {FREEZE_MODES}")
    cases = (parse_list(args.cases, int) if args.cases
             else [int(c) for c in config["experiments"]["cases"]])
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    root = ensure_dir(args.run_dir or Path(config["paths"]["runs"]) / "freezing_ablation")

    exp = config["experiments"]
    checkpoint = model_path(config, exp["modelType"], int(exp["epochs_pretrain"]),
                            "supervised", int(exp["pretrain_samples"][0]), exp["NNchannels"])
    if not Path(checkpoint).exists():
        raise SystemExit(
            f"pretrained U-Net not found:\n  {checkpoint}\n"
            "Run scripts/pretrain.py first (it must use the same sample count and "
            "epoch budget as experiments.pretrain_samples / epochs_pretrain).")

    print("=" * 78)
    print(f"  checkpoint : {checkpoint}")
    print(f"  modes      : {modes}")
    print(f"  cases      : {cases}")
    print(f"  data       : {data_dir}")
    print(f"  output     : {root}")
    print(f"  total      : {len(modes) * len(cases)} inversions")
    print("=" * 78, flush=True)
    if args.dry_run:
        for mode in modes:
            for case_id in cases:
                state = "done" if is_complete(root / mode / f"case{case_id}", case_id) else "todo"
                print(f"  [{state}] {mode} case{case_id}")
        return 0

    failed, started = [], time.perf_counter()
    for mode in modes:
        for case_id in cases:
            run_dir = root / mode / f"case{case_id}"
            if is_complete(run_dir, case_id) and not args.force:
                print(f"[skip] {mode} case{case_id}", flush=True)
                continue
            ensure_dir(run_dir)
            for sub in ("figures", "histories", "outputs"):
                ensure_dir(run_dir / sub)
            run_config = load_config(args.config)
            run_config["experiments"][METHOD]["freeze_mode"] = mode
            if args.epochs is not None:
                run_config["experiments"][METHOD]["epochs"] = int(args.epochs)
            save_experiment_config(run_config, METHOD, case_id, run_dir / "config.yaml")

            print(f"\n=== freeze_mode={mode}  case{case_id} ===", flush=True)
            start = time.perf_counter()
            experiment = result = None
            try:
                experiment = TransferLearningFWIFrozenEncoder(run_config)
                result = experiment.run(case_id, data_dir, run_dir)
                (run_dir / "runtime.txt").write_text(
                    f"mode: {mode}\ncase_id: {case_id}\n"
                    f"runtime_seconds: {time.perf_counter()-start:.6f}\n", encoding="utf-8")
                print(f"  done in {(time.perf_counter()-start)/60:.1f} min", flush=True)
            except Exception as exc:
                failed.append(f"{mode} case{case_id}: {type(exc).__name__}: {exc}")
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            finally:
                del result, experiment
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ------------------------------- report ---------------------------------
    report = ensure_dir(root / "report")
    rows, histories, reconstructions = [], [], []
    for mode in modes:
        for case_id in cases:
            run_dir = root / mode / f"case{case_id}"
            if not is_complete(run_dir, case_id):
                continue
            h, o = run_dir / "histories", run_dir / "outputs"
            mse = np.atleast_1d(np.loadtxt(h / f"{METHOD}_case{case_id}_mse_history.txt", delimiter=","))
            cost = np.atleast_1d(np.loadtxt(h / f"{METHOD}_case{case_id}_cost_history.txt", delimiter=","))
            final = np.asarray(load_hdf(o / f"{METHOD}_case{case_id}_final_gamma.h5"))
            target = np.asarray(load_hdf(o / f"{METHOD}_case{case_id}_target_gamma.h5"))
            trivial = float(((1.0 - target) ** 2).mean())
            m = metrics.all_metrics(final, target)
            rows.append({"mode": mode, "case": case_id, "epochs": len(mse),
                         "mse_final": float(mse[-1]), "mse_best": float(np.min(mse)),
                         "trivial_mse": trivial,
                         "vs_trivial": float(np.min(mse)) / max(trivial, 1e-30),
                         "void_fraction_target": float((target <= 0.5).mean()),
                         "final_cost": float(cost[-1]),
                         **{k: v for k, v in m.items() if k != "void_threshold"}})
            histories.append({"label": mode, "case": case_id, "mse": mse, "trivial": trivial})
            reconstructions.append({"label": mode, "case": case_id,
                                    "final": final, "target": target})

    if rows:
        write_csv(report / "summary.csv", rows)
        agg = aggregate_rows(rows, "mode", ["mse_final", "mse_best", "vs_trivial", "f1", "iou"])
        write_csv(report / "aggregate_by_mode.csv", agg)
        plot_convergence_grid(report / "convergence.png", histories, cases,
                              title="Freezing ablation: FWI convergence by mode")
        plot_reconstruction_gallery(report / "reconstructions.png", reconstructions, cases)
        plot_metric_bars(report / "final_metrics.png", rows, "mode",
                         [("f1", "Void-mask F1"),
                          ("vs_trivial", "gamma MSE / trivial (lower better)")])
        print("\n" + "=" * 78)
        print(f"{'mode':<22}{'n':>4}{'F1 mean':>10}{'F1 std':>9}{'vs trivial':>12}")
        print("-" * 78)
        for entry in agg:
            print(f"{entry['mode']:<22}{entry['n']:>4}{entry.get('f1_mean',float('nan')):>10.4f}"
                  f"{entry.get('f1_std',0):>9.4f}{entry.get('vs_trivial_mean',float('nan')):>12.4f}")
        print("=" * 78)
        print("Read `random_encoder` against `encoder`: if they match, what transfers")
        print("is not the pretrained features (Raghu et al., Karimi et al.).")
        print(f"\nreport: {report}")
    else:
        print("\nno completed runs to report on")

    print(f"\nfinished in {(time.perf_counter()-started)/60:.1f} min")
    if failed:
        print(f"{len(failed)} FAILED:")
        for f in failed:
            print(f"  {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
