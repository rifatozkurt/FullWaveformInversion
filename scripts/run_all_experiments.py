"""
Run a set of FWI experiments (typically the INR family) and produce the
comparison figures and CSVs for the thesis in the SAME directory.

Designed to be one notebook cell:

    !python scripts/run_all_experiments.py \
        --config configs/config_final.yaml \
        --methods inr_siren_fwi,inr_lr_fwi,inr_mpe_fwi,inr_ig_fwi \
        --cases 15000,15001 \
        --data-dir /content/eval \
        --run-dir runs/final/inr

Everything lands under --run-dir:

    <run-dir>/<method>/case<N>/      per-run histories, outputs, figures
    <run-dir>/report/               comparison figures + aggregate CSVs
    <run-dir>/report/summary.csv    one row per (method, case) -- plot it yourself

Completed runs are skipped, so re-running the same command resumes an
interrupted session. Each experiment runs in-process but frees its GPU memory
afterwards; if a single run fails the rest still complete and the failure is
reported at the end.
"""

import argparse
import csv
import gc
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src import metrics
from src.config import load_config, save_experiment_config
from src.io import ensure_dir, load_hdf
from src.registry import EXPERIMENTS
from src.reporting import (aggregate_rows, plot_convergence_grid,
                           plot_metric_bars, plot_reconstruction_gallery,
                           write_csv)

# The four ansätze the thesis actually compares. The `_centered` variants are
# extra and are only run when explicitly requested.
DEFAULT_METHODS = ("inr_siren_fwi", "inr_lr_fwi", "inr_mpe_fwi", "inr_ig_fwi")


def parse_list(text, cast=str):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def run_dir_for(root, method, case_id):
    return Path(root) / method / f"case{case_id}"


def is_complete(run_dir, method, case_id):
    outputs = Path(run_dir) / "outputs"
    histories = Path(run_dir) / "histories"
    return all(p.exists() for p in (
        outputs / f"{method}_case{case_id}_final_gamma.h5",
        outputs / f"{method}_case{case_id}_target_gamma.h5",
        histories / f"{method}_case{case_id}_mse_history.txt",
    ))


def load_run(run_dir, method, case_id):
    histories = Path(run_dir) / "histories"
    outputs = Path(run_dir) / "outputs"
    mse = np.atleast_1d(np.loadtxt(histories / f"{method}_case{case_id}_mse_history.txt", delimiter=","))
    cost = np.atleast_1d(np.loadtxt(histories / f"{method}_case{case_id}_cost_history.txt", delimiter=","))
    final = np.asarray(load_hdf(outputs / f"{method}_case{case_id}_final_gamma.h5"))
    target = np.asarray(load_hdf(outputs / f"{method}_case{case_id}_target_gamma.h5"))
    return mse, cost, final, target


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config_final.yaml")
    # NOTE: plain --methods/--cases, NOT argparse `append` with a default list.
    # `action="append"` with a non-empty default APPENDS to that default, so
    # `--method X` silently yields [defaults..., X] -- the previous version of
    # this script had exactly that bug.
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="Comma-separated method names. Replaces the default entirely.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case ids. Default: experiments.cases from the config.")
    parser.add_argument("--data-dir", default=None,
                        help="Case data. Default: paths.casestudy_data.")
    parser.add_argument("--run-dir", default=None,
                        help="Parent output directory. Default: <paths.runs>/inr_experiments.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override FWI epochs for every method. Omitted -> config value.")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    methods = parse_list(args.methods)
    unknown = [m for m in methods if m not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}\navailable: {sorted(EXPERIMENTS)}")
    cases = (parse_list(args.cases, int) if args.cases
             else [int(c) for c in config["experiments"]["cases"]])
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    root = ensure_dir(args.run_dir or Path(config["paths"]["runs"]) / "inr_experiments")

    if not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")
    missing = [data_dir / f"material{c}.h5" for c in cases
               if not (data_dir / f"material{c}.h5").exists()]
    if missing:
        raise SystemExit("missing case file(s):\n  " + "\n  ".join(str(m) for m in missing))

    print("=" * 78)
    print(f"  methods : {methods}")
    print(f"  cases   : {cases}")
    print(f"  data    : {data_dir}")
    print(f"  output  : {root}")
    print(f"  epochs  : {args.epochs if args.epochs is not None else '(from config)'}")
    print(f"  total   : {len(methods) * len(cases)} runs")
    print("=" * 78, flush=True)
    if args.dry_run:
        for method in methods:
            for case_id in cases:
                state = "done" if is_complete(run_dir_for(root, method, case_id), method, case_id) else "todo"
                print(f"  [{state}] {method} case{case_id}")
        return 0

    failed, started = [], time.perf_counter()
    for method in methods:
        for case_id in cases:
            run_dir = run_dir_for(root, method, case_id)
            if is_complete(run_dir, method, case_id) and not args.force:
                print(f"[skip] {method} case{case_id} (already complete)", flush=True)
                continue
            ensure_dir(run_dir)
            for sub in ("figures", "histories", "outputs"):
                ensure_dir(run_dir / sub)

            run_config = load_config(args.config)
            if args.epochs is not None:
                run_config["experiments"][method]["epochs"] = int(args.epochs)
            save_experiment_config(run_config, method, case_id, run_dir / "config.yaml")

            print(f"\n=== {method} case{case_id} ===", flush=True)
            start = time.perf_counter()
            experiment = result = None
            try:
                experiment = EXPERIMENTS[method](run_config)
                result = experiment.run(case_id, data_dir, run_dir)
                elapsed = time.perf_counter() - start
                (run_dir / "runtime.txt").write_text(
                    f"method: {method}\ncase_id: {case_id}\nruntime_seconds: {elapsed:.6f}\n",
                    encoding="utf-8")
                print(f"  done in {elapsed/60:.1f} min", flush=True)
            except Exception as exc:            # one bad run must not lose the session
                failed.append(f"{method} case{case_id}: {type(exc).__name__}: {exc}")
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            finally:
                del result, experiment
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---------------- aggregate + figures, in the same directory --------------
    if not args.skip_report:
        report = ensure_dir(root / "report")
        rows, histories, reconstructions = [], [], []
        for method in methods:
            for case_id in cases:
                run_dir = run_dir_for(root, method, case_id)
                if not is_complete(run_dir, method, case_id):
                    continue
                mse, cost, final, target = load_run(run_dir, method, case_id)
                trivial = float(((1.0 - target) ** 2).mean())
                m = metrics.all_metrics(final, target)
                rows.append({
                    "method": method, "case": case_id, "epochs": len(mse),
                    "mse_first_epoch": float(mse[0]), "mse_final": float(mse[-1]),
                    "mse_best": float(np.min(mse)),
                    "trivial_mse": trivial,
                    "vs_trivial": float(np.min(mse)) / max(trivial, 1e-30),
                    "void_fraction_target": float((target <= 0.5).mean()),
                    "final_cost": float(cost[-1]),
                    **{k: v for k, v in m.items() if k != "void_threshold"},
                })
                histories.append({"label": method, "case": case_id, "mse": mse, "cost": cost})
                reconstructions.append({"label": method, "case": case_id,
                                        "final": final, "target": target})

        if rows:
            write_csv(report / "summary.csv", rows)
            write_csv(report / "aggregate_by_method.csv",
                      aggregate_rows(rows, group_key="method",
                                     metric_keys=["mse_final", "mse_best", "vs_trivial",
                                                  "f1", "iou", "gamma_mse"]))
            plot_convergence_grid(report / "convergence.png", histories, cases,
                                  ylabel="gamma MSE",
                                  title="INR ansätze: FWI convergence")
            plot_reconstruction_gallery(report / "reconstructions.png", reconstructions, cases)
            plot_metric_bars(report / "final_metrics.png", rows, group_key="method",
                             metrics_to_plot=[("f1", "Void-mask F1"),
                                              ("vs_trivial", "gamma MSE / trivial (lower better)")])
            print(f"\nreport written to {report}", flush=True)
        else:
            print("\nno completed runs to report on", flush=True)

    total = (time.perf_counter() - started) / 60
    print("\n" + "=" * 78)
    print(f"  finished in {total:.1f} min")
    if failed:
        print(f"  {len(failed)} FAILED:")
        for f in failed:
            print(f"    {f}")
        print("  Re-run the same command to retry only those.")
    print(f"  results: {root}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
