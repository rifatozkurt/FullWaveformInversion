import argparse
import copy
import itertools
import time
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.config import load_config, save_experiment_config
from src.io import create_run_dir, ensure_dir
from src.registry import get_experiment


METHODS = ("inr_fwi", "inr_siren_fwi")


def parse_csv(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def format_value(value):
    if isinstance(value, float):
        return "{:.0e}".format(value) if value != 0 else "0"
    return str(value)


def trial_name(method, case_id, trial_index, params):
    parts = [
        f"trial{trial_index:03d}",
        method,
        f"case{case_id}",
        f"lr{format_value(params['lr'])}",
        f"cs{format_value(params['costScaling'])}",
        f"hf{params['hidden_features']}",
        f"hl{params['hidden_layers']}",
    ]
    if method == "inr_siren_fwi":
        parts.append(f"w{format_value(params['omega0'])}")
    return "_".join(parts)


def build_trials(method, args):
    base = {
        "lr": args.learning_rates,
        "costScaling": args.cost_scalings,
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "alpha": args.alphas,
        "beta": args.betas,
        "seed": args.seeds,
    }
    if method == "inr_siren_fwi":
        base["omega0"] = args.omega0s

    keys = list(base)
    for values in itertools.product(*(base[key] for key in keys)):
        yield dict(zip(keys, values))


def apply_trial_config(config, method, args, trial):
    config = copy.deepcopy(config)
    cfg = config["experiments"][method]
    for key, value in trial.items():
        cfg[key] = value
    cfg["epochs"] = args.epochs
    cfg.pop("clipGrad", None)
    return config


def run_trial(config, method, case_id, data_dir, run_dir):
    save_experiment_config(config, method, case_id, run_dir / "config.yaml")
    with (run_dir / "full_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    experiment = get_experiment(method)(config)
    start = time.perf_counter()
    result = experiment.run(case_id, data_dir, run_dir)
    runtime = time.perf_counter() - start

    return {
        "runtime_seconds": runtime,
        "final_cost": float(result.cost_history[-1]),
        "best_cost": float(result.cost_history.min()),
        "final_mse": float(result.mse_history[-1]),
        "best_mse": float(result.mse_history.min()),
    }


def plot_heatmaps(results, output_dir):
    heatmap_dir = ensure_dir(output_dir / "heatmaps")
    metrics = ("final_cost", "best_cost", "final_mse", "best_mse")

    for method, method_df in results.groupby("method"):
        group_cols = ["hidden_features", "hidden_layers", "alpha", "beta", "seed"]
        if method == "inr_siren_fwi":
            group_cols.append("omega0")

        for group_values, group_df in method_df.groupby(group_cols, dropna=False):
            group_values = group_values if isinstance(group_values, tuple) else (group_values,)
            title_bits = [
                f"{name}={format_value(value)}"
                for name, value in zip(group_cols, group_values)
            ]
            suffix = "_".join(
                f"{name}-{format_value(value)}" for name, value in zip(group_cols, group_values)
            )

            for metric in metrics:
                table = group_df.pivot_table(
                    index="costScaling",
                    columns="lr",
                    values=metric,
                    aggfunc="mean",
                ).sort_index(ascending=False)
                if table.empty:
                    continue

                fig, ax = plt.subplots(figsize=(1.4 * len(table.columns) + 3, 1.0 * len(table.index) + 2.5))
                image = ax.imshow(table.values, aspect="auto", cmap="viridis")
                ax.set_title(f"{method} {metric}\n" + ", ".join(title_bits), fontsize=10)
                ax.set_xlabel("lr")
                ax.set_ylabel("costScaling")
                ax.set_xticks(range(len(table.columns)))
                ax.set_xticklabels([format_value(value) for value in table.columns], rotation=45, ha="right")
                ax.set_yticks(range(len(table.index)))
                ax.set_yticklabels([format_value(value) for value in table.index])

                for row_index, cost_scaling in enumerate(table.index):
                    for col_index, lr in enumerate(table.columns):
                        value = table.loc[cost_scaling, lr]
                        if pd.notna(value):
                            ax.text(
                                col_index,
                                row_index,
                                "{:.2e}".format(value),
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                            )

                fig.colorbar(image, ax=ax, label=metric)
                fig.tight_layout()
                fig.savefig(heatmap_dir / f"{method}_{metric}_{suffix}.png", dpi=180)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run compact hyperparameter sweeps for INR_FWI and INR_SIREN_FWI."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--methods", default="inr_fwi,inr_siren_fwi")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--cost-scalings", default="1e8,3e8,1e9,3e9,1e10")
    parser.add_argument("--learning-rates", default="3e-5,1e-4,3e-4")
    parser.add_argument("--hidden-features", default="128")
    parser.add_argument("--hidden-layers", default="4")
    parser.add_argument("--omega0s", default="30")
    parser.add_argument("--alphas", default="-0.5")
    parser.add_argument("--betas", default="0.2")
    parser.add_argument("--seeds", default="50")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    args.methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown_methods = sorted(set(args.methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {unknown_methods}")

    args.cost_scalings = parse_csv(args.cost_scalings, float)
    args.learning_rates = parse_csv(args.learning_rates, float)
    args.hidden_features = parse_csv(args.hidden_features, int)
    args.hidden_layers = parse_csv(args.hidden_layers, int)
    args.omega0s = parse_csv(args.omega0s, float)
    args.alphas = parse_csv(args.alphas, float)
    args.betas = parse_csv(args.betas, float)
    args.seeds = parse_csv(args.seeds, int)

    base_config = load_config(args.config)
    data_dir = Path(base_config["paths"]["casestudy_data"])
    sweep_dir = create_run_dir(
        Path(base_config["paths"]["runs"]) / "tuning",
        prefix=f"inr_tuning_case{args.case}",
    )
    print(f"Saving tuning outputs to {sweep_dir}")

    records = []
    trial_index = 0
    for method in args.methods:
        for trial in build_trials(method, args):
            if args.max_trials is not None and trial_index >= args.max_trials:
                break
            trial_index += 1
            config = apply_trial_config(base_config, method, args, trial)
            run_dir = ensure_dir(sweep_dir / trial_name(method, args.case, trial_index, trial))

            print(f"[{trial_index}] {method}: {trial}")
            metrics = run_trial(config, method, args.case, data_dir, run_dir)
            record = {
                "trial": trial_index,
                "method": method,
                "case_id": args.case,
                "epochs": args.epochs,
                "run_dir": str(run_dir),
                **trial,
                **metrics,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(sweep_dir / "results.csv", index=False)
            plot_heatmaps(pd.DataFrame(records), sweep_dir)

        if args.max_trials is not None and trial_index >= args.max_trials:
            break

    results = pd.DataFrame(records)
    results.to_csv(sweep_dir / "results.csv", index=False)
    plot_heatmaps(results, sweep_dir)
    print(f"Saved results to {sweep_dir / 'results.csv'}")
    print(f"Saved heatmaps to {sweep_dir / 'heatmaps'}")


if __name__ == "__main__":
    main()
