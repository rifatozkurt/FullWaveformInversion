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


METHODS = (
    "inr_fwi",
    "inr_siren_fwi",
    "inr_siren_centered_fwi",
    "inr_lr_fwi",
    "inr_mpe_fwi",
    "inr_mpe_centered_fwi",
    "inr_ig_fwi",
    "inr_ig_centered_fwi",
)
SIREN_METHODS = ("inr_siren_fwi", "inr_siren_centered_fwi")
MPE_METHODS = ("inr_mpe_fwi", "inr_mpe_centered_fwi")
IG_METHODS = ("inr_ig_fwi", "inr_ig_centered_fwi")


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
        f"fb{format_value(params['final_bias'])}",
        f"tv{format_value(params['tv_weight'])}",
        f"tvt{params['tv_type']}",
    ]
    if params["output_mode"] != "voidness":
        parts.append(f"out{params['output_mode']}")
    if method in SIREN_METHODS or method == "inr_lr_fwi":
        parts.append(f"w{format_value(params['omega0'])}")
    if method == "inr_lr_fwi":
        parts.append(f"r{params['rank']}")
        parts.append(f"core{format_value(params['core_init_std'])}")
    if method in MPE_METHODS:
        parts.append(f"lv{params['num_levels']}")
        parts.append(f"br{params['base_resolution']}")
        parts.append(f"pl{format_value(params['per_level_scale'])}")
        parts.append(f"fpl{params['features_per_level']}")
        parts.append(f"grid{format_value(params['grid_init_std'])}")
    if method in IG_METHODS:
        parts.append(f"fa{format_value(params['fusion_alpha'])}")
        parts.append(f"lv{params['num_levels']}")
        parts.append(f"br{params['base_resolution']}")
        parts.append(f"pl{format_value(params['per_level_scale'])}")
        parts.append(f"fpl{params['features_per_level']}")
        parts.append(f"grid{format_value(params['grid_init_std'])}")
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
        "output_mode": args.output_modes,
        "final_bias": args.final_biases,
        "tv_weight": args.tv_weights,
        "tv_type": args.tv_types,
    }
    if method in SIREN_METHODS or method == "inr_lr_fwi":
        base["omega0"] = args.omega0s
    if method == "inr_lr_fwi":
        base["rank"] = args.ranks
        base["core_init_std"] = args.core_init_stds
    if method in MPE_METHODS:
        base["num_levels"] = args.num_levels
        base["base_resolution"] = args.base_resolutions
        base["per_level_scale"] = args.per_level_scales
        base["features_per_level"] = args.features_per_levels
        base["grid_init_std"] = args.grid_init_stds
        base["align_corners"] = args.align_corners_values
        base["swap_grid_coords"] = args.swap_grid_coords_values
    if method in IG_METHODS:
        base["fusion_alpha"] = args.fusion_alphas
        base["num_levels"] = args.num_levels
        base["base_resolution"] = args.base_resolutions
        base["per_level_scale"] = args.per_level_scales
        base["features_per_level"] = args.features_per_levels
        base["grid_init_std"] = args.grid_init_stds
        base["align_corners"] = args.align_corners_values
        base["swap_grid_coords"] = args.swap_grid_coords_values
        base["siren_hidden_features"] = args.siren_hidden_features
        base["siren_hidden_layers"] = args.siren_hidden_layers
        base["siren_out_features"] = args.siren_out_features
        base["omega0"] = args.omega0s
        base["fusion_hidden_features"] = args.fusion_hidden_features
        base["fusion_hidden_layers"] = args.fusion_hidden_layers

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
        group_cols = [
            "hidden_features",
            "hidden_layers",
            "output_mode",
            "final_bias",
            "tv_weight",
            "tv_type",
            "alpha",
            "beta",
            "seed",
        ]
        if method in SIREN_METHODS:
            group_cols.append("omega0")
        if method == "inr_lr_fwi":
            group_cols.extend(["omega0", "rank", "core_init_std"])
        if method in MPE_METHODS:
            group_cols.extend(
                [
                    "num_levels",
                    "base_resolution",
                    "per_level_scale",
                    "features_per_level",
                    "grid_init_std",
                    "align_corners",
                    "swap_grid_coords",
                ]
            )
        if method in IG_METHODS:
            group_cols.extend(
                [
                    "fusion_alpha",
                    "num_levels",
                    "base_resolution",
                    "per_level_scale",
                    "features_per_level",
                    "grid_init_std",
                    "align_corners",
                    "swap_grid_coords",
                    "siren_hidden_features",
                    "siren_hidden_layers",
                    "siren_out_features",
                    "omega0",
                    "fusion_hidden_features",
                    "fusion_hidden_layers",
                ]
            )

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
        description="Run compact hyperparameter sweeps for INR-family FWI methods."
    )
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--methods", default="inr_fwi,inr_siren_fwi")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--cost-scalings", default="1e8,3e8,1e9,3e9,1e10")
    parser.add_argument("--learning-rates", default="3e-5,1e-4,3e-4")
    parser.add_argument("--hidden-features", default="128")
    parser.add_argument("--hidden-layers", default="4")
    parser.add_argument("--omega0s", default="30")
    parser.add_argument("--ranks", default="32")
    parser.add_argument("--core-init-stds", default="1e-3")
    parser.add_argument("--num-levels", default="16")
    parser.add_argument("--base-resolutions", default="50")
    parser.add_argument("--per-level-scales", default="1.05")
    parser.add_argument("--features-per-levels", default="2")
    parser.add_argument("--grid-init-stds", default="1e-4")
    parser.add_argument("--align-corners-values", default="true")
    parser.add_argument("--swap-grid-coords-values", default="false")
    parser.add_argument("--fusion-alphas", default="0.5")
    parser.add_argument("--siren-hidden-features", default="128")
    parser.add_argument("--siren-hidden-layers", default="2")
    parser.add_argument("--siren-out-features", default="128")
    parser.add_argument("--fusion-hidden-features", default="64")
    parser.add_argument("--fusion-hidden-layers", default="2")
    parser.add_argument("--output-modes", default="voidness")
    parser.add_argument("--final-biases", default="-5.0")
    parser.add_argument("--tv-weights", default="0.0")
    parser.add_argument("--tv-types", default="anisotropic")
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
    args.ranks = parse_csv(args.ranks, int)
    args.core_init_stds = parse_csv(args.core_init_stds, float)
    args.num_levels = parse_csv(args.num_levels, int)
    args.base_resolutions = parse_csv(args.base_resolutions, int)
    args.per_level_scales = parse_csv(args.per_level_scales, float)
    args.features_per_levels = parse_csv(args.features_per_levels, int)
    args.grid_init_stds = parse_csv(args.grid_init_stds, float)
    args.align_corners_values = [value.lower() in ("1", "true", "yes", "on") for value in parse_csv(args.align_corners_values, str)]
    args.swap_grid_coords_values = [value.lower() in ("1", "true", "yes", "on") for value in parse_csv(args.swap_grid_coords_values, str)]
    args.fusion_alphas = parse_csv(args.fusion_alphas, float)
    args.siren_hidden_features = parse_csv(args.siren_hidden_features, int)
    args.siren_hidden_layers = parse_csv(args.siren_hidden_layers, int)
    args.siren_out_features = parse_csv(args.siren_out_features, int)
    args.fusion_hidden_features = parse_csv(args.fusion_hidden_features, int)
    args.fusion_hidden_layers = parse_csv(args.fusion_hidden_layers, int)
    args.output_modes = parse_csv(args.output_modes, str)
    args.final_biases = parse_csv(args.final_biases, float)
    args.tv_weights = parse_csv(args.tv_weights, float)
    args.tv_types = parse_csv(args.tv_types, str)
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
