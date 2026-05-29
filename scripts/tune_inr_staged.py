import argparse
import copy
import itertools
import time
import traceback
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.config import load_config, save_experiment_config
from src.io import create_run_dir, ensure_dir
from src.registry import get_experiment


STAGES = (
    "01_siren",
    "02_lr_rank",
    "03_mpe_grid",
    "04_ig_fusion",
)


def parse_csv(value, cast):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def format_value(value):
    if isinstance(value, float):
        return "{:.0e}".format(value) if value != 0 else "0"
    return str(value)


def short_params(params):
    return ", ".join(f"{key}={format_value(value)}" for key, value in params.items())


def markdown_table(rows, columns):
    if not rows:
        return "_No results yet._\n"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "{:.6g}".format(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def update_config(base_config, method, case_id, epochs, params):
    config = copy.deepcopy(base_config)
    config["experiments"][method].update(params)
    config["experiments"][method]["epochs"] = epochs
    config.setdefault("run", {})
    config["run"]["method"] = method
    config["run"]["case_id"] = case_id
    return config


def trial_dir_name(index, stage_name, method, params):
    bits = [f"trial{index:03d}", stage_name, method]
    for key, value in params.items():
        bits.append(f"{key}-{format_value(value)}")
    name = "_".join(bits)
    return name.replace(".", "p").replace("+", "")


def run_trial(base_config, method, case_id, data_dir, run_dir, epochs, params):
    config = update_config(base_config, method, case_id, epochs, params)
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


def save_records(records, sweep_dir):
    df = pd.DataFrame(records)
    df.to_csv(sweep_dir / "results_all.csv", index=False)
    for stage in STAGES:
        stage_df = df[df["stage"] == stage] if not df.empty and "stage" in df else pd.DataFrame()
        stage_dir = ensure_dir(sweep_dir / stage)
        stage_df.to_csv(stage_dir / "results.csv", index=False)
    return df


def best_success(records, stage, metric):
    rows = [
        row for row in records
        if row["stage"] == stage and row.get("status") == "ok" and pd.notna(row.get(metric))
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: row[metric])


def write_summary(sweep_dir, records, carried, current_message=""):
    summary_path = sweep_dir / "TUNING_SUMMARY.md"
    lines = [
        "# Staged INR Tuning Summary",
        "",
        f"Output folder: `{sweep_dir}`",
        "",
    ]
    if current_message:
        lines.extend(["## Current Status", "", current_message, ""])

    lines.extend(["## Carried Best Parameters", ""])
    if carried:
        for stage, params in carried.items():
            lines.append(f"- `{stage}`: {short_params(params)}")
    else:
        lines.append("_No stage has finished yet._")
    lines.append("")

    for stage in STAGES:
        stage_rows = [row for row in records if row["stage"] == stage]
        lines.extend([f"## {stage}", ""])
        if not stage_rows:
            lines.extend(["_Not started._", ""])
            continue

        ok_rows = [row for row in stage_rows if row.get("status") == "ok"]
        fail_rows = [row for row in stage_rows if row.get("status") != "ok"]
        if ok_rows:
            lines.extend([
                "### Best By MSE",
                "",
                markdown_table(
                    sorted(ok_rows, key=lambda row: row["best_mse"])[:10],
                    [
                        "trial",
                        "method",
                        "best_mse",
                        "best_cost",
                        "runtime_seconds",
                        "lr",
                        "costScaling",
                        "omega0",
                        "tv_weight",
                        "rank",
                        "base_resolution",
                        "per_level_scale",
                        "features_per_level",
                        "fusion_alpha",
                    ],
                ),
                "",
            ])
        if fail_rows:
            lines.extend([
                "### Failed Trials",
                "",
                markdown_table(fail_rows, ["trial", "method", "error"]),
                "",
            ])

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def plot_stage_rankings(df, sweep_dir, metric):
    if df.empty:
        return

    for stage, stage_df in df.groupby("stage"):
        stage_df = stage_df[stage_df["status"] == "ok"].sort_values(metric)
        if stage_df.empty:
            continue

        plot_df = stage_df.head(20).copy()
        labels = []
        for _, row in plot_df.iterrows():
            bits = [f"T{int(row['trial'])}"]
            for key in ("lr", "omega0", "tv_weight", "rank", "base_resolution", "per_level_scale", "features_per_level", "fusion_alpha"):
                if key in row and pd.notna(row[key]):
                    bits.append(f"{key}={format_value(row[key])}")
            labels.append("\n".join(bits))

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(plot_df)), 5))
        ax.bar(range(len(plot_df)), plot_df[metric].values, color="#2d6cdf")
        ax.set_title(f"{stage}: best {metric} values", fontsize=13)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(sweep_dir / stage / f"{stage}_{metric}_ranking.png", dpi=180)
        plt.close(fig)


def plot_siren_heatmaps(df, sweep_dir, metric):
    stage_df = df[(df["stage"] == "01_siren") & (df["status"] == "ok")]
    if stage_df.empty:
        return

    for tv_weight, group in stage_df.groupby("tv_weight", dropna=False):
        table = group.pivot_table(index="omega0", columns="lr", values=metric, aggfunc="min")
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.3 * len(table.columns) + 3, 1.0 * len(table.index) + 2.5))
        image = ax.imshow(table.values, aspect="auto", cmap="viridis")
        ax.set_title(f"SIREN {metric}, tv_weight={format_value(tv_weight)}", fontsize=12)
        ax.set_xlabel("lr")
        ax.set_ylabel("omega0")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([format_value(v) for v in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([format_value(v) for v in table.index])
        for i, omega0 in enumerate(table.index):
            for j, lr in enumerate(table.columns):
                value = table.loc[omega0, lr]
                if pd.notna(value):
                    ax.text(j, i, "{:.2e}".format(value), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(sweep_dir / "01_siren" / f"siren_{metric}_tv{format_value(tv_weight)}.png", dpi=180)
        plt.close(fig)


def plot_mpe_heatmaps(df, sweep_dir, metric):
    stage_df = df[(df["stage"] == "03_mpe_grid") & (df["status"] == "ok")]
    if stage_df.empty:
        return

    for fpl, group in stage_df.groupby("features_per_level", dropna=False):
        table = group.pivot_table(index="base_resolution", columns="per_level_scale", values=metric, aggfunc="min")
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.3 * len(table.columns) + 3, 1.0 * len(table.index) + 2.5))
        image = ax.imshow(table.values, aspect="auto", cmap="magma")
        ax.set_title(f"MPE {metric}, features_per_level={fpl}", fontsize=12)
        ax.set_xlabel("per_level_scale")
        ax.set_ylabel("base_resolution")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([format_value(v) for v in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([format_value(v) for v in table.index])
        for i, base_resolution in enumerate(table.index):
            for j, per_level_scale in enumerate(table.columns):
                value = table.loc[base_resolution, per_level_scale]
                if pd.notna(value):
                    ax.text(j, i, "{:.2e}".format(value), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(sweep_dir / "03_mpe_grid" / f"mpe_{metric}_fpl{fpl}.png", dpi=180)
        plt.close(fig)


def update_plots(records, sweep_dir, metric):
    df = save_records(records, sweep_dir)
    plot_stage_rankings(df, sweep_dir, metric)
    plot_siren_heatmaps(df, sweep_dir, metric)
    plot_mpe_heatmaps(df, sweep_dir, metric)


def make_trials(keys, values):
    for combo in itertools.product(*(values[key] for key in keys)):
        yield dict(zip(keys, combo))


def run_stage(stage_name, method, base_config, case_id, data_dir, sweep_dir, epochs, trial_start, trials, records, carried, metric, stop_on_error):
    stage_dir = ensure_dir(sweep_dir / stage_name)
    trial_index = trial_start

    for params in trials:
        trial_index += 1
        run_dir = ensure_dir(stage_dir / trial_dir_name(trial_index, stage_name, method, params))
        message = f"Running {stage_name} trial {trial_index}: `{method}` with {short_params(params)}"
        print(message)
        write_summary(sweep_dir, records, carried, message)

        record = {
            "trial": trial_index,
            "stage": stage_name,
            "method": method,
            "case_id": case_id,
            "epochs": epochs,
            "run_dir": str(run_dir),
            **params,
        }

        try:
            metrics = run_trial(base_config, method, case_id, data_dir, run_dir, epochs, params)
            record.update(metrics)
            record["status"] = "ok"
            record["error"] = ""
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if stop_on_error:
                records.append(record)
                update_plots(records, sweep_dir, metric)
                write_summary(sweep_dir, records, carried, f"Stopped after failed trial {trial_index}.")
                raise

        records.append(record)
        update_plots(records, sweep_dir, metric)
        write_summary(sweep_dir, records, carried, f"Finished trial {trial_index}.")

    return trial_index


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run staged INR-family tuning and carry best parameters between methods."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--selection-metric", default="best_mse", choices=("best_mse", "final_mse", "best_cost", "final_cost"))
    parser.add_argument("--cost-scaling", type=float, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--siren-lrs", default="1e-4,3e-4")
    parser.add_argument("--siren-omega0s", default="0.5,30")
    parser.add_argument("--siren-tv-weights", default="0,1e-5")
    parser.add_argument("--tv-type", default="anisotropic")

    parser.add_argument("--lr-ranks", default="8,16,32")

    parser.add_argument("--mpe-base-resolutions", default="16,32,50")
    parser.add_argument("--mpe-per-level-scales", default="1.05,1.2,1.5")
    parser.add_argument("--mpe-features-per-levels", default="2")

    parser.add_argument("--ig-fusion-alphas", default="0.25,0.5,0.75")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.siren_lrs = parse_csv(args.siren_lrs, float)
    args.siren_omega0s = parse_csv(args.siren_omega0s, float)
    args.siren_tv_weights = parse_csv(args.siren_tv_weights, float)
    args.lr_ranks = parse_csv(args.lr_ranks, int)
    args.mpe_base_resolutions = parse_csv(args.mpe_base_resolutions, int)
    args.mpe_per_level_scales = parse_csv(args.mpe_per_level_scales, float)
    args.mpe_features_per_levels = parse_csv(args.mpe_features_per_levels, int)
    args.ig_fusion_alphas = parse_csv(args.ig_fusion_alphas, float)

    base_config = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else Path(base_config["paths"]["casestudy_data"])
    sweep_dir = create_run_dir("runs/tuning_new", prefix=f"staged_inr_case{args.case}")
    for stage in STAGES:
        ensure_dir(sweep_dir / stage)

    records = []
    carried = {}
    trial_index = 0
    metric = args.selection_metric
    cost_scaling = args.cost_scaling
    if cost_scaling is None:
        cost_scaling = base_config["experiments"]["inr_siren_fwi"]["costScaling"]

    print(f"Saving staged tuning outputs to {sweep_dir}")
    write_summary(sweep_dir, records, carried, "Starting staged tuning.")

    siren_trials = make_trials(
        ["lr", "omega0", "tv_weight"],
        {
            "lr": args.siren_lrs,
            "omega0": args.siren_omega0s,
            "tv_weight": args.siren_tv_weights,
        },
    )
    siren_trials = [
        {
            **trial,
            "costScaling": cost_scaling,
            "tv_type": args.tv_type,
        }
        for trial in siren_trials
    ]
    trial_index = run_stage(
        "01_siren",
        "inr_siren_fwi",
        base_config,
        args.case,
        data_dir,
        sweep_dir,
        args.epochs,
        trial_index,
        siren_trials,
        records,
        carried,
        metric,
        args.stop_on_error,
    )
    best_siren = best_success(records, "01_siren", metric)
    if not best_siren:
        raise RuntimeError("No successful SIREN trials. Cannot continue staged tuning.")
    carried["best_siren"] = {
        "lr": best_siren["lr"],
        "omega0": best_siren["omega0"],
        "costScaling": best_siren["costScaling"],
        "tv_weight": best_siren["tv_weight"],
        "tv_type": best_siren["tv_type"],
    }
    write_summary(sweep_dir, records, carried, "Finished SIREN stage.")

    lr_trials = [
        {
            **carried["best_siren"],
            "rank": rank,
        }
        for rank in args.lr_ranks
    ]
    trial_index = run_stage(
        "02_lr_rank",
        "inr_lr_fwi",
        base_config,
        args.case,
        data_dir,
        sweep_dir,
        args.epochs,
        trial_index,
        lr_trials,
        records,
        carried,
        metric,
        args.stop_on_error,
    )
    best_lr = best_success(records, "02_lr_rank", metric)
    if best_lr:
        carried["best_lr"] = {
            "rank": best_lr["rank"],
            "lr": best_lr["lr"],
            "omega0": best_lr["omega0"],
            "costScaling": best_lr["costScaling"],
            "tv_weight": best_lr["tv_weight"],
            "tv_type": best_lr["tv_type"],
        }
    write_summary(sweep_dir, records, carried, "Finished LR-INR rank stage.")

    mpe_trials = make_trials(
        ["base_resolution", "per_level_scale", "features_per_level"],
        {
            "base_resolution": args.mpe_base_resolutions,
            "per_level_scale": args.mpe_per_level_scales,
            "features_per_level": args.mpe_features_per_levels,
        },
    )
    mpe_trials = [
        {
            "lr": carried["best_siren"]["lr"],
            "costScaling": carried["best_siren"]["costScaling"],
            "tv_weight": carried["best_siren"]["tv_weight"],
            "tv_type": carried["best_siren"]["tv_type"],
            **trial,
        }
        for trial in mpe_trials
    ]
    trial_index = run_stage(
        "03_mpe_grid",
        "inr_mpe_fwi",
        base_config,
        args.case,
        data_dir,
        sweep_dir,
        args.epochs,
        trial_index,
        mpe_trials,
        records,
        carried,
        metric,
        args.stop_on_error,
    )
    best_mpe = best_success(records, "03_mpe_grid", metric)
    if not best_mpe:
        raise RuntimeError("No successful MPE trials. Cannot continue to IG stage.")
    carried["best_mpe"] = {
        "base_resolution": best_mpe["base_resolution"],
        "per_level_scale": best_mpe["per_level_scale"],
        "features_per_level": best_mpe["features_per_level"],
        "lr": best_mpe["lr"],
        "costScaling": best_mpe["costScaling"],
        "tv_weight": best_mpe["tv_weight"],
        "tv_type": best_mpe["tv_type"],
    }
    write_summary(sweep_dir, records, carried, "Finished MPE grid stage.")

    ig_trials = [
        {
            "lr": carried["best_mpe"]["lr"],
            "costScaling": carried["best_mpe"]["costScaling"],
            "tv_weight": carried["best_mpe"]["tv_weight"],
            "tv_type": carried["best_mpe"]["tv_type"],
            "omega0": carried["best_siren"]["omega0"],
            "base_resolution": carried["best_mpe"]["base_resolution"],
            "per_level_scale": carried["best_mpe"]["per_level_scale"],
            "features_per_level": carried["best_mpe"]["features_per_level"],
            "fusion_alpha": fusion_alpha,
        }
        for fusion_alpha in args.ig_fusion_alphas
    ]
    trial_index = run_stage(
        "04_ig_fusion",
        "inr_ig_fwi",
        base_config,
        args.case,
        data_dir,
        sweep_dir,
        args.epochs,
        trial_index,
        ig_trials,
        records,
        carried,
        metric,
        args.stop_on_error,
    )
    best_ig = best_success(records, "04_ig_fusion", metric)
    if best_ig:
        carried["best_ig"] = {
            "fusion_alpha": best_ig["fusion_alpha"],
            "lr": best_ig["lr"],
            "omega0": best_ig["omega0"],
            "base_resolution": best_ig["base_resolution"],
            "per_level_scale": best_ig["per_level_scale"],
            "features_per_level": best_ig["features_per_level"],
            "costScaling": best_ig["costScaling"],
            "tv_weight": best_ig["tv_weight"],
            "tv_type": best_ig["tv_type"],
        }

    update_plots(records, sweep_dir, metric)
    write_summary(sweep_dir, records, carried, "Finished all staged tuning.")
    print(f"Saved all results to {sweep_dir}")
    print(f"Human-readable summary: {sweep_dir / 'TUNING_SUMMARY.md'}")


if __name__ == "__main__":
    main()
