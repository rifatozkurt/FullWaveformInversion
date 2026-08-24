"""
Select INR learning rates on IN-DISTRIBUTION held-out cases.

Run this BEFORE the main INR experiments if you want the hyperparameters chosen
on the same distribution you report on:

    !python scripts/tune_inr_learning_rates.py \
        --config configs/config_final.yaml \
        --data-dir /content/eval \
        --run-dir runs/final/inr_tuning

WHY THIS EXISTS
---------------
The values currently in the config were selected on `data/casestudy/`, which
holds deliberately unusual out-of-distribution shapes. Selection set and
reporting set should not differ. A partial in-distribution re-measurement already
showed the difference is real: IG-FWI scored 0.408x the trivial baseline on the
case-study sample but 0.868x on eval case 15002, and a grid rate of 3e-1 --
merely mediocre on the case study -- DIVERGED in distribution (6.5x).

Each trial runs as its own subprocess so GPU memory is reclaimed by process exit
rather than by hoping `empty_cache()` was enough. Trials run strictly one at a
time. Results stream to CSV as they complete, so an interrupted run keeps
whatever finished.

Ranking is on FINAL gamma-MSE relative to the trivial "no void anywhere"
solution, NOT on relative improvement. That matters: in an earlier sweep the
single worst configuration (diverged, 5x worse than trivial) also had the
HIGHEST relative improvement, because its first step was destructive enough to
leave a lot to recover. Ranking on improvement would have selected it.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.config import load_config
from src.io import ensure_dir, load_hdf
from src.reporting import plot_metric_bars, write_csv

REPO = Path(__file__).resolve().parents[1]

# Brackets around the values already measured, rather than a fresh wide grid:
# orders-of-magnitude separation between convergence and divergence is already
# established, so only the local ordering is in question.
DEFAULT_GRID = {
    "inr_siren_fwi": ("lr", [1e-4, 3e-4, 1e-3]),
    "inr_lr_fwi": ("lr", [1e-4, 3e-4, 1e-3]),
    "inr_mpe_fwi": ("lr", [1e-2, 3e-2, 1e-1]),
    "inr_ig_fwi": ("lr_grid", [1e-2, 3e-2, 1e-1]),
    "inr_siren_centered_fwi": ("lr_bias", [0.1, 1.0, 3.0]),
    "inr_mpe_centered_fwi": ("lr_bias", [0.1, 1.0, 3.0]),
    "inr_ig_centered_fwi": ("lr_bias", [0.1, 1.0, 3.0]),
}

WORKER = r'''
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, r"{repo}")
import numpy as np
from src.config import load_config
from src.io import ensure_dir
from src.registry import get_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--case", type=int, required=True)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--value", type=float, required=True)
a = ap.parse_args()

cfg = load_config(a.config)
cfg["experiments"][a.method]["epochs"] = a.epochs
cfg["experiments"][a.method][a.key] = a.value
run_dir = ensure_dir(a.run_dir)
for sub in ("figures", "histories", "outputs"):
    ensure_dir(run_dir / sub)
try:
    res = get_experiment(a.method)(cfg).run(a.case, Path(a.data_dir), run_dir)
    mse = np.asarray(res.mse_history, dtype=float)
    out = {{"method": a.method, "case": a.case, "key": a.key, "value": a.value,
           "mse_final": float(mse[-1]), "mse_best": float(np.min(mse)),
           "finite": bool(np.all(np.isfinite(mse)))}}
except Exception as exc:
    out = {{"method": a.method, "case": a.case, "key": a.key, "value": a.value,
           "error": f"{{type(exc).__name__}}: {{exc}}"[:200]}}
print("RESULT " + json.dumps(out), flush=True)
'''


def apply_to_config(config_path, best):
    """
    Write the winning values into the YAML in place, preserving comments.

    Text substitution rather than a yaml round-trip, because `yaml.safe_dump`
    discards every comment in the file and this config carries a lot of them.
    Only the exact `key:` line inside the matching `  <method>:` block is
    touched. A rate that failed to beat the trivial solution is NOT written --
    silently installing a value that reconstructs nothing would be worse than
    leaving the previous one in place.
    """
    import re
    import shutil

    path = Path(config_path)
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    text = path.read_text(encoding="utf-8")
    applied, skipped = [], []

    for method, row in sorted(best.items()):
        if row["vs_trivial"] >= 0.99:
            skipped.append(f"{method}: {row['key']}={row['value']:g} "
                           f"({row['vs_trivial']:.3f}x trivial -- not usable)")
            continue
        block = re.search(rf"(^  {re.escape(method)}:\n(?:(?:    .*|\s*#.*)\n)*)", text, re.M)
        if not block:
            skipped.append(f"{method}: no such block in {path.name}")
            continue
        body, key, value = block.group(1), row["key"], row["value"]
        line = f"    {key}: {value:g}"
        if re.search(rf"^    {re.escape(key)}:", body, re.M):
            new_body = re.sub(rf"^    {re.escape(key)}: .*$", line, body, count=1, flags=re.M)
        else:                       # key absent (e.g. lr_bias on a fresh config)
            new_body = re.sub(r"^(    seed: .*\n)", rf"\1{line}\n", body, count=1, flags=re.M)
        if new_body == body:
            skipped.append(f"{method}: could not place {key}")
            continue
        text = text[:block.start()] + new_body + text[block.end():]
        applied.append(f"{method}: {key} -> {value:g}  ({row['vs_trivial']:.3f}x trivial)")

    path.write_text(text, encoding="utf-8")
    return applied, skipped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument("--methods", default=",".join(DEFAULT_GRID),
                        help="Comma-separated subset of the tunable methods.")
    parser.add_argument("--case", type=int, default=None,
                        help="Tuning case. Default: the config case with the median "
                             "void fraction, which is more representative than the first.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Short by design: enough to rank rates and catch divergence.")
    parser.add_argument("--apply", action="store_true",
                        help="Write the winning values straight into --config, in place, "
                             "preserving comments. A .bak copy is kept next to it.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir or config["paths"]["casestudy_data"])
    root = ensure_dir(args.run_dir or Path(config["paths"]["runs"]) / "inr_tuning")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in DEFAULT_GRID]
    if unknown:
        raise SystemExit(f"not tunable here: {unknown}\navailable: {sorted(DEFAULT_GRID)}")

    # Pick a representative case rather than the first one: void fractions vary
    # ~20x across eval cases, and the low ones leave almost nothing to improve on.
    if args.case is not None:
        case_id = args.case
    else:
        candidates = []
        for cid in config["experiments"]["cases"]:
            path = data_dir / f"material{cid}.h5"
            if path.exists():
                target = np.asarray(load_hdf(path), dtype=np.float64)
                candidates.append((float((target < 0.5).mean()), int(cid)))
        if not candidates:
            raise SystemExit(f"no case files found in {data_dir}")
        candidates.sort()
        case_id = candidates[len(candidates) // 2][1]

    target = np.asarray(load_hdf(data_dir / f"material{case_id}.h5"), dtype=np.float64)
    trivial = float(((1.0 - target) ** 2).mean())
    trials = [(m, *DEFAULT_GRID[m][:1], v) for m in methods for v in DEFAULT_GRID[m][1]]

    print("=" * 78)
    print(f"  tuning case   : {case_id}  (void fraction {float((target<0.5).mean())*100:.2f}%)")
    print(f"  trivial MSE   : {trivial:.4e}   <- a rate only 'works' well below this")
    print(f"  data          : {data_dir}")
    print(f"  epochs/trial  : {args.epochs}")
    print(f"  trials        : {len(trials)}")
    print("=" * 78, flush=True)
    if args.dry_run:
        for method, key, value in trials:
            print(f"  {method:<26} {key}={value:g}")
        return 0

    worker_path = root / "_worker.py"
    worker_path.write_text(WORKER.format(repo=str(REPO)), encoding="utf-8")

    rows, started = [], time.perf_counter()
    print(f"{'method':<26}{'setting':<18}{'best mse':>12}{'vs trivial':>12}", flush=True)
    print("-" * 72, flush=True)
    for index, (method, key, value) in enumerate(trials):
        cmd = [sys.executable, str(worker_path), "--config", args.config,
               "--method", method, "--case", str(case_id), "--epochs", str(args.epochs),
               "--data-dir", str(data_dir), "--key", key, "--value", repr(value),
               "--run-dir", str(root / "trials" / f"{method}_{key}{value:g}")]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            rec = json.loads(line[7:]) if line else {
                "method": method, "key": key, "value": value,
                "error": (proc.stderr.strip().splitlines() or ["no RESULT line"])[-1][:200]}
        except subprocess.TimeoutExpired:
            rec = {"method": method, "key": key, "value": value, "error": "timeout"}
        rec["trivial_mse"] = trivial
        if "error" not in rec:
            rec["vs_trivial"] = rec["mse_best"] / max(trivial, 1e-30)
        rows.append(rec)
        write_csv(root / "tuning_results.csv", rows)
        if "error" in rec:
            print(f"{method:<26}{key+'='+format(value,'g'):<18}  ERROR {rec['error'][:40]}", flush=True)
        else:
            print(f"{method:<26}{key+'='+format(value,'g'):<18}"
                  f"{rec['mse_best']:>12.4e}{rec['vs_trivial']:>11.3f}x", flush=True)

    ok = [r for r in rows if "error" not in r]
    if ok:
        plot_metric_bars(root / "tuning_vs_trivial.png", ok, "method",
                         [("vs_trivial", "best gamma MSE / trivial (lower better)")])
        best = {}
        for row in ok:
            key = row["method"]
            if key not in best or row["mse_best"] < best[key]["mse_best"]:
                best[key] = row
        write_csv(root / "best_per_method.csv", list(best.values()))
        print("\n" + "=" * 78)
        print("  BEST PER METHOD")
        print("-" * 78)
        for method, row in best.items():
            flag = "" if row["vs_trivial"] < 0.99 else "   <-- NOT usable (>= trivial)"
            print(f"  {method:<28} {row['key']}: {row['value']:g}"
                  f"      ({row['vs_trivial']:.3f}x trivial){flag}")
        print("=" * 78)
        print("  A value at or above 1.0x has not reconstructed anything at this")
        print("  epoch budget -- treat it as 'no usable rate found', not as a winner.")

        # A ready-to-paste YAML fragment, written whether or not --apply is used,
        # so the numbers survive even if the config is not edited here.
        fragment = ["# Selected by scripts/tune_inr_learning_rates.py",
                    f"# case {case_id}, {args.epochs} FWI epochs, trivial MSE {trivial:.4e}",
                    "experiments:"]
        for method, row in sorted(best.items()):
            fragment.append(f"  {method}:")
            fragment.append(f"    {row['key']}: {row['value']:g}"
                            f"   # {row['vs_trivial']:.3f}x trivial")
        (root / "tuned_values.yaml").write_text("\n".join(fragment) + "\n", encoding="utf-8")
        print(f"\n  YAML fragment: {root/'tuned_values.yaml'}")

        if args.apply:
            applied, skipped = apply_to_config(args.config, best)
            print(f"\n  APPLIED to {args.config}  ({len(applied)} value(s) written)")
            for line in applied:
                print(f"    {line}")
            for line in skipped:
                print(f"    SKIPPED {line}")
            print(f"    backup: {args.config}.bak")
            print("\n  The experiments below will now use these values. Remember to copy")
            print("  the edited config back to Drive -- a Colab checkout is not persistent.")
        else:
            print("\n  Re-run with --apply to write these into the config automatically.")
    print(f"\n  {len(rows)-len(ok)} failed, {len(ok)} ok, {(time.perf_counter()-started)/60:.1f} min")
    print(f"  results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
