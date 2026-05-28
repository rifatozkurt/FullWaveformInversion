from pathlib import Path


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)} min {rem:.1f} s"


def format_run_status(results):
    if not results:
        return "No runs completed."

    lines = ["Completed runs:"]
    for item in results:
        lines.append(
            "- {method} case {case}: {elapsed} -> {run_dir}".format(
                method=item["method_name"],
                case=item["case_id"],
                elapsed=format_seconds(item["elapsed_seconds"]),
                run_dir=Path(item["run_dir"]),
            )
        )
    return "\n".join(lines)
