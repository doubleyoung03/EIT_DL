"""
Build comparison tables from test_runs.jsonl.

Usage (from project root):
    python src/analyze_logs.py
    python src/analyze_logs.py --config config.yaml
    python src/analyze_logs.py --sort-by test_rmse --top-k 30
    python src/analyze_logs.py --log results/test_runs.jsonl

Outputs (default in results/):
    - comparison_runs.csv
    - comparison_models.csv
    - comparison_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


RUN_COLUMNS = [
    "time",
    "model_name",
    "checkpoint",
    "loss_curve_file",
    "reconstruction_file",
    "test_mse",
    "test_rmse",
    "loc_err_mean_mm",
    "loc_err_median_mm",
    "loc_err_p90_mm",
    "n_samples",
    "seed",
]

MODEL_COLUMNS = [
    "model_name",
    "runs",
    "mean_test_mse",
    "mean_test_rmse",
    "std_test_rmse",
    "best_test_rmse",
    "mean_loc_err_mean_mm",
    "best_loc_err_mean_mm",
    "best_checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model comparison tables from JSONL logs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--log", default=None, help="Path to test_runs.jsonl (default resolved from config)")
    parser.add_argument(
        "--sort-by",
        default="test_rmse",
        choices=["time", "test_mse", "test_rmse", "loc_err_mean_mm", "loc_err_p90_mm"],
        help="Sort key for run ranking (default: test_rmse)",
    )
    parser.add_argument("--ascending", action="store_true", help="Sort ascending (default for numeric metrics is ascending)")
    parser.add_argument("--top-k", type=int, default=20, help="Rows shown in markdown top-runs table")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: results_dir from config)")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_time_key(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.log is not None:
        log_path = Path(args.log)
    else:
        log_name = str(cfg.get("test_log_filename", "test_runs.jsonl"))
        log_path = Path(log_name)
        if not log_path.is_absolute():
            log_path = Path(cfg["results_dir"]) / log_path

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["results_dir"])
    return log_path, output_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            records.append(obj)
    return records


def flatten_run(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {}) or {}
    cfg = record.get("config", {}) or {}
    model_name = record.get("model_name") or cfg.get("model_name") or "unknown"

    return {
        "time": str(record.get("time", "")),
        "model_name": str(model_name),
        "checkpoint": str(record.get("checkpoint", "")),
        "loss_curve_file": str(record.get("loss_curve_file", "")),
        "reconstruction_file": str(record.get("reconstruction_file", "")),
        "test_mse": _to_float(metrics.get("test_mse")),
        "test_rmse": _to_float(metrics.get("test_rmse")),
        "loc_err_mean_mm": _to_float(metrics.get("loc_err_mean_mm")),
        "loc_err_median_mm": _to_float(metrics.get("loc_err_median_mm")),
        "loc_err_p90_mm": _to_float(metrics.get("loc_err_p90_mm")),
        "n_samples": _to_int(record.get("n_samples")),
        "seed": _to_int(record.get("seed")),
    }


def sort_runs(rows: list[dict[str, Any]], key: str, ascending: bool) -> list[dict[str, Any]]:
    if key == "time":
        return sorted(rows, key=lambda r: _as_time_key(str(r.get("time", ""))), reverse=not ascending)

    def _numeric_key(r: dict[str, Any]) -> tuple[bool, float]:
        v = r.get(key)
        if v is None:
            return (True, float("inf"))
        return (False, float(v))

    return sorted(rows, key=_numeric_key, reverse=not ascending)


def aggregate_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_name"]), []).append(row)

    out: list[dict[str, Any]] = []
    for model_name, records in by_model.items():
        mse_vals = [r["test_mse"] for r in records if r["test_mse"] is not None]
        rmse_vals = [r["test_rmse"] for r in records if r["test_rmse"] is not None]
        loc_vals = [r["loc_err_mean_mm"] for r in records if r["loc_err_mean_mm"] is not None]

        best_row = None
        best_rmse = None
        for r in records:
            val = r.get("test_rmse")
            if val is None:
                continue
            if best_rmse is None or val < best_rmse:
                best_rmse = float(val)
                best_row = r

        out.append(
            {
                "model_name": model_name,
                "runs": len(records),
                "mean_test_mse": statistics.mean(mse_vals) if mse_vals else None,
                "mean_test_rmse": statistics.mean(rmse_vals) if rmse_vals else None,
                "std_test_rmse": statistics.stdev(rmse_vals) if len(rmse_vals) > 1 else 0.0 if rmse_vals else None,
                "best_test_rmse": min(rmse_vals) if rmse_vals else None,
                "mean_loc_err_mean_mm": statistics.mean(loc_vals) if loc_vals else None,
                "best_loc_err_mean_mm": min(loc_vals) if loc_vals else None,
                "best_checkpoint": str(best_row["checkpoint"]) if best_row is not None else "",
            }
        )

    return sorted(
        out,
        key=lambda r: (r["mean_test_rmse"] is None, r["mean_test_rmse"] if r["mean_test_rmse"] is not None else float("inf")),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def to_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        vals = [_fmt(row.get(c)) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_path, output_dir = resolve_paths(args)
    records = read_jsonl(log_path)

    if not records:
        raise ValueError(f"No valid records found in {log_path}.")

    runs = [flatten_run(r) for r in records]
    sort_ascending = args.ascending or args.sort_by != "time"
    runs_sorted = sort_runs(runs, key=args.sort_by, ascending=sort_ascending)
    models = aggregate_models(runs)

    runs_csv = output_dir / "comparison_runs.csv"
    models_csv = output_dir / "comparison_models.csv"
    report_md = output_dir / "comparison_report.md"

    write_csv(runs_csv, runs_sorted, RUN_COLUMNS)
    write_csv(models_csv, models, MODEL_COLUMNS)

    top_k = max(1, int(args.top_k))
    top_rows = runs_sorted[:top_k]
    report = "\n".join(
        [
            "# EIT Experiment Comparison",
            "",
            f"- log_file: `{log_path}`",
            f"- total_runs: {len(runs_sorted)}",
            f"- sort_by: `{args.sort_by}` ({'ascending' if sort_ascending else 'descending'})",
            "",
            "## Model Summary (lower is better)",
            "",
            to_markdown_table(models, MODEL_COLUMNS),
            "",
            f"## Top {top_k} Runs",
            "",
            to_markdown_table(top_rows, RUN_COLUMNS),
            "",
        ]
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(report, encoding="utf-8")

    print(f"[compare] runs csv    -> {runs_csv}")
    print(f"[compare] models csv  -> {models_csv}")
    print(f"[compare] report md   -> {report_md}")


if __name__ == "__main__":
    main()
