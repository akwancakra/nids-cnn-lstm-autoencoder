"""Compare standardized metrics between baseline and candidate pipelines."""

from __future__ import annotations

import argparse
from glob import glob
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metric(record: dict[str, Any], key: str) -> float:
    if key in record:
        return _to_float(record[key])
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        return _to_float(metrics.get(key))
    return 0.0


def _metric_stats(records: Iterable[dict[str, Any]], key: str) -> dict[str, float]:
    values = [_extract_metric(record, key) for record in records]
    if not values:
        raise ValueError(f"No values found for metric '{key}'.")
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def compare_candidate_vs_baseline(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not baseline_records:
        raise ValueError("baseline_records cannot be empty.")
    if not candidate_records:
        raise ValueError("candidate_records cannot be empty.")

    baseline_f1 = _metric_stats(baseline_records, "f1")
    baseline_fpr = _metric_stats(baseline_records, "fpr")
    candidate_f1 = _metric_stats(candidate_records, "f1")
    candidate_fpr = _metric_stats(candidate_records, "fpr")

    delta_f1_mean = candidate_f1["mean"] - baseline_f1["mean"]
    delta_fpr_mean = candidate_fpr["mean"] - baseline_fpr["mean"]
    candidate_is_better = (delta_f1_mean > 0.0) and (delta_fpr_mean < 0.0)

    return {
        "baseline": {"f1": baseline_f1, "fpr": baseline_fpr},
        "candidate": {"f1": candidate_f1, "fpr": candidate_fpr},
        "delta_f1_mean": delta_f1_mean,
        "delta_fpr_mean": delta_fpr_mean,
        "candidate_is_better": candidate_is_better,
        "gate_rule": "candidate_mean_f1 > baseline_mean_f1 AND candidate_mean_fpr < baseline_mean_fpr",
    }


def _load_json_records(pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(glob(pattern)):
        path = Path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend([item for item in data if isinstance(item, dict)])
        elif isinstance(data, dict):
            records.append(data)
    return records


def _filter_records(records: list[dict[str, Any]], dataset: str | None, mode: str | None) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if dataset is not None and str(record.get("dataset")) != dataset:
            continue
        if mode is not None and str(record.get("mode")) != mode:
            continue
        out.append(record)
    return out


def _to_markdown(result: dict[str, Any], baseline_count: int, candidate_count: int) -> str:
    lines = [
        "# A/B Comparison Result",
        "",
        f"- baseline_records: {baseline_count}",
        f"- candidate_records: {candidate_count}",
        f"- candidate_is_better: {result['candidate_is_better']}",
        "",
        "## Mean Delta",
        "",
        f"- delta_f1_mean: {result['delta_f1_mean']:.6f}",
        f"- delta_fpr_mean: {result['delta_fpr_mean']:.6f}",
        "",
        "## Gate Rule",
        "",
        f"- {result['gate_rule']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--dataset", default="CSE-CIC-IDS2018")
    parser.add_argument("--mode", default="zero_shot")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    baseline_all = _load_json_records(args.baseline_glob)
    candidate_all = _load_json_records(args.candidate_glob)
    baseline = _filter_records(baseline_all, args.dataset, args.mode)
    candidate = _filter_records(candidate_all, args.dataset, args.mode)

    report = compare_candidate_vs_baseline(baseline, candidate)
    report["dataset"] = args.dataset
    report["mode"] = args.mode
    report["baseline_count"] = len(baseline)
    report["candidate_count"] = len(candidate)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(report, len(baseline), len(candidate)), encoding="utf-8")

    print(json.dumps({"candidate_is_better": report["candidate_is_better"]}, indent=2))


if __name__ == "__main__":
    main()
