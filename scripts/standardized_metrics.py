"""Helpers to build a stable, comparable metrics schema across pipelines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr")


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    f1_value = metrics.get("f1")
    if f1_value is None:
        f1_value = metrics.get("f1_score")

    normalized = {
        "accuracy": _to_float(metrics.get("accuracy")),
        "precision": _to_float(metrics.get("precision")),
        "recall": _to_float(metrics.get("recall")),
        "f1": _to_float(f1_value),
        "roc_auc": _to_float(metrics.get("roc_auc")),
        "fpr": _to_float(metrics.get("fpr")),
    }
    return normalized


def compute_code_hash(paths: Sequence[str | Path]) -> str:
    """Compute a deterministic hash from one or more files."""
    hasher = hashlib.sha256()
    for raw_path in sorted(str(Path(p)) for p in paths):
        path = Path(raw_path)
        hasher.update(raw_path.encode("utf-8"))
        if not path.exists():
            hasher.update(b"<missing>")
            continue
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_standardized_record(
    *,
    dataset: str,
    metrics: Mapping[str, Any],
    threshold: float,
    threshold_method: str,
    mode: str,
    seed: int,
    model_path: str,
    tag: str,
    code_hash: str,
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one flattened metrics record with stable keys for comparisons."""
    normalized = _normalize_metrics(metrics)
    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "mode": str(mode),
        "threshold_method": str(threshold_method),
        "threshold": _to_float(threshold),
        "seed": int(seed),
        "tag": str(tag),
        "model_path": str(model_path),
        "code_hash": str(code_hash),
        "config_snapshot": dict(config_snapshot),
    }
    record.update(normalized)

    # Preserve confusion counts when available.
    for key in ("tp", "fp", "tn", "fn"):
        if key in metrics:
            record[key] = int(metrics[key])

    return record
