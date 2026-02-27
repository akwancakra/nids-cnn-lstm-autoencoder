"""Utility helpers for preprocessing and evaluation."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def list_csv_files(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files = list(root.rglob("*.csv"))
    return sorted(files)


def detect_label_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    col_set = {}
    for c in columns:
        key = str(c).strip().lower()
        if key not in col_set:
            col_set[key] = c
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in col_set:
            return col_set[key]
    return None


def sanitize_columns(df: pd.DataFrame, drop_columns: List[str], label_col: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in drop_columns and c != label_col]
    df = df[cols]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_inf_and_nan(df: pd.DataFrame, fillna_value: float) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(fillna_value)


def window_sequences(
    x: np.ndarray,
    y: Optional[np.ndarray],
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if window_size <= 1:
        return x[:, None, :], y

    n = x.shape[0]
    if n < window_size:
        return np.empty((0, window_size, x.shape[1]), dtype=x.dtype), None if y is None else np.empty((0,), dtype=y.dtype)

    windows = []
    labels = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(x[start:end])
        if y is not None:
            labels.append(1 if np.any(y[start:end] == 1) else 0)

    xw = np.stack(windows).astype(np.float32)
    yw = None if y is None else np.asarray(labels, dtype=np.int32)
    return xw, yw


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    val_ratio = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=(1 - val_ratio), random_state=seed, stratify=y_tmp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_shard_files(shard_dir: str | Path) -> List[Path]:
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("*.npz"))


def iter_npz_batches(
    shard_files: List[Path],
    batch_size: int,
    shuffle: bool,
    with_labels: bool,
):
    rng = np.random.default_rng(42)
    for shard_path in shard_files:
        data = np.load(shard_path)
        x = data["x"]
        if with_labels:
            y = data["y"]
        idx = np.arange(len(x))
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, len(x), batch_size):
            sel = idx[start : start + batch_size]
            xb = x[sel]
            if with_labels:
                yb = y[sel]
                yield xb, yb
            else:
                yield xb
