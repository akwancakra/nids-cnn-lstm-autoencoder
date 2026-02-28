"""Preprocess pipeline: streaming clean, feature alignment, scaling, and sharded windowing."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import (
    detect_label_column,
    ensure_dir,
    list_csv_files,
    load_yaml,
    replace_inf_and_nan,
    save_json,
    save_npz,
    setup_logging,
    window_sequences,
)


def get_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="uniform", n_quantiles=1000, random_state=42)
    raise ValueError(f"Unknown scaler: {name}")


def select_features_by_statistics(
    sample_array: np.ndarray | list[list[float]],
    features: List[str],
    enable_nzv: bool,
    nzv_threshold: float,
    enable_corr: bool,
    corr_threshold: float,
) -> Tuple[List[str], dict]:
    x = np.asarray(sample_array, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("sample_array must be 2D")
    if x.shape[1] != len(features):
        raise ValueError("sample_array columns must match features length")

    keep_mask = np.ones(len(features), dtype=bool)
    dropped_nzv: List[str] = []
    dropped_corr: List[str] = []

    if enable_nzv:
        var = np.var(x, axis=0)
        nzv_mask = var <= nzv_threshold
        for idx in np.where(nzv_mask)[0]:
            dropped_nzv.append(features[idx])
        keep_mask &= ~nzv_mask

    if enable_corr:
        kept_idx = np.where(keep_mask)[0]
        if kept_idx.size > 1:
            x_kept = x[:, kept_idx]
            corr = np.corrcoef(x_kept, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            to_drop_local = set()
            for i in range(corr.shape[0]):
                if i in to_drop_local:
                    continue
                for j in range(i + 1, corr.shape[0]):
                    if abs(corr[i, j]) >= corr_threshold:
                        # Keep earlier feature deterministically, drop later one.
                        to_drop_local.add(j)
            for local_j in sorted(to_drop_local):
                global_idx = kept_idx[local_j]
                keep_mask[global_idx] = False
                dropped_corr.append(features[global_idx])

    selected = [f for i, f in enumerate(features) if keep_mask[i]]
    if not selected:
        raise ValueError("All features were dropped by statistical filtering.")

    report = {
        "input_feature_count": len(features),
        "selected_feature_count": len(selected),
        "dropped_nzv": dropped_nzv,
        "dropped_corr": dropped_corr,
        "enable_nzv": bool(enable_nzv),
        "nzv_threshold": float(nzv_threshold),
        "enable_corr": bool(enable_corr),
        "corr_threshold": float(corr_threshold),
    }
    return selected, report


def compute_feature_clip_bounds(
    sample_array: np.ndarray | list[list[float]],
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sample_array, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("sample_array must be 2D")
    if not (0.5 < quantile < 1.0):
        raise ValueError("quantile must be in range (0.5, 1.0)")

    lower_q = (1.0 - quantile) * 100.0
    upper_q = quantile * 100.0
    lower = np.percentile(x, lower_q, axis=0).astype(np.float32)
    upper = np.percentile(x, upper_q, axis=0).astype(np.float32)
    swapped = lower > upper
    if np.any(swapped):
        tmp = lower.copy()
        lower[swapped] = upper[swapped]
        upper[swapped] = tmp[swapped]
    return lower, upper


def clip_features_per_column(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    low = np.asarray(lower, dtype=np.float32)
    high = np.asarray(upper, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError("x must be 2D")
    if low.ndim != 1 or high.ndim != 1:
        raise ValueError("lower/upper must be 1D")
    if arr.shape[1] != low.shape[0] or arr.shape[1] != high.shape[0]:
        raise ValueError("x columns must match lower/upper length")

    return np.clip(arr, low[None, :], high[None, :]).astype(np.float32)


def load_header_columns(csv_path: Path) -> List[str]:
    cols = list(pd.read_csv(csv_path, nrows=0, skipinitialspace=True).columns)
    return [str(c).strip() for c in cols]


def canonical_key(name: str) -> str:
    x = str(name).strip().lower()
    x = x.replace("/", " ")
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x)

    replacements = {
        "dst": "destination",
        "src": "source",
        "byts": "bytes",
        "pkts": "packets",
        "pkt": "packet",
        "cnt": "count",
        "len": "length",
        "avg": "average",
        "tot": "total",
        "seg": "segment",
    }

    tokens = []
    for tok in x.split():
        tok = replacements.get(tok, tok)
        if tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]
        tokens.append(tok)
    tokens.sort()
    return " ".join(tokens)


def build_column_mapper(reference_cols: List[str], target_cols: List[str]) -> dict[str, str]:
    key_to_ref: dict[str, str] = {}
    ambiguous: set[str] = set()
    for col in reference_cols:
        key = canonical_key(col)
        existing = key_to_ref.get(key)
        if existing is not None and existing != col:
            ambiguous.add(key)
        else:
            key_to_ref[key] = col
    for key in ambiguous:
        key_to_ref.pop(key, None)

    ref_lower = {c.lower(): c for c in reference_cols}
    mapper: dict[str, str] = {}
    for col in target_cols:
        lower = col.lower()
        if lower in ref_lower:
            mapper[col] = ref_lower[lower]
            continue
        key = canonical_key(col)
        if key in key_to_ref:
            mapper[col] = key_to_ref[key]
    return mapper


def compute_feature_intersection(
    csv_files: List[Path],
    label_candidates: List[str],
    drop_columns: List[str],
    column_mapper: Optional[dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    feature_intersection: Optional[set[str]] = None
    label_col: Optional[str] = None

    for csv_path in csv_files:
        cols = load_header_columns(csv_path)
        if column_mapper:
            cols = [column_mapper.get(c, c) for c in cols]
        if label_col is None:
            label_col = detect_label_column(cols, label_candidates)
            if label_col is None:
                raise ValueError(f"Label column not found in {csv_path.name}")
        if label_col not in cols:
            raise ValueError(f"Label column {label_col} missing in {csv_path.name}")

        feat_cols = [c for c in cols if c not in drop_columns and c != label_col]
        if feature_intersection is None:
            feature_intersection = set(feat_cols)
        else:
            feature_intersection &= set(feat_cols)

    if not feature_intersection:
        raise ValueError("Feature intersection is empty. Check columns and drop list.")

    return label_col, sorted(feature_intersection)


def iter_chunks(
    csv_path: Path,
    usecols: List[str],
    chunksize: int | None,
    sample_frac: float | None,
    max_rows: int | None,
    column_mapper: Optional[dict[str, str]] = None,
) -> Iterable[pd.DataFrame]:
    raw_cols = list(pd.read_csv(csv_path, nrows=0, skipinitialspace=True).columns)
    normalized_to_raw: dict[str, str] = {}
    for raw in raw_cols:
        key = str(raw).strip()
        if key not in normalized_to_raw:
            normalized_to_raw[key] = raw

    raw_usecols = usecols
    rename_map: dict[str, str] = {}
    if column_mapper:
        wanted = set(usecols)
        selected_raw: List[str] = []
        seen_canonical: set[str] = set()
        for raw in raw_cols:
            raw_clean = str(raw).strip()
            canonical = column_mapper.get(raw_clean, column_mapper.get(raw, raw_clean))
            if canonical in wanted and canonical not in seen_canonical:
                selected_raw.append(raw)
                seen_canonical.add(canonical)
                if canonical != raw:
                    rename_map[raw] = canonical
        missing = wanted - seen_canonical
        if missing:
            raise ValueError(f"Missing expected columns in {csv_path.name}: {sorted(missing)}")
        raw_usecols = selected_raw
    else:
        selected_raw: List[str] = []
        for wanted_col in usecols:
            if wanted_col not in normalized_to_raw:
                raise ValueError(f"Missing expected column '{wanted_col}' in {csv_path.name}")
            raw_col = normalized_to_raw[wanted_col]
            selected_raw.append(raw_col)
            if str(raw_col).strip() != wanted_col:
                rename_map[raw_col] = wanted_col
        raw_usecols = selected_raw

    if chunksize is None:
        df = pd.read_csv(csv_path, usecols=raw_usecols, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        if rename_map:
            df = df.rename(columns=rename_map)
        if sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=42)
        if max_rows is not None:
            df = df.head(max_rows)
        yield df
        return

    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=raw_usecols, chunksize=chunksize, skipinitialspace=True):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        if sample_frac is not None:
            chunk = chunk.sample(frac=sample_frac, random_state=42)
        if max_rows is not None:
            remaining = max_rows - seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        seen += len(chunk)
        if len(chunk) == 0:
            continue
        yield chunk


def prepare_chunk(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    fillna_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    for col in features:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = replace_inf_and_nan(df, fillna_value)

    x = df[features].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()
    return x, y


def fit_scaler_on_cic(
    csv_files: List[Path],
    features: List[str],
    label_col: str,
    benign_label: str,
    fillna_value: float,
    sample_frac: float | None,
    max_rows_per_file: int | None,
    chunksize: int | None,
    scaler,
    scaler_fit_mode: str = "auto",
    column_mapper: Optional[dict[str, str]] = None,
    raw_clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    mode = str(scaler_fit_mode).lower()
    supports_partial_fit = hasattr(scaler, "partial_fit")

    if mode == "auto":
        mode = "stream_partial" if supports_partial_fit else "full_benign"

    if mode == "stream_partial":
        if not supports_partial_fit:
            raise ValueError(
                f"Scaler {scaler.__class__.__name__} does not support partial_fit. "
                "Use preprocess.scaler_fit_mode=full_benign or auto."
            )
        total_benign = 0
        for csv_path in csv_files:
            for chunk in iter_chunks(
                csv_path,
                features + [label_col],
                chunksize,
                sample_frac,
                max_rows_per_file,
                column_mapper=column_mapper,
            ):
                x, y = prepare_chunk(chunk, features, label_col, fillna_value)
                y = pd.Series(y).astype(str).str.upper().to_numpy()
                benign_mask = y == benign_label.upper()
                if np.any(benign_mask):
                    benign_x = x[benign_mask]
                    if raw_clip_bounds is not None:
                        benign_x = clip_features_per_column(benign_x, raw_clip_bounds[0], raw_clip_bounds[1])
                    scaler.partial_fit(benign_x)
                    total_benign += benign_x.shape[0]
        if total_benign == 0:
            raise ValueError("No BENIGN rows found in CIC source data for scaler fitting.")
        logging.info("[PROGRESS] scaler fit mode=stream_partial | benign_rows=%d", total_benign)
        return

    if mode == "full_benign":
        benign_x = collect_benign_sample_from_cic(
            csv_files=csv_files,
            features=features,
            label_col=label_col,
            benign_label=benign_label,
            fillna_value=fillna_value,
            sample_frac=sample_frac,
            max_rows_per_file=max_rows_per_file,
            chunksize=chunksize,
            sample_rows_limit=None,
            column_mapper=column_mapper,
        )
        if benign_x.shape[0] == 0:
            raise ValueError("No BENIGN rows found in CIC source data for scaler fitting.")
        if raw_clip_bounds is not None:
            benign_x = clip_features_per_column(benign_x, raw_clip_bounds[0], raw_clip_bounds[1])
        scaler.fit(benign_x)
        logging.info("[PROGRESS] scaler fit mode=full_benign | benign_rows=%d", benign_x.shape[0])
        return

    raise ValueError(
        f"Unknown preprocess.scaler_fit_mode={scaler_fit_mode}. "
        "Allowed: auto, stream_partial, full_benign."
    )


def collect_benign_sample_from_cic(
    csv_files: List[Path],
    features: List[str],
    label_col: str,
    benign_label: str,
    fillna_value: float,
    sample_frac: float | None,
    max_rows_per_file: int | None,
    chunksize: int | None,
    sample_rows_limit: int | None,
    column_mapper: Optional[dict[str, str]] = None,
) -> np.ndarray:
    if sample_rows_limit is not None and sample_rows_limit <= 0:
        return np.empty((0, len(features)), dtype=np.float32)

    collected: List[np.ndarray] = []
    total = 0
    for csv_path in csv_files:
        for chunk in iter_chunks(
            csv_path,
            features + [label_col],
            chunksize,
            sample_frac,
            max_rows_per_file,
            column_mapper=column_mapper,
        ):
            x, y = prepare_chunk(chunk, features, label_col, fillna_value)
            y = pd.Series(y).astype(str).str.upper().to_numpy()
            benign_x = x[y == benign_label.upper()]
            if benign_x.shape[0] == 0:
                continue

            if sample_rows_limit is not None:
                remaining = sample_rows_limit - total
                if remaining <= 0:
                    break
                if benign_x.shape[0] > remaining:
                    benign_x = benign_x[:remaining]
            collected.append(benign_x)
            total += benign_x.shape[0]
        if sample_rows_limit is not None and total >= sample_rows_limit:
            break

    if not collected:
        return np.empty((0, len(features)), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def iter_windows_from_file(
    csv_path: Path,
    features: List[str],
    label_col: str,
    benign_label: str,
    fillna_value: float,
    sample_frac: float | None,
    max_rows_per_file: int | None,
    chunksize: int | None,
    scaler,
    window_size: int,
    stride: int,
    column_mapper: Optional[dict[str, str]] = None,
    raw_clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    post_scale_clip_abs: float | None = None,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    buffer_x = np.empty((0, len(features)), dtype=np.float32)
    buffer_y = np.empty((0,), dtype=np.int32)

    for chunk in iter_chunks(
        csv_path,
        features + [label_col],
        chunksize,
        sample_frac,
        max_rows_per_file,
        column_mapper=column_mapper,
    ):
        x, y_raw = prepare_chunk(chunk, features, label_col, fillna_value)
        y = (pd.Series(y_raw).astype(str).str.upper() != benign_label.upper()).astype(np.int32).to_numpy()
        if raw_clip_bounds is not None:
            x = clip_features_per_column(x, raw_clip_bounds[0], raw_clip_bounds[1])
        x = scaler.transform(x).astype(np.float32)
        if post_scale_clip_abs is not None:
            x = np.clip(x, -post_scale_clip_abs, post_scale_clip_abs).astype(np.float32)

        if buffer_x.shape[0] > 0:
            x = np.vstack([buffer_x, x])
            y = np.concatenate([buffer_y, y])

        xw, yw = window_sequences(x, y, window_size, stride)
        if xw.shape[0] > 0:
            yield xw, yw

        if window_size > 1:
            buffer_x = x[-(window_size - 1) :]
            buffer_y = y[-(window_size - 1) :]
        else:
            buffer_x = np.empty((0, len(features)), dtype=np.float32)
            buffer_y = np.empty((0,), dtype=np.int32)


def split_files_by_index(n_files: int, test_size: float, val_size: float) -> Tuple[int, int]:
    train_end = max(1, int(n_files * (1 - test_size - val_size)))
    val_end = max(train_end + 1, int(n_files * (1 - test_size)))
    return train_end, val_end


def split_files_by_index_with_calib(
    n_files: int,
    test_size: float,
    val_size: float,
    calib_size: float,
) -> Tuple[int, int, int]:
    if n_files <= 0:
        return 0, 0, 0

    train_end = max(1, int(n_files * (1 - test_size - val_size - calib_size)))
    train_end = min(train_end, n_files)

    val_end = max(train_end + 1, int(n_files * (1 - test_size - calib_size)))
    val_end = min(val_end, n_files)

    calib_end = max(val_end + 1, int(n_files * (1 - test_size)))
    calib_end = min(calib_end, n_files)
    return train_end, val_end, calib_end


def format_duration(seconds: float) -> str:
    sec = max(0, int(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ShardWriter:
    def __init__(
        self,
        root_dir: Path,
        split_dir: Path,
        prefix: str,
        shard_size: int,
        with_labels: bool,
        input_shape: Tuple[int, int],
    ) -> None:
        self.root_dir = root_dir
        self.split_dir = split_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.with_labels = with_labels
        self.input_shape = input_shape
        self.shards = []
        self.total_samples = 0
        self._buf_x = []
        self._buf_y = []
        self._idx = 0
        ensure_dir(self.split_dir)

    def add(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if x.size == 0:
            return
        self._buf_x.append(x)
        if self.with_labels:
            if y is None:
                raise ValueError("Labels required for this writer")
            self._buf_y.append(y)
        self.total_samples += x.shape[0]
        if sum(arr.shape[0] for arr in self._buf_x) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_x:
            return
        x = np.concatenate(self._buf_x, axis=0)
        y = np.concatenate(self._buf_y, axis=0) if self.with_labels else None
        self._buf_x = []
        self._buf_y = []

        for start in range(0, x.shape[0], self.shard_size):
            end = start + self.shard_size
            xb = x[start:end]
            if xb.shape[0] == 0:
                continue
            shard_path = self.split_dir / f"{self.prefix}_{self._idx:04d}.npz"
            if self.with_labels:
                yb = y[start:end]
                save_npz(shard_path, x=xb, y=yb)
            else:
                save_npz(shard_path, x=xb)
            rel_path = shard_path.relative_to(self.root_dir)
            self.shards.append({"path": str(rel_path).replace("\\", "/"), "samples": int(xb.shape[0])})
            self._idx += 1
            # Live progress feedback
            if self._idx % 5 == 0 or self._idx == 1:
                logging.info(
                    "[PROGRESS] Writing shards | %s | shards:%d samples:%d",
                    self.prefix,
                    self._idx,
                    self.total_samples,
                )

    def finalize(self) -> dict:
        logging.info("[PROGRESS] Finalizing shards | %s | flushing buffer...", self.prefix)
        self._flush()
        logging.info("[DONE] Finalized | %s | total_shards:%d total_samples:%d", self.prefix, len(self.shards), self.total_samples)
        return {
            "total_samples": int(self.total_samples),
            "num_shards": len(self.shards),
            "input_shape": list(self.input_shape),
            "shards": self.shards,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging()

    data_raw_cic = Path(cfg["paths"]["data_raw_cic"])
    data_raw_cse = Path(cfg["paths"]["data_raw_cse"])
    data_processed = Path(cfg["paths"]["data_processed"])

    drop_columns = cfg["preprocess"]["drop_columns"]
    label_candidates = cfg["preprocess"]["label_candidates"]
    benign_label = cfg["preprocess"]["benign_label"]
    fillna_value = float(cfg["preprocess"]["fillna_value"])
    window_size = int(cfg["preprocess"]["window_size"])
    stride = int(cfg["preprocess"]["stride"])
    test_size = float(cfg["preprocess"]["test_size"])
    val_size = float(cfg["preprocess"]["val_size"])
    calib_size = float(cfg["preprocess"].get("calib_size", 0.10))
    sample_frac = cfg["preprocess"]["sample_frac"]
    max_rows_per_file = cfg["preprocess"]["max_rows_per_file"]
    chunksize = cfg["preprocess"]["chunksize"]
    split_by_file = bool(cfg["preprocess"].get("split_by_file", True))
    shard_enable = bool(cfg["preprocess"].get("shard_enable", True))
    shard_dir = Path(cfg["preprocess"].get("shard_dir", data_processed / "shards"))
    shard_size = int(cfg["preprocess"].get("shard_size", 50000))
    write_combined = bool(cfg["preprocess"].get("write_combined", False))
    scaler_name = cfg["preprocess"]["scaler"]
    scaler_fit_mode = str(cfg["preprocess"].get("scaler_fit_mode", "auto"))
    feature_filter_cfg = cfg["preprocess"].get("feature_filter", {})
    filter_enable_nzv = bool(feature_filter_cfg.get("enable_nzv", False))
    filter_nzv_threshold = float(feature_filter_cfg.get("nzv_threshold", 1e-6))
    filter_enable_corr = bool(feature_filter_cfg.get("enable_corr", False))
    filter_corr_threshold = float(feature_filter_cfg.get("corr_threshold", 0.95))
    filter_sample_rows = int(feature_filter_cfg.get("sample_rows_limit", 100000))
    scale_guard_cfg = cfg["preprocess"].get("scale_guard", {})
    scale_guard_enable = bool(scale_guard_cfg.get("enable", True))
    raw_clip_quantile = float(scale_guard_cfg.get("raw_clip_quantile", 0.999))
    raw_clip_sample_rows = int(scale_guard_cfg.get("raw_clip_sample_rows", 200000))
    post_scale_clip_abs_raw = scale_guard_cfg.get("post_scale_clip_abs", 20.0)
    post_scale_clip_abs = None if post_scale_clip_abs_raw is None else float(post_scale_clip_abs_raw)
    if post_scale_clip_abs is not None and post_scale_clip_abs <= 0:
        raise ValueError("preprocess.scale_guard.post_scale_clip_abs must be positive or null")

    if not split_by_file:
        logging.warning("split_by_file is disabled, but streaming mode requires file split. Using file split.")
        split_by_file = True

    cic_files = list_csv_files(data_raw_cic)
    cse_files = list_csv_files(data_raw_cse)
    if not cic_files:
        raise FileNotFoundError(f"No CSV files found under {data_raw_cic}")
    if not cse_files:
        raise FileNotFoundError(f"No CSV files found under {data_raw_cse}")

    logging.info("Inspecting feature columns...")
    cic_label_col, cic_features = compute_feature_intersection(cic_files, label_candidates, drop_columns)

    cic_reference_cols = load_header_columns(cic_files[0])
    cse_mapper: dict[str, str] = {}
    for cse_path in cse_files:
        cse_cols = load_header_columns(cse_path)
        cse_mapper.update(build_column_mapper(cic_reference_cols, cse_cols))
    cse_label_col, cse_features = compute_feature_intersection(
        cse_files, label_candidates, drop_columns, column_mapper=cse_mapper
    )

    features = sorted(set(cic_features).intersection(cse_features))
    if not features:
        raise ValueError("No shared features between CIC-IDS2017 and CSE-CIC-IDS2018")

    if filter_enable_nzv or filter_enable_corr:
        logging.info("[STAGE] Applying statistical feature filter on CIC benign sample...")
        sample_x = collect_benign_sample_from_cic(
            cic_files,
            features,
            cic_label_col,
            benign_label,
            fillna_value,
            sample_frac,
            max_rows_per_file,
            chunksize,
            sample_rows_limit=filter_sample_rows,
        )
        if sample_x.shape[0] == 0:
            raise ValueError("Feature filter enabled, but no benign sample rows were collected.")
        features, filter_report = select_features_by_statistics(
            sample_array=sample_x,
            features=features,
            enable_nzv=filter_enable_nzv,
            nzv_threshold=filter_nzv_threshold,
            enable_corr=filter_enable_corr,
            corr_threshold=filter_corr_threshold,
        )
        save_json(data_processed / "feature_filter_report.json", filter_report)
        logging.info(
            "[PROGRESS] Feature filter | input=%d selected=%d dropped_nzv=%d dropped_corr=%d",
            filter_report["input_feature_count"],
            filter_report["selected_feature_count"],
            len(filter_report["dropped_nzv"]),
            len(filter_report["dropped_corr"]),
        )

    logging.info("Feature intersection count: %d", len(features))
    logging.info("[STAGE] %s", "Preprocess started")
    logging.info("[PROGRESS] CIC files: %d | CSE files: %d", len(cic_files), len(cse_files))

    scaler = get_scaler(scaler_name)
    raw_clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    scale_guard_report: Optional[dict] = None
    if scale_guard_enable:
        logging.info("[STAGE] Fitting scale guard bounds on CIC benign sample...")
        guard_sample = collect_benign_sample_from_cic(
            csv_files=cic_files,
            features=features,
            label_col=cic_label_col,
            benign_label=benign_label,
            fillna_value=fillna_value,
            sample_frac=sample_frac,
            max_rows_per_file=max_rows_per_file,
            chunksize=chunksize,
            sample_rows_limit=raw_clip_sample_rows,
            column_mapper=None,
        )
        if guard_sample.shape[0] == 0:
            raise ValueError("Scale guard enabled, but no benign sample rows were collected.")
        raw_lower, raw_upper = compute_feature_clip_bounds(guard_sample, raw_clip_quantile)
        raw_clip_bounds = (raw_lower, raw_upper)
        scale_guard_report = {
            "enabled": True,
            "raw_clip_quantile": raw_clip_quantile,
            "raw_clip_sample_rows": int(guard_sample.shape[0]),
            "post_scale_clip_abs": post_scale_clip_abs,
            "raw_lower": raw_lower.tolist(),
            "raw_upper": raw_upper.tolist(),
        }
        logging.info(
            "[PROGRESS] Scale guard fitted | quantile=%.4f sample_rows=%d post_scale_clip_abs=%s",
            raw_clip_quantile,
            guard_sample.shape[0],
            "None" if post_scale_clip_abs is None else f"{post_scale_clip_abs:.2f}",
        )
    else:
        post_scale_clip_abs = None
        logging.info("[STAGE] Scale guard disabled.")

    scaler_t0 = time.time()
    logging.info("[STAGE] %s", "Fitting scaler on CIC benign data")
    fit_scaler_on_cic(
        cic_files,
        features,
        cic_label_col,
        benign_label,
        fillna_value,
        sample_frac,
        max_rows_per_file,
        chunksize,
        scaler,
        scaler_fit_mode=scaler_fit_mode,
        raw_clip_bounds=raw_clip_bounds,
    )
    logging.info("[DONE] Scaler fit in %s", format_duration(time.time() - scaler_t0))

    data_processed.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, data_processed / "scaler.pkl")
    save_json(data_processed / "feature_columns.json", {"features": features, "label_col": cic_label_col})
    if scale_guard_report is not None:
        save_json(data_processed / "scale_guard_report.json", scale_guard_report)

    if shard_enable:
        shard_root = shard_dir
        ensure_dir(shard_root)

        train_dir = shard_root / "cic" / "train"
        val_dir = shard_root / "cic" / "val"
        calib_dir = shard_root / "cic" / "calib"
        test_dir = shard_root / "cic" / "test"
        cse_dir = shard_root / "cse" / "test"

        input_shape = (window_size, len(features))
        train_writer = ShardWriter(shard_root, train_dir, "cic_train", shard_size, False, input_shape)
        val_writer = ShardWriter(shard_root, val_dir, "cic_val", shard_size, False, input_shape)
        calib_writer = ShardWriter(shard_root, calib_dir, "cic_calib", shard_size, True, input_shape)
        test_writer = ShardWriter(shard_root, test_dir, "cic_test", shard_size, True, input_shape)
        cse_writer = ShardWriter(shard_root, cse_dir, "cse_test", shard_size, True, input_shape)

        train_end, val_end, calib_end = split_files_by_index_with_calib(
            len(cic_files), test_size, val_size, calib_size
        )
        split_files_root = data_processed / "splits"
        save_json(
            split_files_root / "cic_train_files.json",
            {"files": [str(p).replace("\\", "/") for p in cic_files[:train_end]]},
        )
        save_json(
            split_files_root / "cic_val_files.json",
            {"files": [str(p).replace("\\", "/") for p in cic_files[train_end:val_end]]},
        )
        save_json(
            split_files_root / "cic_calib_files.json",
            {"files": [str(p).replace("\\", "/") for p in cic_files[val_end:calib_end]]},
        )
        save_json(
            split_files_root / "cic_test_files.json",
            {"files": [str(p).replace("\\", "/") for p in cic_files[calib_end:]]},
        )

        logging.info("[STAGE] %s", "Windowing CIC-IDS2017 per file (streaming)")
        cic_t0 = time.time()
        n_cic = len(cic_files)
        for idx, csv_path in enumerate(cic_files):
            file_start = time.time()
            pct = ((idx + 1) / n_cic) * 100.0
            logging.info(
                "[PROGRESS] CIC file %d/%d (%.1f%%): %s",
                idx + 1,
                n_cic,
                pct,
                csv_path.name,
            )
            for xw, yw in iter_windows_from_file(
                csv_path,
                features,
                cic_label_col,
                benign_label,
                fillna_value,
                sample_frac,
                max_rows_per_file,
                chunksize,
                scaler,
                window_size,
                stride,
                raw_clip_bounds=raw_clip_bounds,
                post_scale_clip_abs=post_scale_clip_abs,
            ):
                if idx < train_end:
                    train_writer.add(xw[yw == 0])
                elif idx < val_end:
                    val_writer.add(xw[yw == 0])
                elif idx < calib_end:
                    calib_writer.add(xw, yw)
                else:
                    test_writer.add(xw, yw)
            elapsed = time.time() - cic_t0
            avg = elapsed / (idx + 1)
            eta = avg * (n_cic - (idx + 1))
            logging.info(
                "[PROGRESS] CIC accum windows -> train:%d val:%d calib:%d test:%d | file_time:%s | ETA:%s",
                train_writer.total_samples,
                val_writer.total_samples,
                calib_writer.total_samples,
                test_writer.total_samples,
                format_duration(time.time() - file_start),
                format_duration(eta),
            )

        if train_writer.total_samples == 0 or val_writer.total_samples == 0:
            raise ValueError("Train/val benign windows are empty. Check split order or benign label.")

        logging.info("[DONE] CIC windowing in %s", format_duration(time.time() - cic_t0))
        logging.info("[STAGE] %s", "Windowing CSE-CIC-IDS2018 per file (streaming)")
        cse_t0 = time.time()
        n_cse = len(cse_files)
        for idx, csv_path in enumerate(cse_files):
            file_start = time.time()
            pct = ((idx + 1) / n_cse) * 100.0
            logging.info(
                "[PROGRESS] CSE file %d/%d (%.1f%%): %s",
                idx + 1,
                n_cse,
                pct,
                csv_path.name,
            )
            for xw, yw in iter_windows_from_file(
                csv_path,
                features,
                cse_label_col,
                benign_label,
                fillna_value,
                sample_frac,
                max_rows_per_file,
                chunksize,
                scaler,
                window_size,
                stride,
                column_mapper=cse_mapper,
                raw_clip_bounds=raw_clip_bounds,
                post_scale_clip_abs=post_scale_clip_abs,
            ):
                cse_writer.add(xw, yw)
            elapsed = time.time() - cse_t0
            avg = elapsed / (idx + 1)
            eta = avg * (n_cse - (idx + 1))
            logging.info(
                "[PROGRESS] CSE accum windows -> test:%d | file_time:%s | ETA:%s",
                cse_writer.total_samples,
                format_duration(time.time() - file_start),
                format_duration(eta),
            )
        logging.info("[DONE] CSE windowing in %s", format_duration(time.time() - cse_t0))

        logging.info("[STAGE] Finalizing all shards (writing to disk)...")
        finalize_t0 = time.time()
        train_manifest = train_writer.finalize()
        val_manifest = val_writer.finalize()
        calib_manifest = calib_writer.finalize()
        test_manifest = test_writer.finalize()
        cse_manifest = cse_writer.finalize()
        logging.info("[DONE] All shards finalized in %s", format_duration(time.time() - finalize_t0))

        save_json(train_dir / "manifest.json", train_manifest)
        save_json(val_dir / "manifest.json", val_manifest)
        save_json(calib_dir / "manifest.json", calib_manifest)
        save_json(test_dir / "manifest.json", test_manifest)
        save_json(cse_dir / "manifest.json", cse_manifest)

        logging.info("[DONE] Sharded preprocessing complete.")
        logging.info("[PROGRESS] Train benign windows: %d", train_manifest["total_samples"])
        logging.info("[PROGRESS] Val benign windows: %d", val_manifest["total_samples"])
        logging.info("[PROGRESS] Calib windows: %d", calib_manifest["total_samples"])
        logging.info("[PROGRESS] Test windows: %d", test_manifest["total_samples"])
        logging.info("[PROGRESS] CSE test windows: %d", cse_manifest["total_samples"])

        if write_combined:
            logging.warning("write_combined is true but not implemented in streaming mode.")

    else:
        raise ValueError("Shard mode disabled is not supported in streaming preprocess. Enable shard_enable in config.")


if __name__ == "__main__":
    main()
