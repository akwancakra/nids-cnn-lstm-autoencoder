"""Thresholding utilities (percentile)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import iter_npz_batches, load_json, load_yaml, setup_logging


def reconstruction_errors(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    preds = model.predict(x, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(x - preds), axis=(1, 2))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="models/cnn_lstm_ae/best_model.keras")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging()

    data_dir = Path(cfg["paths"]["data_processed"])
    shard_root = Path(cfg["preprocess"].get("shard_dir", data_dir / "shards"))
    val_manifest = shard_root / "cic" / "val" / "manifest.json"

    model = tf.keras.models.load_model(args.model)

    percentile = cfg["threshold"]["percentile"]
    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 256))

    if val_manifest.exists():
        val_info = load_json(val_manifest)
        val_shards = [shard_root / s["path"] for s in val_info["shards"]]
        errors = []
        for xb in iter_npz_batches(val_shards, batch_size, False, False):
            errors.append(reconstruction_errors(model, xb, batch_size=batch_size))
        if not errors:
            raise ValueError("No validation data to compute threshold.")
        all_err = np.concatenate(errors)
        thr = np.percentile(all_err, percentile)
    else:
        val_npz = np.load(data_dir / "cic_val.npz")
        x_val = val_npz["x"].astype(np.float32)
        thr = np.percentile(reconstruction_errors(model, x_val), percentile)

    print(f"Threshold (p{percentile}): {thr}")


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 1:
        try:
            tf.config.set_visible_devices(physical_devices[0], "GPU")
            print("[INFO] Multiple GPU adapters detected. Using only GPU:0 for stability.")
        except Exception as e:
            print(f"[WARN] Could not set single visible GPU: {e}")
    main()
