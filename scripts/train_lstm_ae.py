"""Train LSTM Autoencoder baseline (unsupervised)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import ensure_dir, iter_npz_batches, load_json, load_yaml, save_json, setup_logging


def build_model(input_shape, cfg):
    lstm_units = cfg["training"]["lstm_units"]
    dropout = cfg["training"]["dropout"]
    latent_dim = cfg["training"]["latent_dim"]

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(units, return_sequences=(i < len(lstm_units) - 1))(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(latent_dim, activation="relu")(x)

    x = layers.RepeatVector(input_shape[0])(x)
    for units in reversed(lstm_units):
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    return model


def format_duration(seconds: float) -> str:
    sec = max(0, int(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def to_serializable_history(history_dict: dict) -> dict:
    out = {}
    for k, vals in history_dict.items():
        if isinstance(vals, list):
            out[k] = [float(v) for v in vals]
        else:
            out[k] = float(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging()
    force_cpu = bool(cfg.get("training", {}).get("force_cpu", False))
    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("[STAGE] force_cpu=true -> training will run on CPU.")
        except Exception as e:
            logging.warning("Could not force CPU mode cleanly: %s", e)
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            try:
                tf.config.set_visible_devices(gpus[0], "GPU")
                logging.info("[STAGE] Multiple GPU adapters detected -> using GPU:0 only for stability.")
            except Exception as e:
                logging.warning("Could not limit visible GPUs: %s", e)

    seed = int(cfg["preprocess"]["random_seed"])
    np.random.seed(seed)
    # DirectML on Windows can crash on stateless random ops during layer init.
    # Keep NumPy seed for partial reproducibility; skip tf seed in GPU mode.
    if force_cpu:
        tf.random.set_seed(seed)
    else:
        logging.warning(
            "Skipping tf.random.set_seed in GPU mode to avoid DirectML stateless random kernel conflict."
        )

    data_dir = Path(cfg["paths"]["data_processed"])
    models_dir = Path(cfg["paths"]["models_dir"]) / "lstm_ae"
    results_dir = Path(cfg["paths"]["results_dir"]) / "logs"

    ensure_dir(models_dir)
    ensure_dir(results_dir)

    shard_root = Path(cfg["preprocess"].get("shard_dir", data_dir / "shards"))
    train_manifest_path = shard_root / "cic" / "train" / "manifest.json"
    val_manifest_path = shard_root / "cic" / "val" / "manifest.json"

    use_shards = train_manifest_path.exists() and val_manifest_path.exists()
    if use_shards:
        train_manifest = load_json(train_manifest_path)
        val_manifest = load_json(val_manifest_path)

        input_shape = tuple(train_manifest["input_shape"])
        batch_size = int(cfg["training"]["batch_size"])
        steps_per_epoch = math.ceil(train_manifest["total_samples"] / batch_size)
        val_steps = math.ceil(val_manifest["total_samples"] / batch_size)

        train_shards = [shard_root / s["path"] for s in train_manifest["shards"]]
        val_shards = [shard_root / s["path"] for s in val_manifest["shards"]]

        output_signature = tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32)
        train_ds = tf.data.Dataset.from_generator(
            lambda: iter_npz_batches(train_shards, batch_size, True, False),
            output_signature=output_signature,
        ).map(lambda x: (x, x)).repeat().prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: iter_npz_batches(val_shards, batch_size, False, False),
            output_signature=output_signature,
        ).map(lambda x: (x, x)).repeat().prefetch(tf.data.AUTOTUNE)
        logging.info(
            "[PROGRESS] shard mode | train_samples=%d val_samples=%d batch=%d steps_per_epoch=%d val_steps=%d",
            train_manifest["total_samples"],
            val_manifest["total_samples"],
            batch_size,
            steps_per_epoch,
            val_steps,
        )
    else:
        train_npz = np.load(data_dir / "cic_train.npz")
        val_npz = np.load(data_dir / "cic_val.npz")

        x_train = train_npz["x"].astype(np.float32)
        x_val = val_npz["x"].astype(np.float32)

        input_shape = x_train.shape[1:]
        logging.info(
            "[PROGRESS] npz mode | x_train=%s x_val=%s batch=%d",
            x_train.shape,
            x_val.shape,
            int(cfg["training"]["batch_size"]),
        )

    model = build_model(input_shape, cfg)
    logging.info("[STAGE] Train LSTM AE | input_shape=%s", input_shape)

    lr = float(cfg["training"]["learning_rate"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(cfg["training"]["reduce_lr_factor"]),
            patience=int(cfg["training"]["reduce_lr_patience"]),
            min_lr=float(cfg["training"]["min_lr"]),
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    t0 = time.time()
    if use_shards:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(cfg["training"]["epochs"]),
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=int(cfg["training"]["epochs"]),
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

    model.save(models_dir / "final_model.keras")

    save_json(results_dir / "lstm_history.json", to_serializable_history(history.history))
    save_json(results_dir / "config_snapshot.json", cfg)
    best_val = min(history.history.get("val_loss", [float("nan")]))
    last_epoch = len(history.history.get("loss", []))
    logging.info(
        "[DONE] Train LSTM AE finished | epochs=%d best_val_loss=%.6f duration=%s",
        last_epoch,
        best_val,
        format_duration(time.time() - t0),
    )


if __name__ == "__main__":
    main()
