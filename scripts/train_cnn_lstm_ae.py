"""Train Hybrid CNN-LSTM Autoencoder (unsupervised)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
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


def _resolve_model_variant(cfg: dict) -> dict:
    variant = cfg.get("model_variant", {})
    lstm_backbone = str(variant.get("lstm_backbone", "lstm")).lower()
    if lstm_backbone not in {"lstm", "bilstm"}:
        raise ValueError("model_variant.lstm_backbone must be one of: lstm, bilstm")

    reconstruction_loss = str(variant.get("reconstruction_loss", "mse")).lower()
    if reconstruction_loss not in {"mse", "huber"}:
        raise ValueError("model_variant.reconstruction_loss must be one of: mse, huber")

    kernels = variant.get("multi_scale_kernels", [])
    if kernels is None:
        kernels = []
    if not isinstance(kernels, list):
        raise ValueError("model_variant.multi_scale_kernels must be a list of ints")
    kernels = [int(k) for k in kernels if int(k) > 0]

    dropout_schedule = variant.get("dropout_schedule", [])
    if dropout_schedule is None:
        dropout_schedule = []
    if not isinstance(dropout_schedule, list):
        raise ValueError("model_variant.dropout_schedule must be a list of floats")
    dropout_schedule = [float(v) for v in dropout_schedule]

    return {
        "lstm_backbone": lstm_backbone,
        "use_temporal_attention": bool(variant.get("use_temporal_attention", False)),
        "multi_scale_kernels": kernels,
        "reconstruction_loss": reconstruction_loss,
        "dropout_schedule": dropout_schedule,
    }


def _dropout_for_layer(base_dropout: float, schedule: list[float], idx: int, total: int) -> float:
    if total <= 0:
        return float(base_dropout)
    if not schedule:
        return float(base_dropout)
    if len(schedule) == 1:
        return float(schedule[0])
    start = float(schedule[0])
    end = float(schedule[-1])
    if total == 1:
        return end
    ratio = idx / max(1, total - 1)
    return float(start + ((end - start) * ratio))


def _build_conv_stack(x, cnn_filters: list[int], kernel_size: int):
    for f in cnn_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    return x


def _rnn_layer(units: int, return_sequences: bool, backbone: str):
    if backbone == "bilstm":
        return layers.Bidirectional(layers.LSTM(units, return_sequences=return_sequences))
    return layers.LSTM(units, return_sequences=return_sequences)


def build_model(input_shape, cfg):
    cnn_filters = [int(v) for v in cfg["training"]["cnn_filters"]]
    cnn_kernel = int(cfg["training"]["cnn_kernel_size"])
    lstm_units = [int(v) for v in cfg["training"]["lstm_units"]]
    if not lstm_units:
        raise ValueError("training.lstm_units must contain at least one layer size")
    dropout = float(cfg["training"]["dropout"])
    latent_dim = int(cfg["training"]["latent_dim"])
    variant = _resolve_model_variant(cfg)

    inputs = keras.Input(shape=input_shape)
    multi_scale_kernels = variant["multi_scale_kernels"]
    if multi_scale_kernels:
        conv_branches = [_build_conv_stack(inputs, cnn_filters, int(k)) for k in multi_scale_kernels]
        x = layers.Concatenate(name="multi_scale_concat")(conv_branches) if len(conv_branches) > 1 else conv_branches[0]
    else:
        x = _build_conv_stack(inputs, cnn_filters, cnn_kernel)

    total_recurrent_layers = max(1, len(lstm_units) * 2)
    rec_idx = 0
    for i, units in enumerate(lstm_units):
        is_last = i == (len(lstm_units) - 1)
        return_seq = (not is_last) or variant["use_temporal_attention"]
        x = _rnn_layer(units, return_sequences=return_seq, backbone=variant["lstm_backbone"])(x)
        layer_dropout = _dropout_for_layer(dropout, variant["dropout_schedule"], rec_idx, total_recurrent_layers)
        rec_idx += 1
        x = layers.Dropout(layer_dropout)(x)

    if variant["use_temporal_attention"]:
        attn_score = layers.Dense(1, activation="tanh", name="temporal_attention_score")(x)
        attn_weights = layers.Softmax(axis=1, name="temporal_attention_weights")(attn_score)
        weighted = layers.Multiply(name="temporal_attention_apply")([x, attn_weights])
        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="temporal_attention_context")(weighted)

    x = layers.Dense(latent_dim, activation="relu", name="latent_dense")(x)

    x = layers.RepeatVector(input_shape[0])(x)
    for units in reversed(lstm_units):
        x = _rnn_layer(units, return_sequences=True, backbone=variant["lstm_backbone"])(x)
        layer_dropout = _dropout_for_layer(dropout, variant["dropout_schedule"], rec_idx, total_recurrent_layers)
        rec_idx += 1
        x = layers.Dropout(layer_dropout)(x)

    outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)

    model = keras.Model(inputs, outputs, name="cnn_lstm_autoencoder")
    return model


def _resolve_augmentation_cfg(cfg: dict) -> dict:
    aug = cfg.get("augmentation", {})
    noise_std = float(aug.get("gaussian_noise_std", 0.0))
    mask_ratio = float(aug.get("feature_mask_ratio", 0.0))
    jitter_prob = float(aug.get("temporal_jitter_prob", 0.0))
    jitter_max_shift = int(aug.get("temporal_jitter_max_shift", 0))
    return {
        "gaussian_noise_std": max(0.0, noise_std),
        "feature_mask_ratio": min(max(mask_ratio, 0.0), 1.0),
        "temporal_jitter_prob": min(max(jitter_prob, 0.0), 1.0),
        "temporal_jitter_max_shift": max(0, jitter_max_shift),
    }


def _augment_batch_numpy(x_batch: np.ndarray, aug_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    x_aug = np.array(x_batch, copy=True)
    noise_std = float(aug_cfg["gaussian_noise_std"])
    mask_ratio = float(aug_cfg["feature_mask_ratio"])
    jitter_prob = float(aug_cfg["temporal_jitter_prob"])
    jitter_max_shift = int(aug_cfg["temporal_jitter_max_shift"])

    if noise_std > 0.0:
        x_aug += rng.normal(0.0, noise_std, size=x_aug.shape).astype(np.float32)

    if mask_ratio > 0.0:
        mask = rng.random(size=x_aug.shape) < mask_ratio
        x_aug[mask] = 0.0

    if jitter_prob > 0.0 and jitter_max_shift > 0:
        apply = rng.random(size=(x_aug.shape[0],)) < jitter_prob
        shifts = rng.integers(-jitter_max_shift, jitter_max_shift + 1, size=(x_aug.shape[0],))
        for idx in np.where(apply)[0]:
            shift = int(shifts[idx])
            if shift != 0:
                x_aug[idx] = np.roll(x_aug[idx], shift=shift, axis=0)

    return x_aug.astype(np.float32)


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


def find_latest_periodic_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.keras$")
    latest_path: Path | None = None
    latest_epoch = 0

    if not checkpoint_dir.exists():
        return None, 0

    for ckpt in checkpoint_dir.glob("checkpoint_epoch_*.keras"):
        match = pattern.fullmatch(ckpt.name)
        if not match:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = ckpt

    return latest_path, latest_epoch


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
    has_directml_plugin = importlib.util.find_spec("tensorflow_directml_plugin") is not None
    if force_cpu or not has_directml_plugin:
        tf.random.set_seed(seed)
        logging.info("[STAGE] tf.random.set_seed(%d) enabled.", seed)
    else:
        logging.warning(
            "Skipping tf.random.set_seed because tensorflow_directml_plugin is active "
            "(avoids DirectML stateless random kernel conflict)."
        )
    aug_cfg = _resolve_augmentation_cfg(cfg)
    augmentation_enabled = any(
        [
            aug_cfg["gaussian_noise_std"] > 0.0,
            aug_cfg["feature_mask_ratio"] > 0.0,
            aug_cfg["temporal_jitter_prob"] > 0.0 and aug_cfg["temporal_jitter_max_shift"] > 0,
        ]
    )
    logging.info("[STAGE] Augmentation enabled: %s | cfg=%s", augmentation_enabled, aug_cfg)

    data_dir = Path(cfg["paths"]["data_processed"])
    models_dir = Path(cfg["paths"]["models_dir"]) / "cnn_lstm_ae"
    results_dir = Path(cfg["paths"]["results_dir"]) / "logs"
    checkpoint_dir = models_dir / "checkpoints"

    ensure_dir(models_dir)
    ensure_dir(results_dir)
    ensure_dir(checkpoint_dir)

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
        if augmentation_enabled:
            def train_batch_generator():
                rng = np.random.default_rng(seed)
                for xb in iter_npz_batches(train_shards, batch_size, True, False):
                    yield _augment_batch_numpy(xb, aug_cfg, rng)
        else:
            def train_batch_generator():
                for xb in iter_npz_batches(train_shards, batch_size, True, False):
                    yield xb

        train_ds = (
            tf.data.Dataset.from_generator(
                train_batch_generator,
                output_signature=output_signature,
            )
            .map(lambda x: (x, x))
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

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
        if augmentation_enabled:
            rng = np.random.default_rng(seed)
            x_train = _augment_batch_numpy(x_train, aug_cfg, rng)

        input_shape = x_train.shape[1:]
        logging.info(
            "[PROGRESS] npz mode | x_train=%s x_val=%s batch=%d",
            x_train.shape,
            x_val.shape,
            int(cfg["training"]["batch_size"]),
        )

    # Try to resume from checkpoint (useful if Colab disconnects)
    checkpoint_path = models_dir / "best_model.keras"
    initial_epoch = 0
    history_path = results_dir / "cnn_lstm_history.json"
    resume_path = None

    latest_periodic_ckpt, latest_periodic_epoch = find_latest_periodic_checkpoint(checkpoint_dir)
    if latest_periodic_ckpt is not None:
        resume_path = latest_periodic_ckpt
        initial_epoch = latest_periodic_epoch
        logging.info(
            "[CHECKPOINT] Found periodic checkpoint: %s (epoch=%d)",
            latest_periodic_ckpt,
            latest_periodic_epoch,
        )
    elif checkpoint_path.exists():
        resume_path = checkpoint_path
        logging.info("[CHECKPOINT] Found existing checkpoint: %s", checkpoint_path)

    if resume_path is not None:
        try:
            model = keras.models.load_model(resume_path)
            logging.info("[CHECKPOINT] Loaded model from checkpoint, resuming training")

            # History is optional. If present, keep the larger epoch index.
            if history_path.exists():
                prev_history = load_json(history_path)
                if "loss" in prev_history:
                    hist_epoch = len(prev_history["loss"])
                    if hist_epoch > initial_epoch:
                        initial_epoch = hist_epoch
                    logging.info("[CHECKPOINT] Resuming from epoch %d", initial_epoch)
        except Exception as e:
            logging.warning("[CHECKPOINT] Could not load checkpoint, building new model: %s", e)
            model = build_model(input_shape, cfg)
            initial_epoch = 0
            logging.warning("[CHECKPOINT] Resume disabled due to invalid checkpoint. Training restarts from epoch 0.")
    else:
        logging.info("[STAGE] Building new model (no checkpoint found)")
        model = build_model(input_shape, cfg)
    
    logging.info("[STAGE] Train CNN-LSTM AE | input_shape=%s", input_shape)

    lr = float(cfg["training"]["learning_rate"])
    clipnorm = cfg.get("training", {}).get("clipnorm")
    optimizer_kwargs = {"learning_rate": lr}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = float(clipnorm)
    variant = _resolve_model_variant(cfg)
    if variant["reconstruction_loss"] == "huber":
        reconstruction_loss = keras.losses.Huber()
    else:
        reconstruction_loss = "mse"
    model.compile(optimizer=keras.optimizers.Adam(**optimizer_kwargs), loss=reconstruction_loss)

    # Custom callback for periodic checkpoint (every N epochs)
    class PeriodicCheckpoint(keras.callbacks.Callback):
        def __init__(self, checkpoint_dir, period=5):
            super().__init__()
            self.checkpoint_dir = Path(checkpoint_dir)
            self.period = period
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.period == 0:
                filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.keras"
                self.model.save(filepath)
                print(f"\nSaved periodic checkpoint: {filepath.name}")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["training"]["early_stopping_patience"]),
            restore_best_weights=bool(cfg["training"].get("restore_best_weights", True)),
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        PeriodicCheckpoint(checkpoint_dir, period=5),
    ]

    scheduler_mode = str(cfg["training"].get("lr_scheduler", "reduce_on_plateau")).lower()
    if scheduler_mode == "reduce_on_plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(cfg["training"]["reduce_lr_factor"]),
                patience=int(cfg["training"]["reduce_lr_patience"]),
                min_lr=float(cfg["training"]["min_lr"]),
            )
        )
    elif scheduler_mode == "cosine":
        min_lr = float(cfg["training"]["min_lr"])
        max_lr = lr
        warmup_epochs = int(cfg["training"].get("warmup_epochs", 0))
        total_epochs = int(cfg["training"]["epochs"])

        def cosine_with_optional_warmup(epoch: int, _current_lr: float) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                # Linear warmup from min_lr to max_lr.
                return min_lr + ((max_lr - min_lr) * ((epoch + 1) / warmup_epochs))
            denom = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / denom
            progress = min(max(progress, 0.0), 1.0)
            return min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress)))

        callbacks.append(keras.callbacks.LearningRateScheduler(cosine_with_optional_warmup, verbose=0))
    elif scheduler_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown training.lr_scheduler: {scheduler_mode}")

    target_epochs = int(cfg["training"]["epochs"])
    if initial_epoch >= target_epochs:
        logging.info(
            "[DONE] Training already reached target epochs (initial_epoch=%d, target=%d). Nothing to run.",
            initial_epoch,
            target_epochs,
        )
        return

    t0 = time.time()
    if use_shards:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=target_epochs,
            initial_epoch=initial_epoch,
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
            epochs=target_epochs,
            initial_epoch=initial_epoch,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

    model.save(models_dir / "final_model.keras")

    save_json(results_dir / "cnn_lstm_history.json", to_serializable_history(history.history))
    save_json(results_dir / "config_snapshot.json", cfg)
    best_val = min(history.history.get("val_loss", [float("nan")]))
    completed_epochs = initial_epoch + len(history.history.get("loss", []))
    logging.info(
        "[DONE] Train CNN-LSTM AE finished | epochs=%d best_val_loss=%.6f duration=%s",
        completed_epochs,
        best_val,
        format_duration(time.time() - t0),
    )


if __name__ == "__main__":
    main()
