"""Evaluation metrics, confusion matrix, ROC/AUC, FPR, generalization gap."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import ensure_dir, load_json, load_yaml, save_json, setup_logging


def reconstruction_errors(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    preds = model.predict(x, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(x - preds), axis=(1, 2))
    return errors


def format_duration(seconds: float) -> str:
    sec = max(0, int(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ReservoirSampler:
    def __init__(self, max_size: int, seed: int = 42) -> None:
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)
        self.samples = []
        self.labels = []
        self.seen = 0

    def update(self, values: np.ndarray, labels: np.ndarray | None = None) -> None:
        if self.max_size is None or self.max_size <= 0:
            return
        if labels is None:
            labels = np.zeros_like(values, dtype=np.int32)

        for v, l in zip(values, labels):
            self.seen += 1
            if len(self.samples) < self.max_size:
                self.samples.append(float(v))
                self.labels.append(int(l))
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.max_size:
                    self.samples[j] = float(v)
                    self.labels[j] = int(l)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.samples, dtype=np.float32), np.asarray(self.labels, dtype=np.int32)


def compute_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_roc(y_true, scores, out_path: Path):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = metrics.roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, out_path: Path):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_dist(scores_benign, scores_attack, out_path: Path):
    plt.figure()
    if len(scores_benign) > 0:
        sns.histplot(scores_benign, label="Benign", stat="density", bins=50, color="green", alpha=0.5)
    if len(scores_attack) > 0:
        sns.histplot(scores_attack, label="Attack", stat="density", bins=50, color="red", alpha=0.5)
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_threshold_from_shards(model, shard_files, percentile, batch_size, sample_size):
    sampler = ReservoirSampler(sample_size)
    total = len(shard_files)
    t0 = time.time()
    for idx, shard_path in enumerate(shard_files, start=1):
        data = np.load(shard_path)
        x = data["x"]
        errs = reconstruction_errors(model, x, batch_size=batch_size)
        sampler.update(errs, None)
        pct = (idx / max(1, total)) * 100.0
        logging.info("[PROGRESS] threshold shards %d/%d (%.1f%%)", idx, total, pct)
    errs_sample, _ = sampler.get()
    if errs_sample.size == 0:
        raise ValueError("No validation data to compute threshold.")
    thr = float(np.percentile(errs_sample, percentile))
    logging.info(
        "[DONE] Threshold computed | percentile=p%s value=%.8f duration=%s",
        percentile,
        thr,
        format_duration(time.time() - t0),
    )
    return thr


def collect_error_sample_from_shards(
    model,
    shard_files,
    batch_size: int,
    sample_size: int,
    benign_only: bool,
) -> np.ndarray:
    sampler = ReservoirSampler(sample_size)
    for shard_path in shard_files:
        data = np.load(shard_path)
        x = data["x"]
        if benign_only:
            y = data["y"]
            x = x[y == 0]
            if x.shape[0] == 0:
                continue
        errs = reconstruction_errors(model, x, batch_size=batch_size)
        sampler.update(errs, None)
    return sampler.get()[0]


def sample_target_benign_windows(
    shard_files,
    frac: float,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    collected = []
    total = 0
    for shard_path in shard_files:
        data = np.load(shard_path)
        x = data["x"]
        y = data["y"]
        benign_x = x[y == 0]
        if benign_x.shape[0] == 0:
            continue
        keep = rng.random(benign_x.shape[0]) < frac
        sampled = benign_x[keep]
        if sampled.shape[0] == 0:
            continue
        if max_samples > 0 and total + sampled.shape[0] > max_samples:
            sampled = sampled[: max_samples - total]
        if sampled.shape[0] == 0:
            break
        collected.append(sampled)
        total += sampled.shape[0]
        if max_samples > 0 and total >= max_samples:
            break
    if not collected:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def compute_threshold_value(
    method: str,
    source_errors: np.ndarray,
    target_benign_errors: np.ndarray | None,
    percentile: float,
    k_sigma: float,
) -> float:
    method = method.lower()
    if source_errors.size == 0:
        raise ValueError("source_errors cannot be empty")

    if method == "percentile":
        return float(np.percentile(source_errors, percentile))

    if target_benign_errors is None or target_benign_errors.size == 0:
        raise ValueError(f"Threshold method '{method}' requires target benign errors.")

    if method == "target_percentile":
        return float(np.percentile(target_benign_errors, percentile))
    if method == "target_gaussian":
        mu = float(np.mean(target_benign_errors))
        sigma = float(np.std(target_benign_errors))
        return mu + (k_sigma * sigma)

    raise ValueError(f"Unknown threshold method: {method}")


def eval_shards(model, shard_files, threshold, batch_size, sample_size, label: str):
    tp = fp = tn = fn = 0
    sampler = ReservoirSampler(sample_size)
    sampler_benign = ReservoirSampler(sample_size)
    sampler_attack = ReservoirSampler(sample_size)

    t0 = time.time()
    total = len(shard_files)
    for idx, shard_path in enumerate(shard_files, start=1):
        data = np.load(shard_path)
        xb = data["x"]
        yb = data["y"]
        scores = reconstruction_errors(model, xb, batch_size=batch_size)
        preds = (scores > threshold).astype(np.int32)

        tp += int(np.sum((preds == 1) & (yb == 1)))
        tn += int(np.sum((preds == 0) & (yb == 0)))
        fp += int(np.sum((preds == 1) & (yb == 0)))
        fn += int(np.sum((preds == 0) & (yb == 1)))

        sampler.update(scores, yb)
        if np.any(yb == 0):
            sampler_benign.update(scores[yb == 0], yb[yb == 0])
        if np.any(yb == 1):
            sampler_attack.update(scores[yb == 1], yb[yb == 1])
        pct = (idx / max(1, total)) * 100.0
        logging.info("[PROGRESS] %s shards %d/%d (%.1f%%)", label, idx, total, pct)

    metrics_dict = compute_metrics_from_counts(tp, fp, tn, fn)

    sample_scores, sample_labels = sampler.get()
    if len(np.unique(sample_labels)) >= 2:
        metrics_dict["roc_auc"] = float(metrics.roc_auc_score(sample_labels, sample_scores))
    else:
        metrics_dict["roc_auc"] = None

    logging.info(
        "[DONE] %s eval complete | f1=%.4f fpr=%.4f roc_auc=%s duration=%s",
        label,
        metrics_dict["f1"],
        metrics_dict["fpr"],
        "None" if metrics_dict["roc_auc"] is None else f"{metrics_dict['roc_auc']:.4f}",
        format_duration(time.time() - t0),
    )
    return metrics_dict, sampler, sampler_benign, sampler_attack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="models/cnn_lstm_ae/best_model.keras")
    parser.add_argument("--tag", default="cnn_lstm")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging()
    t0 = time.time()

    data_dir = Path(cfg["paths"]["data_processed"])
    results_dir = Path(cfg["paths"]["results_dir"])
    plots_dir = results_dir / "plots" / args.tag
    metrics_dir = results_dir / "metrics"
    shard_root = Path(cfg["preprocess"].get("shard_dir", data_dir / "shards"))

    ensure_dir(plots_dir)
    ensure_dir(metrics_dir)

    model = tf.keras.models.load_model(args.model)

    eval_cfg = cfg.get("evaluation", {})
    eval_batch = int(eval_cfg.get("batch_size", 256))
    sample_size = int(eval_cfg.get("sample_size", 200000))
    eval_mode = str(eval_cfg.get("mode", "zero_shot")).lower()
    threshold_method = str(eval_cfg.get("threshold_method", "percentile")).lower()
    threshold_k_sigma = float(eval_cfg.get("threshold_k_sigma", 2.5))
    few_shot_frac = float(eval_cfg.get("few_shot_benign_frac", 0.01))
    few_shot_max_samples = int(eval_cfg.get("few_shot_max_samples", 50000))
    few_shot_finetune_epochs = int(eval_cfg.get("few_shot_finetune_epochs", 0))
    few_shot_finetune_lr = float(eval_cfg.get("few_shot_finetune_lr", cfg["training"]["learning_rate"]))
    few_shot_finetune_batch = int(eval_cfg.get("few_shot_finetune_batch_size", eval_batch))

    val_manifest = shard_root / "cic" / "val" / "manifest.json"
    cic_manifest = shard_root / "cic" / "test" / "manifest.json"
    cse_manifest = shard_root / "cse" / "test" / "manifest.json"

    if val_manifest.exists() and cic_manifest.exists() and cse_manifest.exists():
        val_info = load_json(val_manifest)
        cic_info = load_json(cic_manifest)
        cse_info = load_json(cse_manifest)

        logging.info(
            "[STAGE] Evaluate shard mode | val_shards=%d cic_shards=%d cse_shards=%d",
            len(val_info["shards"]),
            len(cic_info["shards"]),
            len(cse_info["shards"]),
        )
        val_shards = [shard_root / s["path"] for s in val_info["shards"]]
        cic_shards = [shard_root / s["path"] for s in cic_info["shards"]]
        cse_shards = [shard_root / s["path"] for s in cse_info["shards"]]

        source_errors = collect_error_sample_from_shards(
            model,
            val_shards,
            batch_size=eval_batch,
            sample_size=sample_size,
            benign_only=False,
        )
        target_benign_errors = None
        if eval_mode == "few_shot":
            adapt_x = sample_target_benign_windows(
                cse_shards,
                frac=few_shot_frac,
                max_samples=few_shot_max_samples,
                seed=int(cfg["preprocess"]["random_seed"]),
            )
            if adapt_x.size == 0:
                raise ValueError("few_shot mode enabled, but no benign adaptation windows were sampled.")
            logging.info("[PROGRESS] few_shot adaptation sample windows: %d", adapt_x.shape[0])

            if few_shot_finetune_epochs > 0:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=few_shot_finetune_lr),
                    loss="mse",
                )
                model.fit(
                    adapt_x,
                    adapt_x,
                    epochs=few_shot_finetune_epochs,
                    batch_size=few_shot_finetune_batch,
                    shuffle=True,
                    verbose=1,
                )

            target_benign_errors = reconstruction_errors(model, adapt_x, batch_size=eval_batch)

        threshold = compute_threshold_value(
            method=threshold_method,
            source_errors=source_errors,
            target_benign_errors=target_benign_errors,
            percentile=float(cfg["threshold"]["percentile"]),
            k_sigma=threshold_k_sigma,
        )
        logging.info(
            "[DONE] Threshold selected | mode=%s method=%s value=%.8f",
            eval_mode,
            threshold_method,
            threshold,
        )

        cic_metrics, cic_sampler, cic_benign, cic_attack = eval_shards(
            model, cic_shards, threshold, eval_batch, sample_size, label="CIC"
        )
        save_json(metrics_dir / f"{args.tag}_cic_metrics.json", cic_metrics)

        cse_metrics, cse_sampler, cse_benign, cse_attack = eval_shards(
            model, cse_shards, threshold, eval_batch, sample_size, label="CSE"
        )
        save_json(metrics_dir / f"{args.tag}_cse_metrics.json", cse_metrics)

        cic_scores, cic_labels = cic_sampler.get()
        cse_scores, cse_labels = cse_sampler.get()

        plot_roc(cic_labels, cic_scores, plots_dir / "roc_cic.png")
        plot_roc(cse_labels, cse_scores, plots_dir / "roc_cse.png")

        plot_confusion(cic_labels, (cic_scores > threshold).astype(int), plots_dir / "cm_cic.png")
        plot_confusion(cse_labels, (cse_scores > threshold).astype(int), plots_dir / "cm_cse.png")

        plot_error_dist(cic_benign.get()[0], cic_attack.get()[0], plots_dir / "err_dist_cic.png")
        plot_error_dist(cse_benign.get()[0], cse_attack.get()[0], plots_dir / "err_dist_cse.png")

    else:
        if eval_mode == "few_shot":
            raise ValueError("few_shot mode currently requires shard manifests.")
        val_npz = np.load(data_dir / "cic_val.npz")
        x_val = val_npz["x"].astype(np.float32)

        cic_test = np.load(data_dir / "cic_test.npz")
        x_cic = cic_test["x"].astype(np.float32)
        y_cic = cic_test["y"].astype(np.int32)

        cse_test = np.load(data_dir / "cse_test.npz")
        x_cse = cse_test["x"].astype(np.float32)
        y_cse = cse_test["y"].astype(np.int32)

        threshold = np.percentile(reconstruction_errors(model, x_val), cfg["threshold"]["percentile"])

        cic_scores = reconstruction_errors(model, x_cic)
        cse_scores = reconstruction_errors(model, x_cse)

        tn, fp, fn, tp = metrics.confusion_matrix(y_cic, (cic_scores > threshold).astype(int), labels=[0, 1]).ravel()
        cic_metrics = compute_metrics_from_counts(tp, fp, tn, fn)
        tn, fp, fn, tp = metrics.confusion_matrix(y_cse, (cse_scores > threshold).astype(int), labels=[0, 1]).ravel()
        cse_metrics = compute_metrics_from_counts(tp, fp, tn, fn)

        try:
            cic_metrics["roc_auc"] = float(metrics.roc_auc_score(y_cic, cic_scores))
        except Exception:
            cic_metrics["roc_auc"] = None
        try:
            cse_metrics["roc_auc"] = float(metrics.roc_auc_score(y_cse, cse_scores))
        except Exception:
            cse_metrics["roc_auc"] = None

        save_json(metrics_dir / f"{args.tag}_cic_metrics.json", cic_metrics)
        save_json(metrics_dir / f"{args.tag}_cse_metrics.json", cse_metrics)

        plot_roc(y_cic, cic_scores, plots_dir / "roc_cic.png")
        plot_roc(y_cse, cse_scores, plots_dir / "roc_cse.png")
        plot_confusion(y_cic, (cic_scores > threshold).astype(int), plots_dir / "cm_cic.png")
        plot_confusion(y_cse, (cse_scores > threshold).astype(int), plots_dir / "cm_cse.png")
        plot_error_dist(cic_scores[y_cic == 0], cic_scores[y_cic == 1], plots_dir / "err_dist_cic.png")
        plot_error_dist(cse_scores[y_cse == 0], cse_scores[y_cse == 1], plots_dir / "err_dist_cse.png")

    generalization_gap = {
        "f1_gap": cic_metrics["f1"] - cse_metrics["f1"],
        "auc_gap": (cic_metrics.get("roc_auc") or 0) - (cse_metrics.get("roc_auc") or 0),
        "accuracy_gap": cic_metrics["accuracy"] - cse_metrics["accuracy"],
        "mode": eval_mode,
        "threshold_method": threshold_method,
        "threshold": float(threshold),
    }
    save_json(metrics_dir / f"{args.tag}_generalization_gap.json", generalization_gap)
    logging.info(
        "[DONE] Evaluation finished | f1_gap=%.4f auc_gap=%.4f acc_gap=%.4f duration=%s",
        generalization_gap["f1_gap"],
        generalization_gap["auc_gap"],
        generalization_gap["accuracy_gap"],
        format_duration(time.time() - t0),
    )


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 1:
        try:
            tf.config.set_visible_devices(physical_devices[0], "GPU")
            physical_devices = [physical_devices[0]]
            print("[INFO] Multiple GPU adapters detected. Using only GPU:0 for stability.")
        except Exception as e:
            print(f"[WARN] Could not set single visible GPU: {e}")
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    main()
