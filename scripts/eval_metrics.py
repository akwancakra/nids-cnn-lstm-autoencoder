"""Evaluation metrics, thresholding, and plots for CIC/CSE shard pipelines."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - allows unit tests without tensorflow
    tf = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import ensure_dir, load_json, load_yaml, save_json, setup_logging


def format_duration(seconds: float) -> str:
    sec = max(0, int(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def compute_reconstruction_scores(
    x: np.ndarray,
    preds: np.ndarray,
    score_mode: str = "recon_mse",
) -> np.ndarray:
    mode = str(score_mode).lower()
    diff = x - preds

    if mode in {"recon_mse", "hybrid_recon_latent"}:
        return np.mean(np.square(diff), axis=(1, 2)).astype(np.float32)

    if mode == "recon_huber":
        delta = 1.0
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        huber = 0.5 * np.square(quadratic) + (delta * linear)
        return np.mean(huber, axis=(1, 2)).astype(np.float32)

    raise ValueError(f"Unknown score_mode: {score_mode}")


def combine_hybrid_scores(recon_scores: np.ndarray, latent_scores: np.ndarray, alpha: float) -> np.ndarray:
    if recon_scores.shape != latent_scores.shape:
        raise ValueError("recon_scores and latent_scores must have the same shape")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    return (alpha * recon_scores + (1.0 - alpha) * latent_scores).astype(np.float32)


def flatten_latent(emb: np.ndarray) -> np.ndarray:
    if emb.ndim == 1:
        return emb[:, None]
    if emb.ndim == 2:
        return emb
    return emb.reshape(emb.shape[0], -1)


def build_latent_model(model):
    if tf is None:
        raise ModuleNotFoundError("tensorflow is required for hybrid_recon_latent scoring")
    try:
        latent_layer = model.get_layer("latent")
    except Exception as exc:
        raise ValueError("Model does not have a layer named 'latent' for hybrid scoring.") from exc
    return tf.keras.Model(inputs=model.input, outputs=latent_layer.output)


def compute_scores_batch(
    model,
    xb: np.ndarray,
    batch_size: int,
    score_mode: str,
    latent_model=None,
    latent_mean: Optional[np.ndarray] = None,
    latent_std: Optional[np.ndarray] = None,
    hybrid_alpha: float = 0.7,
) -> np.ndarray:
    preds = model.predict(xb, batch_size=batch_size, verbose=0)
    recon_scores = compute_reconstruction_scores(xb, preds, score_mode=score_mode)

    if str(score_mode).lower() != "hybrid_recon_latent":
        return recon_scores

    if latent_model is None or latent_mean is None or latent_std is None:
        raise ValueError("hybrid_recon_latent requires latent_model and latent reference statistics.")

    latent = latent_model.predict(xb, batch_size=batch_size, verbose=0)
    latent = flatten_latent(np.asarray(latent, dtype=np.float32))
    z = (latent - latent_mean[None, :]) / latent_std[None, :]
    latent_scores = np.mean(np.square(z), axis=1).astype(np.float32)
    return combine_hybrid_scores(recon_scores, latent_scores, alpha=hybrid_alpha)


def collect_latent_reference_from_shards(latent_model, shard_files, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_vec = None
    sum_sq = None

    total = len(shard_files)
    last_log_pct = 0.0
    for idx, shard_path in enumerate(shard_files, start=1):
        data = np.load(shard_path)
        x = data["x"]
        y = data["y"] if "y" in data else None
        if y is not None:
            x = x[y == 0]
        if x.shape[0] == 0:
            continue

        emb = latent_model.predict(x, batch_size=batch_size, verbose=0)
        emb = flatten_latent(np.asarray(emb, dtype=np.float32))
        if emb.shape[0] == 0:
            continue

        if sum_vec is None:
            sum_vec = np.zeros(emb.shape[1], dtype=np.float64)
            sum_sq = np.zeros(emb.shape[1], dtype=np.float64)
        sum_vec += np.sum(emb, axis=0)
        sum_sq += np.sum(np.square(emb), axis=0)
        count += emb.shape[0]

        pct = (idx / max(1, total)) * 100.0
        if idx == 1 or idx == total or (pct - last_log_pct >= 5.0):
            logging.info("[PROGRESS] Computing latent reference | shards:%d/%d (%.1f%%) samples:%d", idx, total, pct, count)
            last_log_pct = pct

    if count == 0 or sum_vec is None or sum_sq is None:
        raise ValueError("No benign windows available to compute latent reference statistics.")

    mean = (sum_vec / count).astype(np.float32)
    var = np.maximum(sum_sq / count - np.square(mean), 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


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


def choose_threshold_from_labeled_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    guardrail_fpr_max: float | None = None,
) -> float:
    if scores.size == 0 or labels.size == 0:
        raise ValueError("scores and labels cannot be empty.")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have same length.")
    if len(np.unique(labels)) < 2:
        raise ValueError("source_calib threshold methods require both benign and attack labels.")

    unique_scores = np.unique(scores.astype(np.float64))
    candidates = []
    if unique_scores.size == 1:
        s = float(unique_scores[0])
        candidates = [s - 1e-9, s, s + 1e-9]
    else:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates = [float(unique_scores[0] - 1e-9)] + [float(v) for v in mids] + [float(unique_scores[-1] + 1e-9)]

    best_thr = candidates[0]
    best_key = (-1.0, -1.0, -1.0)
    for thr in candidates:
        preds = (scores > thr).astype(np.int32)
        tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        met = compute_metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)

        if guardrail_fpr_max is not None and met["fpr"] > guardrail_fpr_max:
            continue

        # Maximize F1, then recall, then precision.
        key = (met["f1"], met["recall"], met["precision"])
        if key > best_key:
            best_key = key
            best_thr = thr

    if best_key[0] < 0:
        # No candidate passed guardrail, fallback to strictest threshold.
        return float(unique_scores[-1] + 1e-9)
    return float(best_thr)


def compute_threshold_value(
    method: str,
    source_errors: np.ndarray,
    source_labels: np.ndarray | None,
    target_benign_errors: np.ndarray | None,
    percentile: float,
    k_sigma: float,
    guardrail_fpr_max: float | None = None,
) -> float:
    aliases = {
        "percentile": "source_percentile",
        "gaussian": "source_gaussian",
    }
    method = aliases.get(str(method).lower(), str(method).lower())
    if source_errors.size == 0:
        raise ValueError("source_errors cannot be empty")

    if method == "source_percentile":
        return float(np.percentile(source_errors, percentile))
    if method == "source_gaussian":
        return float(np.mean(source_errors) + (k_sigma * np.std(source_errors)))
    if method == "source_calib_f1":
        if source_labels is None:
            raise ValueError("source_calib_f1 requires source_labels.")
        return choose_threshold_from_labeled_scores(source_errors, source_labels, guardrail_fpr_max=None)
    if method == "source_calib_guardrail":
        if source_labels is None:
            raise ValueError("source_calib_guardrail requires source_labels.")
        return choose_threshold_from_labeled_scores(
            source_errors,
            source_labels,
            guardrail_fpr_max=float(guardrail_fpr_max) if guardrail_fpr_max is not None else 0.20,
        )

    # Backward compatibility with old target-based modes.
    if target_benign_errors is None or target_benign_errors.size == 0:
        raise ValueError(f"Threshold method '{method}' requires target benign errors.")
    if method == "target_percentile":
        return float(np.percentile(target_benign_errors, percentile))
    if method == "target_gaussian":
        mu = float(np.mean(target_benign_errors))
        sigma = float(np.std(target_benign_errors))
        return mu + (k_sigma * sigma)

    raise ValueError(f"Unknown threshold method: {method}")


def collect_scores_from_shards(
    model,
    shard_files,
    batch_size: int,
    score_mode: str,
    latent_model=None,
    latent_mean: Optional[np.ndarray] = None,
    latent_std: Optional[np.ndarray] = None,
    hybrid_alpha: float = 0.7,
) -> tuple[np.ndarray, np.ndarray | None]:
    all_scores = []
    all_labels = []
    has_labels = None

    total = len(shard_files)
    last_log_pct = 0.0
    for idx, shard_path in enumerate(shard_files, start=1):
        data = np.load(shard_path)
        x = data["x"]
        y = data["y"] if "y" in data else None
        has_labels = y is not None if has_labels is None else has_labels
        scores = compute_scores_batch(
            model=model,
            xb=x,
            batch_size=batch_size,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )
        all_scores.append(scores)
        if y is not None:
            all_labels.append(y.astype(np.int32))

        pct = (idx / max(1, total)) * 100.0
        if idx == 1 or idx == total or (pct - last_log_pct >= 5.0):
            logging.info("[PROGRESS] Collecting scores | shards:%d/%d (%.1f%%)", idx, total, pct)
            last_log_pct = pct

    if not all_scores:
        raise ValueError("No scores collected from shard files.")
    score_arr = np.concatenate(all_scores, axis=0).astype(np.float32)
    if has_labels and all_labels:
        return score_arr, np.concatenate(all_labels, axis=0).astype(np.int32)
    return score_arr, None


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
    plt.title("Score Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def eval_shards(
    model,
    shard_files,
    threshold: float,
    batch_size: int,
    sample_size: int,
    label: str,
    score_mode: str,
    latent_model=None,
    latent_mean: Optional[np.ndarray] = None,
    latent_std: Optional[np.ndarray] = None,
    hybrid_alpha: float = 0.7,
):
    tp = fp = tn = fn = 0
    sampler = ReservoirSampler(sample_size)
    sampler_benign = ReservoirSampler(sample_size)
    sampler_attack = ReservoirSampler(sample_size)

    t0 = time.time()
    total = len(shard_files)
    last_log_pct = 0.0
    for idx, shard_path in enumerate(shard_files, start=1):
        data = np.load(shard_path)
        xb = data["x"]
        yb = data["y"].astype(np.int32)
        scores = compute_scores_batch(
            model=model,
            xb=xb,
            batch_size=batch_size,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )
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
        # Log every ~2% or at milestones to avoid spam
        if idx == 1 or idx == total or (pct - last_log_pct >= 2.0):
            logging.info("[PROGRESS] %s shards %d/%d (%.1f%%)", label, idx, total, pct)
            last_log_pct = pct

    metrics_dict = compute_metrics_from_counts(tp, fp, tn, fn)

    sample_scores, sample_labels = sampler.get()
    if len(np.unique(sample_labels)) >= 2:
        metrics_dict["roc_auc"] = float(metrics.roc_auc_score(sample_labels, sample_scores))
        metrics_dict["pr_auc"] = float(metrics.average_precision_score(sample_labels, sample_scores))
    else:
        metrics_dict["roc_auc"] = None
        metrics_dict["pr_auc"] = None

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
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold override")
    args = parser.parse_args()

    if tf is None:
        raise ModuleNotFoundError("tensorflow is required to run evaluation.")

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

    model = tf.keras.models.load_model(args.model, compile=False)

    eval_cfg = cfg.get("evaluation", {})
    eval_batch = int(eval_cfg.get("batch_size", 256))
    sample_size = int(eval_cfg.get("sample_size", 0))
    eval_mode = str(eval_cfg.get("mode", "zero_shot")).lower()
    threshold_method = str(eval_cfg.get("threshold_method", "source_percentile")).lower()
    threshold_k_sigma = float(eval_cfg.get("threshold_k_sigma", 2.5))
    source_calib_split = str(eval_cfg.get("source_calib_split", "calib")).lower()
    guardrail_fpr_max = float(eval_cfg.get("guardrail_fpr_max", 0.20))
    score_mode = str(eval_cfg.get("score_mode", "recon_mse")).lower()
    hybrid_alpha = float(eval_cfg.get("hybrid_alpha", 0.7))

    if eval_mode != "zero_shot":
        raise ValueError("Sprint 3 policy enforces evaluation.mode=zero_shot.")

    source_manifest = shard_root / "cic" / source_calib_split / "manifest.json"
    cic_manifest = shard_root / "cic" / "test" / "manifest.json"
    cse_manifest = shard_root / "cse" / "test" / "manifest.json"

    if source_manifest.exists() and cic_manifest.exists() and cse_manifest.exists():
        source_info = load_json(source_manifest)
        cic_info = load_json(cic_manifest)
        cse_info = load_json(cse_manifest)

        source_shards = [shard_root / s["path"] for s in source_info["shards"]]
        cic_shards = [shard_root / s["path"] for s in cic_info["shards"]]
        cse_shards = [shard_root / s["path"] for s in cse_info["shards"]]

        logging.info(
            "[STAGE] Evaluate shard mode | source_split=%s source_shards=%d cic_shards=%d cse_shards=%d",
            source_calib_split,
            len(source_shards),
            len(cic_shards),
            len(cse_shards),
        )

        latent_model = None
        latent_mean = None
        latent_std = None
        if score_mode == "hybrid_recon_latent":
            latent_model = build_latent_model(model)
            latent_mean, latent_std = collect_latent_reference_from_shards(
                latent_model=latent_model,
                shard_files=source_shards,
                batch_size=eval_batch,
            )
            logging.info("[DONE] Latent reference fitted | dims=%d", latent_mean.shape[0])

        source_scores, source_labels = collect_scores_from_shards(
            model=model,
            shard_files=source_shards,
            batch_size=eval_batch,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )

        target_benign_errors = None
        if threshold_method in ("target_percentile", "target_gaussian"):
            cse_scores_pre, cse_labels_pre = collect_scores_from_shards(
                model=model,
                shard_files=cse_shards,
                batch_size=eval_batch,
                score_mode=score_mode,
                latent_model=latent_model,
                latent_mean=latent_mean,
                latent_std=latent_std,
                hybrid_alpha=hybrid_alpha,
            )
            if cse_labels_pre is None:
                raise ValueError("target_percentile/target_gaussian require CSE shards with labels.")
            benign_mask = cse_labels_pre == 0
            target_benign_errors = cse_scores_pre[benign_mask].astype(np.float64)
            sample_frac = float(eval_cfg.get("target_benign_sample_frac", 0.0))
            if 0 < sample_frac < 1.0 and target_benign_errors.size > 0:
                orig_n = target_benign_errors.size
                rng = np.random.default_rng(42)
                n = max(1, int(orig_n * sample_frac))
                idx = rng.choice(orig_n, size=min(n, orig_n), replace=False)
                target_benign_errors = target_benign_errors[idx]
                logging.info("[STAGE] target_benign_sample_frac=%.2f -> subsampled %d of %d CSE benign", sample_frac, target_benign_errors.size, orig_n)

        if args.threshold is not None:
            threshold = float(args.threshold)
            logging.info("[OVERRIDE] Using manual threshold from CLI: %.8f", threshold)
        else:
            threshold = compute_threshold_value(
                method=threshold_method,
                source_errors=source_scores,
                source_labels=source_labels,
                target_benign_errors=target_benign_errors,
                percentile=float(cfg["threshold"]["percentile"]),
                k_sigma=threshold_k_sigma,
                guardrail_fpr_max=guardrail_fpr_max,
            )

        logging.info(
            "[DONE] Threshold selected | method=%s value=%.8f source_split=%s",
            threshold_method,
            threshold,
            source_calib_split,
        )

        cic_metrics, cic_sampler, cic_benign, cic_attack = eval_shards(
            model=model,
            shard_files=cic_shards,
            threshold=threshold,
            batch_size=eval_batch,
            sample_size=sample_size,
            label="CIC",
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )
        save_json(metrics_dir / f"{args.tag}_cic_metrics.json", cic_metrics)

        cse_metrics, cse_sampler, cse_benign, cse_attack = eval_shards(
            model=model,
            shard_files=cse_shards,
            threshold=threshold,
            batch_size=eval_batch,
            sample_size=sample_size,
            label="CSE",
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
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
        val_npz = np.load(data_dir / "cic_val.npz")
        x_val = val_npz["x"].astype(np.float32)

        cic_test = np.load(data_dir / "cic_test.npz")
        x_cic = cic_test["x"].astype(np.float32)
        y_cic = cic_test["y"].astype(np.int32)

        cse_test = np.load(data_dir / "cse_test.npz")
        x_cse = cse_test["x"].astype(np.float32)
        y_cse = cse_test["y"].astype(np.int32)

        latent_model = None
        latent_mean = None
        latent_std = None
        if score_mode == "hybrid_recon_latent":
            latent_model = build_latent_model(model)
            lat = latent_model.predict(x_val, batch_size=eval_batch, verbose=0)
            lat = flatten_latent(np.asarray(lat, dtype=np.float32))
            latent_mean = np.mean(lat, axis=0).astype(np.float32)
            latent_std = np.maximum(np.std(lat, axis=0).astype(np.float32), 1e-8)

        val_scores = compute_scores_batch(
            model=model,
            xb=x_val,
            batch_size=eval_batch,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )
        if args.threshold is not None:
            threshold = float(args.threshold)
            logging.info("[OVERRIDE] Using manual threshold from CLI: %.8f", threshold)
        else:
            threshold = compute_threshold_value(
                method=threshold_method,
                source_errors=val_scores,
                source_labels=None,
                target_benign_errors=None,
                percentile=float(cfg["threshold"]["percentile"]),
                k_sigma=threshold_k_sigma,
                guardrail_fpr_max=guardrail_fpr_max,
            )

        cic_scores = compute_scores_batch(
            model=model,
            xb=x_cic,
            batch_size=eval_batch,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )
        cse_scores = compute_scores_batch(
            model=model,
            xb=x_cse,
            batch_size=eval_batch,
            score_mode=score_mode,
            latent_model=latent_model,
            latent_mean=latent_mean,
            latent_std=latent_std,
            hybrid_alpha=hybrid_alpha,
        )

        tn, fp, fn, tp = metrics.confusion_matrix(y_cic, (cic_scores > threshold).astype(int), labels=[0, 1]).ravel()
        cic_metrics = compute_metrics_from_counts(tp, fp, tn, fn)
        tn, fp, fn, tp = metrics.confusion_matrix(y_cse, (cse_scores > threshold).astype(int), labels=[0, 1]).ravel()
        cse_metrics = compute_metrics_from_counts(tp, fp, tn, fn)

        try:
            cic_metrics["roc_auc"] = float(metrics.roc_auc_score(y_cic, cic_scores))
            cic_metrics["pr_auc"] = float(metrics.average_precision_score(y_cic, cic_scores))
        except Exception:
            cic_metrics["roc_auc"] = None
            cic_metrics["pr_auc"] = None
        try:
            cse_metrics["roc_auc"] = float(metrics.roc_auc_score(y_cse, cse_scores))
            cse_metrics["pr_auc"] = float(metrics.average_precision_score(y_cse, cse_scores))
        except Exception:
            cse_metrics["roc_auc"] = None
            cse_metrics["pr_auc"] = None

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
        "pr_auc_gap": (cic_metrics.get("pr_auc") or 0) - (cse_metrics.get("pr_auc") or 0),
        "accuracy_gap": cic_metrics["accuracy"] - cse_metrics["accuracy"],
        "mode": eval_mode,
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "score_mode": score_mode,
        "source_calib_split": source_calib_split,
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
    if tf is not None:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if gpus:
            print("[STAGE] Using GPU for evaluation (devices=%d)." % len(gpus))
        if len(gpus) > 1:
            try:
                tf.config.set_visible_devices(gpus[0], "GPU")
                print("[STAGE] Multiple GPUs -> using GPU:0 for eval.")
            except Exception as e:
                print("[WARN] Could not limit visible GPUs: %s" % e)
    main()

