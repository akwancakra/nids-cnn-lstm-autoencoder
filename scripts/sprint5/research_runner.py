"""Research runner for Sprint 5 (domain shift, target-aware threshold)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STAGES = {"preprocess", "train", "eval"}
ALLOWED_STAGE_NAMES = {"stage0", "stage1", "stage2", "stage3", "stage4"}
ALLOWED_CONDITIONS = {"always", "gate_pass"}
MODEL_VARIANT_TO_SUBDIR = {
    "hybrid": "cnn_lstm_ae",
    "baseline": "lstm_ae",
}


class CommandExecutionError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str, duration_sec: float) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        self.duration_sec = float(duration_sec)
        super().__init__(
            f"Command failed rc={returncode} after {duration_sec:.2f}s: {' '.join(cmd)}"
        )


class StageLockViolationError(RuntimeError):
    pass


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_dict_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_csv_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def is_finite_number(value: Any) -> bool:
    v = to_float(value)
    if v is None:
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def run_tag(run: dict[str, Any]) -> str:
    return str(run.get("tag") or run["run_id"])


def model_subdir(model_variant: str) -> str:
    return MODEL_VARIANT_TO_SUBDIR[str(model_variant).lower()]


def derive_isolated_paths(run_id: str) -> dict[str, str]:
    processed_dir = Path("data") / "sprint5" / run_id / "processed"
    return {
        "data_processed": str(processed_dir).replace("\\", "/"),
        "shard_dir": str(processed_dir / "shards").replace("\\", "/"),
        "models_dir": str(Path("models") / "sprint5" / run_id).replace("\\", "/"),
        "results_dir": str(Path("results") / "sprint5" / run_id).replace("\\", "/"),
    }


def apply_run_isolation(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("paths", {})
    out.setdefault("preprocess", {})
    isolated = derive_isolated_paths(run_id)
    out["paths"]["data_processed"] = isolated["data_processed"]
    out["paths"]["models_dir"] = isolated["models_dir"]
    out["paths"]["results_dir"] = isolated["results_dir"]
    out["preprocess"]["shard_dir"] = isolated["shard_dir"]
    out["preprocess"]["shard_enable"] = True
    return out


def apply_data_reuse_inputs(cfg: dict[str, Any], source_run_id: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("paths", {})
    out.setdefault("preprocess", {})
    source = derive_isolated_paths(source_run_id)
    out["paths"]["data_processed"] = source["data_processed"]
    out["preprocess"]["shard_dir"] = source["shard_dir"]
    out["preprocess"]["shard_enable"] = True
    return out


def default_model_path_from_cfg(cfg: dict[str, Any], model_variant: str) -> str:
    model_dir = Path(cfg["paths"]["models_dir"]) / model_subdir(model_variant)
    return str(model_dir / "best_model.keras").replace("\\", "/")


def metrics_paths(root: Path, cfg: dict[str, Any], tag: str) -> tuple[Path, Path, Path]:
    metrics_dir = root / Path(cfg["paths"]["results_dir"]) / "metrics"
    return (
        metrics_dir / f"{tag}_cic_metrics.json",
        metrics_dir / f"{tag}_cse_metrics.json",
        metrics_dir / f"{tag}_generalization_gap.json",
    )


def metrics_exist(root: Path, cfg: dict[str, Any], tag: str) -> bool:
    return all(path.exists() for path in metrics_paths(root, cfg, tag))


def ensure_run_status(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    if run_id not in state:
        state[run_id] = {
            "retry_count": 0,
            "terminal_status": "pending",
            "last_error": "",
            "hash_violation": False,
            "wall_time_sec": 0.0,
            "attempt_logs": [],
        }
    return state[run_id]


def load_run_status_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_run_status_map(path: Path, state: dict[str, Any]) -> None:
    save_json(path, state)


def classify_failure_kind(exc: Exception) -> str:
    if isinstance(exc, StageLockViolationError):
        return "experiment"
    if not isinstance(exc, CommandExecutionError):
        return "experiment"

    if exc.returncode in {137, 143, -9}:  # killed / preempted
        return "infra"

    text = f"{exc.stdout}\n{exc.stderr}\n{exc}".lower()
    infra_keywords = [
        "runtime disconnected",
        "connection reset",
        "connection aborted",
        "timed out",
        "deadline exceeded",
        "transport endpoint",
        "i/o error",
        "input/output error",
        "stale file handle",
        "preempt",
        "killed",
        "resource exhausted",
        "drive mount",
        "permission denied: /content/drive",
        "no space left",
    ]
    for kw in infra_keywords:
        if kw in text:
            return "infra"
    return "experiment"


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> float:
    print("[CMD]", " ".join(cmd), flush=True)
    if dry_run:
        return 0.0

    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    lines: list[str] = []
    for line in proc.stdout or []:
        lines.append(line)
        print(line, end="", flush=True)
    ret = proc.wait()
    dt = time.time() - t0

    out_text = "".join(lines)
    if ret != 0:
        raise CommandExecutionError(
            cmd=cmd,
            returncode=ret,
            stdout=out_text,
            stderr="",
            duration_sec=dt,
        )
    return dt


def wait_for_file_stable(
    path: Path,
    required_same_checks: int = 2,
    delay_sec: float = 1.0,
    timeout_sec: float = 20.0,
) -> bool:
    end_time = time.time() + timeout_sec
    last_state: tuple[int, int] | None = None
    same = 0

    while time.time() < end_time:
        if not path.exists():
            time.sleep(delay_sec)
            continue
        st = path.stat()
        current = (int(st.st_size), int(st.st_mtime_ns))
        if current == last_state:
            same += 1
        else:
            same = 0
            last_state = current

        if same >= required_same_checks:
            return True
        time.sleep(delay_sec)
    return False


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def lock_root_path() -> Path:
    colab_root = Path("/content")
    if colab_root.exists():
        return colab_root / "sprint5_lock"
    return Path(tempfile.gettempdir()) / "sprint5_lock"


def prepare_stage3_locked_model(source_model: Path, run_id: str) -> tuple[Path, str]:
    if not source_model.exists():
        raise FileNotFoundError(f"Stage3 model not found: {source_model}")

    lock_root = lock_root_path() / run_id
    lock_root.mkdir(parents=True, exist_ok=True)
    locked_model = lock_root / source_model.name
    if locked_model.exists():
        try:
            locked_model.chmod(0o666)
        except Exception:
            pass
        locked_model.unlink()
    shutil.copy2(source_model, locked_model)

    if not wait_for_file_stable(locked_model, required_same_checks=2, delay_sec=1.0, timeout_sec=30.0):
        raise RuntimeError(f"Locked model did not stabilize in time: {locked_model}")

    make_read_only(locked_model)
    pre_hash = hash_file_sha256(locked_model)
    return locked_model, pre_hash


def verify_stage3_locked_model(path: Path, expected_hash: str) -> bool:
    if not wait_for_file_stable(path, required_same_checks=2, delay_sec=1.0, timeout_sec=30.0):
        return False
    post_hash = hash_file_sha256(path)
    return post_hash == expected_hash


def canonicalize_config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.pop("runtime_contract", None)
    return out


def compute_base_config_hash(base_cfg: dict[str, Any]) -> str:
    return hash_dict_sha256(base_cfg)


def compute_effective_config_hash(cfg: dict[str, Any]) -> str:
    return hash_dict_sha256(canonicalize_config_for_hash(cfg))


def add_contract_hashes(cfg: dict[str, Any], base_hash: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("runtime_contract", {})
    out["runtime_contract"]["base_config_hash"] = base_hash
    out["runtime_contract"]["effective_config_hash"] = compute_effective_config_hash(out)
    return out


def resolve_profile_cfg(profile_spec: Any, root: Path) -> dict[str, Any]:
    if isinstance(profile_spec, str):
        profile_path = root / profile_spec
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_path}")
        return load_yaml(profile_path)

    if isinstance(profile_spec, dict):
        profile_path_raw = profile_spec.get("profile_path")
        if profile_path_raw:
            profile_path = root / str(profile_path_raw)
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile file not found: {profile_path}")
            file_cfg = load_yaml(profile_path)
            inline = {k: v for k, v in profile_spec.items() if k != "profile_path"}
            return deep_update(file_cfg, inline)
        return deepcopy(profile_spec)

    raise ValueError(f"Unsupported profile spec type: {type(profile_spec).__name__}")


def validate_registry(registry: dict[str, Any]) -> None:
    profiles = registry.get("profiles")
    runs = registry.get("runs")

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("registry.profiles must be a non-empty mapping")
    if not isinstance(runs, list) or not runs:
        raise ValueError("registry.runs must be a non-empty list")

    seen: set[str] = set()
    for run in runs:
        run_id = str(run.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Each run must define non-empty run_id")
        if run_id in seen:
            raise ValueError(f"Duplicate run_id detected: {run_id}")
        seen.add(run_id)

        profile_name = str(run.get("profile", "")).strip()
        if profile_name not in profiles:
            raise ValueError(f"Run {run_id} references unknown profile: {profile_name}")

        stage_name = str(run.get("stage_name", "")).strip()
        if stage_name and stage_name not in ALLOWED_STAGE_NAMES:
            raise ValueError(f"Run {run_id} has invalid stage_name={stage_name}")

        stages = run.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"Run {run_id} must define non-empty stages list")
        invalid = [s for s in stages if s not in ALLOWED_STAGES]
        if invalid:
            raise ValueError(f"Run {run_id} has invalid stages: {invalid}")

        condition = str(run.get("condition", "always"))
        if condition not in ALLOWED_CONDITIONS:
            raise ValueError(f"Run {run_id} has invalid condition={condition}")

        variant = str(run.get("model_variant", "hybrid")).lower()
        if variant not in MODEL_VARIANT_TO_SUBDIR:
            raise ValueError(f"Run {run_id} has invalid model_variant={variant}")


def should_run(run: dict[str, Any], selected_ids: set[str] | None, selected_stage_names: set[str] | None) -> bool:
    if selected_ids is not None and run["run_id"] not in selected_ids:
        return False
    if selected_stage_names is not None and run.get("stage_name") not in selected_stage_names:
        return False
    return bool(run.get("active", False))


def passes_constraints(row: dict[str, Any], cic_fpr_max: float, cse_precision_min: float) -> bool:
    cse_prec = to_float(row.get("cse_prec"))
    cic_fpr = to_float(row.get("cic_fpr"))
    if cse_prec is None or cic_fpr is None:
        return False
    return cse_prec >= cse_precision_min and cic_fpr <= cic_fpr_max


def sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(to_float(r.get("cse_rec")) or -1.0),
            float(to_float(r.get("cse_f1")) or -1.0),
            -float(to_float(r.get("cic_fpr")) or 1.0),
        ),
        reverse=True,
    )


def select_stage_candidates(stage_rows: list[dict[str, Any]], meta: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    valid_rows = [r for r in stage_rows if bool(r.get("valid_metrics", False))]
    if not valid_rows:
        return [], "none"

    ranking = meta.get("ranking_constraints", {})
    hard = ranking.get("hard", {})
    fallback = ranking.get("fallback", {})

    hard_rows = [
        r
        for r in valid_rows
        if passes_constraints(
            r,
            cic_fpr_max=float(hard.get("cic_fpr_max", 0.10)),
            cse_precision_min=float(hard.get("cse_precision_min", 0.30)),
        )
    ]
    if hard_rows:
        return sort_candidate_rows(hard_rows), "hard"

    fallback_rows = [
        r
        for r in valid_rows
        if passes_constraints(
            r,
            cic_fpr_max=float(fallback.get("cic_fpr_max", 0.12)),
            cse_precision_min=float(fallback.get("cse_precision_min", 0.25)),
        )
    ]
    if fallback_rows:
        return sort_candidate_rows(fallback_rows), "fallback"

    return sort_candidate_rows(valid_rows), "worst_case"


def extract_metric_rows(cic: dict[str, Any], cse: dict[str, Any], gap: dict[str, Any]) -> dict[str, float | None]:
    cic_acc = to_float(cic.get("accuracy"))
    cic_prec = to_float(cic.get("precision"))
    cic_rec = to_float(cic.get("recall"))
    cic_f1 = to_float(cic.get("f1", cic.get("f1_score")))
    cic_fpr = to_float(cic.get("fpr"))
    cic_auc = to_float(cic.get("roc_auc", cic.get("auc")))

    cse_acc = to_float(cse.get("accuracy"))
    cse_prec = to_float(cse.get("precision"))
    cse_rec = to_float(cse.get("recall"))
    cse_f1 = to_float(cse.get("f1", cse.get("f1_score")))
    cse_fpr = to_float(cse.get("fpr"))
    cse_auc = to_float(cse.get("roc_auc", cse.get("auc")))

    return {
        "threshold": to_float(gap.get("threshold")),
        "cic_acc": cic_acc,
        "cic_prec": cic_prec,
        "cic_rec": cic_rec,
        "cic_f1": cic_f1,
        "cic_fpr": cic_fpr,
        "cic_auc": cic_auc,
        "cse_acc": cse_acc,
        "cse_prec": cse_prec,
        "cse_rec": cse_rec,
        "cse_f1": cse_f1,
        "cse_fpr": cse_fpr,
        "cse_auc": cse_auc,
        "f1_gap": to_float(gap.get("f1_gap")),
        "auc_gap": to_float(gap.get("auc_gap")),
        "accuracy_gap": to_float(gap.get("accuracy_gap")),
        "score_mode": gap.get("score_mode"),
        "threshold_method": gap.get("threshold_method"),
    }


def valid_metrics_fields(metrics_row: dict[str, Any]) -> bool:
    required = ["cse_rec", "cse_prec", "cse_f1", "cic_fpr", "cic_f1"]
    for key in required:
        if not is_finite_number(metrics_row.get(key)):
            return False
    return True


def flag_collapse(cse_rec: float | None, cse_prec: float | None, meta: dict[str, Any]) -> bool:
    collapse = meta.get("gate_rules", {}).get("collapse_flag", {})
    recall_gte = float(collapse.get("recall_gte", 0.95))
    precision_lt = float(collapse.get("precision_lt", 0.20))
    if cse_rec is None or cse_prec is None:
        return False
    return (cse_rec >= recall_gte) and (cse_prec < precision_lt)


def compute_stage_validity(
    rows: list[dict[str, Any]],
    stage_rules: dict[str, Any],
    gate_pass: bool,
) -> dict[str, dict[str, Any]]:
    stage_map: dict[str, dict[str, Any]] = {}

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(str(row.get("stage_name", "")), []).append(row)

    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        required = int(stage_rules.get(stage_name, 0))
        stage_rows = by_stage.get(stage_name, [])
        valid_count = len([r for r in stage_rows if bool(r.get("valid_metrics", False))])

        if stage_name == "stage4" and not gate_pass:
            stage_map[stage_name] = {
                "required": required,
                "valid_count": valid_count,
                "passed": True,
                "status": "skipped_by_design",
            }
            continue

        passed = valid_count >= required
        stage_map[stage_name] = {
            "required": required,
            "valid_count": valid_count,
            "passed": passed,
            "status": "ok" if passed else "inconclusive",
        }

    return stage_map


def collect_history_pairs(root: Path, cic_fpr_cap: float) -> dict[str, Any]:
    recalls_all: list[float] = []
    recalls_guard: list[float] = []
    best_guard_recall = -1.0
    best_guard_path = ""

    include_sprints = ["sprint4", "sprint5"]
    results_root = (root / "results").resolve()
    for sprint in include_sprints:
        sprint_dir = results_root / sprint
        if not sprint_dir.exists():
            continue
        for cse_path in sprint_dir.rglob("*_cse_metrics.json"):
            stem = cse_path.name.replace("_cse_metrics.json", "")
            cic_path = cse_path.with_name(f"{stem}_cic_metrics.json")
            if not cic_path.exists():
                continue

            try:
                cse = load_json(cse_path)
                cic = load_json(cic_path)
            except Exception:
                continue

            cse_rec = to_float(cse.get("recall"))
            cic_fpr_val = to_float(cic.get("fpr"))
            if cse_rec is None or cic_fpr_val is None:
                continue

            recalls_all.append(cse_rec)
            if cic_fpr_val <= cic_fpr_cap:
                recalls_guard.append(cse_rec)
                if cse_rec > best_guard_recall:
                    best_guard_recall = cse_rec
                    best_guard_path = str(cse_path).replace("\\", "/")

    def percentile(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        arr = sorted(vals)
        if len(arr) == 1:
            return arr[0]
        pos = (len(arr) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(arr) - 1)
        alpha = pos - lo
        return arr[lo] * (1.0 - alpha) + arr[hi] * alpha

    return {
        "count_all": len(recalls_all),
        "count_under_guardrail": len(recalls_guard),
        "recall_p50": percentile(recalls_guard or recalls_all, 0.50),
        "recall_p90": percentile(recalls_guard or recalls_all, 0.90),
        "best_recall_under_guardrail": None if best_guard_recall < 0 else best_guard_recall,
        "best_recall_path": best_guard_path,
    }


def _assert_history_populated(root: Path, history_summary: dict[str, Any]) -> None:
    """Fail early if we have complete metric pairs on disk but history is empty (include/parse bug)."""
    count_all = history_summary.get("count_all", 0) or 0
    count_guard = history_summary.get("count_under_guardrail", 0) or 0
    if count_all > 0 or count_guard > 0:
        return
    results_root = (root / "results").resolve()
    for sprint in ("sprint4", "sprint5"):
        sprint_dir = results_root / sprint
        if not sprint_dir.exists():
            continue
        for cse_path in sprint_dir.rglob("*_cse_metrics.json"):
            stem = cse_path.name.replace("_cse_metrics.json", "")
            cic_path = cse_path.with_name(f"{stem}_cic_metrics.json")
            if not cic_path.exists():
                continue
            try:
                cse = load_json(cse_path)
                cic = load_json(cic_path)
            except Exception:
                continue
            if to_float(cse.get("recall")) is not None and to_float(cic.get("fpr")) is not None:
                raise RuntimeError(
                    f"[BUG] Found valid pair under results/{sprint}/: {cse_path.name} + {cic_path.name} "
                    "but history count_all=0, count_under_guardrail=0. "
                    "Check collect_history_pairs include logic or path normalization."
                )
    return None


def compute_adaptive_recall_target(meta: dict[str, Any], history_summary: dict[str, Any]) -> tuple[float, str]:
    gate = meta.get("gate_rules", {}).get("adaptive_recall", {})
    floor = float(gate.get("fixed_floor", 0.40))
    plus = float(gate.get("plus_baseline", 0.05))

    baseline_meta = meta.get("baseline_history", {})
    baseline = to_float(baseline_meta.get("best_recall_under_cic_fpr_cap"))
    if baseline is None:
        baseline = to_float(history_summary.get("best_recall_under_guardrail"))

    if baseline is None:
        return floor, "fixed_floor_only"

    target = max(floor, baseline + plus)
    return float(target), f"max({floor:.4f}, baseline({baseline:.4f})+{plus:.4f})"


def build_run_row(
    root: Path,
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_state: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    tag = run_tag(run)
    cic_path, cse_path, gap_path = metrics_paths(root, run_cfg, tag)

    cic = load_json(cic_path) if cic_path.exists() else {}
    cse = load_json(cse_path) if cse_path.exists() else {}
    gap = load_json(gap_path) if gap_path.exists() else {}

    row = {
        "run_id": run["run_id"],
        "stage_name": run.get("stage_name", ""),
        "profile": run.get("profile", ""),
        "model_variant": run.get("model_variant", "hybrid"),
        "active": bool(run.get("active", False)),
        "condition": run.get("condition", "always"),
        "stages": ",".join(run.get("stages", [])),
        "tag": tag,
        "terminal_status": run_state.get("terminal_status", "pending"),
        "retry_count": int(run_state.get("retry_count", 0)),
        "hash_violation": bool(run_state.get("hash_violation", False)),
        "last_error": run_state.get("last_error", ""),
        "wall_time_sec": to_float(run_state.get("wall_time_sec")),
    }
    row.update(extract_metric_rows(cic, cse, gap))

    status = str(row["terminal_status"])
    base_valid = (
        status == "success"
        and cic_path.exists()
        and cse_path.exists()
        and gap_path.exists()
        and valid_metrics_fields(row)
        and not bool(row["hash_violation"])
    )
    row["valid_metrics"] = base_valid
    row["collapse_flag"] = flag_collapse(row.get("cse_rec"), row.get("cse_prec"), meta)

    ranking = meta.get("ranking_constraints", {})
    hard = ranking.get("hard", {})
    fallback = ranking.get("fallback", {})
    row["constraint_pass"] = base_valid and passes_constraints(
        row,
        cic_fpr_max=float(hard.get("cic_fpr_max", 0.10)),
        cse_precision_min=float(hard.get("cse_precision_min", 0.30)),
    )
    row["fallback_pass"] = base_valid and passes_constraints(
        row,
        cic_fpr_max=float(fallback.get("cic_fpr_max", 0.12)),
        cse_precision_min=float(fallback.get("cse_precision_min", 0.25)),
    )
    row["stage_validity_pass"] = None
    row["gate_pass"] = None
    return row


def select_best_run_id_for_stage(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
    stage_name: str,
    rank: int = 1,
) -> str | None:
    stage_rows = []
    for run in runs:
        if run.get("stage_name") != stage_name:
            continue
        cfg = run_cfg_lookup.get(run["run_id"])
        if cfg is None:
            continue
        state = ensure_run_status(run_state_map, run["run_id"])
        stage_rows.append(build_run_row(root, run, cfg, state, meta))

    stage_rules = meta.get("stage_validity_rules", {})
    required = int(stage_rules.get(stage_name, 0))
    valid_count = len([r for r in stage_rows if bool(r.get("valid_metrics", False))])
    if required > 0 and valid_count < required:
        return None

    candidates, _mode = select_stage_candidates(stage_rows, meta)
    if not candidates:
        return None
    idx = max(0, rank - 1)
    if idx >= len(candidates):
        return None
    return str(candidates[idx]["run_id"])


def resolve_model_path_for_eval(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
    run: dict[str, Any],
    run_cfg: dict[str, Any],
) -> str:
    explicit = run.get("model_path")
    if explicit:
        return str(explicit)

    source_run = run.get("reuse_model_from")
    if source_run:
        source_cfg = run_cfg_lookup[source_run]
        source_spec = next(r for r in runs if r["run_id"] == source_run)
        source_variant = str(source_spec.get("model_variant", "hybrid")).lower()
        return default_model_path_from_cfg(source_cfg, source_variant)

    source_stage = run.get("reuse_model_from_best_stage")
    if source_stage:
        rank = int(run.get("reuse_stage_rank", 1))
        selected = select_best_run_id_for_stage(
            root=root,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            run_state_map=run_state_map,
            meta=meta,
            stage_name=source_stage,
            rank=rank,
        )
        if not selected:
            raise ValueError(
                f"Run {run['run_id']} requires best model from stage={source_stage}, but source stage is inconclusive."
            )
        source_cfg = run_cfg_lookup[selected]
        source_spec = next(r for r in runs if r["run_id"] == selected)
        source_variant = str(source_spec.get("model_variant", "hybrid")).lower()
        return default_model_path_from_cfg(source_cfg, source_variant)

    variant = str(run.get("model_variant", "hybrid")).lower()
    return default_model_path_from_cfg(run_cfg, variant)


def materialize_run_config(
    root: Path,
    base_cfg: dict[str, Any],
    base_hash: str,
    resolved_profiles: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
    run_spec: dict[str, Any],
    allow_unresolved_dynamic: bool = False,
) -> dict[str, Any]:
    run_id = run_spec["run_id"]
    profile_name = str(run_spec.get("profile", "")).strip()
    if profile_name not in resolved_profiles:
        raise ValueError(f"Run {run_id} references unknown profile: {profile_name}")

    cfg_base = deep_update(base_cfg, resolved_profiles[profile_name])

    template_stage = run_spec.get("template_from_best_stage")
    if template_stage:
        rank = int(run_spec.get("reuse_stage_rank", 1))
        source_run_id = select_best_run_id_for_stage(
            root=root,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            run_state_map=run_state_map,
            meta=meta,
            stage_name=template_stage,
            rank=rank,
        )
        if source_run_id and source_run_id in run_cfg_lookup:
            cfg_base = deepcopy(run_cfg_lookup[source_run_id])
        elif not allow_unresolved_dynamic:
            raise ValueError(
                f"Run {run_id} requires template_from_best_stage={template_stage}, but source stage is inconclusive."
            )

    cfg = deep_update(cfg_base, run_spec.get("overrides", {}))
    cfg = apply_run_isolation(cfg, run_id)

    reuse_data_from = run_spec.get("reuse_data_from")
    if reuse_data_from:
        cfg = apply_data_reuse_inputs(cfg, str(reuse_data_from))

    reuse_data_stage = run_spec.get("reuse_data_from_best_stage")
    if reuse_data_stage:
        rank = int(run_spec.get("reuse_stage_rank", 1))
        source_run_id = select_best_run_id_for_stage(
            root=root,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            run_state_map=run_state_map,
            meta=meta,
            stage_name=reuse_data_stage,
            rank=rank,
        )
        if source_run_id:
            cfg = apply_data_reuse_inputs(cfg, source_run_id)
        elif not allow_unresolved_dynamic:
            raise ValueError(
                f"Run {run_id} requires best data from stage={reuse_data_stage}, but source stage is inconclusive."
            )

    cfg = add_contract_hashes(cfg, base_hash)
    return cfg


def verify_config_contract(cfg: dict[str, Any], base_hash: str) -> None:
    contract = cfg.get("runtime_contract", {})
    expected_base = str(contract.get("base_config_hash", ""))
    expected_effective = str(contract.get("effective_config_hash", ""))

    if expected_base != base_hash:
        raise ValueError("Config contract mismatch: base_config_hash does not match current base config.")

    current_effective = compute_effective_config_hash(cfg)
    if expected_effective != current_effective:
        raise ValueError("Config contract mismatch: effective_config_hash validation failed.")


def run_single_experiment(
    root: Path,
    python_exe: str,
    runs: list[dict[str, Any]],
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_cfg_path: Path,
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
    dry_run: bool,
) -> float:
    stages = run.get("stages", [])
    variant = str(run.get("model_variant", "hybrid")).lower()
    tag = run_tag(run)
    total_sec = 0.0

    for stage in stages:
        if stage == "preprocess":
            total_sec += run_cmd(
                [python_exe, "scripts/sprint5/preprocess.py", "--config", str(run_cfg_path)],
                cwd=root,
                dry_run=dry_run,
            )
        elif stage == "train":
            total_sec += run_cmd(
                [
                    python_exe,
                    "scripts/sprint5/train.py",
                    "--config",
                    str(run_cfg_path),
                    "--variant",
                    variant,
                ],
                cwd=root,
                dry_run=dry_run,
            )
        elif stage == "eval":
            model_path = resolve_model_path_for_eval(
                root=root,
                runs=runs,
                run_cfg_lookup=run_cfg_lookup,
                run_state_map=run_state_map,
                meta=meta,
                run=run,
                run_cfg=run_cfg,
            )

            eval_model_path = model_path
            lock_hash = None
            locked_local_path = None

            if str(run.get("stage_name", "")) == "stage3":
                locked_local_path, lock_hash = prepare_stage3_locked_model(Path(model_path), run["run_id"])
                eval_model_path = str(locked_local_path).replace("\\", "/")

            total_sec += run_cmd(
                [
                    python_exe,
                    "scripts/sprint5/eval.py",
                    "--config",
                    str(run_cfg_path),
                    "--model",
                    eval_model_path,
                    "--tag",
                    tag,
                ],
                cwd=root,
                dry_run=dry_run,
            )

            if lock_hash is not None and locked_local_path is not None and not dry_run:
                if not verify_stage3_locked_model(locked_local_path, lock_hash):
                    state = ensure_run_status(run_state_map, run["run_id"])
                    state["hash_violation"] = True
                    raise StageLockViolationError(
                        f"Stage3 lock hash mismatch for run {run['run_id']} ({locked_local_path})"
                    )
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    return total_sec


def compute_gate_state(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = run_cfg_lookup.get(run["run_id"])
        if cfg is None:
            continue
        state = ensure_run_status(run_state_map, run["run_id"])
        rows.append(build_run_row(root, run, cfg, state, meta))

    stage3_rows = [r for r in rows if r.get("stage_name") == "stage3"]
    candidates, mode = select_stage_candidates(stage3_rows, meta)

    history = collect_history_pairs(
        root=root,
        cic_fpr_cap=float(meta.get("gate_rules", {}).get("cic_fpr_max", 0.10)),
    )
    _assert_history_populated(root, history)
    adaptive_target, expr = compute_adaptive_recall_target(meta, history)

    out = {
        "gate_pass": False,
        "reason": "stage3_no_candidate",
        "candidate_mode": mode,
        "adaptive_recall_target": adaptive_target,
        "adaptive_target_reasoning": expr,
        "best_stage3_run_id": None,
        "best_stage3_metrics": None,
        "history_summary": history,
    }

    if not candidates:
        return out

    best = candidates[0]
    out["best_stage3_run_id"] = best.get("run_id")
    out["best_stage3_metrics"] = {
        "cse_recall": best.get("cse_rec"),
        "cse_precision": best.get("cse_prec"),
        "cse_f1": best.get("cse_f1"),
        "cic_fpr": best.get("cic_fpr"),
        "threshold": best.get("threshold"),
    }

    cse_recall = float(to_float(best.get("cse_rec")) or 0.0)
    cse_precision = float(to_float(best.get("cse_prec")) or 0.0)
    cic_fpr = float(to_float(best.get("cic_fpr")) or 1.0)

    precision_min = float(meta.get("gate_rules", {}).get("cse_precision_min", 0.30))
    cic_fpr_max = float(meta.get("gate_rules", {}).get("cic_fpr_max", 0.10))

    gate_pass = (
        cse_recall >= adaptive_target
        and cse_precision >= precision_min
        and cic_fpr <= cic_fpr_max
    )

    out["gate_pass"] = gate_pass
    out["reason"] = "passed" if gate_pass else "failed_constraints"
    return out


def annotate_stage_validity(rows: list[dict[str, Any]], stage_validity: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        stage_name = str(row.get("stage_name", ""))
        row["stage_validity_pass"] = bool(stage_validity.get(stage_name, {}).get("passed", True))


def summarize(
    root: Path,
    registry: dict[str, Any],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    summary_csv: Path,
    report_md: Path,
    gate_json: Path,
) -> None:
    meta = registry.get("meta", {})
    runs = registry.get("runs", [])

    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = run_cfg_lookup.get(run["run_id"])
        if cfg is None:
            continue
        state = ensure_run_status(run_state_map, run["run_id"])
        row = build_run_row(root, run, cfg, state, meta)
        rows.append(row)

    gate_state = compute_gate_state(root, runs, run_cfg_lookup, run_state_map, meta)
    stage_validity = compute_stage_validity(
        rows=rows,
        stage_rules=meta.get("stage_validity_rules", {}),
        gate_pass=bool(gate_state.get("gate_pass", False)),
    )

    for row in rows:
        row["gate_pass"] = bool(gate_state.get("gate_pass", False)) if row.get("stage_name") == "stage3" else None
    annotate_stage_validity(rows, stage_validity)

    fields = [
        "run_id",
        "stage_name",
        "profile",
        "model_variant",
        "active",
        "condition",
        "stages",
        "tag",
        "terminal_status",
        "retry_count",
        "hash_violation",
        "valid_metrics",
        "stage_validity_pass",
        "constraint_pass",
        "fallback_pass",
        "collapse_flag",
        "gate_pass",
        "threshold",
        "threshold_method",
        "score_mode",
        "cic_acc",
        "cic_prec",
        "cic_rec",
        "cic_f1",
        "cic_fpr",
        "cic_auc",
        "cse_acc",
        "cse_prec",
        "cse_rec",
        "cse_f1",
        "cse_fpr",
        "cse_auc",
        "f1_gap",
        "auc_gap",
        "accuracy_gap",
        "wall_time_sec",
        "last_error",
    ]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    history_summary = gate_state.get("history_summary", {})
    gate_out = {
        "generated_at": datetime.now().isoformat(),
        "gate_pass": bool(gate_state.get("gate_pass", False)),
        "reason": gate_state.get("reason"),
        "adaptive_recall_target": gate_state.get("adaptive_recall_target"),
        "adaptive_target_reasoning": gate_state.get("adaptive_target_reasoning"),
        "best_stage3_run_id": gate_state.get("best_stage3_run_id"),
        "best_stage3_metrics": gate_state.get("best_stage3_metrics"),
        "pivot_recommendation": None,
        "stage_validity": stage_validity,
        "sprint_history_summary": {
            "recall_p50": history_summary.get("recall_p50"),
            "recall_p90": history_summary.get("recall_p90"),
            "best_recall_under_guardrail": history_summary.get("best_recall_under_guardrail"),
            "best_recall_path": history_summary.get("best_recall_path"),
            "count_all": history_summary.get("count_all"),
            "count_under_guardrail": history_summary.get("count_under_guardrail"),
        },
    }
    if not gate_out["gate_pass"]:
        gate_out["pivot_recommendation"] = meta.get("pivot_policy", {}).get("on_gate_fail", "USAD")

    save_json(gate_json, gate_out)

    ok_rows = [r for r in rows if r.get("valid_metrics")]
    lines = [
        "# RESEARCH REPORT CSE F1 - SPRINT 4",
        "",
        f"- generated_at: {datetime.now().isoformat()}",
        f"- source_summary: `{summary_csv.as_posix()}`",
        f"- gate_decision: `{gate_json.as_posix()}`",
        "",
        "## Ringkasan / Summary",
        "",
        "- Objective: maximize CSE recall dengan guardrail CIC FPR rendah.",
        "- Objective (EN): maximize CSE recall under low CIC-FPR guardrail.",
        f"- Gate pass: `{gate_out['gate_pass']}` (reason: `{gate_out['reason']}`).",
        f"- Adaptive recall target: `{gate_out['adaptive_recall_target']}` ({gate_out['adaptive_target_reasoning']}).",
        "",
        "## Stage Validity",
        "",
        "| stage | required_valid_runs | valid_runs | status |",
        "| --- | ---: | ---: | --- |",
    ]

    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        st = stage_validity.get(stage_name, {})
        lines.append(
            f"| {stage_name} | {st.get('required', 0)} | {st.get('valid_count', 0)} | {st.get('status', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Best Candidate Per Stage",
            "",
            "| stage | run_id | mode | cse_recall | cse_precision | cse_f1 | cic_fpr |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for stage_name in ["stage0", "stage1", "stage2", "stage3", "stage4"]:
        stage_rows = [r for r in ok_rows if r.get("stage_name") == stage_name]
        if not stage_rows:
            lines.append(f"| {stage_name} | - | - | - | - | - | - |")
            continue
        candidates, mode = select_stage_candidates(stage_rows, meta)
        if not candidates:
            lines.append(f"| {stage_name} | - | - | - | - | - | - |")
            continue
        best = candidates[0]
        lines.append(
            "| {stage} | {run_id} | {mode} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {fpr:.4f} |".format(
                stage=stage_name,
                run_id=best["run_id"],
                mode=mode,
                rec=float(best["cse_rec"]),
                prec=float(best["cse_prec"]),
                f1=float(best["cse_f1"]),
                fpr=float(best["cic_fpr"]),
            )
        )

    lines.extend(
        [
            "",
            "## Operational Notes",
            "",
            "- ID Primary: Terminologi teknis tetap English untuk konsistensi.",
            "- EN Mirror: Technical keywords are intentionally kept in English.",
            "",
            "## Run Status",
            "",
            f"- total_runs: {len(rows)}",
            f"- success: {len([r for r in rows if r['terminal_status'] == 'success'])}",
            f"- failed_experiment: {len([r for r in rows if r['terminal_status'] == 'failed_experiment'])}",
            f"- failed_infra_exhausted: {len([r for r in rows if r['terminal_status'] == 'failed_infra_exhausted'])}",
            f"- pending_or_skipped: {len([r for r in rows if r['terminal_status'] not in {'success','failed_experiment','failed_infra_exhausted'}])}",
            "",
        ]
    )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def generated_cfg_path(generated_dir: Path, run_id: str) -> Path:
    return generated_dir / f"{run_id}.yaml"


def estimate_stage2_eta(
    root: Path,
    runs: list[dict[str, Any]],
    run_state_map: dict[str, Any],
    out_path: Path,
) -> None:
    stage0_ids = [r["run_id"] for r in runs if r.get("stage_name") == "stage0"]
    stage2_runs = [r for r in runs if r.get("stage_name") == "stage2"]

    stage0_durations = []
    for run_id in stage0_ids:
        state = ensure_run_status(run_state_map, run_id)
        if str(state.get("terminal_status")) == "success":
            dur = to_float(state.get("wall_time_sec"))
            if dur is not None and dur > 0:
                stage0_durations.append(dur)

    baseline = sum(stage0_durations) / len(stage0_durations) if stage0_durations else 3600.0

    multipliers = {
        "s4_m01_baseline_arch": 1.00,
        "s4_m02_dropout_030": 1.02,
        "s4_m03_latent_64": 1.08,
        "s4_m04_cnn_64_128_128": 1.35,
        "s4_m05_lstm_192_96": 1.22,
        "s4_m06_loss_huber": 1.04,
        "s4_m07_loss_mse_mae_mix_a07": 1.05,
        "s4_m08_hybrid_score_alpha07": 1.08,
    }

    per_run = []
    total_p50 = 0.0
    for run in stage2_runs:
        run_id = run["run_id"]
        mult = multipliers.get(run_id, 1.10)
        est = baseline * mult
        per_run.append({"run_id": run_id, "multiplier": mult, "eta_sec": est})
        total_p50 += est

    eta = {
        "generated_at": datetime.now().isoformat(),
        "note": "Coarse ETA only (no false precision). Warmup benchmark is optional and may vary in Colab.",
        "baseline_source": "stage0_success_average_or_default_3600s",
        "baseline_sec": baseline,
        "p50_total_sec": total_p50,
        "p90_total_sec": total_p50 * 1.25,
        "runs": per_run,
    }
    save_json(out_path, eta)


def execute_run_with_retry(
    root: Path,
    python_exe: str,
    runs: list[dict[str, Any]],
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_cfg_path: Path,
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_state_map: dict[str, Any],
    meta: dict[str, Any],
    dry_run: bool,
) -> None:
    run_id = run["run_id"]
    state = ensure_run_status(run_state_map, run_id)
    max_retry = int(meta.get("retry_policy", {}).get("infra_retry_free_max", 2))

    while True:
        try:
            elapsed = run_single_experiment(
                root=root,
                python_exe=python_exe,
                runs=runs,
                run=run,
                run_cfg=run_cfg,
                run_cfg_path=run_cfg_path,
                run_cfg_lookup=run_cfg_lookup,
                run_state_map=run_state_map,
                meta=meta,
                dry_run=dry_run,
            )
            state["terminal_status"] = "success"
            state["wall_time_sec"] = float(to_float(state.get("wall_time_sec")) or 0.0) + float(elapsed)
            state["last_error"] = ""
            break
        except Exception as exc:
            kind = classify_failure_kind(exc)
            state["attempt_logs"].append(
                {
                    "ts": datetime.now().isoformat(),
                    "kind": kind,
                    "error": str(exc),
                }
            )
            if kind == "infra" and int(state.get("retry_count", 0)) < max_retry:
                state["retry_count"] = int(state.get("retry_count", 0)) + 1
                state["terminal_status"] = "retrying_infra"
                state["last_error"] = str(exc)
                print(
                    f"[RETRY] run_id={run_id} infra retry {state['retry_count']}/{max_retry} | {exc}",
                    file=sys.stderr,
                )
                continue

            if kind == "infra":
                state["terminal_status"] = "failed_infra_exhausted"
            else:
                state["terminal_status"] = "failed_experiment"
            state["last_error"] = str(exc)
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="research/sprint5/run_registry.yaml")
    parser.add_argument("--base-config", default="config/sprint5/base.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--generated-config-dir", default=None)
    parser.add_argument("--run-ids", default=None)
    parser.add_argument("--stage-names", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--summary-csv", default="results/sprint5/summary.csv")
    parser.add_argument("--report-md", default="docs/sprint5/RESEARCH_REPORT_CSE_F1_SPRINT5.md")
    parser.add_argument("--gate-json", default="results/sprint5/gate_decision.json")
    parser.add_argument("--status-json", default="results/sprint5/runtime/run_status.json")
    parser.add_argument("--eta-json", default="results/sprint5/eta_stage2.json")
    args = parser.parse_args()

    root = ROOT
    registry_path = root / args.registry
    base_cfg_path = root / args.base_config
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_cfg_path}")

    registry = load_yaml(registry_path)
    validate_registry(registry)

    base_cfg = load_yaml(base_cfg_path)
    base_hash = compute_base_config_hash(base_cfg)

    meta = registry.get("meta", {})
    expected_base = meta.get("base_config_hash")
    if expected_base and str(expected_base) != base_hash:
        raise ValueError(
            f"Registry base_config_hash mismatch. expected={expected_base} actual={base_hash}"
        )

    runs = registry["runs"]
    profiles = registry["profiles"]
    resolved_profiles: dict[str, dict[str, Any]] = {}
    for name, spec in profiles.items():
        resolved_profiles[name] = resolve_profile_cfg(spec, root)

    if args.generated_config_dir:
        generated_dir = root / args.generated_config_dir
    else:
        generated_dir = registry_path.parent / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    status_json = root / args.status_json
    run_state_map = load_run_status_map(status_json)

    selected_ids = parse_csv_set(args.run_ids)
    selected_stage_names = parse_csv_set(args.stage_names)

    run_cfg_lookup: dict[str, dict[str, Any]] = {}

    for run in runs:
        cfg = materialize_run_config(
            root=root,
            base_cfg=base_cfg,
            base_hash=base_hash,
            resolved_profiles=resolved_profiles,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            run_state_map=run_state_map,
            meta=meta,
            run_spec=run,
            allow_unresolved_dynamic=True,
        )
        run_cfg_lookup[run["run_id"]] = cfg
        save_yaml(generated_cfg_path(generated_dir, run["run_id"]), cfg)

    if not args.summarize_only:
        if selected_stage_names is None or "stage2" in selected_stage_names:
            estimate_stage2_eta(root, runs, run_state_map, root / args.eta_json)

        for run in runs:
            if not should_run(run, selected_ids, selected_stage_names):
                continue

            run_id = run["run_id"]
            state = ensure_run_status(run_state_map, run_id)

            if str(run.get("condition", "always")) == "gate_pass":
                gate_state = compute_gate_state(root, runs, run_cfg_lookup, run_state_map, meta)
                if not bool(gate_state.get("gate_pass", False)):
                    state["terminal_status"] = "skipped_gate"
                    state["last_error"] = "Gate not passed; stage4 skipped by design."
                    save_run_status_map(status_json, run_state_map)
                    print(f"[SKIP] {run_id} gate_pass condition not met")
                    continue

            try:
                cfg = materialize_run_config(
                    root=root,
                    base_cfg=base_cfg,
                    base_hash=base_hash,
                    resolved_profiles=resolved_profiles,
                    runs=runs,
                    run_cfg_lookup=run_cfg_lookup,
                    run_state_map=run_state_map,
                    meta=meta,
                    run_spec=run,
                    allow_unresolved_dynamic=False,
                )
            except ValueError as exc:
                state["terminal_status"] = "skipped_inconclusive"
                state["last_error"] = str(exc)
                save_run_status_map(status_json, run_state_map)
                print(f"[SKIP] {run_id} | {exc}")
                continue

            verify_config_contract(cfg, base_hash)
            run_cfg_lookup[run_id] = cfg
            cfg_path = generated_cfg_path(generated_dir, run_id)
            save_yaml(cfg_path, cfg)

            tag = run_tag(run)
            if args.skip_existing and "eval" in run.get("stages", []) and metrics_exist(root, cfg, tag):
                if state.get("terminal_status") != "success":
                    state["terminal_status"] = "success"
                save_run_status_map(status_json, run_state_map)
                print(f"[SKIP] {run_id} metrics already exist for tag={tag}")
                continue

            print(f"[RUN] {run_id} | stage={run.get('stage_name')} | stages={run.get('stages')} | tag={tag}")
            execute_run_with_retry(
                root=root,
                python_exe=args.python_exe,
                runs=runs,
                run=run,
                run_cfg=cfg,
                run_cfg_path=cfg_path,
                run_cfg_lookup=run_cfg_lookup,
                run_state_map=run_state_map,
                meta=meta,
                dry_run=args.dry_run,
            )
            save_run_status_map(status_json, run_state_map)

    if not args.no_summarize:
        summary_csv = root / args.summary_csv
        report_md = root / args.report_md
        gate_json = root / args.gate_json
        summarize(
            root=root,
            registry=registry,
            run_cfg_lookup=run_cfg_lookup,
            run_state_map=run_state_map,
            summary_csv=summary_csv,
            report_md=report_md,
            gate_json=gate_json,
        )
        save_run_status_map(status_json, run_state_map)
        print(f"[DONE] summary: {summary_csv.as_posix()}")
        print(f"[DONE] report: {report_md.as_posix()}")
        print(f"[DONE] gate: {gate_json.as_posix()}")


if __name__ == "__main__":
    main()
