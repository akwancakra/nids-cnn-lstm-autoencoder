"""Isolated research runner for Sprint2 experiments.

This runner separates config and artifacts per run_id to improve reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STAGES = {"preprocess", "train", "eval"}
MODEL_VARIANT_TO_TRAIN_SCRIPT = {
    "hybrid": "scripts/train_cnn_lstm_ae.py",
    "baseline": "scripts/train_lstm_ae.py",
}
MODEL_VARIANT_TO_SUBDIR = {
    "hybrid": "cnn_lstm_ae",
    "baseline": "lstm_ae",
}


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
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def should_run(run: dict[str, Any], selected_ids: set[str] | None) -> bool:
    if selected_ids is not None:
        return run["run_id"] in selected_ids
    return bool(run.get("active", False))


def derive_isolated_paths(run_id: str) -> dict[str, str]:
    processed_dir = Path("data") / "research" / run_id / "processed"
    return {
        "data_processed": str(processed_dir).replace("\\", "/"),
        "shard_dir": str(processed_dir / "shards").replace("\\", "/"),
        "models_dir": str(Path("models") / "research" / run_id).replace("\\", "/"),
        "results_dir": str(Path("results") / "research" / run_id).replace("\\", "/"),
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


def apply_eval_reuse_inputs(cfg: dict[str, Any], source_run_id: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("paths", {})
    out.setdefault("preprocess", {})

    source_paths = derive_isolated_paths(source_run_id)
    out["paths"]["data_processed"] = source_paths["data_processed"]
    out["preprocess"]["shard_dir"] = source_paths["shard_dir"]
    out["preprocess"]["shard_enable"] = True
    return out


def build_run_config(
    base_cfg: dict[str, Any],
    profile_cfg: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    cfg = deep_update(base_cfg, profile_cfg)
    cfg = deep_update(cfg, run_spec.get("overrides", {}))
    cfg = apply_run_isolation(cfg, run_spec["run_id"])

    reuse_source = run_spec.get("reuse_artifacts_from")
    if reuse_source:
        cfg = apply_eval_reuse_inputs(cfg, reuse_source)

    return cfg


def validate_registry(registry: dict[str, Any]) -> None:
    profiles = registry.get("profiles")
    runs = registry.get("runs")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("registry.profiles must be a non-empty mapping")
    if not isinstance(runs, list) or not runs:
        raise ValueError("registry.runs must be a non-empty list")

    seen_ids: set[str] = set()
    run_lookup: dict[str, dict[str, Any]] = {}

    for run in runs:
        run_id = str(run.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Each run must define non-empty run_id")
        if run_id in seen_ids:
            raise ValueError(f"Duplicate run_id detected: {run_id}")
        seen_ids.add(run_id)
        run_lookup[run_id] = run

        profile_name = str(run.get("profile", "")).strip()
        if profile_name not in profiles:
            raise ValueError(f"Run {run_id} references unknown profile: {profile_name}")

        stages = run.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"Run {run_id} must define non-empty stages list")
        invalid_stages = [s for s in stages if s not in ALLOWED_STAGES]
        if invalid_stages:
            raise ValueError(f"Run {run_id} has invalid stages: {invalid_stages}")

        model_variant = str(run.get("model_variant", "hybrid")).lower()
        if model_variant not in MODEL_VARIANT_TO_TRAIN_SCRIPT:
            raise ValueError(f"Run {run_id} has invalid model_variant: {model_variant}")

    for run in runs:
        run_id = run["run_id"]
        reuse_source = run.get("reuse_artifacts_from")
        if not reuse_source:
            continue
        if reuse_source not in run_lookup:
            raise ValueError(f"Run {run_id} reuse_artifacts_from references missing run: {reuse_source}")
        if reuse_source == run_id:
            raise ValueError(f"Run {run_id} cannot reuse artifacts from itself")
        stages = run.get("stages", [])
        if set(stages) != {"eval"}:
            raise ValueError(
                f"Run {run_id} with reuse_artifacts_from must be eval-only (stages=['eval'])"
            )


def generated_cfg_path(generated_dir: Path, run_id: str) -> Path:
    return generated_dir / f"{run_id}.yaml"


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
            inline_overrides = {k: v for k, v in profile_spec.items() if k != "profile_path"}
            return deep_update(file_cfg, inline_overrides)
        return deepcopy(profile_spec)

    raise ValueError(f"Unsupported profile spec type: {type(profile_spec).__name__}")


def run_tag(run: dict[str, Any]) -> str:
    return str(run.get("tag") or run["run_id"])


def model_subdir(model_variant: str) -> str:
    key = str(model_variant).lower()
    return MODEL_VARIANT_TO_SUBDIR[key]


def default_model_path_from_cfg(cfg: dict[str, Any], model_variant: str) -> str:
    model_dir = Path(cfg["paths"]["models_dir"]) / model_subdir(model_variant)
    return str(model_dir / "best_model.keras").replace("\\", "/")


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def metrics_paths(root: Path, cfg: dict[str, Any], tag: str) -> tuple[Path, Path, Path]:
    results_dir = root / Path(cfg["paths"]["results_dir"])
    metrics_dir = results_dir / "metrics"
    return (
        metrics_dir / f"{tag}_cic_metrics.json",
        metrics_dir / f"{tag}_cse_metrics.json",
        metrics_dir / f"{tag}_generalization_gap.json",
    )


def metrics_exist(root: Path, cfg: dict[str, Any], tag: str) -> bool:
    return all(path.exists() for path in metrics_paths(root, cfg, tag))


def resolve_model_path_for_eval(
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_lookup: dict[str, dict[str, Any]],
) -> str:
    explicit = run.get("model_path")
    if explicit:
        return str(explicit)

    reuse_source = run.get("reuse_artifacts_from")
    if reuse_source:
        source_run = run_lookup[reuse_source]
        source_cfg = run_cfg_lookup[reuse_source]
        source_variant = str(source_run.get("model_variant", "hybrid")).lower()
        return default_model_path_from_cfg(source_cfg, source_variant)

    variant = str(run.get("model_variant", "hybrid")).lower()
    return default_model_path_from_cfg(run_cfg, variant)


def run_single_experiment(
    root: Path,
    python_exe: str,
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_cfg_path: Path,
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_lookup: dict[str, dict[str, Any]],
    dry_run: bool,
) -> None:
    stages = run.get("stages", [])
    tag = run_tag(run)
    variant = str(run.get("model_variant", "hybrid")).lower()

    for stage in stages:
        if stage == "preprocess":
            run_cmd(
                [python_exe, "scripts/preprocess.py", "--config", str(run_cfg_path)],
                cwd=root,
                dry_run=dry_run,
            )
        elif stage == "train":
            train_script = MODEL_VARIANT_TO_TRAIN_SCRIPT[variant]
            run_cmd(
                [python_exe, train_script, "--config", str(run_cfg_path)],
                cwd=root,
                dry_run=dry_run,
            )
        elif stage == "eval":
            model_path = resolve_model_path_for_eval(run, run_cfg, run_cfg_lookup, run_lookup)
            run_cmd(
                [
                    python_exe,
                    "scripts/eval_metrics.py",
                    "--config",
                    str(run_cfg_path),
                    "--model",
                    model_path,
                    "--tag",
                    tag,
                ],
                cwd=root,
                dry_run=dry_run,
            )
        else:
            raise ValueError(f"Unsupported stage: {stage}")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize(
    root: Path,
    registry: dict[str, Any],
    run_cfg_lookup: dict[str, dict[str, Any]],
    summary_csv: Path,
    report_md: Path,
) -> None:
    meta = registry.get("meta", {})
    runs = registry["runs"]
    guardrail_cse_fpr_max = float(meta.get("guardrail_cse_fpr_max", 0.20))
    guardrail_cse_f1_min = float(meta.get("guardrail_cse_f1_min", 0.24))
    w_f1 = float(meta.get("weight_f1_gap", 0.5))
    w_auc = float(meta.get("weight_auc_gap", 0.3))
    w_acc = float(meta.get("weight_accuracy_gap", 0.2))

    rows: list[dict[str, Any]] = []
    for run in runs:
        run_id = run["run_id"]
        cfg = run_cfg_lookup[run_id]
        tag = run_tag(run)
        cic_path, cse_path, gap_path = metrics_paths(root, cfg, tag)
        status = "ok" if cic_path.exists() and cse_path.exists() and gap_path.exists() else "pending"
        cic = load_json(cic_path) if cic_path.exists() else {}
        cse = load_json(cse_path) if cse_path.exists() else {}
        gap = load_json(gap_path) if gap_path.exists() else {}
        ev = cfg.get("evaluation", {})

        row = {
            "run_id": run_id,
            "profile": run.get("profile"),
            "active": bool(run.get("active", False)),
            "stages": ",".join(run.get("stages", [])),
            "tag": tag,
            "reuse_artifacts_from": run.get("reuse_artifacts_from"),
            "model_variant": run.get("model_variant", "hybrid"),
            "mode": ev.get("mode"),
            "threshold_method": ev.get("threshold_method"),
            "threshold": to_float(gap.get("threshold")),
            "cic_f1": to_float(cic.get("f1")),
            "cic_fpr": to_float(cic.get("fpr")),
            "cic_auc": to_float(cic.get("roc_auc")),
            "cse_f1": to_float(cse.get("f1")),
            "cse_fpr": to_float(cse.get("fpr")),
            "cse_auc": to_float(cse.get("roc_auc")),
            "f1_gap": to_float(gap.get("f1_gap")),
            "auc_gap": to_float(gap.get("auc_gap")),
            "accuracy_gap": to_float(gap.get("accuracy_gap")),
            "composite_gap_score": None,
            "passes_guardrail": None,
            "status": status,
            "notes": run.get("notes", ""),
        }

        if row["f1_gap"] is not None and row["auc_gap"] is not None and row["accuracy_gap"] is not None:
            row["composite_gap_score"] = (
                w_f1 * abs(float(row["f1_gap"]))
                + w_auc * abs(float(row["auc_gap"]))
                + w_acc * abs(float(row["accuracy_gap"]))
            )
        if row["cse_f1"] is not None and row["cse_fpr"] is not None:
            row["passes_guardrail"] = (
                float(row["cse_f1"]) >= guardrail_cse_f1_min
                and float(row["cse_fpr"]) <= guardrail_cse_fpr_max
            )
        rows.append(row)

        # Per-run lightweight summary file.
        run_report = root / "reports" / "research" / run_id / "SUMMARY.md"
        run_report.parent.mkdir(parents=True, exist_ok=True)
        run_lines = [
            f"# Run Summary: {run_id}",
            "",
            f"- profile: {run.get('profile')}",
            f"- stages: {','.join(run.get('stages', []))}",
            f"- tag: {tag}",
            f"- status: {status}",
            f"- reuse_artifacts_from: {run.get('reuse_artifacts_from')}",
            "",
            "## Metrics",
            "",
            f"- cic_f1: {row.get('cic_f1')}",
            f"- cic_fpr: {row.get('cic_fpr')}",
            f"- cic_auc: {row.get('cic_auc')}",
            f"- cse_f1: {row.get('cse_f1')}",
            f"- cse_fpr: {row.get('cse_fpr')}",
            f"- cse_auc: {row.get('cse_auc')}",
            f"- f1_gap: {row.get('f1_gap')}",
            f"- auc_gap: {row.get('auc_gap')}",
            f"- accuracy_gap: {row.get('accuracy_gap')}",
            f"- composite_gap_score: {row.get('composite_gap_score')}",
            f"- passes_guardrail: {row.get('passes_guardrail')}",
            "",
        ]
        run_report.write_text("\n".join(run_lines), encoding="utf-8")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        [r for r in rows if r["status"] == "ok" and r["composite_gap_score"] is not None],
        key=lambda r: float(r["composite_gap_score"]),
    )

    lines: list[str] = []
    lines.append("# RESEARCH_REPORT_CSE_F1_SPRINT2")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- source_summary: `{summary_csv.as_posix()}`")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append(f"- cse_f1 >= {guardrail_cse_f1_min:.4f}")
    lines.append(f"- cse_fpr <= {guardrail_cse_fpr_max:.4f}")
    lines.append("")
    lines.append("## Ranking (Composite Gap, lower is better)")
    lines.append("")
    lines.append("| rank | run_id | profile | mode | threshold_method | score | cse_f1 | cse_fpr | guardrail |")
    lines.append("| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for idx, row in enumerate(ranked, start=1):
        guardrail = "PASS" if row.get("passes_guardrail") else "FAIL"
        lines.append(
            "| {rank} | {run_id} | {profile} | {mode} | {threshold_method} | {score:.4f} | {cse_f1:.4f} | {cse_fpr:.4f} | {guardrail} |".format(
                rank=idx,
                run_id=row["run_id"],
                profile=row["profile"],
                mode=row["mode"],
                threshold_method=row["threshold_method"],
                score=float(row["composite_gap_score"]),
                cse_f1=float(row["cse_f1"]),
                cse_fpr=float(row["cse_fpr"]),
                guardrail=guardrail,
            )
        )
    if not ranked:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    lines.append("")

    pending = [r["run_id"] for r in rows if r["status"] != "ok"]
    lines.append("## Pending Runs")
    lines.append("")
    if pending:
        for run_id in pending:
            lines.append(f"- {run_id}")
    else:
        lines.append("- none")
    lines.append("")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="research/sprint2/run_registry.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--generated-config-dir", default=None)
    parser.add_argument("--run-ids", default=None, help="Comma-separated run IDs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--summary-csv", default="results/research/sprint2/summary.csv")
    parser.add_argument("--report-md", default="docs/RESEARCH_REPORT_CSE_F1_SPRINT2.md")
    args = parser.parse_args()

    root = ROOT
    registry_path = root / args.registry
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    base_cfg_path = root / args.base_config
    if not base_cfg_path.exists():
        # Backward-compat fallback for existing setups.
        fallback = root / "config.yaml"
        if not fallback.exists():
            raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
        base_cfg_path = fallback

    registry = load_yaml(registry_path)
    validate_registry(registry)

    base_cfg = load_yaml(base_cfg_path)
    profiles = registry["profiles"]
    resolved_profiles: dict[str, dict[str, Any]] = {}
    for profile_name, profile_spec in profiles.items():
        resolved_profiles[profile_name] = resolve_profile_cfg(profile_spec, root)
    runs = registry["runs"]

    if args.generated_config_dir:
        generated_dir = root / args.generated_config_dir
    else:
        generated_dir = registry_path.parent / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = parse_run_ids(args.run_ids)
    run_lookup = {run["run_id"]: run for run in runs}
    run_cfg_lookup: dict[str, dict[str, Any]] = {}

    for run in runs:
        profile_cfg = resolved_profiles[run["profile"]]
        run_cfg = build_run_config(base_cfg, profile_cfg, run)
        run_cfg_lookup[run["run_id"]] = run_cfg
        save_yaml(generated_cfg_path(generated_dir, run["run_id"]), run_cfg)

    if not args.summarize_only:
        for run in runs:
            if not should_run(run, selected_ids):
                continue
            run_id = run["run_id"]
            tag = run_tag(run)
            run_cfg = run_cfg_lookup[run_id]
            run_cfg_path = generated_cfg_path(generated_dir, run_id)

            if args.skip_existing and "eval" in run.get("stages", []) and metrics_exist(root, run_cfg, tag):
                print(f"[SKIP] {run_id} metrics already exist for tag={tag}")
                continue

            print(f"[RUN] {run_id} | stages={run.get('stages', [])} | tag={tag}")
            run_single_experiment(
                root=root,
                python_exe=args.python_exe,
                run=run,
                run_cfg=run_cfg,
                run_cfg_path=run_cfg_path,
                run_cfg_lookup=run_cfg_lookup,
                run_lookup=run_lookup,
                dry_run=args.dry_run,
            )

    if not args.no_summarize:
        summary_csv = root / args.summary_csv
        report_md = root / args.report_md
        summarize(
            root=root,
            registry=registry,
            run_cfg_lookup=run_cfg_lookup,
            summary_csv=summary_csv,
            report_md=report_md,
        )
        print(f"[DONE] summary: {summary_csv.as_posix()}")
        print(f"[DONE] report: {report_md.as_posix()}")


if __name__ == "__main__":
    main()
