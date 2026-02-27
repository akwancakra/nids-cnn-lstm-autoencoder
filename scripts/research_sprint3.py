"""Research runner for Sprint 3 recovery (CNN-LSTM AE, zero-shot, strict split)."""

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
ALLOWED_STAGE_NAMES = {"stage0", "stage1", "stage2", "stage3", "stage4"}
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


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_csv_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


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


def apply_data_reuse_inputs(cfg: dict[str, Any], source_run_id: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("paths", {})
    out.setdefault("preprocess", {})
    source_paths = derive_isolated_paths(source_run_id)
    out["paths"]["data_processed"] = source_paths["data_processed"]
    out["preprocess"]["shard_dir"] = source_paths["shard_dir"]
    out["preprocess"]["shard_enable"] = True
    return out


def run_tag(run: dict[str, Any]) -> str:
    return str(run.get("tag") or run["run_id"])


def model_subdir(model_variant: str) -> str:
    key = str(model_variant).lower()
    return MODEL_VARIANT_TO_SUBDIR[key]


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


def extract_metric_rows(cic: dict[str, Any], cse: dict[str, Any]) -> dict[str, float | None]:
    cic_acc = to_float(cic.get("accuracy"))
    cic_prec = to_float(cic.get("precision"))
    cic_rec = to_float(cic.get("recall"))
    cic_f1 = to_float(cic.get("f1", cic.get("f1_score")))
    cse_acc = to_float(cse.get("accuracy"))
    cse_prec = to_float(cse.get("precision"))
    cse_rec = to_float(cse.get("recall"))
    cse_f1 = to_float(cse.get("f1", cse.get("f1_score")))
    return {
        "cic_acc": cic_acc,
        "cic_prec": cic_prec,
        "cic_rec": cic_rec,
        "cic_f1": cic_f1,
        "cse_acc": cse_acc,
        "cse_prec": cse_prec,
        "cse_rec": cse_rec,
        "cse_f1": cse_f1,
    }


def compute_stage_objective(stage_name: str, metrics_row: dict[str, float | None]) -> tuple[float | None, float | None]:
    if stage_name == "stage1":
        vals = [
            metrics_row["cic_acc"],
            metrics_row["cic_prec"],
            metrics_row["cic_rec"],
            metrics_row["cic_f1"],
            metrics_row["cse_acc"],
        ]
        if any(v is None for v in vals):
            return None, None
        return min(vals), float(metrics_row["cse_f1"]) if metrics_row["cse_f1"] is not None else 0.0

    vals = [
        metrics_row["cic_acc"],
        metrics_row["cic_prec"],
        metrics_row["cic_rec"],
        metrics_row["cic_f1"],
        metrics_row["cse_acc"],
        metrics_row["cse_prec"],
        metrics_row["cse_rec"],
        metrics_row["cse_f1"],
    ]
    if any(v is None for v in vals):
        return None, None
    return min(vals), sum(vals) / len(vals)


def build_run_row(
    root: Path,
    run: dict[str, Any],
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    tag = run_tag(run)
    cic_path, cse_path, gap_path = metrics_paths(root, run_cfg, tag)
    status = "ok" if cic_path.exists() and cse_path.exists() and gap_path.exists() else "pending"
    cic = load_json(cic_path) if cic_path.exists() else {}
    cse = load_json(cse_path) if cse_path.exists() else {}
    gap = load_json(gap_path) if gap_path.exists() else {}
    row = {
        "run_id": run["run_id"],
        "stage_name": run.get("stage_name", ""),
        "profile": run.get("profile"),
        "model_variant": run.get("model_variant", "hybrid"),
        "active": bool(run.get("active", False)),
        "stages": ",".join(run.get("stages", [])),
        "tag": tag,
        "status": status,
        "threshold": to_float(gap.get("threshold")),
        "score_mode": gap.get("score_mode"),
        "threshold_method": gap.get("threshold_method"),
    }
    row.update(extract_metric_rows(cic, cse))
    row["f1_gap"] = to_float(gap.get("f1_gap"))
    row["auc_gap"] = to_float(gap.get("auc_gap"))
    row["accuracy_gap"] = to_float(gap.get("accuracy_gap"))
    primary, secondary = compute_stage_objective(row["stage_name"], row)
    row["stage_primary"] = primary
    row["stage_secondary"] = secondary
    return row


def select_best_run_id_for_stage(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    stage_name: str,
    rank: int = 1,
) -> str | None:
    candidates = []
    for run in runs:
        if run.get("stage_name") != stage_name:
            continue
        cfg = run_cfg_lookup.get(run["run_id"])
        if cfg is None:
            continue
        row = build_run_row(root, run, cfg)
        if row["status"] != "ok" or row["stage_primary"] is None:
            continue
        candidates.append((run["run_id"], float(row["stage_primary"]), float(row["stage_secondary"])))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    idx = max(0, rank - 1)
    if idx >= len(candidates):
        return None
    return candidates[idx][0]


def resolve_model_path_for_eval(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
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
        selected = select_best_run_id_for_stage(root, runs, run_cfg_lookup, source_stage, rank=rank)
        if not selected:
            raise ValueError(
                f"Run {run['run_id']} requires best model from stage={source_stage}, but no completed run found."
            )
        source_cfg = run_cfg_lookup[selected]
        source_spec = next(r for r in runs if r["run_id"] == selected)
        source_variant = str(source_spec.get("model_variant", "hybrid")).lower()
        return default_model_path_from_cfg(source_cfg, source_variant)

    variant = str(run.get("model_variant", "hybrid")).lower()
    return default_model_path_from_cfg(run_cfg, variant)


def should_run(run: dict[str, Any], selected_ids: set[str] | None, selected_stage_names: set[str] | None) -> bool:
    if selected_ids is not None and run["run_id"] not in selected_ids:
        return False
    if selected_stage_names is not None and run.get("stage_name") not in selected_stage_names:
        return False
    return bool(run.get("active", False))


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


def materialize_run_config(
    root: Path,
    base_cfg: dict[str, Any],
    resolved_profiles: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    run_spec: dict[str, Any],
    allow_unresolved_dynamic: bool = False,
) -> dict[str, Any]:
    run_id = run_spec["run_id"]
    profile_name = str(run_spec.get("profile", "")).strip()
    if profile_name not in resolved_profiles:
        raise ValueError(f"Run {run_id} references unknown profile: {profile_name}")

    profile_cfg = resolved_profiles[profile_name]
    cfg_base = deep_update(base_cfg, profile_cfg)

    template_stage = run_spec.get("template_from_best_stage")
    if template_stage:
        rank = int(run_spec.get("reuse_stage_rank", 1))
        source_run_id = select_best_run_id_for_stage(root, runs, run_cfg_lookup, template_stage, rank=rank)
        if source_run_id and source_run_id in run_cfg_lookup:
            cfg_base = deepcopy(run_cfg_lookup[source_run_id])
        elif not allow_unresolved_dynamic:
            raise ValueError(
                f"Run {run_id} requires template_from_best_stage={template_stage}, but no completed run is available."
            )

    cfg = deep_update(cfg_base, run_spec.get("overrides", {}))
    cfg = apply_run_isolation(cfg, run_id)

    reuse_data_from = run_spec.get("reuse_data_from")
    if reuse_data_from:
        cfg = apply_data_reuse_inputs(cfg, str(reuse_data_from))

    reuse_data_stage = run_spec.get("reuse_data_from_best_stage")
    if reuse_data_stage:
        rank = int(run_spec.get("reuse_stage_rank", 1))
        source_run_id = select_best_run_id_for_stage(root, runs, run_cfg_lookup, reuse_data_stage, rank=rank)
        if source_run_id:
            cfg = apply_data_reuse_inputs(cfg, source_run_id)
        elif not allow_unresolved_dynamic:
            raise ValueError(
                f"Run {run_id} requires best data from stage={reuse_data_stage}, but no completed run found."
            )

    return cfg


def run_single_experiment(
    root: Path,
    python_exe: str,
    runs: list[dict[str, Any]],
    run: dict[str, Any],
    run_cfg: dict[str, Any],
    run_cfg_path: Path,
    run_cfg_lookup: dict[str, dict[str, Any]],
    dry_run: bool,
) -> None:
    stages = run.get("stages", [])
    variant = str(run.get("model_variant", "hybrid")).lower()
    tag = run_tag(run)

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
            model_path = resolve_model_path_for_eval(root, runs, run_cfg_lookup, run, run_cfg)
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


def validate_registry(registry: dict[str, Any]) -> None:
    profiles = registry.get("profiles")
    runs = registry.get("runs")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("registry.profiles must be a non-empty mapping")
    if not isinstance(runs, list) or not runs:
        raise ValueError("registry.runs must be a non-empty list")

    seen_ids: set[str] = set()
    for run in runs:
        run_id = str(run.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Each run must define non-empty run_id")
        if run_id in seen_ids:
            raise ValueError(f"Duplicate run_id detected: {run_id}")
        seen_ids.add(run_id)

        profile_name = str(run.get("profile", "")).strip()
        if profile_name not in profiles:
            raise ValueError(f"Run {run_id} references unknown profile: {profile_name}")

        stage_name = str(run.get("stage_name", "")).strip()
        if stage_name and stage_name not in ALLOWED_STAGE_NAMES:
            raise ValueError(f"Run {run_id} has invalid stage_name={stage_name}")

        stages = run.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"Run {run_id} must define non-empty stages list")
        invalid_stages = [s for s in stages if s not in ALLOWED_STAGES]
        if invalid_stages:
            raise ValueError(f"Run {run_id} has invalid stages: {invalid_stages}")

        model_variant = str(run.get("model_variant", "hybrid")).lower()
        if model_variant not in MODEL_VARIANT_TO_TRAIN_SCRIPT:
            raise ValueError(f"Run {run_id} has invalid model_variant: {model_variant}")


def summarize(
    root: Path,
    runs: list[dict[str, Any]],
    run_cfg_lookup: dict[str, dict[str, Any]],
    summary_csv: Path,
    report_md: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = run_cfg_lookup.get(run["run_id"])
        if cfg is None:
            continue
        rows.append(build_run_row(root, run, cfg))

    if not rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", encoding="utf-8") as f:
            f.write("run_id,status\n")
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text("# RESEARCH_REPORT_CSE_F1_SPRINT3\n\n- no rows generated\n", encoding="utf-8")
        return

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok" and r["stage_primary"] is not None]
    lines = [
        "# RESEARCH_REPORT_CSE_F1_SPRINT3",
        "",
        f"- generated_at: {datetime.now().isoformat()}",
        f"- source_summary: `{summary_csv.as_posix()}`",
        "",
        "## Stage Best Candidates",
        "",
        "| stage | run_id | primary | secondary | cse_f1 | cse_acc | cic_f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for stage_name in ["stage0", "stage1", "stage2", "stage3", "stage4"]:
        stage_rows = [r for r in ok_rows if r["stage_name"] == stage_name]
        if not stage_rows:
            lines.append(f"| {stage_name} | - | - | - | - | - | - |")
            continue
        stage_rows.sort(key=lambda r: (float(r["stage_primary"]), float(r["stage_secondary"])), reverse=True)
        best = stage_rows[0]
        lines.append(
            "| {stage} | {run_id} | {primary:.4f} | {secondary:.4f} | {cse_f1:.4f} | {cse_acc:.4f} | {cic_f1:.4f} |".format(
                stage=stage_name,
                run_id=best["run_id"],
                primary=float(best["stage_primary"]),
                secondary=float(best["stage_secondary"]),
                cse_f1=float(best["cse_f1"]),
                cse_acc=float(best["cse_acc"]),
                cic_f1=float(best["cic_f1"]),
            )
        )

    lines.extend(
        [
            "",
            "## Run Status",
            "",
            f"- total_runs: {len(rows)}",
            f"- completed_with_metrics: {len([r for r in rows if r['status'] == 'ok'])}",
            f"- pending: {len([r for r in rows if r['status'] != 'ok'])}",
            "",
        ]
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def generated_cfg_path(generated_dir: Path, run_id: str) -> Path:
    return generated_dir / f"{run_id}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="research/sprint3/run_registry.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--generated-config-dir", default=None)
    parser.add_argument("--run-ids", default=None, help="Comma-separated run IDs")
    parser.add_argument("--stage-names", default=None, help="Comma-separated stage names (stage0..stage4)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--summary-csv", default="results/research/sprint3/summary.csv")
    parser.add_argument("--report-md", default="docs/RESEARCH_REPORT_CSE_F1_SPRINT3.md")
    args = parser.parse_args()

    root = ROOT
    registry_path = root / args.registry
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    base_cfg_path = root / args.base_config
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    registry = load_yaml(registry_path)
    validate_registry(registry)

    base_cfg = load_yaml(base_cfg_path)
    profiles = registry["profiles"]
    runs = registry["runs"]
    resolved_profiles: dict[str, dict[str, Any]] = {}
    for profile_name, profile_spec in profiles.items():
        resolved_profiles[profile_name] = resolve_profile_cfg(profile_spec, root)

    if args.generated_config_dir:
        generated_dir = root / args.generated_config_dir
    else:
        generated_dir = registry_path.parent / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = parse_csv_set(args.run_ids)
    selected_stage_names = parse_csv_set(args.stage_names)
    run_cfg_lookup: dict[str, dict[str, Any]] = {}

    # Initial pass builds static configs to support summarize and path lookups.
    for run in runs:
        cfg = materialize_run_config(
            root=root,
            base_cfg=base_cfg,
            resolved_profiles=resolved_profiles,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            run_spec=run,
            allow_unresolved_dynamic=True,
        )
        run_cfg_lookup[run["run_id"]] = cfg
        save_yaml(generated_cfg_path(generated_dir, run["run_id"]), cfg)

    if not args.summarize_only:
        for run in runs:
            if not should_run(run, selected_ids, selected_stage_names):
                continue

            run_id = run["run_id"]
            cfg = materialize_run_config(
                root=root,
                base_cfg=base_cfg,
                resolved_profiles=resolved_profiles,
                runs=runs,
                run_cfg_lookup=run_cfg_lookup,
                run_spec=run,
                allow_unresolved_dynamic=False,
            )
            run_cfg_lookup[run_id] = cfg
            cfg_path = generated_cfg_path(generated_dir, run_id)
            save_yaml(cfg_path, cfg)

            tag = run_tag(run)
            if args.skip_existing and "eval" in run.get("stages", []) and metrics_exist(root, cfg, tag):
                print(f"[SKIP] {run_id} metrics already exist for tag={tag}")
                continue

            print(f"[RUN] {run_id} | stage_name={run.get('stage_name')} | stages={run.get('stages', [])} | tag={tag}")
            run_single_experiment(
                root=root,
                python_exe=args.python_exe,
                runs=runs,
                run=run,
                run_cfg=cfg,
                run_cfg_path=cfg_path,
                run_cfg_lookup=run_cfg_lookup,
                dry_run=args.dry_run,
            )

    if not args.no_summarize:
        summary_csv = root / args.summary_csv
        report_md = root / args.report_md
        summarize(
            root=root,
            runs=runs,
            run_cfg_lookup=run_cfg_lookup,
            summary_csv=summary_csv,
            report_md=report_md,
        )
        print(f"[DONE] summary: {summary_csv.as_posix()}")
        print(f"[DONE] report: {report_md.as_posix()}")


if __name__ == "__main__":
    main()

