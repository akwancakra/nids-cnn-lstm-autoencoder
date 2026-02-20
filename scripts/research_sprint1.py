"""Runner and summarizer for CSE-F1 sprint1 research experiments.

Usage examples:
  python scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --dry-run
  python scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --run-ids rs02,rs03
  python scripts/research_sprint1.py --plan research/sprint1/experiments.yaml --base-config config.yaml --summarize-only
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


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
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


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_run_ids(run_ids: str | None) -> set[str] | None:
    if not run_ids:
        return None
    return {x.strip() for x in run_ids.split(",") if x.strip()}


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str | None) -> list[float]:
    out: list[float] = []
    for item in parse_csv_list(raw):
        out.append(float(item))
    return out


def should_run(exp: dict[str, Any], selected_ids: set[str] | None) -> bool:
    if selected_ids is not None:
        return exp["id"] in selected_ids
    return bool(exp.get("active", False))


def apply_isolated_paths(cfg: dict[str, Any], exp_id: str) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    data_processed = Path("data") / "research" / exp_id / "processed"
    cfg.setdefault("paths", {})
    cfg["paths"]["data_processed"] = str(data_processed).replace("\\", "/")
    cfg["paths"]["models_dir"] = str((Path("models") / "research" / exp_id)).replace("\\", "/")
    cfg["paths"]["results_dir"] = str((Path("results") / "research" / exp_id)).replace("\\", "/")

    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["shard_dir"] = str((data_processed / "shards")).replace("\\", "/")
    cfg["preprocess"]["shard_enable"] = True
    return cfg


def build_cfg(base_cfg: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    cfg = deep_update(base_cfg, exp.get("overrides", {}))
    if exp.get("isolate_paths", False):
        cfg = apply_isolated_paths(cfg, exp["id"])
    return cfg


def generated_cfg_path(generated_dir: Path, exp_id: str) -> Path:
    return generated_dir / f"{exp_id}.yaml"


def exp_tag(exp: dict[str, Any]) -> str:
    return str(exp.get("tag") or exp["id"])


def metrics_tag(exp: dict[str, Any]) -> str:
    return str(exp.get("metrics_tag") or exp_tag(exp))


def metrics_paths(root: Path, cfg: dict[str, Any], tag: str) -> tuple[Path, Path, Path]:
    results_dir = root / Path(cfg["paths"]["results_dir"])
    metrics_dir = results_dir / "metrics"
    return (
        metrics_dir / f"{tag}_cic_metrics.json",
        metrics_dir / f"{tag}_cse_metrics.json",
        metrics_dir / f"{tag}_generalization_gap.json",
    )


def metrics_exist(root: Path, cfg: dict[str, Any], tag: str) -> bool:
    return all(p.exists() for p in metrics_paths(root, cfg, tag))


def build_threshold_sweep_experiments(
    base_cfg: dict[str, Any],
    experiments: list[dict[str, Any]],
    candidate_tags: list[str],
    percentiles: list[float],
    methods: list[str],
    k_sigmas: list[float],
) -> list[dict[str, Any]]:
    by_tag: dict[str, dict[str, Any]] = {exp_tag(e): e for e in experiments}
    generated: list[dict[str, Any]] = []
    seen_ids = {e["id"] for e in experiments}

    for cand_tag in candidate_tags:
        src_exp = by_tag.get(cand_tag)
        if src_exp is None:
            print(f"[WARN] sweep candidate tag not found in plan: {cand_tag}")
            continue

        src_cfg = build_cfg(base_cfg, src_exp)
        src_eval = deepcopy(src_cfg.get("evaluation", {}))
        src_threshold = deepcopy(src_cfg.get("threshold", {}))
        model_path = str(Path(src_cfg["paths"]["models_dir"]) / "cnn_lstm_ae" / "best_model.keras").replace("\\", "/")
        base_preprocess = src_cfg.get("preprocess", {})

        for method in methods:
            m = method.strip().lower()
            if m == "target_gaussian":
                for k in k_sigmas:
                    sid = f"swp_{cand_tag}_{m}_k{str(k).replace('.', 'p')}"
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)

                    ev = deepcopy(src_eval)
                    ev["mode"] = "zero_shot"
                    ev["threshold_method"] = m
                    ev["threshold_k_sigma"] = float(k)
                    ev["few_shot_finetune_epochs"] = 0

                    generated.append(
                        {
                            "id": sid,
                            "active": True,
                            "phase": "S_sweep",
                            "description": f"Eval-only threshold sweep from {cand_tag} ({m}, k={k}).",
                            "pipeline": ["eval"],
                            "tag": sid,
                            "model_path": model_path,
                            "overrides": {
                                "paths": deepcopy(src_cfg["paths"]),
                                "preprocess": {
                                    "shard_dir": base_preprocess.get("shard_dir"),
                                    "random_seed": base_preprocess.get("random_seed", 42),
                                },
                                "evaluation": ev,
                                "threshold": deepcopy(src_threshold),
                            },
                        }
                    )
                continue

            for pct in percentiles:
                sid = f"swp_{cand_tag}_{m}_p{str(pct).replace('.', 'p')}"
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                ev = deepcopy(src_eval)
                ev["mode"] = "zero_shot"
                ev["threshold_method"] = m
                ev["few_shot_finetune_epochs"] = 0

                thr = deepcopy(src_threshold)
                thr["percentile"] = float(pct)

                generated.append(
                    {
                        "id": sid,
                        "active": True,
                        "phase": "S_sweep",
                        "description": f"Eval-only threshold sweep from {cand_tag} ({m}, p={pct}).",
                        "pipeline": ["eval"],
                        "tag": sid,
                        "model_path": model_path,
                        "overrides": {
                            "paths": deepcopy(src_cfg["paths"]),
                            "preprocess": {
                                "shard_dir": base_preprocess.get("shard_dir"),
                                "random_seed": base_preprocess.get("random_seed", 42),
                            },
                            "evaluation": ev,
                            "threshold": thr,
                        },
                    }
                )
    return generated


def run_experiment(
    root: Path,
    python_exe: str,
    cfg_path: Path,
    cfg: dict[str, Any],
    exp: dict[str, Any],
    default_model: str,
    dry_run: bool,
) -> None:
    pipeline = exp.get("pipeline", ["eval"])
    tag = exp_tag(exp)
    model_path = exp.get("model_path", default_model)

    for step in pipeline:
        if step == "preprocess":
            cmd = [python_exe, "scripts/preprocess.py", "--config", str(cfg_path)]
            run_cmd(cmd, cwd=root, dry_run=dry_run)
        elif step == "train":
            cmd = [python_exe, "scripts/train_cnn_lstm_ae.py", "--config", str(cfg_path)]
            run_cmd(cmd, cwd=root, dry_run=dry_run)
            model_path = str(Path(cfg["paths"]["models_dir"]) / "cnn_lstm_ae" / "best_model.keras")
        elif step == "eval":
            cmd = [
                python_exe,
                "scripts/eval_metrics.py",
                "--config",
                str(cfg_path),
                "--model",
                model_path,
                "--tag",
                tag,
            ]
            run_cmd(cmd, cwd=root, dry_run=dry_run)
        else:
            raise ValueError(f"Unknown pipeline step: {step}")


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def summarize(
    root: Path,
    base_cfg: dict[str, Any],
    experiments: list[dict[str, Any]],
    generated_dir: Path,
    summary_csv: Path,
    report_md: Path,
    meta: dict[str, Any],
    summary_cfg: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        cfg_path = generated_cfg_path(generated_dir, exp["id"])
        cfg = load_yaml(cfg_path) if cfg_path.exists() else build_cfg(base_cfg, exp)
        tag = metrics_tag(exp)
        cic_p, cse_p, gap_p = metrics_paths(root, cfg, tag)
        status = "ok" if (cic_p.exists() and cse_p.exists() and gap_p.exists()) else "pending"

        cic = load_json(cic_p) if cic_p.exists() else {}
        cse = load_json(cse_p) if cse_p.exists() else {}
        gap = load_json(gap_p) if gap_p.exists() else {}
        ev = cfg.get("evaluation", {})
        ff = cfg.get("preprocess", {}).get("feature_filter", {})
        sg = cfg.get("preprocess", {}).get("scale_guard", {})

        row = {
            "id": exp["id"],
            "active": bool(exp.get("active", False)),
            "phase": exp.get("phase", ""),
            "pipeline": ",".join(exp.get("pipeline", ["eval"])),
            "tag": tag,
            "mode": ev.get("mode"),
            "threshold_method": ev.get("threshold_method"),
            "threshold_k_sigma": ev.get("threshold_k_sigma"),
            "few_shot_benign_frac": ev.get("few_shot_benign_frac"),
            "few_shot_max_samples": ev.get("few_shot_max_samples"),
            "few_shot_finetune_epochs": ev.get("few_shot_finetune_epochs"),
            "few_shot_finetune_lr": ev.get("few_shot_finetune_lr"),
            "feature_filter_enable_corr": ff.get("enable_corr"),
            "feature_filter_corr_threshold": ff.get("corr_threshold"),
            "scale_guard_quantile": sg.get("raw_clip_quantile"),
            "scale_guard_clip_abs": sg.get("post_scale_clip_abs"),
            "cic_f1": to_float(cic.get("f1")),
            "cic_fpr": to_float(cic.get("fpr")),
            "cic_auc": to_float(cic.get("roc_auc")),
            "cse_f1": to_float(cse.get("f1")),
            "cse_fpr": to_float(cse.get("fpr")),
            "cse_auc": to_float(cse.get("roc_auc")),
            "f1_gap": to_float(gap.get("f1_gap")),
            "auc_gap": to_float(gap.get("auc_gap")),
            "accuracy_gap": to_float(gap.get("accuracy_gap")),
            "selected_threshold": to_float(gap.get("threshold")),
            "composite_gap_score": None,
            "passes_guardrail": None,
            "rank_gap_balanced": None,
            "status": status,
            "notes": exp.get("description", ""),
        }
        rows.append(row)

    w_f1 = float(summary_cfg["weight_f1_gap"])
    w_auc = float(summary_cfg["weight_auc_gap"])
    w_acc = float(summary_cfg["weight_accuracy_gap"])
    guardrail_cse_fpr_max = float(summary_cfg["guardrail_cse_fpr_max"])
    guardrail_cse_f1_min = float(summary_cfg["guardrail_cse_f1_min"])

    for row in rows:
        if (
            row["f1_gap"] is not None
            and row["auc_gap"] is not None
            and row["accuracy_gap"] is not None
        ):
            row["composite_gap_score"] = (
                w_f1 * abs(float(row["f1_gap"]))
                + w_auc * abs(float(row["auc_gap"]))
                + w_acc * abs(float(row["accuracy_gap"]))
            )
        if row["cse_fpr"] is not None and row["cse_f1"] is not None:
            row["passes_guardrail"] = (
                float(row["cse_fpr"]) <= guardrail_cse_fpr_max
                and float(row["cse_f1"]) >= guardrail_cse_f1_min
            )

    ranked_gap = sorted(
        [r for r in rows if r["status"] == "ok" and r["composite_gap_score"] is not None],
        key=lambda x: float(x["composite_gap_score"]),
    )
    for idx, row in enumerate(ranked_gap, start=1):
        row["rank_gap_balanced"] = idx

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    baseline_id = meta.get("baseline_id")
    baseline = next((r for r in rows if r["id"] == baseline_id and r["status"] == "ok"), None)
    valid_rows = [r for r in rows if r["status"] == "ok" and r["composite_gap_score"] is not None]
    ranked = sorted(valid_rows, key=lambda x: float(x["composite_gap_score"]))

    def guardrail_ok(r: dict[str, Any]) -> bool:
        return bool(r.get("passes_guardrail"))

    best_guardrail = next((r for r in ranked if guardrail_ok(r)), None)
    pending_ids = [r["id"] for r in rows if r["status"] != "ok"]

    lines: list[str] = []
    lines.append("# RESEARCH_REPORT_CSE_F1_SPRINT1")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- source_summary: `{summary_csv.as_posix()}`")
    lines.append("")
    lines.append("## Sprint Goal")
    lines.append("")
    lines.append("- Primary KPI: minimize domain gap using composite gap score.")
    lines.append(
        f"- Composite score = {w_f1:.2f}*|f1_gap| + {w_auc:.2f}*|auc_gap| + {w_acc:.2f}*|accuracy_gap|."
    )
    lines.append(
        f"- Guardrails: cse_fpr <= {guardrail_cse_fpr_max:.4f} and cse_f1 >= {guardrail_cse_f1_min:.4f}."
    )
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    if baseline is None:
        lines.append(f"- Baseline `{baseline_id}` not found or metrics missing.")
    else:
        lines.append(
            "- id={id} | cse_f1={cse_f1:.4f} | cse_fpr={cse_fpr:.4f} | cic_f1={cic_f1:.4f}".format(
                **baseline
            )
        )
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append(f"- total_runs_in_plan: {len(rows)}")
    lines.append(f"- completed_with_metrics: {len(valid_rows)}")
    lines.append(f"- pending: {len(pending_ids)}")
    lines.append("")
    if pending_ids:
        lines.append("Pending IDs:")
        for pid in pending_ids:
            lines.append(f"- {pid}")
        lines.append("")

    lines.append("## Ranking (by Composite Gap Score, lower is better)")
    lines.append("")
    lines.append(
        "| rank | id | phase | mode | threshold_method | score | cse_f1 | cse_fpr | f1_gap | auc_gap | guardrail |"
    )
    lines.append("| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in ranked[:10]:
        gr = "PASS" if guardrail_ok(r) else "FAIL"
        lines.append(
            "| {rank} | {id} | {phase} | {mode} | {threshold_method} | {score:.4f} | {cse_f1:.4f} | {cse_fpr:.4f} | {f1_gap:.4f} | {auc_gap:.4f} | {gr} |".format(
                rank=int(r["rank_gap_balanced"]),
                id=r["id"],
                phase=r["phase"],
                mode=r["mode"],
                threshold_method=r["threshold_method"],
                score=float(r["composite_gap_score"]),
                cse_f1=float(r["cse_f1"]),
                cse_fpr=float(r["cse_fpr"]),
                f1_gap=float(r["f1_gap"]) if r["f1_gap"] is not None else float("nan"),
                auc_gap=float(r["auc_gap"]) if r["auc_gap"] is not None else float("nan"),
                gr=gr,
            )
        )
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if best_guardrail is None:
        lines.append(
            "- No configuration passes guardrails yet. Run pending experiments first, then reassess."
        )
    else:
        lines.append(
            "- Current best guardrail-safe candidate: `{}` (score={:.4f}, cse_f1={:.4f}, cse_fpr={:.4f}, f1_gap={:.4f}, auc_gap={:.4f}).".format(
                best_guardrail["id"],
                float(best_guardrail["composite_gap_score"]),
                float(best_guardrail["cse_f1"]),
                float(best_guardrail["cse_fpr"]),
                float(best_guardrail["f1_gap"]),
                float(best_guardrail["auc_gap"]),
            )
        )
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Complete pending runs in the plan or sweep set.")
    lines.append("- Re-run top-2 gap-balanced candidates once for variance check.")
    lines.append("- If no guardrail-safe gains, run one preprocess retrain candidate only.")
    lines.append("")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="research/sprint1/experiments.yaml")
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--default-model", default="models/cnn_lstm_ae/best_model.keras")
    parser.add_argument("--generated-config-dir", default="research/sprint1/generated_configs")
    parser.add_argument("--run-ids", default=None, help="Comma-separated experiment IDs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--summary-csv", default="results/metrics/summary_sprint1.csv")
    parser.add_argument("--report-md", default="docs/RESEARCH_REPORT_CSE_F1_SPRINT1.md")
    parser.add_argument("--sweep-threshold", action="store_true")
    parser.add_argument("--candidate-tags", default=None, help="Comma-separated source tags for eval-only sweep")
    parser.add_argument("--sweep-percentiles", default=None, help="Comma-separated percentiles for threshold sweep")
    parser.add_argument("--sweep-methods", default=None, help="Comma-separated methods for threshold sweep")
    parser.add_argument("--sweep-k-sigmas", default=None, help="Comma-separated k-sigma values for target_gaussian")
    parser.add_argument("--guardrail-cse-fpr", type=float, default=None)
    parser.add_argument("--guardrail-cse-f1", type=float, default=None)
    parser.add_argument("--weight-f1-gap", type=float, default=None)
    parser.add_argument("--weight-auc-gap", type=float, default=None)
    parser.add_argument("--weight-accuracy-gap", type=float, default=None)
    args = parser.parse_args()

    root = ROOT
    base_cfg = load_yaml(root / args.base_config)
    plan = load_yaml(root / args.plan)
    experiments: list[dict[str, Any]] = plan.get("experiments", [])
    meta: dict[str, Any] = plan.get("meta", {})
    generated_dir = root / args.generated_config_dir
    generated_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_threshold:
        default_candidates = [exp_tag(e) for e in experiments if str(e.get("phase", "")).startswith("B_")]
        candidate_tags = parse_csv_list(args.candidate_tags) or meta.get("sweep_candidate_tags", default_candidates)
        percentiles = parse_csv_floats(args.sweep_percentiles) or [97.0, 98.0, 99.0, 99.5]
        methods = parse_csv_list(args.sweep_methods) or ["percentile", "target_percentile", "target_gaussian"]
        k_sigmas = parse_csv_floats(args.sweep_k_sigmas) or [1.8, 2.0, 2.2]
        sweep_experiments = build_threshold_sweep_experiments(
            base_cfg=base_cfg,
            experiments=experiments,
            candidate_tags=candidate_tags,
            percentiles=percentiles,
            methods=methods,
            k_sigmas=k_sigmas,
        )
        experiments = experiments + sweep_experiments
        print(
            f"[INFO] threshold sweep generated runs={len(sweep_experiments)} "
            f"for tags={','.join(candidate_tags)}"
        )

    summary_cfg = {
        "guardrail_cse_fpr_max": args.guardrail_cse_fpr
        if args.guardrail_cse_fpr is not None
        else float(meta.get("guardrail_cse_fpr_max", 0.20)),
        "guardrail_cse_f1_min": args.guardrail_cse_f1
        if args.guardrail_cse_f1 is not None
        else float(meta.get("guardrail_cse_f1_min", 0.24)),
        "weight_f1_gap": args.weight_f1_gap
        if args.weight_f1_gap is not None
        else float(meta.get("weight_f1_gap", 0.5)),
        "weight_auc_gap": args.weight_auc_gap
        if args.weight_auc_gap is not None
        else float(meta.get("weight_auc_gap", 0.3)),
        "weight_accuracy_gap": args.weight_accuracy_gap
        if args.weight_accuracy_gap is not None
        else float(meta.get("weight_accuracy_gap", 0.2)),
    }

    selected_ids = parse_run_ids(args.run_ids)

    if not args.summarize_only:
        for exp in experiments:
            if not should_run(exp, selected_ids):
                continue
            cfg = build_cfg(base_cfg, exp)
            cfg_path = generated_cfg_path(generated_dir, exp["id"])
            save_yaml(cfg_path, cfg)

            tag = exp_tag(exp)
            if args.skip_existing and metrics_exist(root, cfg, tag):
                print(f"[SKIP] {exp['id']} metrics already exist for tag={tag}")
                continue

            print(f"[RUN] {exp['id']} | pipeline={exp.get('pipeline', ['eval'])} | tag={tag}")
            run_experiment(
                root=root,
                python_exe=args.python_exe,
                cfg_path=cfg_path,
                cfg=cfg,
                exp=exp,
                default_model=args.default_model,
                dry_run=args.dry_run,
            )

    if not args.no_summarize:
        summarize(
            root=root,
            base_cfg=base_cfg,
            experiments=experiments,
            generated_dir=generated_dir,
            summary_csv=root / args.summary_csv,
            report_md=root / args.report_md,
            meta=meta,
            summary_cfg=summary_cfg,
        )
        print(f"[DONE] summary: {args.summary_csv}")
        print(f"[DONE] report: {args.report_md}")


if __name__ == "__main__":
    main()
