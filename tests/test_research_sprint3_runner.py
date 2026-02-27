import tempfile
import unittest
from pathlib import Path

from scripts import research_sprint3


class TestResearchSprint3Runner(unittest.TestCase):
    def test_apply_run_isolation_sets_expected_paths(self):
        base_cfg = {"paths": {}, "preprocess": {}}
        cfg = research_sprint3.apply_run_isolation(base_cfg, run_id="s3_demo")
        self.assertEqual(cfg["paths"]["data_processed"], "data/research/s3_demo/processed")
        self.assertEqual(cfg["preprocess"]["shard_dir"], "data/research/s3_demo/processed/shards")
        self.assertEqual(cfg["paths"]["models_dir"], "models/research/s3_demo")
        self.assertEqual(cfg["paths"]["results_dir"], "results/research/s3_demo")

    def test_compute_stage_objective_stage1(self):
        row = {
            "cic_acc": 0.91,
            "cic_prec": 0.92,
            "cic_rec": 0.93,
            "cic_f1": 0.94,
            "cse_acc": 0.85,
            "cse_prec": 0.81,
            "cse_rec": 0.82,
            "cse_f1": 0.83,
        }
        primary, secondary = research_sprint3.compute_stage_objective("stage1", row)
        self.assertAlmostEqual(primary, 0.85, places=6)
        self.assertAlmostEqual(secondary, 0.83, places=6)

    def test_compute_stage_objective_stage2(self):
        row = {
            "cic_acc": 0.91,
            "cic_prec": 0.92,
            "cic_rec": 0.93,
            "cic_f1": 0.94,
            "cse_acc": 0.85,
            "cse_prec": 0.86,
            "cse_rec": 0.87,
            "cse_f1": 0.88,
        }
        primary, secondary = research_sprint3.compute_stage_objective("stage2", row)
        self.assertAlmostEqual(primary, 0.85, places=6)
        self.assertAlmostEqual(secondary, sum(row.values()) / 8.0, places=6)

    def test_materialize_run_config_reuse_data_from(self):
        base_cfg = {
            "paths": {"data_processed": "data/processed", "models_dir": "models", "results_dir": "results"},
            "preprocess": {"shard_dir": "data/processed/shards", "shard_enable": True},
            "evaluation": {"mode": "zero_shot"},
        }
        resolved_profiles = {"p0": {}}
        runs = [
            {
                "run_id": "s3_a",
                "profile": "p0",
                "stages": ["preprocess", "train", "eval"],
                "stage_name": "stage1",
            },
            {
                "run_id": "s3_b",
                "profile": "p0",
                "stages": ["eval"],
                "stage_name": "stage2",
                "reuse_data_from": "s3_a",
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_cfg_lookup = {
                "s3_a": research_sprint3.apply_run_isolation(base_cfg, "s3_a"),
            }
            cfg = research_sprint3.materialize_run_config(
                root=root,
                base_cfg=base_cfg,
                resolved_profiles=resolved_profiles,
                runs=runs,
                run_cfg_lookup=run_cfg_lookup,
                run_spec=runs[1],
                allow_unresolved_dynamic=False,
            )
            self.assertEqual(cfg["paths"]["data_processed"], "data/research/s3_a/processed")
            self.assertEqual(cfg["paths"]["results_dir"], "results/research/s3_b")

    def test_validate_registry_rejects_invalid_stage_name(self):
        registry = {
            "profiles": {"p0": {}},
            "runs": [
                {
                    "run_id": "s3_a",
                    "profile": "p0",
                    "stages": ["eval"],
                    "stage_name": "not_stage",
                }
            ],
        }
        with self.assertRaises(ValueError):
            research_sprint3.validate_registry(registry)


if __name__ == "__main__":
    unittest.main()

