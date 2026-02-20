import unittest
import tempfile
from pathlib import Path

from scripts import research_sprint2


class TestResearchSprint2Runner(unittest.TestCase):
    def test_apply_run_isolation_sets_expected_paths(self):
        base_cfg = {"paths": {}, "preprocess": {}}
        cfg = research_sprint2.apply_run_isolation(base_cfg, run_id="s2_demo")

        self.assertEqual(cfg["paths"]["data_processed"], "data/research/s2_demo/processed")
        self.assertEqual(cfg["preprocess"]["shard_dir"], "data/research/s2_demo/processed/shards")
        self.assertEqual(cfg["paths"]["models_dir"], "models/research/s2_demo")
        self.assertEqual(cfg["paths"]["results_dir"], "results/research/s2_demo")
        self.assertTrue(cfg["preprocess"]["shard_enable"])

    def test_build_run_config_merges_base_profile_and_overrides(self):
        base_cfg = {
            "paths": {"results_dir": "results"},
            "preprocess": {"random_seed": 42, "shard_enable": True},
            "evaluation": {"mode": "zero_shot", "threshold_method": "percentile"},
            "threshold": {"percentile": 99.0},
        }
        profile_cfg = {
            "evaluation": {"mode": "few_shot", "threshold_method": "target_percentile"},
            "threshold": {"percentile": 98.0},
        }
        run_spec = {
            "run_id": "s2_hybrid_fewshot",
            "profile": "hybrid_few_shot",
            "stages": ["preprocess", "train", "eval"],
            "overrides": {"preprocess": {"random_seed": 1234}},
        }

        cfg = research_sprint2.build_run_config(base_cfg, profile_cfg, run_spec)
        self.assertEqual(cfg["evaluation"]["mode"], "few_shot")
        self.assertEqual(cfg["threshold"]["percentile"], 98.0)
        self.assertEqual(cfg["preprocess"]["random_seed"], 1234)
        self.assertEqual(cfg["paths"]["results_dir"], "results/research/s2_hybrid_fewshot")

    def test_apply_eval_reuse_inputs_uses_source_data_but_keeps_own_results(self):
        cfg = research_sprint2.apply_run_isolation({"paths": {}, "preprocess": {}}, run_id="s2_eval_only")
        reused = research_sprint2.apply_eval_reuse_inputs(cfg, source_run_id="s2_source")

        self.assertEqual(reused["paths"]["data_processed"], "data/research/s2_source/processed")
        self.assertEqual(reused["preprocess"]["shard_dir"], "data/research/s2_source/processed/shards")
        self.assertEqual(reused["paths"]["results_dir"], "results/research/s2_eval_only")

    def test_validate_registry_rejects_unknown_profile(self):
        registry = {
            "profiles": {"p1": {}},
            "runs": [
                {
                    "run_id": "s2_a",
                    "profile": "missing_profile",
                    "stages": ["eval"],
                }
            ],
        }

        with self.assertRaises(ValueError):
            research_sprint2.validate_registry(registry)

    def test_validate_registry_rejects_non_eval_reuse(self):
        registry = {
            "profiles": {"p1": {}},
            "runs": [
                {
                    "run_id": "s2_a",
                    "profile": "p1",
                    "stages": ["preprocess", "train", "eval"],
                },
                {
                    "run_id": "s2_b",
                    "profile": "p1",
                    "stages": ["train", "eval"],
                    "reuse_artifacts_from": "s2_a",
                },
            ],
        }

        with self.assertRaises(ValueError):
            research_sprint2.validate_registry(registry)

    def test_resolve_profile_cfg_from_file_with_inline_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile_path = root / "profiles" / "hybrid.yaml"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(
                "evaluation:\n"
                "  mode: zero_shot\n"
                "  threshold_method: percentile\n",
                encoding="utf-8",
            )

            profile_spec = {
                "profile_path": "profiles/hybrid.yaml",
                "evaluation": {"threshold_method": "target_percentile"},
            }
            cfg = research_sprint2.resolve_profile_cfg(profile_spec, root=root)
            self.assertEqual(cfg["evaluation"]["mode"], "zero_shot")
            self.assertEqual(cfg["evaluation"]["threshold_method"], "target_percentile")


if __name__ == "__main__":
    unittest.main()
