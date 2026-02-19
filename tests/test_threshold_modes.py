import unittest
import numpy as np

from scripts.eval_metrics import compute_threshold_value, validate_eval_policy


class TestThresholdingModes(unittest.TestCase):
    def test_source_percentile_threshold(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        thr = compute_threshold_value(
            method="source_percentile",
            source_errors=errors,
            target_benign_errors=None,
            percentile=75,
            k_sigma=2.5,
        )
        self.assertAlmostEqual(thr, 0.325, places=6)

    def test_source_gaussian_k_sigma(self):
        source = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        thr = compute_threshold_value(
            method="source_gaussian",
            source_errors=source,
            target_benign_errors=None,
            percentile=95,
            k_sigma=2.0,
        )
        self.assertAlmostEqual(thr, float(np.mean(source) + 2.0 * np.std(source)), places=6)

    def test_source_evt_returns_finite_value(self):
        source = np.array([0.1, 0.2, 0.3, 0.8, 1.2, 2.5], dtype=np.float32)
        thr = compute_threshold_value(
            method="source_evt",
            source_errors=source,
            target_benign_errors=None,
            percentile=95,
            k_sigma=2.0,
        )
        self.assertTrue(np.isfinite(thr))
        self.assertGreater(thr, 0.0)

    def test_target_percentile_requires_target_errors(self):
        with self.assertRaises(ValueError):
            compute_threshold_value(
                method="target_percentile",
                source_errors=np.array([0.1, 0.2]),
                target_benign_errors=None,
                percentile=95,
                k_sigma=2.5,
            )

    def test_target_gaussian_k_sigma(self):
        source = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        target = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        thr = compute_threshold_value(
            method="target_gaussian",
            source_errors=source,
            target_benign_errors=target,
            percentile=95,
            k_sigma=2.0,
        )
        self.assertAlmostEqual(thr, 0.2, places=6)

    def test_zero_shot_strict_blocks_target_threshold_methods(self):
        with self.assertRaises(ValueError):
            validate_eval_policy(
                eval_mode="zero_shot",
                threshold_method="target_percentile",
                zero_shot_strict=True,
            )

    def test_zero_shot_strict_allows_source_threshold_methods(self):
        validate_eval_policy(
            eval_mode="zero_shot",
            threshold_method="source_percentile",
            zero_shot_strict=True,
        )


if __name__ == "__main__":
    unittest.main()
