import unittest
import numpy as np

from scripts.eval_metrics import compute_threshold_value


class TestThresholdingModes(unittest.TestCase):
    def test_source_percentile_threshold(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        thr = compute_threshold_value(
            method="source_percentile",
            source_errors=errors,
            source_labels=None,
            target_benign_errors=None,
            percentile=75,
            k_sigma=2.5,
        )
        self.assertAlmostEqual(thr, 0.325, places=6)

    def test_source_gaussian_threshold(self):
        errors = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        thr = compute_threshold_value(
            method="source_gaussian",
            source_errors=errors,
            source_labels=None,
            target_benign_errors=None,
            percentile=95,
            k_sigma=1.0,
        )
        self.assertAlmostEqual(thr, float(np.mean(errors) + np.std(errors)), places=6)

    def test_source_calib_f1_uses_source_labels(self):
        errors = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        thr = compute_threshold_value(
            method="source_calib_f1",
            source_errors=errors,
            source_labels=labels,
            target_benign_errors=None,
            percentile=95,
            k_sigma=2.0,
        )
        # Any midpoint in (0.2, 0.8) yields perfect split.
        self.assertGreater(thr, 0.2)
        self.assertLess(thr, 0.8)

    def test_source_calib_guardrail_respects_fpr(self):
        errors = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        thr = compute_threshold_value(
            method="source_calib_guardrail",
            source_errors=errors,
            source_labels=labels,
            target_benign_errors=None,
            percentile=95,
            k_sigma=2.0,
            guardrail_fpr_max=0.0,
        )
        self.assertGreater(thr, 0.2)

    def test_source_calib_requires_labels(self):
        with self.assertRaises(ValueError):
            compute_threshold_value(
                method="source_calib_f1",
                source_errors=np.array([0.1, 0.2], dtype=np.float32),
                source_labels=None,
                target_benign_errors=None,
                percentile=95,
                k_sigma=2.5,
            )

    def test_percentile_threshold(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        thr = compute_threshold_value(
            method="percentile",
            source_errors=errors,
            source_labels=None,
            target_benign_errors=None,
            percentile=75,
            k_sigma=2.5,
        )
        self.assertAlmostEqual(thr, 0.325, places=6)

    def test_target_percentile_requires_target_errors(self):
        with self.assertRaises(ValueError):
            compute_threshold_value(
                method="target_percentile",
                source_errors=np.array([0.1, 0.2]),
                source_labels=None,
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
            source_labels=None,
            target_benign_errors=target,
            percentile=95,
            k_sigma=2.0,
        )
        self.assertAlmostEqual(thr, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
