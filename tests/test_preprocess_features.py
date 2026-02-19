import unittest
import numpy as np

from scripts.preprocess import (
    clip_features_per_column,
    compute_feature_clip_bounds,
    get_scaler,
    select_features_by_statistics,
)


class TestPreprocessFeatureSelection(unittest.TestCase):
    def test_get_scaler_supports_robust(self):
        scaler = get_scaler("robust")
        self.assertEqual(scaler.__class__.__name__, "RobustScaler")

    def test_select_features_by_statistics_drops_nzv_and_high_corr(self):
        features = ["a", "b", "c"]
        sample = [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0],
        ]
        selected, report = select_features_by_statistics(
            sample_array=sample,
            features=features,
            enable_nzv=True,
            nzv_threshold=1e-6,
            enable_corr=True,
            corr_threshold=0.95,
        )
        self.assertEqual(selected, ["a"])
        self.assertIn("c", report["dropped_nzv"])
        self.assertIn("b", report["dropped_corr"])

    def test_compute_feature_clip_bounds_and_clip_features(self):
        sample = np.array(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [1000.0, 13.0],
            ],
            dtype=np.float32,
        )
        lower, upper = compute_feature_clip_bounds(sample, quantile=0.90)
        x = np.array([[-100.0, 0.0], [2000.0, 20.0]], dtype=np.float32)
        clipped = clip_features_per_column(x, lower, upper)

        self.assertTrue(np.all(clipped >= lower[None, :] - 1e-6))
        self.assertTrue(np.all(clipped <= upper[None, :] + 1e-6))

    def test_compute_feature_clip_bounds_rejects_bad_quantile(self):
        with self.assertRaises(ValueError):
            compute_feature_clip_bounds(np.array([[1.0], [2.0]], dtype=np.float32), quantile=1.0)


if __name__ == "__main__":
    unittest.main()
