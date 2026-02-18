import unittest

from scripts.preprocess import get_scaler, select_features_by_statistics


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


if __name__ == "__main__":
    unittest.main()
