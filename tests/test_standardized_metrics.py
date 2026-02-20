import unittest

from scripts.standardized_metrics import build_standardized_record


class TestStandardizedMetrics(unittest.TestCase):
    def test_build_standardized_record_has_required_fields(self):
        metrics = {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.9,
            "f1": 0.79,
            "roc_auc": 0.81,
            "fpr": 0.2,
        }
        record = build_standardized_record(
            dataset="CSE-CIC-IDS2018",
            metrics=metrics,
            threshold=0.123,
            threshold_method="source_percentile",
            mode="zero_shot",
            seed=1234,
            model_path="models/cnn_lstm_ae/best_model.keras",
            tag="cnn_lstm",
            code_hash="abc123",
            config_snapshot={"training": {"epochs": 10}},
        )

        self.assertEqual(record["dataset"], "CSE-CIC-IDS2018")
        self.assertEqual(record["mode"], "zero_shot")
        self.assertEqual(record["threshold_method"], "source_percentile")
        self.assertEqual(record["threshold"], 0.123)
        self.assertEqual(record["seed"], 1234)
        self.assertEqual(record["f1"], 0.79)
        self.assertEqual(record["model_path"], "models/cnn_lstm_ae/best_model.keras")
        self.assertIn("created_at", record)

    def test_build_standardized_record_maps_f1_score_alias(self):
        metrics = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "roc_auc": 1.0,
            "fpr": 0.0,
        }
        record = build_standardized_record(
            dataset="CIC-IDS2017",
            metrics=metrics,
            threshold=0.5,
            threshold_method="source_gaussian",
            mode="few_shot",
            seed=42,
            model_path="model.keras",
            tag="x",
            code_hash="h",
            config_snapshot={},
        )
        self.assertEqual(record["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
