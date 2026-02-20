import unittest

from scripts.compare_ab import compare_candidate_vs_baseline


class TestCompareAB(unittest.TestCase):
    def test_candidate_wins_when_f1_higher_and_fpr_lower(self):
        baseline = [
            {"f1": 0.60, "fpr": 0.30, "dataset": "CSE-CIC-IDS2018"},
            {"f1": 0.62, "fpr": 0.28, "dataset": "CSE-CIC-IDS2018"},
        ]
        candidate = [
            {"f1": 0.70, "fpr": 0.20, "dataset": "CSE-CIC-IDS2018"},
            {"f1": 0.68, "fpr": 0.22, "dataset": "CSE-CIC-IDS2018"},
        ]

        result = compare_candidate_vs_baseline(baseline, candidate)
        self.assertTrue(result["candidate_is_better"])
        self.assertGreater(result["delta_f1_mean"], 0.0)
        self.assertLess(result["delta_fpr_mean"], 0.0)

    def test_candidate_fails_gate_when_fpr_is_worse(self):
        baseline = [{"f1": 0.60, "fpr": 0.20, "dataset": "CSE-CIC-IDS2018"}]
        candidate = [{"f1": 0.70, "fpr": 0.30, "dataset": "CSE-CIC-IDS2018"}]

        result = compare_candidate_vs_baseline(baseline, candidate)
        self.assertFalse(result["candidate_is_better"])


if __name__ == "__main__":
    unittest.main()
