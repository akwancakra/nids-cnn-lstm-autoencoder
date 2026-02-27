import unittest
from scripts.sprint4 import research_runner


class TestSprint4RunnerValidity(unittest.TestCase):
    def test_valid_metrics_fields_true_when_finite(self):
        row = {
            'cse_rec': 0.4,
            'cse_prec': 0.3,
            'cse_f1': 0.34,
            'cic_fpr': 0.08,
            'cic_f1': 0.5,
        }
        self.assertTrue(research_runner.valid_metrics_fields(row))

    def test_valid_metrics_fields_false_when_missing(self):
        row = {
            'cse_rec': 0.4,
            'cse_prec': None,
            'cse_f1': 0.34,
            'cic_fpr': 0.08,
            'cic_f1': 0.5,
        }
        self.assertFalse(research_runner.valid_metrics_fields(row))

    def test_stage_validity_pass_and_inconclusive(self):
        rows = []
        for i in range(8):
            rows.append({'run_id': f'r{i}', 'stage_name': 'stage2', 'valid_metrics': i < 6})

        out = research_runner.compute_stage_validity(
            rows=rows,
            stage_rules={'stage2': 6, 'stage3': 5, 'stage4': 2},
            gate_pass=True,
        )
        self.assertTrue(out['stage2']['passed'])
        self.assertEqual(out['stage2']['status'], 'ok')

        rows_bad = []
        for i in range(8):
            rows_bad.append({'run_id': f'r{i}', 'stage_name': 'stage2', 'valid_metrics': i < 5})

        out_bad = research_runner.compute_stage_validity(
            rows=rows_bad,
            stage_rules={'stage2': 6, 'stage3': 5, 'stage4': 2},
            gate_pass=True,
        )
        self.assertFalse(out_bad['stage2']['passed'])
        self.assertEqual(out_bad['stage2']['status'], 'inconclusive')

    def test_validate_registry_rejects_invalid_condition(self):
        registry = {
            'profiles': {'p1': {}},
            'runs': [
                {
                    'run_id': 'x',
                    'profile': 'p1',
                    'stage_name': 'stage1',
                    'stages': ['eval'],
                    'condition': 'invalid_cond',
                }
            ],
        }
        with self.assertRaises(ValueError):
            research_runner.validate_registry(registry)


if __name__ == '__main__':
    unittest.main()
