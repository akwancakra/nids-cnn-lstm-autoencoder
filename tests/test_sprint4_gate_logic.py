import unittest

from scripts.sprint4 import research_runner


class TestSprint4GateLogic(unittest.TestCase):
    def test_adaptive_recall_target_from_baseline(self):
        meta = {
            'gate_rules': {'adaptive_recall': {'fixed_floor': 0.40, 'plus_baseline': 0.05}},
            'baseline_history': {'best_recall_under_cic_fpr_cap': 0.3442},
        }
        target, expr = research_runner.compute_adaptive_recall_target(meta, history_summary={})
        self.assertAlmostEqual(target, 0.40, places=6)
        self.assertIn('max(', expr)

    def test_select_stage_candidates_hard_then_fallback(self):
        meta = {
            'ranking_constraints': {
                'hard': {'cic_fpr_max': 0.10, 'cse_precision_min': 0.30},
                'fallback': {'cic_fpr_max': 0.12, 'cse_precision_min': 0.25},
            }
        }
        rows = [
            {'run_id': 'a', 'valid_metrics': True, 'cse_rec': 0.30, 'cse_f1': 0.28, 'cse_prec': 0.31, 'cic_fpr': 0.09},
            {'run_id': 'b', 'valid_metrics': True, 'cse_rec': 0.35, 'cse_f1': 0.30, 'cse_prec': 0.24, 'cic_fpr': 0.08},
        ]
        cand, mode = research_runner.select_stage_candidates(rows, meta)
        self.assertEqual(mode, 'hard')
        self.assertEqual(cand[0]['run_id'], 'a')

        rows2 = [
            {'run_id': 'c', 'valid_metrics': True, 'cse_rec': 0.31, 'cse_f1': 0.29, 'cse_prec': 0.26, 'cic_fpr': 0.11},
            {'run_id': 'd', 'valid_metrics': True, 'cse_rec': 0.29, 'cse_f1': 0.27, 'cse_prec': 0.24, 'cic_fpr': 0.09},
        ]
        cand2, mode2 = research_runner.select_stage_candidates(rows2, meta)
        self.assertEqual(mode2, 'fallback')
        self.assertEqual(cand2[0]['run_id'], 'c')

    def test_collapse_flag(self):
        meta = {'gate_rules': {'collapse_flag': {'recall_gte': 0.95, 'precision_lt': 0.20}}}
        self.assertTrue(research_runner.flag_collapse(0.96, 0.10, meta))
        self.assertFalse(research_runner.flag_collapse(0.90, 0.10, meta))


if __name__ == '__main__':
    unittest.main()
