import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sprint4 import research_runner


class TestSprint4RetryPolicy(unittest.TestCase):
    def test_classify_infra_by_return_code(self):
        err = research_runner.CommandExecutionError(['x'], 137, '', '', 1.0)
        self.assertEqual(research_runner.classify_failure_kind(err), 'infra')

    def test_classify_experiment_default(self):
        err = research_runner.CommandExecutionError(['x'], 1, 'traceback', '', 1.0)
        self.assertEqual(research_runner.classify_failure_kind(err), 'experiment')

    def test_execute_run_with_retry_infra_then_success(self):
        run = {'run_id': 'r1', 'stage_name': 'stage1', 'stages': []}
        state = {'r1': {'retry_count': 0, 'terminal_status': 'pending', 'last_error': '', 'hash_violation': False, 'wall_time_sec': 0.0, 'attempt_logs': []}}

        side_effect = [
            research_runner.CommandExecutionError(['x'], 137, '', '', 1.0),
            research_runner.CommandExecutionError(['x'], 143, '', '', 1.0),
            0.5,
        ]

        with patch('scripts.sprint4.research_runner.run_single_experiment', side_effect=side_effect):
            research_runner.execute_run_with_retry(
                root=Path('.'),
                python_exe='python',
                runs=[run],
                run=run,
                run_cfg={},
                run_cfg_path=Path('cfg.yaml'),
                run_cfg_lookup={},
                run_state_map=state,
                meta={'retry_policy': {'infra_retry_free_max': 2}},
                dry_run=False,
            )

        self.assertEqual(state['r1']['terminal_status'], 'success')
        self.assertEqual(state['r1']['retry_count'], 2)

    def test_execute_run_with_retry_infra_exhausted(self):
        run = {'run_id': 'r2', 'stage_name': 'stage1', 'stages': []}
        state = {'r2': {'retry_count': 0, 'terminal_status': 'pending', 'last_error': '', 'hash_violation': False, 'wall_time_sec': 0.0, 'attempt_logs': []}}

        side_effect = [
            research_runner.CommandExecutionError(['x'], 137, '', '', 1.0),
            research_runner.CommandExecutionError(['x'], 137, '', '', 1.0),
            research_runner.CommandExecutionError(['x'], 137, '', '', 1.0),
        ]

        with patch('scripts.sprint4.research_runner.run_single_experiment', side_effect=side_effect):
            research_runner.execute_run_with_retry(
                root=Path('.'),
                python_exe='python',
                runs=[run],
                run=run,
                run_cfg={},
                run_cfg_path=Path('cfg.yaml'),
                run_cfg_lookup={},
                run_state_map=state,
                meta={'retry_policy': {'infra_retry_free_max': 2}},
                dry_run=False,
            )

        self.assertEqual(state['r2']['terminal_status'], 'failed_infra_exhausted')


if __name__ == '__main__':
    unittest.main()
