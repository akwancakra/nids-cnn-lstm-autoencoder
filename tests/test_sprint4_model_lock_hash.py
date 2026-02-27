import tempfile
import unittest
from pathlib import Path

from scripts.sprint4 import research_runner


class TestSprint4ModelLockHash(unittest.TestCase):
    def test_prepare_and_verify_locked_model(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'model.keras'
            src.write_bytes(b'abc123')

            locked, pre_hash = research_runner.prepare_stage3_locked_model(src, 'run_a')
            self.assertTrue(locked.exists())
            self.assertTrue(research_runner.verify_stage3_locked_model(locked, pre_hash))

    def test_verify_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'model.keras'
            src.write_bytes(b'abc123')

            locked, pre_hash = research_runner.prepare_stage3_locked_model(src, 'run_b')
            locked.chmod(0o644)
            with locked.open('ab') as f:
                f.write(b'changed')

            self.assertFalse(research_runner.verify_stage3_locked_model(locked, pre_hash))


if __name__ == '__main__':
    unittest.main()
