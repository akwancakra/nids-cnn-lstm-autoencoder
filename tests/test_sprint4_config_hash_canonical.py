import unittest

from scripts.sprint4 import research_runner


class TestSprint4ConfigHashCanonical(unittest.TestCase):
    def test_hash_is_stable_across_key_order(self):
        a = {'z': 1, 'a': {'y': 2, 'x': 3}}
        b = {'a': {'x': 3, 'y': 2}, 'z': 1}
        self.assertEqual(research_runner.hash_dict_sha256(a), research_runner.hash_dict_sha256(b))

    def test_effective_hash_ignores_runtime_contract(self):
        base = {'paths': {'x': 'y'}, 'evaluation': {'sample_size': 0}}
        h1 = research_runner.compute_effective_config_hash(base)

        with_contract = {
            'paths': {'x': 'y'},
            'evaluation': {'sample_size': 0},
            'runtime_contract': {'base_config_hash': 'a', 'effective_config_hash': 'b'},
        }
        h2 = research_runner.compute_effective_config_hash(with_contract)
        self.assertEqual(h1, h2)

    def test_add_contract_hashes(self):
        cfg = {'paths': {'x': 'y'}}
        out = research_runner.add_contract_hashes(cfg, base_hash='base123')
        self.assertIn('runtime_contract', out)
        self.assertEqual(out['runtime_contract']['base_config_hash'], 'base123')
        self.assertTrue(isinstance(out['runtime_contract']['effective_config_hash'], str))


if __name__ == '__main__':
    unittest.main()
