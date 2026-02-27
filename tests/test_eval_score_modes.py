import unittest
import numpy as np

from scripts.eval_metrics import compute_reconstruction_scores, combine_hybrid_scores


class TestEvalScoreModes(unittest.TestCase):
    def test_recon_mse_scores(self):
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        preds = np.array([[[0.0, 2.0], [3.0, 2.0]]], dtype=np.float32)
        scores = compute_reconstruction_scores(x, preds, score_mode="recon_mse")
        self.assertAlmostEqual(float(scores[0]), 1.25, places=6)

    def test_recon_huber_scores(self):
        x = np.array([[[1.0], [3.0]]], dtype=np.float32)
        preds = np.array([[[0.0], [2.0]]], dtype=np.float32)
        scores = compute_reconstruction_scores(x, preds, score_mode="recon_huber")
        self.assertTrue(float(scores[0]) > 0.0)

    def test_hybrid_score_combination(self):
        recon = np.array([0.2, 0.8], dtype=np.float32)
        latent = np.array([0.1, 0.3], dtype=np.float32)
        out = combine_hybrid_scores(recon, latent, alpha=0.7)
        np.testing.assert_allclose(out, np.array([0.17, 0.65], dtype=np.float32), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

