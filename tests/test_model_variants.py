import unittest

import tensorflow as tf

from scripts.train_cnn_lstm_ae import build_model


def _base_cfg():
    return {
        "training": {
            "cnn_filters": [8],
            "cnn_kernel_size": 3,
            "lstm_units": [8],
            "dropout": 0.1,
            "latent_dim": 4,
        },
        "model_variant": {
            "lstm_backbone": "lstm",
            "use_temporal_attention": False,
            "multi_scale_kernels": [],
            "reconstruction_loss": "mse",
        },
    }


class TestModelVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            # Best effort only; test should still run on environments without GPU support.
            pass

    def test_bilstm_backbone_uses_bidirectional_layers(self):
        cfg = _base_cfg()
        cfg["model_variant"]["lstm_backbone"] = "bilstm"
        model = build_model((10, 6), cfg)
        has_bidir = any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers)
        self.assertTrue(has_bidir)

    def test_temporal_attention_variant_builds(self):
        cfg = _base_cfg()
        cfg["model_variant"]["use_temporal_attention"] = True
        model = build_model((10, 6), cfg)
        layer_names = [layer.name for layer in model.layers]
        self.assertIn("temporal_attention_weights", layer_names)

    def test_multi_scale_variant_preserves_output_shape(self):
        cfg = _base_cfg()
        cfg["model_variant"]["multi_scale_kernels"] = [3, 5]
        model = build_model((10, 6), cfg)
        self.assertEqual(tuple(model.output_shape[1:]), (10, 6))


if __name__ == "__main__":
    unittest.main()
