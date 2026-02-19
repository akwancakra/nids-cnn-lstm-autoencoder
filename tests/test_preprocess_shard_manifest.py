import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.preprocess import ShardWriter


class TestShardManifestMetadata(unittest.TestCase):
    def test_manifest_contains_domain_and_source_file_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "shards"
            split = root / "cic" / "test"
            writer = ShardWriter(
                root_dir=root,
                split_dir=split,
                prefix="cic_test",
                shard_size=2,
                with_labels=True,
                input_shape=(10, 3),
                domain_id="cic",
            )

            x1 = np.zeros((2, 10, 3), dtype=np.float32)
            y1 = np.zeros((2,), dtype=np.int32)
            writer.add(x1, y1, source_file_group="file_a")

            x2 = np.ones((2, 10, 3), dtype=np.float32)
            y2 = np.ones((2,), dtype=np.int32)
            writer.add(x2, y2, source_file_group="file_b")

            manifest = writer.finalize()
            self.assertEqual(manifest["domain_id"], "cic")
            self.assertEqual(manifest["source_file_groups"], ["file_a", "file_b"])
            self.assertEqual(len(manifest["shards"]), 2)
            self.assertEqual(manifest["shards"][0]["source_file_group"], "file_a")
            self.assertEqual(manifest["shards"][0]["domain_id"], "cic")
            self.assertEqual(manifest["shards"][1]["source_file_group"], "file_b")
            self.assertEqual(manifest["shards"][1]["domain_id"], "cic")


if __name__ == "__main__":
    unittest.main()
