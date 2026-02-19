import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from scripts.preprocess import fit_scaler_on_cic


class TestScalerFitModes(unittest.TestCase):
    def _write_csv(self, path: Path, rows):
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def test_auto_mode_uses_benign_only_for_standard_scaler(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / 'cic.csv'
            self._write_csv(
                csv_path,
                [
                    {'f1': 1.0, 'Label': 'BENIGN'},
                    {'f1': 2.0, 'Label': 'BENIGN'},
                    {'f1': 100.0, 'Label': 'DoS'},
                ],
            )
            scaler = StandardScaler()
            fit_scaler_on_cic(
                csv_files=[csv_path],
                features=['f1'],
                label_col='Label',
                benign_label='BENIGN',
                fillna_value=0.0,
                sample_frac=None,
                max_rows_per_file=None,
                chunksize=None,
                scaler=scaler,
                scaler_fit_mode='auto',
            )
            self.assertAlmostEqual(float(scaler.mean_[0]), 1.5, places=6)

    def test_full_benign_mode_for_robust_scaler(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / 'cic.csv'
            self._write_csv(
                csv_path,
                [
                    {'f1': 1.0, 'Label': 'BENIGN'},
                    {'f1': 2.0, 'Label': 'BENIGN'},
                    {'f1': 10.0, 'Label': 'BENIGN'},
                    {'f1': 1000.0, 'Label': 'DoS'},
                ],
            )
            scaler = RobustScaler()
            fit_scaler_on_cic(
                csv_files=[csv_path],
                features=['f1'],
                label_col='Label',
                benign_label='BENIGN',
                fillna_value=0.0,
                sample_frac=None,
                max_rows_per_file=None,
                chunksize=None,
                scaler=scaler,
                scaler_fit_mode='full_benign',
            )
            self.assertAlmostEqual(float(scaler.center_[0]), 2.0, places=6)

    def test_full_benign_raises_if_no_benign_rows(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / 'cic.csv'
            self._write_csv(
                csv_path,
                [
                    {'f1': 10.0, 'Label': 'DoS'},
                    {'f1': 20.0, 'Label': 'PortScan'},
                ],
            )
            scaler = RobustScaler()
            with self.assertRaises(ValueError):
                fit_scaler_on_cic(
                    csv_files=[csv_path],
                    features=['f1'],
                    label_col='Label',
                    benign_label='BENIGN',
                    fillna_value=0.0,
                    sample_frac=None,
                    max_rows_per_file=None,
                    chunksize=None,
                    scaler=scaler,
                    scaler_fit_mode='full_benign',
                )


if __name__ == '__main__':
    unittest.main()
