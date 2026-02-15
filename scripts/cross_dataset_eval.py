"""Cross-dataset evaluation wrapper (CIC-IDS2017 -> CSE-CIC-IDS2018)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_metrics import main as eval_main


if __name__ == "__main__":
    eval_main()