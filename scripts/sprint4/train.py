"""Sprint 4 wrapper for train stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["hybrid", "baseline"])
    args = parser.parse_args()

    script = "scripts/train_cnn_lstm_ae.py" if args.variant == "hybrid" else "scripts/train_lstm_ae.py"
    cmd = [sys.executable, script, "--config", args.config]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
