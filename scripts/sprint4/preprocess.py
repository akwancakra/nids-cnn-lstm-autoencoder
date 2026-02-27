"""Sprint 4 wrapper for preprocess stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cmd = [sys.executable, "scripts/preprocess.py", "--config", args.config]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
