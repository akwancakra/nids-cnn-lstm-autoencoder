"""Sprint 4 wrapper for evaluation stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/eval_metrics.py",
        "--config",
        args.config,
        "--model",
        args.model,
        "--tag",
        args.tag,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
