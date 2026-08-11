#!/usr/bin/env python3
"""Run stationary-prefix trimming followed by hesitation episode removal."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trimmed-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    source = args.source.resolve()
    trimmed_output = args.trimmed_output.resolve()
    output = args.output.resolve()
    run(
        [
            sys.executable,
            str(script_dir / "trim_initial_stationary.py"),
            "--source",
            str(source),
            "--output",
            str(trimmed_output),
            "--workers",
            str(args.workers),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "remove_hesitation_episodes.py"),
            "--source",
            str(trimmed_output),
            "--output",
            str(output),
        ]
    )
    shutil.rmtree(trimmed_output)
    print(f"removed intermediate dataset: {trimmed_output}", flush=True)


if __name__ == "__main__":
    main()
