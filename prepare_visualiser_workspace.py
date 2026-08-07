#!/usr/bin/env python3
"""Adapt trajectory splitter segments to a State Visualiser workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/home/geek/share3/agilex_make_breakfast_380")
DEFAULT_SEGMENTS = PROJECT_DIR / "output" / "visualiser_segments.json"
DEFAULT_WORKSPACE = PROJECT_DIR / "output" / "visualiser_workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a State Visualiser workspace for split segments."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    if not (dataset / "meta" / "info.json").is_file():
        raise ValueError(f"Not a LeRobot dataset: {dataset}")

    payload = json.loads(args.segments.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", {})
    if not episodes:
        raise ValueError(f"No episodes in {args.segments}")
    labels = sorted(
        {
            str(segment["label"]).strip()
            for segments in episodes.values()
            for segment in segments
            if str(segment.get("label", "")).strip()
        }
    )
    if not labels:
        raise ValueError("No segment labels found")

    state_dir = workspace / ".state_visualiser" / dataset.name
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dataset = workspace / f"processed_{dataset.name}"
    adapted_segments = {
        "version": 1,
        "source_dataset": str(dataset),
        "output_dataset": str(output_dataset),
        "episodes": episodes,
    }
    config = {
        "version": 1,
        "source_dataset": str(dataset),
        "task_labels": ["default", *labels],
    }
    segments_path = state_dir / "segments.json"
    config_path = state_dir / "config.json"
    segments_path.write_text(
        json.dumps(adapted_segments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset: {dataset}")
    print(f"Workspace: {workspace}")
    print(f"Episodes: {len(episodes)}")
    print(f"Segments: {sum(len(items) for items in episodes.values())}")
    print(f"Labels: {len(labels)}")
    print(f"State: {segments_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
