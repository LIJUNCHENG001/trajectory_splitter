#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess

import pyarrow.parquet as pq


CAMERA_KEYS = (
    "observation.image.top",
    "observation.image.left_wrist",
    "observation.image.right_wrist",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a converted LeRobot v2.1 dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--decode-samples", type=int, default=0)
    args = parser.parse_args()
    root = args.dataset.resolve()

    info = read_json(root / "meta" / "info.json")
    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    stats = read_jsonl(root / "meta" / "episodes_stats.jsonl")
    mappings = read_jsonl(root / "meta" / "source_mapping.jsonl")
    expected_episodes = int(info["total_episodes"])
    if not len(episodes) == len(stats) == len(mappings) == expected_episodes:
        raise ValueError("metadata files disagree on episode count")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError("tasks.jsonl disagrees with total_tasks")

    global_index = 0
    total_frames = 0
    for episode_index, episode in enumerate(episodes):
        if int(episode["episode_index"]) != episode_index:
            raise ValueError(f"non-contiguous episode index at {episode_index}")
        length = int(episode["length"])
        chunk = episode_index // int(info["chunks_size"])
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            raise ValueError(f"{parquet_path}: row count differs from metadata")
        expected_frames = list(range(length))
        if table["frame_index"].to_pylist() != expected_frames:
            raise ValueError(f"{parquet_path}: frame_index is not contiguous")
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"{parquet_path}: wrong episode_index")
        expected_indices = list(range(global_index, global_index + length))
        if table["index"].to_pylist() != expected_indices:
            raise ValueError(f"{parquet_path}: global index is not contiguous")
        task_indices = set(table["task_index"].to_pylist())
        mapping_task = int(mappings[episode_index]["subtask_index"])
        if task_indices != {mapping_task}:
            raise ValueError(f"{parquet_path}: task_index differs from source mapping")
        if episode["tasks"] != [tasks[mapping_task]["task"]]:
            raise ValueError(f"{parquet_path}: task text differs from task_index")
        global_index += length
        total_frames += length

    if total_frames != int(info["total_frames"]):
        raise ValueError("Parquet rows disagree with total_frames")

    sample_count = min(max(args.decode_samples, 0), expected_episodes)
    sampled = random.Random(42).sample(range(expected_episodes), sample_count)
    for episode_index in sampled:
        length = int(episodes[episode_index]["length"])
        chunk = episode_index // int(info["chunks_size"])
        for camera in CAMERA_KEYS:
            path = root / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{episode_index:06d}.mp4"
            actual = frame_count(path)
            if actual != length:
                raise ValueError(f"{path}: expected {length} frames, found {actual}")

    # This catches metadata compatibility failures before training starts.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata("local/subtask_vla", root=root)
    dataset = LeRobotDataset("local/subtask_vla", root=root)
    if len(dataset) != total_frames or metadata.total_episodes != expected_episodes:
        raise ValueError("LeRobot loader reports inconsistent dataset dimensions")
    if sample_count:
        for index in (0, total_frames - 1):
            item = dataset[index]
            if item["task"] not in {task["task"] for task in tasks}:
                raise ValueError("LeRobot loader returned an unknown task")

    print(
        json.dumps(
            {
                "status": "ok",
                "episodes": expected_episodes,
                "frames": total_frames,
                "tasks": len(tasks),
                "decoded_video_episodes": sample_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
