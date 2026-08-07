#!/usr/bin/env python3
"""Synchronize manually edited cut times to frames and downstream outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/home/geek/share3/agilex_make_breakfast_380")
DEFAULT_OUTPUT = PROJECT_DIR / "output"
SEGMENT_LABELS = [
    "阶段1_切分点1前",
    "阶段2_切分点1至2",
    "阶段3_切分点2至3",
    "阶段4_切分点3至4",
    "阶段5_切分点4后",
]
REWARD_KEYS = [
    "success",
    "failure",
    "pick_success",
    "place_success",
    "empty_pick",
    "empty_place",
    "drop",
    "collision",
]
SUB_TASK_DESCRIPTIONS = [
    "Pick up all bread pieces from the bread rack and insert them into the toaster.",
    "Push down the toaster's front lever to activate the toaster, then return the right gripper to its initial resting configuration.",
    "Pour drink from the water bottle into the cup, return the bottle upright, and return the left gripper to its initial resting configuration.",
    "Remove all toasted bread from the toaster and place it on the plate.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map manually edited cut_times to frames and synchronize outputs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output root used by unspecified downstream paths.",
    )
    parser.add_argument(
        "--summary", type=Path, default=None
    )
    parser.add_argument(
        "--cut-points", type=Path, default=None
    )
    parser.add_argument(
        "--visualiser-segments",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--split-output", type=Path, default=None
    )
    parser.add_argument(
        "--parquet-output", type=Path, default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def nearest_frames(timestamps: np.ndarray, cut_times: list[float]) -> list[int]:
    frames = [int(np.abs(timestamps - value).argmin()) for value in cut_times]
    if frames != sorted(frames) or len(set(frames)) != 4:
        raise ValueError(f"cut times map to non-increasing frames: {frames}")
    if frames[0] <= 0 or frames[-1] >= len(timestamps):
        raise ValueError(f"cut frames must leave non-empty edge segments: {frames}")
    return frames


def make_segments(cut_times: list[float], duration: float) -> list[dict]:
    boundaries = [0.0, *cut_times, duration]
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"cut times are not strictly increasing: {cut_times}")
    return [
        {
            "id": f"auto_segment_{number}",
            "start": round(start, 6),
            "end": round(end, 6),
            "label": label,
        }
        for number, (start, end, label) in enumerate(
            zip(boundaries, boundaries[1:], SEGMENT_LABELS), 1
        )
    ]


def make_split_json(
    episode_index: int,
    cut_times: list[float],
    cut_frames: list[int],
    duration: float,
    frame_count: int,
) -> dict:
    return {
        "index": episode_index,
        "name": f"episode_{episode_index:06d}.mp4",
        "sub_task": {
            "subtask": SUB_TASK_DESCRIPTIONS,
            "start_time": [round(value, 3) for value in cut_times],
            "end_time": [
                round(cut_times[1], 3),
                round(cut_times[2], 3),
                round(cut_times[3], 3),
                round(duration, 3),
            ],
            "start_frame": cut_frames,
            "end_frame": [cut_frames[1], cut_frames[2], cut_frames[3], frame_count],
        },
        "idle_frames": {
            "start_time": [],
            "end_time": [],
            "start_frame": [],
            "end_frame": [],
        },
        "reward": {key: [] for key in REWARD_KEYS},
    }


def rewrite_parquet_segments(
    table, episode_index: int, cut_frames: list[int], output: Path
) -> None:
    episode_dir = output / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [0, *cut_frames, table.num_rows]
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        path = episode_dir / f"segment_{number:02d}.parquet"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pq.write_table(table.slice(start, end - start), temporary, compression="zstd")
        os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.summary = args.summary or args.output / "split_summary.json"
    args.cut_points = args.cut_points or args.output / "cut_points.csv"
    args.visualiser_segments = (
        args.visualiser_segments or args.output / "visualiser_segments.json"
    )
    args.split_output = args.split_output or args.output / "split"
    args.parquet_output = args.parquet_output or args.output / "parquet"
    info = json.loads((args.dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    chunk_size = int(info["chunks_size"])
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    csv_rows, csv_fields = read_csv(args.cut_points)
    csv_by_episode = {int(row["episode_index"]): row for row in csv_rows}
    visualiser = json.loads(args.visualiser_segments.read_text(encoding="utf-8"))

    changed: list[dict] = []
    prepared: list[tuple[dict, object, list[float], list[int], float]] = []
    for episode in summary["episodes"]:
        episode_index = int(episode["episode_index"])
        cut_times = [float(value) for value in episode["cut_times"]]
        if len(cut_times) != 4:
            raise ValueError(f"episode {episode_index}: expected 4 cut times")
        parquet_path = args.dataset / info["data_path"].format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        table = pq.read_table(parquet_path)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
        cut_frames = nearest_frames(timestamps, cut_times)
        duration = table.num_rows / fps
        old_frames = [int(value) for value in episode["cut_frames"]]
        csv_times = [
            float(csv_by_episode[episode_index][f"cut{number}_time"])
            for number in range(1, 5)
        ]
        if old_frames != cut_frames or not np.allclose(
            csv_times, cut_times, atol=1e-9
        ):
            changed.append(
                {
                    "episode_index": episode_index,
                    "old_frames": old_frames,
                    "new_frames": cut_frames,
                    "cut_times": cut_times,
                }
            )
        prepared.append((episode, table, cut_times, cut_frames, duration))

    print(f"Episodes: {len(prepared)}; manually changed: {len(changed)}")
    for item in changed:
        print(
            f"  episode {item['episode_index']:06d}: "
            f"{item['old_frames']} -> {item['new_frames']}"
        )
    if args.dry_run:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.summary.parent / "manual_sync_backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for path in (args.summary, args.cut_points, args.visualiser_segments):
        shutil.copy2(path, backup / path.name)
    for item in changed:
        split_path = args.split_output / f"episode_{item['episode_index']:06d}.json"
        if split_path.is_file():
            shutil.copy2(split_path, backup / split_path.name)

    changed_indices = {item["episode_index"] for item in changed}
    for episode, table, cut_times, cut_frames, duration in prepared:
        episode_index = int(episode["episode_index"])
        episode["cut_frames"] = cut_frames
        episode["segments"] = make_segments(cut_times, duration)
        row = csv_by_episode[episode_index]
        for number, (frame, cut_time) in enumerate(
            zip(cut_frames, cut_times), 1
        ):
            row[f"cut{number}_frame"] = frame
            row[f"cut{number}_time"] = cut_time
        visualiser["episodes"][str(episode_index)] = episode["segments"]
        atomic_json(
            args.split_output / f"episode_{episode_index:06d}.json",
            make_split_json(
                episode_index, cut_times, cut_frames, duration, table.num_rows
            ),
        )
        if episode_index in changed_indices:
            rewrite_parquet_segments(
                table, episode_index, cut_frames, args.parquet_output
            )

    atomic_json(args.summary, summary)
    atomic_csv(args.cut_points, csv_rows, csv_fields)
    atomic_json(args.visualiser_segments, visualiser)
    print(f"Synchronized outputs; backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
