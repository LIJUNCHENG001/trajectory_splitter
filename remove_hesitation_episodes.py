#!/usr/bin/env python3
"""Detect insertion hesitation and export a LeRobot dataset without bad episodes."""

from __future__ import annotations

import argparse
import csv
import errno
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trim_initial_stationary import (
    numeric_stats,
    read_json,
    read_jsonl,
    replace_column,
    write_json,
    write_jsonl,
)

END_COLUMN = "observation.state.end"
GRIPPER_COLUMN = "observation.gripper_position"


@dataclass(frozen=True)
class Hesitation:
    episode_index: int
    episode_frames: int
    insertion_start_frame: int
    release_frame: int
    pause_start_frame: int
    pause_end_frame: int

    @property
    def pause_frames(self) -> int:
        return self.pause_end_frame - self.pause_start_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove episodes with a long internal pause during bread insertion."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--smooth-frames", type=int, default=7)
    parser.add_argument("--speed-threshold-mm-s", type=float, default=10.5)
    parser.add_argument("--min-pause-seconds", type=float, default=1.0)
    parser.add_argument("--ignore-opening-seconds", type=float, default=0.2)
    parser.add_argument("--release-delta", type=float, default=0.12)
    parser.add_argument("--release-sustain-frames", type=int, default=3)
    return parser.parse_args()


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth_positions(values: np.ndarray, window: int) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
    return (cumulative[window:] - cumulative[:-window]) / window


def first_release_frame(
    gripper: np.ndarray, delta: float, sustain_frames: int
) -> int:
    opened = gripper > gripper[0] + delta
    for start, end in true_runs(opened):
        if end - start >= sustain_frames:
            return start
    return len(gripper) - 1


def detect_episode(
    path: Path,
    fps: float,
    smooth_frames: int,
    speed_threshold_mm_s: float,
    min_pause_seconds: float,
    ignore_opening_seconds: float,
    release_delta: float,
    release_sustain_frames: int,
) -> Hesitation | None:
    table = pq.read_table(path, columns=[END_COLUMN, GRIPPER_COLUMN])
    ends = np.asarray(table[END_COLUMN].to_pylist(), dtype=float)
    grippers = np.asarray(table[GRIPPER_COLUMN].to_pylist(), dtype=float)
    if ends.ndim != 2 or ends.shape[1] < 9:
        raise ValueError(f"{path}: {END_COLUMN} must contain both 6-DoF arms")
    if grippers.ndim != 2 or grippers.shape[1] < 2:
        raise ValueError(f"{path}: {GRIPPER_COLUMN} must contain both grippers")

    right_xyz = smooth_positions(ends[:, 6:9], smooth_frames)
    release_frame = first_release_frame(
        grippers[:, 1], release_delta, release_sustain_frames
    )
    insertion_start = int(np.argmax(right_xyz[: release_frame + 1, 2]))
    speed_m_per_frame = np.linalg.norm(np.diff(right_xyz, axis=0), axis=1)
    threshold_m_per_frame = speed_threshold_mm_s / 1000.0 / fps
    minimum_frames = math.ceil(min_pause_seconds * fps)

    candidates: list[tuple[int, int]] = []
    insertion_low_speed = (
        speed_m_per_frame[insertion_start:release_frame] < threshold_m_per_frame
    )
    for relative_start, relative_end in true_runs(insertion_low_speed):
        pause_start = insertion_start + relative_start
        pause_end = insertion_start + relative_end
        if pause_end >= release_frame:
            continue
        if pause_end - pause_start < minimum_frames:
            continue
        if pause_start / fps <= ignore_opening_seconds:
            continue
        candidates.append((pause_start, pause_end))

    if not candidates:
        return None
    pause_start, pause_end = max(candidates, key=lambda item: item[1] - item[0])
    return Hesitation(
        episode_index=int(path.stem.removeprefix("episode_")),
        episode_frames=table.num_rows,
        insertion_start_frame=insertion_start,
        release_frame=release_frame,
        pause_start_frame=pause_start,
        pause_end_frame=pause_end,
    )


def detect_all(source: Path, info: dict[str, Any], args: argparse.Namespace) -> list[Hesitation]:
    paths = sorted(
        (source / "data").rglob("episode_*.parquet"),
        key=lambda path: int(path.stem.removeprefix("episode_")),
    )
    expected = list(range(int(info["total_episodes"])))
    actual = [int(path.stem.removeprefix("episode_")) for path in paths]
    if actual != expected:
        raise ValueError("episode parquet indices do not match meta/info.json")

    fps = float(info["fps"])
    results: list[Hesitation] = []
    for position, path in enumerate(paths, start=1):
        result = detect_episode(
            path,
            fps,
            args.smooth_frames,
            args.speed_threshold_mm_s,
            args.min_pause_seconds,
            args.ignore_opening_seconds,
            args.release_delta,
            args.release_sustain_frames,
        )
        if result is not None:
            results.append(result)
        if position % 25 == 0 or position == len(paths):
            print(f"detect: {position}/{len(paths)} episodes", flush=True)
    return results


REPORT_FIELDS = [
    "episode_index",
    "episode_frames",
    "insertion_start_frame",
    "insertion_start_seconds",
    "release_frame",
    "release_seconds",
    "pause_start_frame",
    "pause_start_seconds",
    "pause_end_frame",
    "pause_end_seconds",
    "pause_frames",
    "pause_seconds",
]


def write_report(path: Path, results: list[Hesitation], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "episode_index": item.episode_index,
                    "episode_frames": item.episode_frames,
                    "insertion_start_frame": item.insertion_start_frame,
                    "insertion_start_seconds": item.insertion_start_frame / fps,
                    "release_frame": item.release_frame,
                    "release_seconds": item.release_frame / fps,
                    "pause_start_frame": item.pause_start_frame,
                    "pause_start_seconds": item.pause_start_frame / fps,
                    "pause_end_frame": item.pause_end_frame,
                    "pause_end_seconds": item.pause_end_frame / fps,
                    "pause_frames": item.pause_frames,
                    "pause_seconds": item.pause_frames / fps,
                }
            )


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def build_info(
    source_info: dict[str, Any], episode_count: int, total_frames: int, video_count: int
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    chunk_size = int(info["chunks_size"])
    info.update(
        {
            "total_episodes": episode_count,
            "total_frames": total_frames,
            "total_videos": video_count,
            "total_chunks": math.ceil(episode_count / chunk_size),
            "splits": {"train": f"0:{episode_count}"},
        }
    )
    return info


def export_filtered_dataset(
    source: Path,
    output: Path,
    source_info: dict[str, Any],
    excluded: list[Hesitation],
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("output cannot be the source or a directory inside the source")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    chunk_size = int(source_info["chunks_size"])
    features = source_info["features"]
    camera_keys = [name for name, feature in features.items() if feature["dtype"] == "video"]
    source_episodes = {
        int(item["episode_index"]): item
        for item in read_jsonl(source / "meta" / "episodes.jsonl")
    }
    source_stats = {
        int(item["episode_index"]): item["stats"]
        for item in read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }
    mapping_path = source / "meta" / "source_mapping.jsonl"
    source_mappings = (
        {
            int(item["episode_index"]): item
            for item in read_jsonl(mapping_path)
        }
        if mapping_path.is_file()
        else {}
    )
    excluded_indices = {item.episode_index for item in excluded}
    kept_indices = [
        index
        for index in range(int(source_info["total_episodes"]))
        if index not in excluded_indices
    ]

    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    total_frames = 0
    try:
        for new_index, old_index in enumerate(kept_indices):
            source_parquet = (
                source
                / "data"
                / f"chunk-{old_index // chunk_size:03d}"
                / f"episode_{old_index:06d}.parquet"
            )
            table = pq.read_table(source_parquet)
            length = table.num_rows
            table = replace_column(
                table, "episode_index", np.full(length, new_index, dtype=np.int64)
            )
            table = replace_column(
                table,
                "index",
                np.arange(total_frames, total_frames + length, dtype=np.int64),
            )
            output_parquet = (
                staging
                / "data"
                / f"chunk-{new_index // chunk_size:03d}"
                / f"episode_{new_index:06d}.parquet"
            )
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_parquet, compression="zstd")

            source_episode = source_episodes[old_index]
            episodes.append(
                {
                    "episode_index": new_index,
                    "tasks": source_episode["tasks"],
                    "length": length,
                }
            )
            stats = numeric_stats(table, features)
            for camera in camera_keys:
                stats[camera] = json.loads(json.dumps(source_stats[old_index][camera]))
            episode_stats.append({"episode_index": new_index, "stats": stats})

            mapping = dict(source_mappings.get(old_index, {}))
            mapping["episode_index"] = new_index
            mapping["filtered_source_episode_index"] = old_index
            mappings.append(mapping)

            for camera in camera_keys:
                source_video = (
                    source
                    / "videos"
                    / f"chunk-{old_index // chunk_size:03d}"
                    / camera
                    / f"episode_{old_index:06d}.mp4"
                )
                output_video = (
                    staging
                    / "videos"
                    / f"chunk-{new_index // chunk_size:03d}"
                    / camera
                    / f"episode_{new_index:06d}.mp4"
                )
                link_or_copy(source_video, output_video)

            total_frames += length
            if (new_index + 1) % 25 == 0 or new_index + 1 == len(kept_indices):
                print(f"export: {new_index + 1}/{len(kept_indices)} episodes", flush=True)

        info = build_info(
            source_info,
            len(kept_indices),
            total_frames,
            len(kept_indices) * len(camera_keys),
        )
        write_json(staging / "meta" / "info.json", info)
        write_jsonl(
            staging / "meta" / "tasks.jsonl",
            read_jsonl(source / "meta" / "tasks.jsonl"),
        )
        write_jsonl(staging / "meta" / "episodes.jsonl", episodes)
        write_jsonl(staging / "meta" / "episodes_stats.jsonl", episode_stats)
        write_jsonl(staging / "meta" / "source_mapping.jsonl", mappings)
        write_report(
            staging / "meta" / "removed_hesitation_episodes.csv",
            excluded,
            float(source_info["fps"]),
        )

        validate_output(staging, info, camera_keys)
        staging.rename(output)
    except BaseException:
        print(f"export stopped; partial output remains in {staging}", file=sys.stderr)
        raise


def validate_output(root: Path, info: dict[str, Any], camera_keys: list[str]) -> None:
    chunk_size = int(info["chunks_size"])
    total_frames = 0
    for episode_index in range(int(info["total_episodes"])):
        parquet_path = (
            root
            / "data"
            / f"chunk-{episode_index // chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path, columns=["episode_index", "index"])
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"{parquet_path}: wrong episode_index")
        expected = list(range(total_frames, total_frames + table.num_rows))
        if table["index"].to_pylist() != expected:
            raise ValueError(f"{parquet_path}: global index is not contiguous")
        for camera in camera_keys:
            video_path = (
                root
                / "videos"
                / f"chunk-{episode_index // chunk_size:03d}"
                / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
        total_frames += table.num_rows
    if total_frames != int(info["total_frames"]):
        raise ValueError("total_frames does not match exported parquet rows")


def validate_args(args: argparse.Namespace) -> None:
    if args.detect_only:
        if args.output is not None:
            raise ValueError("--detect-only cannot be combined with --output")
        if args.report is None:
            raise ValueError("--detect-only requires --report")
    elif args.output is None:
        raise ValueError("dataset export requires --output (or use --detect-only)")
    if args.smooth_frames <= 0 or args.smooth_frames % 2 == 0:
        raise ValueError("--smooth-frames must be a positive odd number")
    if args.speed_threshold_mm_s <= 0 or args.min_pause_seconds <= 0:
        raise ValueError("speed threshold and pause duration must be positive")
    if args.ignore_opening_seconds < 0:
        raise ValueError("--ignore-opening-seconds cannot be negative")
    if args.release_delta <= 0 or args.release_sustain_frames <= 0:
        raise ValueError("release parameters must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = args.source.resolve()
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot v2.1 dataset: {source}")
    info = read_json(info_path)
    results = detect_all(source, info, args)
    if args.report is not None:
        write_report(args.report.resolve(), results, float(info["fps"]))

    summary = {
        "source": str(source),
        "episodes": int(info["total_episodes"]),
        "removed_episodes": [item.episode_index for item in results],
        "kept_episodes": int(info["total_episodes"]) - len(results),
    }
    if not args.detect_only:
        output = args.output.resolve()
        export_filtered_dataset(source, output, info, results)
        summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
