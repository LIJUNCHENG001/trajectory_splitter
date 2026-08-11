#!/usr/bin/env python3
"""Trim initial stationary frames and rebuild a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

JOINT_COLUMN = "observation.state.joint"
GRIPPER_COLUMN = "observation.gripper_position"


@dataclass(frozen=True)
class TrimPoint:
    episode_index: int
    source_length: int
    motion_frame: int | None
    trim_frame: int
    trigger: str
    cut_source: str = "auto"

    @property
    def output_length(self) -> int:
        return self.source_length - self.trim_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and remove initial stationary frames from LeRobot v2.1 episodes."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only write --report; do not create a dataset.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--cuts",
        type=Path,
        help="CSV previously produced by --detect-only; edited trim_frame values override detection.",
    )
    parser.add_argument("--joint-threshold", type=float, default=0.005)
    parser.add_argument("--gripper-threshold", type=float, default=0.01)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--pre-roll-frames", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def first_sustained_onset(
    values: np.ndarray, threshold: float, sustain_frames: int
) -> int | None:
    deviation = np.max(np.abs(values - values[0]), axis=1)
    moving = deviation > threshold
    if len(moving) < sustain_frames:
        return None
    run_lengths = np.convolve(
        moving.astype(np.int8), np.ones(sustain_frames, dtype=np.int8), mode="valid"
    )
    matches = np.flatnonzero(run_lengths == sustain_frames)
    return int(matches[0]) if len(matches) else None


def detect_trim_point(
    path: Path,
    joint_threshold: float,
    gripper_threshold: float,
    sustain_frames: int,
    pre_roll_frames: int,
) -> TrimPoint:
    table = pq.read_table(path, columns=[JOINT_COLUMN, GRIPPER_COLUMN])
    joints = np.asarray(table[JOINT_COLUMN].to_pylist(), dtype=float)
    grippers = np.asarray(table[GRIPPER_COLUMN].to_pylist(), dtype=float)
    signals = {
        "observation_joint": first_sustained_onset(
            joints, joint_threshold, sustain_frames
        ),
        "observation_gripper": first_sustained_onset(
            grippers, gripper_threshold, sustain_frames
        ),
    }
    detected = [(frame, name) for name, frame in signals.items() if frame is not None]
    episode_index = int(path.stem.removeprefix("episode_"))
    if not detected:
        return TrimPoint(episode_index, table.num_rows, None, 0, "none")
    motion_frame = min(frame for frame, _ in detected)
    trigger = ",".join(
        sorted(name for frame, name in detected if frame == motion_frame)
    )
    trim_frame = max(0, motion_frame - pre_roll_frames)
    return TrimPoint(episode_index, table.num_rows, motion_frame, trim_frame, trigger)


def detect_all(
    source: Path, info: dict[str, Any], args: argparse.Namespace
) -> list[TrimPoint]:
    paths = sorted(
        (source / "data").rglob("episode_*.parquet"),
        key=lambda path: int(path.stem.removeprefix("episode_")),
    )
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        paths = paths[: args.max_episodes]
    if not paths:
        raise ValueError("no episode parquet files found")
    expected_indices = list(range(len(paths)))
    actual_indices = [int(path.stem.removeprefix("episode_")) for path in paths]
    if actual_indices != expected_indices:
        raise ValueError("episode indices must be contiguous and start at zero")

    points: list[TrimPoint] = []
    for position, path in enumerate(paths, start=1):
        points.append(
            detect_trim_point(
                path,
                args.joint_threshold,
                args.gripper_threshold,
                args.sustain_frames,
                args.pre_roll_frames,
            )
        )
        if position % 25 == 0 or position == len(paths):
            print(f"detect: {position}/{len(paths)} episodes", flush=True)
    return points


def load_cut_overrides(path: Path) -> dict[int, int]:
    overrides: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_index = int(row["episode_index"])
            if episode_index in overrides:
                raise ValueError(f"{path}: duplicate episode {episode_index}")
            overrides[episode_index] = int(row["trim_frame"])
    return overrides


def apply_cut_overrides(
    points: list[TrimPoint], overrides: dict[int, int]
) -> list[TrimPoint]:
    expected = {point.episode_index for point in points}
    missing = expected - overrides.keys()
    if missing:
        raise ValueError(f"cuts CSV is missing episodes: {sorted(missing)[:10]}")
    result = []
    for point in points:
        trim_frame = overrides[point.episode_index]
        if not 0 <= trim_frame < point.source_length:
            raise ValueError(
                f"episode {point.episode_index}: trim_frame {trim_frame} is outside "
                f"[0, {point.source_length})"
            )
        result.append(replace(point, trim_frame=trim_frame, cut_source="csv"))
    return result


REPORT_FIELDS = [
    "episode_index",
    "source_length",
    "motion_frame",
    "motion_time",
    "trim_frame",
    "trim_time",
    "removed_frames",
    "output_length",
    "trigger",
    "cut_source",
]


def write_report(path: Path, points: list[TrimPoint], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "episode_index": point.episode_index,
                    "source_length": point.source_length,
                    "motion_frame": ""
                    if point.motion_frame is None
                    else point.motion_frame,
                    "motion_time": ""
                    if point.motion_frame is None
                    else point.motion_frame / fps,
                    "trim_frame": point.trim_frame,
                    "trim_time": point.trim_frame / fps,
                    "removed_frames": point.trim_frame,
                    "output_length": point.output_length,
                    "trigger": point.trigger,
                    "cut_source": point.cut_source,
                }
            )


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Parquet table has no required column {name!r}")
    return table.set_column(
        index, name, pa.array(values, type=table.schema.field(index).type)
    )


def numeric_stats(table: pa.Table, feature_info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, feature in feature_info.items():
        if feature["dtype"] in {"video", "image", "string"}:
            continue
        if key not in table.column_names:
            continue
        values = np.asarray(table[key].to_pylist())
        keepdims = values.ndim == 1
        result[key] = {
            "min": np.min(values, axis=0, keepdims=keepdims).tolist(),
            "max": np.max(values, axis=0, keepdims=keepdims).tolist(),
            "mean": np.mean(values, axis=0, keepdims=keepdims).tolist(),
            "std": np.std(values, axis=0, keepdims=keepdims).tolist(),
            "count": [len(values)],
        }
    return result


def image_sample_count(length: int) -> int:
    return max(min(length, 100), min(int(length**0.75), 10_000))


def convert_parquet(
    source: Path,
    staging: Path,
    point: TrimPoint,
    global_start_index: int,
    fps: float,
    features: dict[str, Any],
    camera_keys: list[str],
    source_image_stats: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any]:
    source_path = (
        source
        / "data"
        / f"chunk-{point.episode_index // chunk_size:03d}"
        / f"episode_{point.episode_index:06d}.parquet"
    )
    table = pq.read_table(source_path).slice(point.trim_frame, point.output_length)
    table = replace_column(
        table, "timestamp", np.arange(point.output_length, dtype=np.float32) / fps
    )
    table = replace_column(
        table, "frame_index", np.arange(point.output_length, dtype=np.int64)
    )
    table = replace_column(
        table,
        "episode_index",
        np.full(point.output_length, point.episode_index, dtype=np.int64),
    )
    table = replace_column(
        table,
        "index",
        np.arange(
            global_start_index,
            global_start_index + point.output_length,
            dtype=np.int64,
        ),
    )
    output_path = (
        staging
        / "data"
        / f"chunk-{point.episode_index // chunk_size:03d}"
        / f"episode_{point.episode_index:06d}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")

    stats = numeric_stats(table, features)
    for camera in camera_keys:
        stats[camera] = json.loads(json.dumps(source_image_stats[camera]))
        stats[camera]["count"] = [image_sample_count(point.output_length)]
    return {"episode_index": point.episode_index, "stats": stats}


def video_frame_count(path: Path) -> int:
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


def convert_videos(
    source: Path,
    staging: Path,
    point: TrimPoint,
    camera_keys: list[str],
    chunk_size: int,
    fps: float,
    crf: int,
    preset: str,
) -> None:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for camera in camera_keys:
        command.extend(
            [
                "-i",
                str(
                    source
                    / "videos"
                    / f"chunk-{point.episode_index // chunk_size:03d}"
                    / camera
                    / f"episode_{point.episode_index:06d}.mp4"
                ),
            ]
        )
    filters = [
        f"[{index}:v]trim=start_frame={point.trim_frame}:"
        f"end_frame={point.source_length},setpts=PTS-STARTPTS[o{index}]"
        for index in range(len(camera_keys))
    ]
    command.extend(["-filter_complex", ";".join(filters)])
    output_paths: list[Path] = []
    for index, camera in enumerate(camera_keys):
        output_path = (
            staging
            / "videos"
            / f"chunk-{point.episode_index // chunk_size:03d}"
            / camera
            / f"episode_{point.episode_index:06d}.mp4"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.append(output_path)
        command.extend(
            [
                "-map",
                f"[o{index}]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-fps_mode",
                "cfr",
                "-g",
                str(round(fps)),
                "-keyint_min",
                str(round(fps)),
                "-sc_threshold",
                "0",
                "-threads",
                "2",
                "-frames:v",
                str(point.output_length),
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    subprocess.run(command, check=True)
    for output_path in output_paths:
        actual = video_frame_count(output_path)
        if actual != point.output_length:
            raise RuntimeError(
                f"{output_path}: expected {point.output_length} frames, found {actual}"
            )


def build_info(
    source_info: dict[str, Any],
    points: list[TrimPoint],
    camera_keys: list[str],
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    chunk_size = int(info["chunks_size"])
    info.update(
        {
            "total_episodes": len(points),
            "total_frames": sum(point.output_length for point in points),
            "total_videos": len(points) * len(camera_keys),
            "total_chunks": math.ceil(len(points) / chunk_size),
            "splits": {"train": f"0:{len(points)}"},
        }
    )
    for camera in camera_keys:
        video_info = info["features"][camera]["info"]
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.fps"] = info["fps"]
    return info


def validate_staging(
    staging: Path,
    info: dict[str, Any],
    points: list[TrimPoint],
    camera_keys: list[str],
) -> None:
    chunk_size = int(info["chunks_size"])
    fps = float(info["fps"])
    global_index = 0
    for point in points:
        parquet_path = (
            staging
            / "data"
            / f"chunk-{point.episode_index // chunk_size:03d}"
            / f"episode_{point.episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        if table.num_rows != point.output_length:
            raise ValueError(f"{parquet_path}: wrong row count")
        if table["frame_index"].to_pylist() != list(range(point.output_length)):
            raise ValueError(f"{parquet_path}: frame_index is not contiguous")
        expected_index = list(range(global_index, global_index + point.output_length))
        if table["index"].to_pylist() != expected_index:
            raise ValueError(f"{parquet_path}: global index is not contiguous")
        timestamps = np.asarray(table["timestamp"], dtype=float)
        expected_timestamps = np.arange(point.output_length) / fps
        if not np.allclose(timestamps, expected_timestamps, atol=1e-5):
            raise ValueError(f"{parquet_path}: timestamps are invalid")
        for camera in camera_keys:
            video_path = (
                staging
                / "videos"
                / f"chunk-{point.episode_index // chunk_size:03d}"
                / camera
                / f"episode_{point.episode_index:06d}.mp4"
            )
            if video_frame_count(video_path) != point.output_length:
                raise ValueError(f"{video_path}: frame count does not match parquet")
        global_index += point.output_length
    if global_index != int(info["total_frames"]):
        raise ValueError("total_frames does not match exported parquet rows")


def export_dataset(
    source: Path,
    output: Path,
    points: list[TrimPoint],
    source_info: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError(
            "output cannot be the source dataset or one of its subdirectories"
        )
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    camera_keys = [
        name
        for name, feature in source_info["features"].items()
        if feature["dtype"] == "video"
    ]
    if not camera_keys:
        raise ValueError("source dataset has no video features")
    chunk_size = int(source_info["chunks_size"])
    fps = float(source_info["fps"])
    source_episodes = {
        int(item["episode_index"]): item
        for item in read_jsonl(source / "meta" / "episodes.jsonl")
    }
    source_stats = {
        int(item["episode_index"]): item["stats"]
        for item in read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }
    tasks = read_jsonl(source / "meta" / "tasks.jsonl")
    info = build_info(source_info, points, camera_keys)

    try:
        episodes = []
        mappings = []
        episode_stats = []
        global_index = 0
        for position, point in enumerate(points, start=1):
            source_episode = source_episodes[point.episode_index]
            episodes.append(
                {
                    "episode_index": point.episode_index,
                    "tasks": source_episode["tasks"],
                    "length": point.output_length,
                }
            )
            mappings.append(
                {
                    "episode_index": point.episode_index,
                    "source_episode_index": point.episode_index,
                    "source_start_frame": point.trim_frame,
                    "source_end_frame": point.source_length,
                    "source_length": point.source_length,
                    "length": point.output_length,
                    "trigger": point.trigger,
                    "cut_source": point.cut_source,
                }
            )
            episode_stats.append(
                convert_parquet(
                    source,
                    staging,
                    point,
                    global_index,
                    fps,
                    source_info["features"],
                    camera_keys,
                    source_stats[point.episode_index],
                    chunk_size,
                )
            )
            global_index += point.output_length
            if position % 25 == 0 or position == len(points):
                print(f"parquet: {position}/{len(points)} episodes", flush=True)

        write_json(staging / "meta" / "info.json", info)
        write_jsonl(staging / "meta" / "tasks.jsonl", tasks)
        write_jsonl(staging / "meta" / "episodes.jsonl", episodes)
        write_jsonl(staging / "meta" / "episodes_stats.jsonl", episode_stats)
        write_jsonl(staging / "meta" / "source_mapping.jsonl", mappings)
        write_report(staging / "meta" / "trim_points.csv", points, fps)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    convert_videos,
                    source,
                    staging,
                    point,
                    camera_keys,
                    chunk_size,
                    fps,
                    args.crf,
                    args.preset,
                ): point.episode_index
                for point in points
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                episode_index = futures[future]
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"video conversion failed for episode {episode_index}"
                    ) from error
                if completed % 10 == 0 or completed == len(points):
                    print(f"video: {completed}/{len(points)} episodes", flush=True)

        validate_staging(staging, info, points, camera_keys)
        staging.rename(output)
    except BaseException:
        print(
            f"conversion stopped; partial output remains in {staging}", file=sys.stderr
        )
        raise


def validate_args(args: argparse.Namespace) -> None:
    if args.detect_only:
        if args.output is not None:
            raise ValueError("--detect-only cannot be combined with --output")
        if args.report is None:
            raise ValueError("--detect-only requires --report")
        if args.cuts is not None:
            raise ValueError("--detect-only cannot be combined with --cuts")
    elif args.output is None:
        raise ValueError("dataset export requires --output (or use --detect-only)")
    if args.joint_threshold <= 0 or args.gripper_threshold <= 0:
        raise ValueError("movement thresholds must be positive")
    if args.sustain_frames <= 0 or args.pre_roll_frames < 0:
        raise ValueError("frame parameters are invalid")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = args.source.resolve()
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot v2.1 dataset: {source}")
    source_info = read_json(info_path)
    points = detect_all(source, source_info, args)
    if args.cuts is not None:
        points = apply_cut_overrides(points, load_cut_overrides(args.cuts.resolve()))
    fps = float(source_info["fps"])
    if args.report is not None:
        write_report(args.report.resolve(), points, fps)

    removed = sum(point.trim_frame for point in points)
    summary = {
        "episodes": len(points),
        "source_frames": sum(point.source_length for point in points),
        "removed_frames": removed,
        "removed_seconds": removed / fps,
        "output_frames": sum(point.output_length for point in points),
        "median_trim_frames": float(np.median([point.trim_frame for point in points])),
    }
    if args.detect_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output = args.output.resolve()
    export_dataset(source, output, points, source_info, args)
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
