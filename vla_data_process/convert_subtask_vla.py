#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_KEYS = (
    "observation.image.top",
    "observation.image.left_wrist",
    "observation.image.right_wrist",
)


@dataclass(frozen=True)
class Segment:
    output_episode: int
    source_episode: int
    subtask_index: int
    subtask_id: str
    task: str
    start_frame: int
    end_frame: int
    global_start_index: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert labeled Breakfast intervals into LeRobot sub-task episodes."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="Directory containing episode_*.json annotations (default: SOURCE/split).",
    )
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-source-episodes", type=int)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse frame-verified videos in an existing .<output>.building directory.",
    )
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


def load_segments(
    source: Path,
    split_dir: Path,
    task_spec_path: Path,
    max_source_episodes: int | None,
) -> tuple[list[Segment], list[dict[str, Any]], dict[str, Any]]:
    info = read_json(source / "meta" / "info.json")
    source_episodes = {
        int(item["episode_index"]): item
        for item in read_jsonl(source / "meta" / "episodes.jsonl")
    }
    task_spec = read_json(task_spec_path)
    sub_tasks = task_spec["sub_tasks"]
    if not sub_tasks:
        raise ValueError("TaskSpec has no sub_tasks")

    annotation_paths = sorted(split_dir.glob("episode_*.json"))
    if max_source_episodes is not None:
        if max_source_episodes <= 0:
            raise ValueError("--max-source-episodes must be positive")
        annotation_paths = annotation_paths[:max_source_episodes]

    raw_segments: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        annotation = read_json(annotation_path)
        source_episode = int(annotation["index"])
        episode_meta = source_episodes.get(source_episode)
        if episode_meta is None:
            raise ValueError(f"{annotation_path}: episode is missing from episodes.jsonl")
        episode_length = int(episode_meta["length"])
        # task_end is a status-only marker; VLA segmentation intentionally uses
        # only the four intervals under sub_task.
        labels = annotation.get("sub_task", {})
        descriptions = labels.get("subtask", [])
        starts = labels.get("start_frame", [])
        ends = labels.get("end_frame", [])
        expected = len(sub_tasks)
        if not (len(descriptions) == len(starts) == len(ends) == expected):
            raise ValueError(
                f"{annotation_path}: expected {expected} labeled intervals, got "
                f"{len(descriptions)}/{len(starts)}/{len(ends)}"
            )

        previous_end = None
        for subtask_index, (description, start, end) in enumerate(
            zip(descriptions, starts, ends, strict=True)
        ):
            task = str(sub_tasks[subtask_index]["description"])
            if description != task:
                raise ValueError(
                    f"{annotation_path}: subtask {subtask_index} description differs "
                    "from the TaskSpec"
                )
            start, end = int(start), int(end)
            if not 0 <= start < end <= episode_length:
                raise ValueError(
                    f"{annotation_path}: invalid interval [{start}, {end}) for "
                    f"episode length {episode_length}"
                )
            if previous_end is not None and start != previous_end:
                raise ValueError(
                    f"{annotation_path}: labeled intervals have a gap or overlap at frame {start}"
                )
            previous_end = end
            raw_segments.append(
                {
                    "source_episode": source_episode,
                    "subtask_index": subtask_index,
                    "subtask_id": str(sub_tasks[subtask_index]["id"]),
                    "task": task,
                    "start_frame": start,
                    "end_frame": end,
                }
            )

    segments: list[Segment] = []
    global_index = 0
    for output_episode, raw in enumerate(raw_segments):
        segment = Segment(
            output_episode=output_episode,
            global_start_index=global_index,
            **raw,
        )
        segments.append(segment)
        global_index += segment.length
    return segments, sub_tasks, info


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Parquet table has no required column {name!r}")
    target_type = table.schema.field(index).type
    return table.set_column(index, name, pa.array(values, type=target_type))


def numeric_stats(table: pa.Table, feature_info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, feature in feature_info.items():
        if feature["dtype"] in {"video", "image", "string"}:
            continue
        if key not in table.column_names:
            continue
        values = np.asarray(table[key].to_pylist())
        axis = 0
        keepdims = values.ndim == 1
        result[key] = {
            "min": np.min(values, axis=axis, keepdims=keepdims).tolist(),
            "max": np.max(values, axis=axis, keepdims=keepdims).tolist(),
            "mean": np.mean(values, axis=axis, keepdims=keepdims).tolist(),
            "std": np.std(values, axis=axis, keepdims=keepdims).tolist(),
            "count": [len(values)],
        }
    return result


def image_sample_count(length: int) -> int:
    return max(min(length, 100), min(int(length**0.75), 10_000))


def convert_parquet_group(
    source: Path,
    staging: Path,
    source_episode: int,
    segments: list[Segment],
    fps: int,
    source_image_stats: dict[str, Any],
    features: dict[str, Any],
) -> list[dict[str, Any]]:
    source_path = (
        source
        / "data"
        / f"chunk-{source_episode // 1000:03d}"
        / f"episode_{source_episode:06d}.parquet"
    )
    source_table = pq.read_table(source_path)
    output_stats: list[dict[str, Any]] = []
    for segment in segments:
        table = source_table.slice(segment.start_frame, segment.length)
        table = replace_column(
            table, "timestamp", np.arange(segment.length, dtype=np.float32) / fps
        )
        table = replace_column(table, "frame_index", np.arange(segment.length, dtype=np.int64))
        table = replace_column(
            table, "episode_index", np.full(segment.length, segment.output_episode, dtype=np.int64)
        )
        table = replace_column(
            table,
            "index",
            np.arange(
                segment.global_start_index,
                segment.global_start_index + segment.length,
                dtype=np.int64,
            ),
        )
        table = replace_column(
            table, "task_index", np.full(segment.length, segment.subtask_index, dtype=np.int64)
        )

        output_path = (
            staging
            / "data"
            / f"chunk-{segment.output_episode // 1000:03d}"
            / f"episode_{segment.output_episode:06d}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="zstd")

        stats = numeric_stats(table, features)
        count = image_sample_count(segment.length)
        for camera in CAMERA_KEYS:
            stats[camera] = dict(source_image_stats[camera])
            stats[camera]["count"] = [count]
        output_stats.append({"episode_index": segment.output_episode, "stats": stats})
    return output_stats


def video_frame_count(path: Path) -> int:
    command = [
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
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def convert_video_group(
    source: Path,
    staging: Path,
    source_episode: int,
    segments: list[Segment],
    fps: int,
    crf: int,
    preset: str,
) -> None:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for camera in CAMERA_KEYS:
        command.extend(
            [
                "-i",
                str(
                    source
                    / "videos"
                    / f"chunk-{source_episode // 1000:03d}"
                    / camera
                    / f"episode_{source_episode:06d}.mp4"
                ),
            ]
        )

    filters: list[str] = []
    for camera_index in range(len(CAMERA_KEYS)):
        split_outputs = "".join(
            f"[s{camera_index}_{i}]" for i in range(len(segments))
        )
        filters.append(
            f"[{camera_index}:v]split={len(segments)}{split_outputs}"
        )
        for i, segment in enumerate(segments):
            filters.append(
                f"[s{camera_index}_{i}]trim=start_frame={segment.start_frame}:"
                f"end_frame={segment.end_frame},setpts=PTS-STARTPTS[o{camera_index}_{i}]"
            )
    command.extend(["-filter_complex", ";".join(filters)])

    for camera_index, camera in enumerate(CAMERA_KEYS):
        for i, segment in enumerate(segments):
            output_path = (
                staging
                / "videos"
                / f"chunk-{segment.output_episode // 1000:03d}"
                / camera
                / f"episode_{segment.output_episode:06d}.mp4"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    "-map",
                    f"[o{camera_index}_{i}]",
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
                    str(fps),
                    "-keyint_min",
                    str(fps),
                    "-sc_threshold",
                    "0",
                    "-threads",
                    "2",
                    "-frames:v",
                    str(segment.length),
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
    subprocess.run(command, check=True)

    for camera in CAMERA_KEYS:
        for segment in segments:
            output_path = (
                staging
                / "videos"
                / f"chunk-{segment.output_episode // 1000:03d}"
                / camera
                / f"episode_{segment.output_episode:06d}.mp4"
            )
            actual = video_frame_count(output_path)
            if actual != segment.length:
                raise RuntimeError(
                    f"{output_path}: expected {segment.length} frames, found {actual}"
                )


def video_group_is_complete(
    staging: Path, segments: list[Segment]
) -> bool:
    paths: list[tuple[Path, int]] = []
    for camera in CAMERA_KEYS:
        for segment in segments:
            path = (
                staging
                / "videos"
                / f"chunk-{segment.output_episode // 1000:03d}"
                / camera
                / f"episode_{segment.output_episode:06d}.mp4"
            )
            if not path.is_file():
                return False
            paths.append((path, segment.length))
    try:
        return all(video_frame_count(path) == length for path, length in paths)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def build_info(
    source_info: dict[str, Any], total_episodes: int, total_frames: int, total_tasks: int
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    chunks_size = int(info["chunks_size"])
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_episodes * len(CAMERA_KEYS),
            "total_chunks": math.ceil(total_episodes / chunks_size),
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    for camera in CAMERA_KEYS:
        video_info = info["features"][camera]["info"]
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.fps"] = int(info["fps"])
    return info


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)

    split_dir = args.split_dir.resolve() if args.split_dir else source / "split"
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    segments, sub_tasks, source_info = load_segments(
        source, split_dir, args.task_spec.resolve(), args.max_source_episodes
    )
    if not segments:
        raise ValueError("no labeled segments found")
    fps = int(source_info["fps"])
    staging = output.with_name(f".{output.name}.building")
    if staging.exists() and not args.resume:
        raise FileExistsError(
            f"staging directory from another run already exists: {staging}"
        )
    staging.mkdir(parents=True, exist_ok=args.resume)

    by_source: dict[int, list[Segment]] = {}
    for segment in segments:
        by_source.setdefault(segment.source_episode, []).append(segment)
    source_stats = {
        int(item["episode_index"]): item["stats"]
        for item in read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }

    try:
        tasks = [
            {"task_index": index, "task": str(subtask["description"])}
            for index, subtask in enumerate(sub_tasks)
        ]
        episodes = [
            {
                "episode_index": segment.output_episode,
                "tasks": [segment.task],
                "length": segment.length,
            }
            for segment in segments
        ]
        mappings = [
            {
                "episode_index": segment.output_episode,
                "source_episode_index": segment.source_episode,
                "subtask_index": segment.subtask_index,
                "subtask_id": segment.subtask_id,
                "task": segment.task,
                "source_start_frame": segment.start_frame,
                "source_end_frame": segment.end_frame,
                "length": segment.length,
            }
            for segment in segments
        ]
        info = build_info(source_info, len(segments), sum(s.length for s in segments), len(tasks))
        write_json(staging / "meta" / "info.json", info)
        write_jsonl(staging / "meta" / "tasks.jsonl", tasks)
        write_jsonl(staging / "meta" / "episodes.jsonl", episodes)
        write_jsonl(staging / "meta" / "source_mapping.jsonl", mappings)

        episode_stats: list[dict[str, Any]] = []
        total_groups = len(by_source)
        for completed, (source_episode, group) in enumerate(by_source.items(), start=1):
            episode_stats.extend(
                convert_parquet_group(
                    source,
                    staging,
                    source_episode,
                    group,
                    fps,
                    source_stats[source_episode],
                    source_info["features"],
                )
            )
            if completed % 25 == 0 or completed == total_groups:
                print(f"parquet: {completed}/{total_groups} source episodes", flush=True)
        episode_stats.sort(key=lambda item: item["episode_index"])
        write_jsonl(staging / "meta" / "episodes_stats.jsonl", episode_stats)

        pending_groups: dict[int, list[Segment]] = {}
        reused_groups = 0
        for source_episode, group in by_source.items():
            if args.resume and video_group_is_complete(staging, group):
                reused_groups += 1
            else:
                pending_groups[source_episode] = group
        if args.resume:
            print(
                f"video resume: reused {reused_groups}/{total_groups} source episodes",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    convert_video_group,
                    source,
                    staging,
                    source_episode,
                    group,
                    fps,
                    args.crf,
                    args.preset,
                ): source_episode
                for source_episode, group in pending_groups.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                source_episode = futures[future]
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"video conversion failed for source episode {source_episode}"
                    ) from error
                if completed % 10 == 0 or completed == len(pending_groups):
                    print(
                        f"video: {reused_groups + completed}/{total_groups} source episodes",
                        flush=True,
                    )

        staging.rename(output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "source_episodes": len(by_source),
                    "episodes": len(segments),
                    "frames": sum(segment.length for segment in segments),
                    "tasks": len(tasks),
                },
                indent=2,
            )
        )
    except BaseException:
        print(f"conversion stopped; partial output remains in {staging}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
