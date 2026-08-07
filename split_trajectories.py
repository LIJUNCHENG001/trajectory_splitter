#!/usr/bin/env python3
"""Detect four motion events and split LeRobot trajectories into five parts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("/home/geek/share3/agilex_make_breakfast_380")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
JOINT_COLUMN = "observation.state.joint"
GRIPPER_COLUMN = "observation.gripper_position"
SEGMENT_LABELS = [
    "阶段1_切分点1前",
    "阶段2_切分点1至2",
    "阶段3_切分点2至3",
    "阶段4_切分点3至4",
    "阶段5_切分点4后",
]


@dataclass(frozen=True)
class Config:
    fps: float
    smooth_frames: int = 5
    baseline_frames: int = 8
    mutation_change: float = 0.05
    large_motor_change: float = 0.12
    baseline_tolerance: float = 0.006
    gripper_drop: float = 0.12
    gripper_reopen: float = 0.12
    gripper_peak_slack: float = 0.01


@dataclass(frozen=True)
class Cuts:
    cut1_right_arm_mutation: int
    cut2_third_right_gripper_close: int
    cut3_left_arm_mutation: int
    cut4_two_left_motors_reverse_change: int

    def indices(self) -> list[int]:
        return [
            self.cut1_right_arm_mutation,
            self.cut2_third_right_gripper_close,
            self.cut3_left_arm_mutation,
            self.cut4_two_left_motors_reverse_change,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Breakfast 380 trajectories using four signal events."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Write summaries/visualiser annotations without split parquet files.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(float, copy=True)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(signal.astype(float), (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def find_mutation(
    signal: np.ndarray,
    start: int,
    config: Config,
    name: str,
    change: float | None = None,
) -> int:
    threshold = config.mutation_change if change is None else change
    values = smooth(signal, config.smooth_frames)
    baseline_stop = min(len(values), start + config.baseline_frames)
    if baseline_stop - start < 2:
        raise ValueError(f"{name}: not enough frames for a baseline")
    baseline = float(np.median(values[start:baseline_stop]))
    last_stable = start
    for index in range(start + 1, len(values)):
        deviation = abs(values[index] - baseline)
        if deviation <= config.baseline_tolerance:
            last_stable = index
        if deviation >= threshold:
            return last_stable
    raise ValueError(f"{name}: no change of at least {threshold} found")


def find_gripper_closures(signal: np.ndarray, config: Config) -> list[int]:
    """Return closing onset indices using drop/reopen hysteresis."""
    values = smooth(signal, config.smooth_frames)
    closures: list[int] = []
    armed = True
    peak = trough = float(values[0])
    peak_index = 0

    for index in range(1, len(values)):
        value = float(values[index])
        if armed:
            peak = max(peak, value)
            if value >= peak - config.gripper_peak_slack:
                peak_index = index
            if peak - value >= config.gripper_drop:
                closures.append(peak_index)
                armed = False
                trough = value
        else:
            trough = min(trough, value)
            if value - trough >= config.gripper_reopen:
                armed = True
                peak = value
                peak_index = index
    return closures


def find_any_motor_mutation(
    signals: np.ndarray,
    names: list[str],
    config: Config,
    *,
    start: int = 0,
    reverse: bool = False,
) -> tuple[int, str]:
    """Find the earliest mutation among motors in the requested scan direction."""
    candidates = []
    for column, name in enumerate(names):
        signal = signals[:, column]
        try:
            if reverse:
                reversed_index = find_mutation(signal[::-1], 0, config, name)
                index = len(signal) - 1 - reversed_index
            else:
                index = find_mutation(signal, start, config, name)
            candidates.append((index, name))
        except ValueError:
            continue
    if not candidates:
        direction = "backward" if reverse else "forward"
        raise ValueError(f"no motor mutation found while scanning {direction}")
    selector = max if reverse else min
    return selector(candidates, key=lambda item: item[0])


def find_two_motor_reverse_change(
    signals: np.ndarray, names: list[str], config: Config
) -> tuple[int, list[str]]:
    """Return the point where a second left motor changes while scanning backward."""
    candidates = []
    for column, name in enumerate(names):
        try:
            reversed_index = find_mutation(
                signals[:, column][::-1],
                0,
                config,
                name,
                change=config.large_motor_change,
            )
            index = len(signals) - 1 - reversed_index
            candidates.append((index, name))
        except ValueError:
            continue
    if len(candidates) < 2:
        raise ValueError(
            f"only {len(candidates)} left motors have a reverse change of at least "
            f"{config.large_motor_change}; need 2"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[1][0], [candidates[0][1], candidates[1][1]]


def detect_cuts(
    table, config: Config, joint_names: list[str], gripper_names: list[str]
) -> tuple[Cuts, int, dict[str, str]]:
    joints = np.asarray(table[JOINT_COLUMN].to_pylist(), dtype=float)
    grippers = np.asarray(table[GRIPPER_COLUMN].to_pylist(), dtype=float)
    left_names = [name for name in joint_names if name.endswith("_left")]
    right_names = [name for name in joint_names if name.endswith("_right")]
    left_joints = joints[:, [joint_names.index(name) for name in left_names]]
    right_joints = joints[:, [joint_names.index(name) for name in right_names]]
    right_gripper = grippers[:, gripper_names.index("right_gripper_percent")]

    cut1, cut1_motor = find_any_motor_mutation(
        right_joints, right_names, config
    )
    # The visualiser cannot represent an empty 0–0 s segment when motion
    # already exists in the first smoothed window.
    cut1 = max(1, cut1)
    closures = find_gripper_closures(right_gripper, config)
    if len(closures) < 3:
        raise ValueError(f"right gripper: found {len(closures)} closures, need 3")
    cut2 = closures[2]
    cut3, cut3_motor = find_any_motor_mutation(
        left_joints, left_names, config, start=cut2 + 1
    )
    cut4, cut4_motors = find_two_motor_reverse_change(
        left_joints, left_names, config
    )
    cuts = Cuts(cut1, cut2, cut3, cut4)
    if cuts.indices() != sorted(cuts.indices()) or len(set(cuts.indices())) != 4:
        raise ValueError(f"cut points are not strictly increasing: {cuts.indices()}")
    return cuts, len(closures), {
        "cut1_motor": cut1_motor,
        "cut3_motor": cut3_motor,
        "cut4_motors": ",".join(cut4_motors),
    }


def write_segments(table, episode_index: int, cuts: Cuts, output: Path, overwrite: bool) -> list[str]:
    boundaries = [0, *cuts.indices(), table.num_rows]
    written: list[str] = []
    episode_dir = output / "parquet" / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for segment_index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        path = episode_dir / f"segment_{segment_index:02d}.parquet"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
        pq.write_table(table.slice(start, end - start), path, compression="zstd")
        written.append(str(path))
    return written


def annotation_segments(timestamps: np.ndarray, row_count: int, cuts: Cuts, fps: float) -> list[dict]:
    boundaries = [0, *cuts.indices(), row_count]
    duration = row_count / fps
    segments = []
    for number, (start, end, label) in enumerate(
        zip(boundaries, boundaries[1:], SEGMENT_LABELS), 1
    ):
        start_time = float(timestamps[start]) if start < row_count else duration
        end_time = float(timestamps[end]) if end < row_count else duration
        segments.append(
            {
                "id": f"auto_segment_{number}",
                "start": round(start_time, 6),
                "end": round(end_time, 6),
                "label": label,
            }
        )
    return segments


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "episode_index",
        "status",
        "cut1_frame", "cut1_time",
        "cut1_motor",
        "cut2_frame", "cut2_time",
        "cut3_frame", "cut3_time",
        "cut3_motor",
        "cut4_frame", "cut4_time",
        "cut4_motors",
        "right_gripper_closures",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    info_path = args.dataset / "meta" / "info.json"
    if not info_path.is_file():
        print(f"错误：找不到 {info_path}", file=sys.stderr)
        return 2
    info = json.loads(info_path.read_text(encoding="utf-8"))
    config = Config(fps=float(info["fps"]))
    joint_names = info["features"][JOINT_COLUMN]["names"]
    gripper_names = info["features"][GRIPPER_COLUMN]["names"]
    paths = sorted(
        (args.dataset / "data").rglob("episode_*.parquet"),
        key=lambda path: int(path.stem.removeprefix("episode_")),
    )
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if not paths:
        print("错误：没有找到 parquet 文件", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    annotations: dict[str, list[dict]] = {}
    details: list[dict] = []

    for position, path in enumerate(paths, 1):
        episode_index = int(path.stem.removeprefix("episode_"))
        print(f"[{position}/{len(paths)}] episode {episode_index:06d}", flush=True)
        row = {"episode_index": episode_index, "status": "error", "error": ""}
        try:
            table = pq.read_table(path)
            timestamps = np.asarray(table["timestamp"], dtype=float)
            cuts, closure_count, trigger_sources = detect_cuts(
                table, config, joint_names, gripper_names
            )
            cut_times = [float(timestamps[index]) for index in cuts.indices()]
            row.update(
                {
                    "status": "ok",
                    "cut1_frame": cuts.indices()[0], "cut1_time": cut_times[0],
                    "cut1_motor": trigger_sources["cut1_motor"],
                    "cut2_frame": cuts.indices()[1], "cut2_time": cut_times[1],
                    "cut3_frame": cuts.indices()[2], "cut3_time": cut_times[2],
                    "cut3_motor": trigger_sources["cut3_motor"],
                    "cut4_frame": cuts.indices()[3], "cut4_time": cut_times[3],
                    "cut4_motors": trigger_sources["cut4_motors"],
                    "right_gripper_closures": closure_count,
                }
            )
            annotations[str(episode_index)] = annotation_segments(
                timestamps, table.num_rows, cuts, config.fps
            )
            files = [] if args.detect_only else write_segments(
                table, episode_index, cuts, args.output, args.overwrite
            )
            details.append(
                {
                    "episode_index": episode_index,
                    "source": str(path),
                    "cut_frames": cuts.indices(),
                    "cut_times": cut_times,
                    "trigger_sources": trigger_sources,
                    "segments": annotations[str(episode_index)],
                    "split_files": files,
                }
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"  失败：{row['error']}", file=sys.stderr)
        rows.append(row)

    write_csv(args.output / "cut_points.csv", rows)
    (args.output / "split_summary.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "config": asdict(config),
                "episodes": details,
                "errors": [row for row in rows if row["status"] != "ok"],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (args.output / "visualiser_segments.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_dataset": str(args.dataset.resolve()),
                "output_dataset": str(args.dataset.resolve()),
                "episodes": annotations,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"完成：成功 {len(rows) - failed}，失败 {failed}，输出 {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
