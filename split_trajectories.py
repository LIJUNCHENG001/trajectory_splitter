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
END_STATE_COLUMN = "observation.state.end"
ACTION_COLUMN = "actions"


@dataclass(frozen=True)
class Config:
    fps: float
    smooth_frames: int = 5
    baseline_frames: int = 8
    mutation_change: float = 0.05
    baseline_tolerance: float = 0.006
    gripper_drop: float = 0.12
    gripper_reopen: float = 0.12
    gripper_peak_slack: float = 0.01
    gripper_contact_gap: float = 0.02
    gripper_contact_max_position: float = 0.12
    cut2_release_distance: float = 0.10
    cut2_max_search_seconds: float = 1.0
    max_single_joint_step: float = 0.3
    min_joint_action_ratio: float = 5.0
    arm_stationary_step: float = 0.006
    left_start_min_motion_seconds: float = 0.5
    min_action_motion_seconds: float = 0.5
    min_state_stationary_seconds: float = 0.5
    state_cut_delay_seconds: float = 0.5
    max_arm_lead_seconds: float = 1.0
    cut4_ignore_tail_seconds: float = 5.0


class EpisodeQualityRejection(ValueError):
    """Raised when an episode violates a data-quality constraint."""


class RecordingDiscontinuity(EpisodeQualityRejection):
    """Raised when consecutive robot states imply a recording jump."""


class EarlyArmOverlap(EpisodeQualityRejection):
    """Raised when the other arm starts too early during an arm return."""


class InsufficientReleaseDistance(EpisodeQualityRejection):
    """Raised when the gripper does not visibly separate from the bread."""


@dataclass(frozen=True)
class Cuts:
    cut1_right_arm_mutation: int
    cut2_bread_release_completion: int
    cut3_right_arm_return_stop: int
    cut4_left_arm_return_stop: int

    def indices(self) -> list[int]:
        return [
            self.cut1_right_arm_mutation,
            self.cut2_bread_release_completion,
            self.cut3_right_arm_return_stop,
            self.cut4_left_arm_return_stop,
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
        help="Write cut summaries without split parquet files.",
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
) -> int:
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
        if deviation >= config.mutation_change:
            return last_stable
    raise ValueError(f"{name}: no change of at least {config.mutation_change} found")


def find_gripper_closures(
    signal: np.ndarray, action_signal: np.ndarray, config: Config
) -> tuple[list[int], list[int], list[int]]:
    """Return closing peaks, contact peaks, and contact-release frames."""
    values = smooth(signal, config.smooth_frames)
    action_values = smooth(action_signal, config.smooth_frames)
    closures: list[int] = []
    contact_closures: list[int] = []
    contact_release_frames: list[int] = []
    armed = True
    peak = trough = float(values[0])
    peak_index = 0
    action_trough = float(action_values[0])

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
                action_trough = float(action_values[index])
        else:
            trough = min(trough, value)
            action_trough = min(action_trough, float(action_values[index]))
            if value - trough >= config.gripper_reopen:
                if (
                    trough <= config.gripper_contact_max_position
                    and trough - action_trough >= config.gripper_contact_gap
                ):
                    contact_closures.append(closures[-1])
                    contact_release_frames.append(index)
                armed = True
                peak = value
                peak_index = index
    return closures, contact_closures, contact_release_frames


def find_release_distance_cut(
    positions: np.ndarray, release_frame: int, config: Config
) -> int:
    """Return the first frame 10 cm away from the confirmed release position."""
    values = np.column_stack(
        [
            smooth(positions[:, column], config.smooth_frames)
            for column in range(positions.shape[1])
        ]
    )
    search_stop = min(
        len(values),
        release_frame + round(config.cut2_max_search_seconds * config.fps) + 1,
    )
    distances = np.linalg.norm(
        values[release_frame:search_stop] - values[release_frame], axis=1
    )
    matches = np.flatnonzero(distances >= config.cut2_release_distance)
    if not len(matches):
        raise InsufficientReleaseDistance(
            "right arm: does not move "
            f"{config.cut2_release_distance:.3f}m away from bread within "
            f"{config.cut2_max_search_seconds:.3f}s after release"
        )
    return release_frame + int(matches[0])


def find_any_motor_mutation(
    signals: np.ndarray,
    names: list[str],
    config: Config,
    *,
    start: int = 0,
) -> tuple[int, str]:
    """Find the earliest mutation among the requested motors."""
    candidates = []
    for column, name in enumerate(names):
        try:
            index = find_mutation(signals[:, column], start, config, name)
            candidates.append((index, name))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("no motor mutation found while scanning forward")
    return min(candidates, key=lambda item: item[0])


def find_sustained_arm_motion_start(
    signals: np.ndarray,
    names: list[str],
    config: Config,
    *,
    start: int = 0,
) -> tuple[int, str]:
    """Return the start of the first sustained visible-speed arm motion."""
    values = np.column_stack(
        [
            smooth(signals[:, column], config.smooth_frames)
            for column in range(signals.shape[1])
        ]
    )
    steps = np.abs(np.diff(values, axis=0))
    required_frames = max(1, round(config.left_start_min_motion_seconds * config.fps))
    run_length = 0
    for index in range(start, len(steps)):
        if np.max(steps[index]) >= config.arm_stationary_step:
            run_length += 1
            if run_length >= required_frames:
                first = index - run_length + 1
                motor_steps = np.max(steps[first : index + 1], axis=0)
                motor = names[int(np.argmax(motor_steps))]
                return first, motor
        else:
            run_length = 0
    raise ValueError("arm: no sustained motion found while scanning forward")


def find_sustained_action_motion_runs(
    action_signals: np.ndarray, config: Config
) -> list[tuple[int, int]]:
    """Return action-motion runs that last at least the configured duration."""
    action_values = np.column_stack(
        [
            smooth(action_signals[:, column], config.smooth_frames)
            for column in range(action_signals.shape[1])
        ]
    )
    action_steps = np.abs(np.diff(action_values, axis=0))
    moving_mask = np.max(action_steps, axis=1) >= config.arm_stationary_step
    min_motion_frames = max(1, round(config.min_action_motion_seconds * config.fps))
    motion_runs: list[tuple[int, int]] = []
    run_start = None
    for index in range(len(moving_mask) + 1):
        is_moving = index < len(moving_mask) and moving_mask[index]
        if is_moving and run_start is None:
            run_start = index
        elif not is_moving and run_start is not None:
            if index - run_start >= min_motion_frames:
                motion_runs.append((run_start, index - 1))
            run_start = None
    return motion_runs


def find_arm_stop_from_action_and_state(
    action_signals: np.ndarray,
    state_signals: np.ndarray,
    names: list[str],
    config: Config,
    *,
    motion_search_start: int,
    motion_start_before: int,
) -> tuple[int, list[str]]:
    """Use action to select a motion, then cut after its confirmed state stop."""
    motion_runs = [
        (run_start, run_end)
        for run_start, run_end in find_sustained_action_motion_runs(
            action_signals, config
        )
        if run_end >= motion_search_start and run_start < motion_start_before
    ]
    if not motion_runs:
        raise ValueError("arm: no sustained action motion found in the search window")

    _, action_last_moving = motion_runs[-1]
    state_values = np.column_stack(
        [
            smooth(state_signals[:, column], config.smooth_frames)
            for column in range(state_signals.shape[1])
        ]
    )
    state_steps = np.abs(np.diff(state_values, axis=0))
    state_motion = np.max(state_steps, axis=1) >= config.arm_stationary_step
    stationary_steps = max(
        1, round(config.min_state_stationary_seconds * config.fps) - 1
    )
    stable_start = None
    candidates = range(
        action_last_moving + 1,
        len(state_motion) - stationary_steps + 1,
    )
    for candidate in candidates:
        if not np.any(state_motion[candidate : candidate + stationary_steps]):
            stable_start = candidate
            break
    if stable_start is None:
        raise ValueError("arm: state has no sustained stop after action motion")
    cut_delay_frames = round(config.state_cut_delay_seconds * config.fps)
    frame = stable_start + cut_delay_frames
    if frame >= len(state_values):
        raise ValueError("arm: confirmed state stop is beyond the episode")
    previous_motion = np.flatnonzero(state_motion[:stable_start])
    last_moving = int(previous_motion[-1]) if len(previous_motion) else stable_start
    motors = [
        name
        for column, name in enumerate(names)
        if state_steps[last_moving, column] >= config.arm_stationary_step
    ]
    return frame, motors


def find_optional_task_end(
    right_actions: np.ndarray,
    right_states: np.ndarray,
    right_names: list[str],
    config: Config,
    *,
    cut4: int,
) -> tuple[int | None, list[str], str]:
    """Detect the final right-arm return without affecting trajectory splitting."""
    try:
        frame, motors = find_arm_stop_from_action_and_state(
            right_actions,
            right_states,
            right_names,
            config,
            motion_search_start=cut4 + 1,
            motion_start_before=len(right_states),
        )
        return frame, motors, ""
    except ValueError as error:
        return None, [], str(error)


def reject_early_arm_overlap(
    action_signals: np.ndarray,
    returning_stable_start: int,
    search_start: int,
    config: Config,
    *,
    moving_arm: str,
    returning_arm: str,
) -> None:
    """Reject sustained motion starting too early before the other arm is stable."""
    max_lead_frames = round(config.max_arm_lead_seconds * config.fps)
    for run_start, run_end in find_sustained_action_motion_runs(action_signals, config):
        if (
            run_end < search_start
            or not run_start <= returning_stable_start <= run_end
        ):
            continue
        lead_frames = returning_stable_start - run_start
        if lead_frames > max_lead_frames:
            raise EarlyArmOverlap(
                f"{moving_arm} arm starts {lead_frames / config.fps:.3f}s before "
                f"{returning_arm} arm return stability "
                f"(frames {run_start}->{returning_stable_start}), exceeds "
                f"{config.max_arm_lead_seconds:.3f}s"
            )


def reject_recording_discontinuity(
    joints: np.ndarray, action_joints: np.ndarray, config: Config
) -> None:
    """Reject an episode with an implausible one-frame joint-state jump."""
    if len(joints) < 2:
        return
    joint_deltas = np.diff(joints, axis=0)
    joint_steps = np.linalg.norm(joint_deltas, axis=1)
    max_single_joint_steps = np.max(np.abs(joint_deltas), axis=1)
    action_steps = np.linalg.norm(np.diff(action_joints, axis=0), axis=1)
    ratios = np.divide(
        joint_steps,
        action_steps,
        out=np.full_like(joint_steps, np.inf),
        where=action_steps > 0,
    )
    candidates = np.flatnonzero(
        (max_single_joint_steps >= config.max_single_joint_step)
        & (ratios >= config.min_joint_action_ratio)
    )
    if len(candidates):
        frame = int(candidates[np.argmax(joint_steps[candidates])])
        raise RecordingDiscontinuity(
            f"joint-state jump {joint_steps[frame]:.6f}, max single-joint jump "
            f"{max_single_joint_steps[frame]:.6f}, action jump "
            f"{action_steps[frame]:.6f} (ratio {ratios[frame]:.2f}) at frame "
            f"{frame}->{frame + 1}"
        )


def detect_cuts(
    table, config: Config, joint_names: list[str], gripper_names: list[str]
) -> tuple[Cuts, int | None, int, dict[str, str]]:
    joints = np.asarray(table[JOINT_COLUMN].to_pylist(), dtype=float)
    grippers = np.asarray(table[GRIPPER_COLUMN].to_pylist(), dtype=float)
    end_states = np.asarray(table[END_STATE_COLUMN].to_pylist(), dtype=float)
    actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=float)
    action_joints = np.concatenate((actions[:, :6], actions[:, 7:13]), axis=1)
    reject_recording_discontinuity(joints, action_joints, config)
    left_names = [name for name in joint_names if name.endswith("_left")]
    right_names = [name for name in joint_names if name.endswith("_right")]
    left_joints = joints[:, [joint_names.index(name) for name in left_names]]
    right_joints = joints[:, [joint_names.index(name) for name in right_names]]
    left_actions = actions[:, :6]
    right_actions = actions[:, 7:13]
    right_gripper = grippers[:, gripper_names.index("right_gripper_percent")]
    right_gripper_action = actions[:, 13]

    cut1, cut1_motor = find_any_motor_mutation(right_joints, right_names, config)
    # Keep the first segment non-empty when motion exists in the first window.
    cut1 = max(1, cut1)
    closures, contact_closures, contact_release_frames = find_gripper_closures(
        right_gripper, right_gripper_action, config
    )
    if len(contact_closures) < 2:
        raise ValueError(
            f"right gripper: found {len(contact_closures)} object-contact closures, need 2"
        )
    cut2 = find_release_distance_cut(
        end_states[:, 6:9], contact_release_frames[1], config
    )
    left_start, cut3_motor = find_sustained_arm_motion_start(
        left_actions, left_names, config, start=cut2 + 1
    )
    cut3, cut3_return_motors = find_arm_stop_from_action_and_state(
        right_actions,
        right_joints,
        right_names,
        config,
        motion_search_start=cut2 + 1,
        motion_start_before=left_start,
    )
    state_cut_delay_frames = round(config.state_cut_delay_seconds * config.fps)
    right_stable_start = cut3 - state_cut_delay_frames
    reject_early_arm_overlap(
        left_actions,
        right_stable_start,
        cut2,
        config,
        moving_arm="left",
        returning_arm="right",
    )
    ignored_tail_frames = round(config.cut4_ignore_tail_seconds * config.fps)
    cut4_search_start = cut3 + 1
    cut4_search_stop = len(left_joints) - ignored_tail_frames
    if cut4_search_stop - cut4_search_start < config.baseline_frames:
        raise ValueError("episode is too short for cut4 tail exclusion")
    cut4, cut4_motors = find_arm_stop_from_action_and_state(
        left_actions,
        left_joints,
        left_names,
        config,
        motion_search_start=cut4_search_start,
        motion_start_before=cut4_search_stop,
    )
    left_stable_start = cut4 - state_cut_delay_frames
    reject_early_arm_overlap(
        right_actions,
        left_stable_start,
        cut3,
        config,
        moving_arm="right",
        returning_arm="left",
    )
    cuts = Cuts(cut1, cut2, cut3, cut4)
    if cuts.indices() != sorted(cuts.indices()) or len(set(cuts.indices())) != 4:
        raise ValueError(f"cut points are not strictly increasing: {cuts.indices()}")
    task_end, task_end_motors, task_end_error = find_optional_task_end(
        right_actions,
        right_joints,
        right_names,
        config,
        cut4=cut4,
    )
    return (
        cuts,
        task_end,
        len(closures),
        {
            "cut1_motor": cut1_motor,
            "cut3_return_motors": ",".join(cut3_return_motors),
            "cut3_left_start_motor": cut3_motor,
            "cut3_left_start_frame": str(left_start),
            "cut4_motors": ",".join(cut4_motors),
            "task_end_motors": ",".join(task_end_motors),
            "task_end_error": task_end_error,
        },
    )


def write_segments(
    table, episode_index: int, cuts: Cuts, output: Path, overwrite: bool
) -> list[str]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "episode_index",
        "status",
        "cut1_frame",
        "cut1_time",
        "cut1_motor",
        "cut2_frame",
        "cut2_time",
        "cut3_frame",
        "cut3_time",
        "cut3_return_motors",
        "cut3_left_start_frame",
        "cut3_left_start_motor",
        "cut4_frame",
        "cut4_time",
        "cut4_motors",
        "task_end_status",
        "task_end_frame",
        "task_end_time",
        "task_end_motors",
        "task_end_error",
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
    details: list[dict] = []

    for position, path in enumerate(paths, 1):
        episode_index = int(path.stem.removeprefix("episode_"))
        print(f"[{position}/{len(paths)}] episode {episode_index:06d}", flush=True)
        row = {"episode_index": episode_index, "status": "error", "error": ""}
        try:
            table = pq.read_table(path)
            timestamps = np.asarray(table["timestamp"], dtype=float)
            cuts, task_end, closure_count, trigger_sources = detect_cuts(
                table, config, joint_names, gripper_names
            )
            cut_times = [float(timestamps[index]) for index in cuts.indices()]
            task_end_time = (
                float(timestamps[task_end]) if task_end is not None else None
            )
            row.update(
                {
                    "status": "ok",
                    "cut1_frame": cuts.indices()[0],
                    "cut1_time": cut_times[0],
                    "cut1_motor": trigger_sources["cut1_motor"],
                    "cut2_frame": cuts.indices()[1],
                    "cut2_time": cut_times[1],
                    "cut3_frame": cuts.indices()[2],
                    "cut3_time": cut_times[2],
                    "cut3_return_motors": trigger_sources["cut3_return_motors"],
                    "cut3_left_start_frame": trigger_sources["cut3_left_start_frame"],
                    "cut3_left_start_motor": trigger_sources["cut3_left_start_motor"],
                    "cut4_frame": cuts.indices()[3],
                    "cut4_time": cut_times[3],
                    "cut4_motors": trigger_sources["cut4_motors"],
                    "task_end_status": (
                        "ok" if task_end is not None else "unavailable"
                    ),
                    "task_end_frame": task_end,
                    "task_end_time": task_end_time,
                    "task_end_motors": trigger_sources["task_end_motors"],
                    "task_end_error": trigger_sources["task_end_error"],
                    "right_gripper_closures": closure_count,
                }
            )
            files = (
                []
                if args.detect_only
                else write_segments(
                    table, episode_index, cuts, args.output, args.overwrite
                )
            )
            details.append(
                {
                    "episode_index": episode_index,
                    "source": str(path),
                    "cut_frames": cuts.indices(),
                    "cut_times": cut_times,
                    "task_end": {
                        "status": "ok" if task_end is not None else "unavailable",
                        "frame": task_end,
                        "time": task_end_time,
                        "motors": trigger_sources["task_end_motors"],
                        "error": trigger_sources["task_end_error"],
                    },
                    "trigger_sources": trigger_sources,
                    "split_files": files,
                }
            )
        except EpisodeQualityRejection as error:
            row["status"] = "rejected"
            row["error"] = str(error)
            print(f"  质检剔除：{row['error']}", file=sys.stderr)
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
                "rejected": [row for row in rows if row["status"] == "rejected"],
                "errors": [row for row in rows if row["status"] == "error"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rejected = sum(row["status"] == "rejected" for row in rows)
    failed = sum(row["status"] == "error" for row in rows)
    succeeded = len(rows) - rejected - failed
    print(
        f"完成：成功 {succeeded}，质检剔除 {rejected}，失败 {failed}，"
        f"输出 {args.output}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
