#!/usr/bin/env python3

import unittest

import numpy as np

from split_trajectories import (
    Config,
    EarlyArmOverlap,
    InsufficientReleaseDistance,
    RecordingDiscontinuity,
    find_arm_stop_from_action_and_state,
    find_gripper_closures,
    find_optional_task_end,
    find_release_distance_cut,
    find_sustained_arm_motion_start,
    reject_early_arm_overlap,
    reject_recording_discontinuity,
)


class RecordingDiscontinuityTest(unittest.TestCase):
    def test_accepts_step_below_threshold(self) -> None:
        joints = np.zeros((2, 12))
        actions = np.zeros((2, 12))
        joints[1, :2] = 0.25
        reject_recording_discontinuity(joints, actions, Config(fps=30))

    def test_accepts_action_driven_fast_motion(self) -> None:
        joints = np.zeros((2, 12))
        actions = np.zeros((2, 12))
        joints[1, 0] = 0.4
        actions[1, 0] = 0.1
        reject_recording_discontinuity(joints, actions, Config(fps=30))

    def test_rejects_step_at_threshold(self) -> None:
        joints = np.zeros((2, 12))
        actions = np.zeros((2, 12))
        joints[1, 4] = 0.3
        actions[1, 4] = 0.06
        with self.assertRaisesRegex(
            RecordingDiscontinuity,
            r"0\.300000.*single-joint jump 0\.300000.*ratio 5\.00.*frame 0->1",
        ):
            reject_recording_discontinuity(joints, actions, Config(fps=30))


class GripperClosureTest(unittest.TestCase):
    def test_excludes_empty_closure_from_object_contacts(self) -> None:
        state = np.array(
            [
                0.8,
                0.8,
                0.8,
                0.6,
                0.07,
                0.07,
                0.3,
                0.8,
                0.8,
                0.5,
                0.0,
                0.0,
                0.3,
                0.8,
                0.8,
                0.5,
                0.07,
                0.07,
                0.3,
                0.8,
                0.8,
                0.5,
                0.0,
            ]
        )
        action = state.copy()
        action[[4, 5, 16, 17]] = 0.0

        closures, contact_closures, release_frames = find_gripper_closures(
            state, action, Config(fps=30, smooth_frames=1)
        )

        self.assertEqual(closures, [2, 8, 14, 20])
        self.assertEqual(contact_closures, [2, 14])
        self.assertEqual(release_frames, [6, 18])

    def test_excludes_partial_closure_without_object_contact(self) -> None:
        state = np.array([0.8, 0.8, 0.6, 0.2, 0.2, 0.4, 0.8])
        action = np.array([0.8, 0.8, 0.6, 0.18, 0.18, 0.4, 0.8])

        closures, contact_closures, _ = find_gripper_closures(
            state, action, Config(fps=30, smooth_frames=1)
        )

        self.assertEqual(closures, [1])
        self.assertEqual(contact_closures, [])

    def test_cut2_uses_first_frame_ten_centimetres_from_release(self) -> None:
        positions = np.zeros((40, 3))
        positions[11:, 0] = np.arange(29) * 0.02

        frame = find_release_distance_cut(
            positions, release_frame=10, config=Config(fps=30, smooth_frames=1)
        )

        self.assertEqual(frame, 16)

    def test_cut2_rejects_when_distance_is_not_reached_within_one_second(self) -> None:
        positions = np.zeros((60, 3))
        positions[41:, 0] = 0.2

        with self.assertRaisesRegex(
            InsufficientReleaseDistance, r"0\.100m.*1\.000s"
        ):
            find_release_distance_cut(
                positions, release_frame=10, config=Config(fps=30, smooth_frames=1)
            )


class ArmReturnTest(unittest.TestCase):
    def test_rejects_arm_start_more_than_1_0_seconds_early(self) -> None:
        actions = np.zeros((100, 1))
        actions[10:61, 0] = np.arange(51) * 0.02
        actions[61:, 0] = 1.0

        with self.assertRaisesRegex(EarlyArmOverlap, r"1\.333s.*exceeds 1\.000s"):
            reject_early_arm_overlap(
                actions,
                returning_stable_start=50,
                search_start=0,
                config=Config(fps=30, smooth_frames=1),
                moving_arm="right",
                returning_arm="left",
            )

    def test_accepts_arm_start_exactly_1_0_seconds_early(self) -> None:
        actions = np.zeros((100, 1))
        actions[20:61, 0] = np.arange(41) * 0.02
        actions[61:, 0] = 0.8

        reject_early_arm_overlap(
            actions,
            returning_stable_start=50,
            search_start=0,
            config=Config(fps=30, smooth_frames=1),
            moving_arm="left",
            returning_arm="right",
        )

    def test_motion_start_ignores_drift_and_brief_jitter(self) -> None:
        actions = np.zeros((100, 2))
        actions[:, 0] = np.arange(100) * 0.002
        actions[30, 0] += 0.02
        actions[50:71, 0] += np.arange(21) * 0.02
        actions[71:, 0] += 0.4

        frame, motor = find_sustained_arm_motion_start(
            actions,
            ["motor_0", "motor_1"],
            Config(fps=30, smooth_frames=1),
            start=10,
        )

        self.assertEqual(frame, 50)
        self.assertEqual(motor, "motor_0")

    def test_motion_start_ignores_short_preadjustment(self) -> None:
        actions = np.zeros((100, 1))
        actions[20:29, 0] = np.arange(9) * 0.02
        actions[29:40, 0] = 0.16
        actions[40:61, 0] = 0.16 + np.arange(21) * 0.02
        actions[61:, 0] = 0.56

        frame, motor = find_sustained_arm_motion_start(
            actions, ["motor_0"], Config(fps=30, smooth_frames=1), start=10
        )

        self.assertEqual(frame, 40)
        self.assertEqual(motor, "motor_0")

    def test_action_selects_motion_and_state_locates_stop(self) -> None:
        actions = np.zeros((120, 1))
        actions[5:26, 0] = np.arange(21) * 0.02
        actions[26:, 0] = 0.4
        states = np.zeros((120, 1))
        states[8:30, 0] = np.arange(22) * 0.02
        states[30:, 0] = 0.42

        frame, motors = find_arm_stop_from_action_and_state(
            actions,
            states,
            ["motor_0"],
            Config(fps=30, smooth_frames=1),
            motion_search_start=0,
            motion_start_before=40,
        )

        self.assertEqual(frame, 44)
        self.assertEqual(motors, ["motor_0"])

    def test_motion_may_start_before_search_window_and_overlap_anchor(self) -> None:
        actions = np.zeros((100, 1))
        actions[10:41, 0] = np.arange(31) * 0.02
        actions[41:, 0] = 0.6
        states = actions.copy()

        frame, _ = find_arm_stop_from_action_and_state(
            actions,
            states,
            ["motor_0"],
            Config(fps=30, smooth_frames=1),
            motion_search_start=20,
            motion_start_before=30,
        )

        self.assertEqual(frame, 55)

    def test_ignores_short_action_burst_after_sustained_motion(self) -> None:
        actions = np.zeros((100, 1))
        actions[10:31, 0] = np.arange(21) * 0.02
        actions[31:50, 0] = 0.4
        actions[50:56, 0] = 0.4 + np.arange(6) * 0.02
        actions[56:, 0] = 0.5
        states = actions.copy()

        frame, _ = find_arm_stop_from_action_and_state(
            actions,
            states,
            ["motor_0"],
            Config(fps=30, smooth_frames=1),
            motion_search_start=0,
            motion_start_before=80,
        )

        self.assertEqual(frame, 45)

    def test_ignores_state_pause_shorter_than_half_a_second(self) -> None:
        actions = np.zeros((100, 1))
        actions[10:31, 0] = np.arange(21) * 0.02
        actions[31:, 0] = 0.4
        states = actions.copy()
        states[40:47, 0] = 0.4 + np.arange(7) * 0.02
        states[47:, 0] = 0.52

        frame, _ = find_arm_stop_from_action_and_state(
            actions,
            states,
            ["motor_0"],
            Config(fps=30, smooth_frames=1),
            motion_search_start=0,
            motion_start_before=80,
        )

        self.assertEqual(frame, 61)

    def test_task_end_reuses_right_arm_stop_rule(self) -> None:
        actions = np.zeros((120, 1))
        actions[50:71, 0] = np.arange(21) * 0.02
        actions[71:, 0] = 0.4
        states = np.zeros((120, 1))
        states[53:75, 0] = np.arange(22) * 0.02
        states[75:, 0] = 0.42

        frame, motors, error = find_optional_task_end(
            actions,
            states,
            ["motor_0_right"],
            Config(fps=30, smooth_frames=1),
            cut4=40,
        )

        self.assertEqual(frame, 89)
        self.assertEqual(motors, ["motor_0_right"])
        self.assertEqual(error, "")

    def test_missing_task_end_does_not_fail_split_detection(self) -> None:
        actions = np.zeros((80, 1))
        actions[50:80, 0] = np.arange(30) * 0.02

        frame, motors, error = find_optional_task_end(
            actions,
            actions,
            ["motor_0_right"],
            Config(fps=30, smooth_frames=1),
            cut4=40,
        )

        self.assertIsNone(frame)
        self.assertEqual(motors, [])
        self.assertIn("no sustained stop", error)


if __name__ == "__main__":
    unittest.main()
