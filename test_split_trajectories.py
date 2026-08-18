#!/usr/bin/env python3

import unittest

import numpy as np

from split_trajectories import Config, RecordingDiscontinuity, reject_recording_discontinuity


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


if __name__ == "__main__":
    unittest.main()
