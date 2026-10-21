# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SO101 calibration storage."""

import contextlib
import io
import pathlib
import sys
import tempfile
import types
from unittest import mock

from absl.testing import absltest
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import get_calibration


def _motor(motor_id: int) -> dict[str, int]:
  return {
      "id": motor_id,
      "zero": 2011,
      "range_min": 800,
      "range_max": 3200,
  }


def _all_motors() -> dict[str, dict[str, int]]:
  return {name: _motor(constant.MOTOR_IDS[name]) for name in constant.JOINTS}


class _ScriptedBus:
  """Replays a fixed sequence of raw Present_Position frames."""

  def __init__(self, frames):
    self.frames = list(frames)
    self.reads = 0

  def sync_read(self, register, motors, normalize=True):
    assert register == "Present_Position" and not normalize
    frame = self.frames[self.reads]
    self.reads += 1
    return {name: frame[name] for name in motors}


class CaptureMiddleTest(absltest.TestCase):

  @mock.patch("builtins.input", return_value="")
  def test_capture_middle_calls_set_half_turn_homings(self, _):
    bus = mock.Mock()
    bus.sync_read.return_value = {name: 2047 for name in constant.JOINTS}

    with contextlib.redirect_stdout(io.StringIO()):
      middle = get_calibration._capture_middle(bus)

    bus.set_half_turn_homings.assert_called_once()
    bus.sync_read.assert_called_once_with("Present_Position", normalize=False)
    self.assertEqual(middle["shoulder_pan"], 2047)


class RecordRangesTest(absltest.TestCase):

  def _record(self, motors, frames):
    """Runs _record_ranges over the frames with lerobot's console stubbed."""
    bus = _ScriptedBus(frames)
    utils = types.SimpleNamespace(
        enter_pressed=lambda: bus.reads >= len(bus.frames),
        move_cursor_up=lambda lines: None,
    )
    lerobot_utils = types.SimpleNamespace(utils=utils)
    modules = {
        "lerobot": types.SimpleNamespace(utils=lerobot_utils),
        "lerobot.utils": lerobot_utils,
        "lerobot.utils.utils": utils,
    }
    with mock.patch.dict(sys.modules, modules):
      with contextlib.redirect_stdout(io.StringIO()):
        return get_calibration._record_ranges(bus, list(motors))

  def test_records_min_and_max_during_sweep(self):
    range_mins, range_maxes = self._record(
        ["shoulder_pan"],
        [{"shoulder_pan": raw} for raw in (2047, 1500, 3200, 2047)],
    )

    self.assertEqual(range_mins, {"shoulder_pan": 1500})
    self.assertEqual(range_maxes, {"shoulder_pan": 3200})

  def test_rejects_motors_that_never_moved(self):
    with self.assertRaisesRegex(ValueError, "wrist_flex"):
      self._record(
          ["elbow_flex", "wrist_flex"],
          [
              {"elbow_flex": raw, "wrist_flex": 1000}
              for raw in (2000, 1500, 2600)
          ],
      )


class CalibrationStorageTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    patcher = mock.patch.object(
        get_calibration.paths, "local_dir", return_value=pathlib.Path(tmp.name)
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_round_trips_one_arm(self):
    motors = _all_motors()
    get_calibration.write("follower", motors)

    self.assertEqual(get_calibration.load("follower"), motors)

  def test_writing_one_arm_preserves_the_other(self):
    get_calibration.write("follower", _all_motors())
    get_calibration.write("leader", _all_motors())

    self.assertEqual(get_calibration.load("follower"), _all_motors())
    self.assertEqual(get_calibration.load("leader"), _all_motors())

  def test_missing_file_names_the_calibrate_command(self):
    with self.assertRaisesRegex(
        get_calibration.CalibrationError, "real.get_calibration"
    ):
      get_calibration.load("follower")

  def test_missing_role_names_the_role(self):
    get_calibration.write("follower", _all_motors())

    with self.assertRaisesRegex(get_calibration.CalibrationError, "leader"):
      get_calibration.load("leader")

  def test_missing_motor_is_reported(self):
    motors = _all_motors()
    del motors["wrist_flex"]
    get_calibration.write("follower", motors)

    with self.assertRaisesRegex(get_calibration.CalibrationError, "wrist_flex"):
      get_calibration.load("follower")

  def test_incomplete_motor_is_reported(self):
    motors = _all_motors()
    motors["gripper"] = {"id": 6}  # Missing offset and range fields.
    get_calibration.write("follower", motors)

    with self.assertRaisesRegex(get_calibration.CalibrationError, "incomplete"):
      get_calibration.load("follower")

  def test_serial_mismatch_is_refused(self):
    get_calibration.write("follower", _all_motors(), serial_number="OLD123")

    with self.assertRaisesRegex(get_calibration.CalibrationError, "OLD123"):
      get_calibration.load("follower", expected_serial="NEW456")

  def test_matching_serial_is_accepted(self):
    get_calibration.write("follower", _all_motors(), serial_number="ARM1")

    self.assertEqual(
        get_calibration.load("follower", expected_serial="ARM1"),
        _all_motors(),
    )

  def test_missing_stored_serial_is_tolerated(self):
    get_calibration.write("follower", _all_motors())  # Older file.

    self.assertEqual(
        get_calibration.load("follower", expected_serial="ARM1"),
        _all_motors(),
    )

  def test_serials_are_kept_per_role(self):
    get_calibration.write("follower", _all_motors(), serial_number="ARM1")
    get_calibration.write("leader", _all_motors(), serial_number="ARM2")

    get_calibration.load("follower", expected_serial="ARM1")
    get_calibration.load("leader", expected_serial="ARM2")
    with self.assertRaisesRegex(get_calibration.CalibrationError, "ARM2"):
      get_calibration.load("leader", expected_serial="ARM1")

  def test_motor_ids_follow_canonical_joint_order(self):
    self.assertEqual(list(constant.MOTOR_IDS), list(constant.JOINTS))
    self.assertEqual(
        sorted(constant.MOTOR_IDS.values()),
        list(range(1, len(constant.JOINTS) + 1)),
    )


if __name__ == "__main__":
  absltest.main()
