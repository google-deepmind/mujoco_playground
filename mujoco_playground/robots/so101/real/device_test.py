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

"""Tests for the SO101 REAF device and its raw-tick frame."""

from absl.testing import absltest
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device
import numpy as np

_RADIANS_PER_TICK = 2 * np.pi / constant.TICKS_PER_TURN

# Every joint zeroed at the encoder center with a quarter turn of travel
# either side: -90..+90 degrees.
_ZERO = constant.TICKS_CENTER
_QUARTER_TURN = constant.TICKS_PER_TURN // 4
_GRIPPER_SPAN = 2 * _QUARTER_TURN
_CALIBRATION = {
    name: {
        "id": i + 1,
        "zero": _ZERO,
        "range_min": _ZERO - _QUARTER_TURN,
        "range_max": _ZERO + _QUARTER_TURN,
    }
    for i, name in enumerate(constant.JOINTS)
}


class _FakeBus:
  """Stands in for a raw FeetechMotorsBus."""

  def __init__(self):
    self.positions = {name: _ZERO for name in constant.JOINTS}
    self.written: dict[str, int] | None = None

  def sync_read(self, register, normalize=True):
    assert register == "Present_Position" and not normalize
    return dict(self.positions)

  def sync_write(self, register, values, normalize=True):
    assert register == "Goal_Position" and not normalize
    self.written = dict(values)


class _FakeArm:
  """Stands in for a connected lerobot robot."""

  def __init__(self):
    self.bus = _FakeBus()
    self.disconnected = False

  def disconnect(self):
    self.disconnected = True


class ReadJointsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.bus = _FakeBus()

  def test_readings_are_radians_relative_to_zero(self):
    self.bus.positions["shoulder_pan"] = _ZERO + _QUARTER_TURN

    joints = device.read_joints(self.bus, _CALIBRATION)

    np.testing.assert_allclose(joints["shoulder_pan"], np.pi / 2)
    np.testing.assert_allclose(joints["shoulder_lift"], 0.0)

  def test_readings_unwrap_across_the_encoder_boundary(self):
    calibration = {n: dict(m) for n, m in _CALIBRATION.items()}
    calibration["shoulder_pan"] = {
        "id": 1,
        "zero": 100,
        "range_min": 3596,
        "range_max": 700,
    }
    self.bus.positions["shoulder_pan"] = 4000  # Just below zero, across 0.

    joints = device.read_joints(self.bus, calibration)

    np.testing.assert_allclose(joints["shoulder_pan"], -196 * _RADIANS_PER_TICK)

  def test_gripper_reads_percent_of_its_calibrated_travel(self):
    self.bus.positions["gripper"] = round(
        _ZERO - _QUARTER_TURN + 0.42 * _GRIPPER_SPAN
    )

    joints = device.read_joints(self.bus, _CALIBRATION)

    np.testing.assert_allclose(joints["gripper"], 42.0, atol=0.1)


class WriteGoalsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.bus = _FakeBus()

  def test_goals_round_trip_through_the_bus(self):
    goals = {n: 0.0 for n in constant.JOINTS}
    goals["elbow_flex"] = 0.5
    goals["gripper"] = 60.0

    device.write_goals(self.bus, _CALIBRATION, goals)
    assert self.bus.written is not None
    self.bus.positions = self.bus.written  # The arm arrives exactly.
    back = device.read_joints(self.bus, _CALIBRATION)

    np.testing.assert_allclose(back["elbow_flex"], 0.5, atol=1e-3)
    np.testing.assert_allclose(back["gripper"], 60.0, atol=0.1)

  def test_goals_are_clamped_to_the_calibrated_range(self):
    goals = {"shoulder_pan": np.pi}  # 180 degrees, beyond the +-90 travel.

    device.write_goals(self.bus, _CALIBRATION, goals)
    assert self.bus.written is not None

    self.assertEqual(self.bus.written["shoulder_pan"], _ZERO + _QUARTER_TURN)


class SO101FollowerDeviceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.arm = _FakeArm()
    self.device = device.SO101FollowerDevice(self.arm, _CALIBRATION)

  def test_measurements_use_the_protocol_keys_and_dtype(self):
    self.arm.bus.positions["shoulder_pan"] = _ZERO + _QUARTER_TURN
    measured = self.device.get_measurements()
    expected_keys = set(constant.JOINT_POSITION_KEYS.values()) | {
        constant.FOLLOWER_COMMAND_TIMESTAMP
    }
    self.assertEqual(set(measured), expected_keys)
    reading = np.asarray(measured["follower/joint_position/shoulder_pan"])
    np.testing.assert_allclose(reading, np.pi / 2)
    self.assertEqual(reading.dtype, np.float32)

  def test_commands_map_the_protocol_keys_to_bus_goals(self):
    goal = np.zeros(6)
    goal[1] = np.pi / 4

    self.device.set_commands(
        dict(zip(constant.GOAL_POSITION_KEYS.values(), goal, strict=True))
    )
    assert self.arm.bus.written is not None

    self.assertEqual(
        self.arm.bus.written["shoulder_lift"], _ZERO + _QUARTER_TURN // 2
    )

  def test_command_spec_bounds_come_from_calibration(self):
    spec = self.device.commands_spec()
    pan = spec["follower/goal_position/shoulder_pan"]
    gripper = spec["follower/goal_position/gripper"]

    np.testing.assert_allclose(pan.minimum, -np.pi / 2)
    np.testing.assert_allclose(pan.maximum, np.pi / 2)
    np.testing.assert_allclose(gripper.minimum, 0.0)
    np.testing.assert_allclose(gripper.maximum, 100.0)

  def test_close_disconnects_the_robot(self):
    self.device.close()

    self.assertTrue(self.arm.disconnected)


if __name__ == "__main__":
  absltest.main()
