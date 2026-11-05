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

"""Tests for the YAM REAF devices and I2RT SDK wrapper."""

import sys
import tempfile
import threading
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from etils import epath
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.real import device
import numpy as np


class _FakeChain:

  def __init__(self):
    self.command_lock = threading.Lock()
    self.running = True


class _FakeRobot:
  """Stands in for a connected `i2rt.robots.motor_chain_robot.MotorChainRobot`."""

  def __init__(self, pose: np.ndarray = constant.REST_POSE):
    self.positions = np.array(pose, dtype=float)
    self.written: np.ndarray | None = None
    self.closed = False
    self._stop_event = threading.Event()
    self.motor_chain = _FakeChain()

  def get_joint_pos(self) -> np.ndarray:
    return self.positions.copy()

  def command_joint_pos(self, joint_pos: np.ndarray) -> None:
    self.written = np.array(joint_pos, dtype=float)

  def close(self) -> None:
    self.closed = True


class ReadJointsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.robot = _FakeRobot(constant.READY_POSE)

  def test_readings_match_canonical_joint_order(self):
    joints = device.read_joints(self.robot)

    self.assertEqual(tuple(joints), constant.JOINTS)
    for joint, expected in zip(
        constant.JOINTS, constant.READY_POSE, strict=True
    ):
      np.testing.assert_allclose(joints[joint], expected)


class WriteGoalsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.robot = _FakeRobot()

  def test_goals_round_trip_through_the_robot(self):
    goals = {n: 0.0 for n in constant.JOINTS}
    goals["joint2"] = 1.2
    goals["gripper"] = 0.6

    device.write_goals(self.robot, goals)
    assert self.robot.written is not None
    self.robot.positions = self.robot.written
    back = device.read_joints(self.robot)

    np.testing.assert_allclose(back["joint2"], 1.2)
    np.testing.assert_allclose(back["gripper"], 0.6)

  def test_goals_are_clamped_to_joint_bounds(self):
    goals = {n: 0.0 for n in constant.JOINTS}
    goals["joint1"] = 10.0  # Beyond upper limit (3.14159).
    goals["joint2"] = -1.0  # Below lower limit (0.0).
    goals["gripper"] = 1.5  # Above normalized max (1.0).

    device.write_goals(self.robot, goals)
    assert self.robot.written is not None

    np.testing.assert_allclose(
        self.robot.written[0], constant.JOINT_LIMITS["joint1"][1]
    )
    np.testing.assert_allclose(
        self.robot.written[1], constant.JOINT_LIMITS["joint2"][0]
    )
    np.testing.assert_allclose(self.robot.written[-1], constant.GRIPPER_MAX)


class YamFollowerDeviceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.robot = _FakeRobot()
    self.device = device.YamFollowerDevice(self.robot)

  def test_measurements_use_the_protocol_keys_and_dtype(self):
    self.robot.positions[0] = 0.75
    measured = self.device.get_measurements()
    expected_keys = set(constant.JOINT_POSITION_KEYS.values()) | {
        constant.FOLLOWER_COMMAND_TIMESTAMP
    }
    self.assertEqual(set(measured), expected_keys)
    reading = np.asarray(measured["follower/joint_position/joint1"])
    np.testing.assert_allclose(reading, 0.75)
    self.assertEqual(reading.dtype, np.float32)

  def test_commands_map_the_protocol_keys_to_robot_goals(self):
    goal = np.array(constant.READY_POSE, dtype=np.float32)
    goal[-1] = 0.5

    self.device.set_commands(
        dict(zip(constant.GOAL_POSITION_KEYS.values(), goal, strict=True))
    )

    assert self.robot.written is not None
    np.testing.assert_allclose(self.robot.written, goal)

  def test_command_spec_bounds_match_joint_bounds(self):
    spec = self.device.commands_spec()
    j1 = spec["follower/goal_position/joint1"]
    gripper = spec["follower/goal_position/gripper"]

    np.testing.assert_allclose(
        (j1.minimum, j1.maximum), constant.JOINT_LIMITS["joint1"]
    )
    np.testing.assert_allclose(
        (gripper.minimum, gripper.maximum),
        (constant.GRIPPER_MIN, constant.GRIPPER_MAX),
    )

  def test_close_stops_threads_and_closes_the_robot(self):
    self.device.close()

    self.assertTrue(self.robot._stop_event.is_set())
    self.assertFalse(self.robot.motor_chain.running)
    self.assertTrue(self.robot.closed)


class WristCameraDeviceTest(absltest.TestCase):

  def test_create_wrist_camera_device_uses_resolved_camera(self):
    fake_dev = mock.MagicMock()
    with (
        mock.patch.object(
            device.get_cameras, "resolve", return_value={"wrist": "/dev/video4"}
        ),
        mock.patch.object(
            device.real,
            "create_opencv_camera_device",
            return_value=fake_dev,
        ) as create_mock,
    ):
      result = device.create_wrist_camera_device()

    self.assertIs(result, fake_dev)
    create_mock.assert_called_once_with(
        index_or_path="/dev/video4",
        name=constant.WRIST_CAMERA,
        config=device.OpenCVCameraConfig(
            width=constant.CAMERA_WIDTH,
            height=constant.CAMERA_HEIGHT,
            fps=constant.CAMERA_FPS,
        ),
    )


def _fake_i2rt_modules(
    fake_robot: _FakeRobot,
) -> tuple[dict[str, Any], mock.MagicMock]:
  fake_get_robot: Any = types.ModuleType("i2rt.robots.get_robot")
  fake_get_robot.get_yam_robot = mock.MagicMock(return_value=fake_robot)
  fake_utils: Any = types.ModuleType("i2rt.robots.utils")
  fake_utils.ArmType = lambda x: f"ArmType({x})"
  fake_utils.GripperType = lambda x: f"GripperType({x})"
  fake_robots: Any = types.ModuleType("i2rt.robots")
  fake_robots.get_robot = fake_get_robot
  fake_robots.utils = fake_utils
  fake_i2rt: Any = types.ModuleType("i2rt")
  fake_i2rt.robots = fake_robots
  modules = {
      "i2rt": fake_i2rt,
      "i2rt.robots": fake_robots,
      "i2rt.robots.get_robot": fake_get_robot,
      "i2rt.robots.utils": fake_utils,
  }
  return modules, fake_get_robot.get_yam_robot


class CheckCanUpTest(absltest.TestCase):

  def test_linux_checks_sysfs_operstate(self) -> None:
    tmp = epath.Path(self.enterContext(tempfile.TemporaryDirectory()))
    operstate = tmp / "operstate"
    with (
        mock.patch.object(device.sys, "platform", "linux"),
        mock.patch.object(device.epath, "Path", return_value=operstate),
    ):
      with self.assertRaisesRegex(RuntimeError, "No CAN interface 'can0'"):
        device.check_can_up("can0")

      operstate.write_text("down\n")
      with self.assertRaisesRegex(RuntimeError, "sudo ip link set can0 up"):
        device.check_can_up("can0")

      operstate.write_text("up\n")
      device.check_can_up("can0")

  def test_darwin_delegates_to_gs_usb_darwin(self) -> None:
    with (
        mock.patch.object(device.sys, "platform", "darwin"),
        mock.patch.object(device.gs_usb_darwin, "check_can_up") as darwin_mock,
    ):
      device.check_can_up("can0")

    darwin_mock.assert_called_once_with("can0")


class ConnectRobotTest(absltest.TestCase):

  def test_connect_robot_linux_skips_macos_backend(self) -> None:
    fake_robot = _FakeRobot()
    i2rt_mods, get_yam_mock = _fake_i2rt_modules(fake_robot)

    with (
        mock.patch.object(device.sys, "platform", "linux"),
        mock.patch.object(device, "check_can_up") as check_mock,
        mock.patch.object(
            device.gs_usb_darwin, "register_socketcan_backend"
        ) as register_mock,
        mock.patch.dict(sys.modules, i2rt_mods),
    ):
      result = device.connect_robot("can0")

    self.assertIs(result, fake_robot)
    check_mock.assert_called_once_with("can0")
    register_mock.assert_not_called()
    get_yam_mock.assert_called_once_with(
        channel="can0",
        arm_type=f"ArmType({constant.ARM_TYPE})",
        gripper_type=f"GripperType({constant.GRIPPER_TYPE})",
        zero_gravity_mode=False,
    )

  def test_connect_robot_darwin_registers_gs_usb_backend(self) -> None:
    fake_robot = _FakeRobot()
    i2rt_mods, get_yam_mock = _fake_i2rt_modules(fake_robot)

    with (
        mock.patch.object(device.sys, "platform", "darwin"),
        mock.patch.object(device, "check_can_up") as check_mock,
        mock.patch.object(
            device.gs_usb_darwin, "register_socketcan_backend"
        ) as register_mock,
        mock.patch.dict(sys.modules, i2rt_mods),
    ):
      result = device.connect_robot("can0")

    self.assertIs(result, fake_robot)
    check_mock.assert_called_once_with("can0")
    register_mock.assert_called_once_with()
    get_yam_mock.assert_called_once_with(
        channel="can0",
        arm_type=f"ArmType({constant.ARM_TYPE})",
        gripper_type=f"GripperType({constant.GRIPPER_TYPE})",
        zero_gravity_mode=False,
    )


if __name__ == "__main__":
  absltest.main()
