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

"""Tests that the simulated YAM device layer matches the real device layer."""

import os
import types
from typing import Any

from absl.testing import absltest
import mujoco
from mujoco_playground import real
from mujoco_playground import wrapper as wrapper_lib
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.real import device as real_device
from mujoco_playground.robots.yam.sim import constant as sim_constant
from mujoco_playground.robots.yam.sim import device as sim_device
from mujoco_playground.robots.yam.sim import scene as yam_scene
import numpy as np
from reaf.simulation import torch as sim
from reaf.simulation.torch import flat_scene as build


def setUpModule():
  os.environ.setdefault("MUJOCO_GL", "osmesa")


class _FakeRobot:
  """Stands in for a connected `i2rt.robots.motor_chain_robot.MotorChainRobot`."""

  def __init__(self, pose: np.ndarray = constant.REST_POSE):
    self.positions = np.array(pose, dtype=float)
    self.written: np.ndarray | None = None

  def get_joint_pos(self) -> np.ndarray:
    return self.positions.copy()

  def command_joint_pos(self, joint_pos: np.ndarray) -> None:
    self.written = np.array(joint_pos, dtype=float)
    self.positions = self.written.copy()


def _follower_device_config(base_spec: mujoco.MjSpec) -> build.Environment:
  cfg = build.Environment(base_scene=base_spec)
  cfg.sim.nworld = 1
  cfg.sim.xpu_device = "cpu"
  cfg.sim.impl = "cpp"
  cfg.sim.control_timestep = constant.CONTROL_PERIOD

  cfg.device_layer.devices["follower"] = build.Component(
      sim_device.YamFollower,
  )
  cfg.device_layer.devices["wrist_camera"] = build.Component(
      sim.MujocoCameraDevice,
      camera_name=sim_constant.WRIST_CAM_FULLNAME,
      width=constant.CAMERA_WIDTH,
      height=constant.CAMERA_HEIGHT,
      render_rate_hz=constant.CAMERA_FPS,
      image_output_name_prefix=constant.WRIST_CAMERA,
      rgb_output_name_suffix="",
  )
  return cfg


def _sim_device_layer() -> wrapper_lib.NumpyDeviceLayerWrapper:
  return wrapper_lib.NumpyDeviceLayerWrapper(
      wrapper_lib.SingleDeviceLayerWrapper(
          build.build_device_layer(
              _follower_device_config(yam_scene.yam_aloha_table())
          )
      )
  )


def _real_device_layer() -> Any:
  """The real device layer with mocks."""
  fake_camera = types.SimpleNamespace(
      config=types.SimpleNamespace(
          width=constant.CAMERA_WIDTH, height=constant.CAMERA_HEIGHT
      )
  )
  return real_device.create_device_layer(
      real_device.YamFollowerDevice(_FakeRobot()),
      real.OpenCVCameraDevice(fake_camera, name=constant.WRIST_CAMERA),
      control_period_seconds=constant.CONTROL_PERIOD,
  )


class ProtocolTest(absltest.TestCase):
  """The simulated device layer must be a drop-in for the real one."""

  def setUp(self):
    super().setUp()
    self.sim = _sim_device_layer()
    self.real = _real_device_layer()

  def test_measurement_keys_shapes_dtypes_match(self):
    sim_spec = self.sim.measurements_spec()
    real_spec = self.real.measurements_spec()

    self.assertContainsSubset(real_spec, sim_spec)  # real ⊆ sim, extras OK
    for key, spec in real_spec.items():  # shared keys: exact match
      self.assertEqual(sim_spec[key].shape, spec.shape, key)
      self.assertEqual(sim_spec[key].dtype, spec.dtype, key)

  def test_command_keys_shapes_dtypes_match(self):
    sim_spec = self.sim.commands_spec()
    real_spec = self.real.commands_spec()

    self.assertEqual(set(sim_spec), set(real_spec))
    for key, spec in real_spec.items():
      self.assertEqual(sim_spec[key].shape, spec.shape, key)
      self.assertEqual(sim_spec[key].dtype, spec.dtype, key)
      self.assertIs(type(sim_spec[key]), type(spec), key)
      np.testing.assert_allclose(
          sim_spec[key].minimum, spec.minimum, err_msg=key
      )
      np.testing.assert_allclose(
          sim_spec[key].maximum, spec.maximum, err_msg=key
      )


class SimFollowerTest(absltest.TestCase):

  def test_goals_move_the_arm(self):
    layer = _sim_device_layer()
    layer.begin_stepping()
    goal = dict(
        zip(
            constant.GOAL_POSITION_KEYS.values(),
            np.asarray(constant.READY_POSE),
            strict=True,
        )
    )

    for _ in range(100):  # 2.0 seconds at 50 Hz.
      measured = layer.step(goal)

    for joint, key in constant.JOINT_POSITION_KEYS.items():
      if joint == constant.GRIPPER_JOINT:
        continue  # Normalized [0, 1] scale; checked below.
      np.testing.assert_allclose(
          measured[key],
          goal[f"follower/goal_position/{joint}"],
          atol=0.05,
          err_msg=joint,
      )

  def test_gripper_round_trips_through_the_normalized_scale(self):
    layer = _sim_device_layer()
    layer.begin_stepping()
    goal = dict(
        zip(
            constant.GOAL_POSITION_KEYS.values(),
            np.asarray(constant.REST_POSE),
            strict=True,
        )
    )
    goal[constant.GOAL_POSITION_KEYS[constant.GRIPPER_JOINT]] = 0.4

    for _ in range(75):
      measured = layer.step(goal)

    gripper = measured[constant.JOINT_POSITION_KEYS[constant.GRIPPER_JOINT]]
    self.assertBetween(float(gripper), 0.3, 0.5)

    # Closed targets the squeeze overdrive past the stop, but the pads stop
    # each other, so the measurement stays on the [0, 1] scale and reads
    # closed.
    goal[constant.GOAL_POSITION_KEYS[constant.GRIPPER_JOINT]] = 0.0
    for _ in range(75):
      measured = layer.step(goal)
    gripper = measured[constant.JOINT_POSITION_KEYS[constant.GRIPPER_JOINT]]
    self.assertAlmostEqual(float(gripper), constant.GRIPPER_CLOSED, places=2)

  def test_gripper_unit_conversions_round_trip(self):
    meters = sim_constant.gripper_norm_to_m(0.42)
    np.testing.assert_allclose(meters, 0.42 * sim_constant.GRIPPER_M_RANGE[1])
    np.testing.assert_allclose(sim_constant.gripper_m_to_norm(meters), 0.42)


if __name__ == "__main__":
  absltest.main()
