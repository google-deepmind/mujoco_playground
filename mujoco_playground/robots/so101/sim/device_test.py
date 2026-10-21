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

"""Tests that the simulated SO101 device layer matches the real device layer."""

import os
import types
from typing import Any

from absl.testing import absltest
from mujoco_playground import real
from mujoco_playground import wrapper as wrapper_lib
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as real_device
from mujoco_playground.robots.so101.sim import device as sim_device
from mujoco_playground.robots.so101.sim import scene as so101_scene
import numpy as np
from reaf.simulation.torch import flat_scene as build


def setUpModule():
  os.environ.setdefault("MUJOCO_GL", "osmesa")


def _sim_device_layer() -> wrapper_lib.NumpyDeviceLayerWrapper:
  return wrapper_lib.NumpyDeviceLayerWrapper(
      wrapper_lib.SingleDeviceLayerWrapper(
          build.build_device_layer(
              sim_device.follower_device_config(
                  so101_scene.so101_aloha_table(),
                  control_period_seconds=0.02,
              ),
          )
      )
  )


def _real_device_layer() -> Any:
  """The real device layer with mocks."""
  quarter = constant.TICKS_PER_TURN // 4
  calibration = {
      name: {
          "id": i + 1,
          "zero": constant.TICKS_CENTER,
          "range_min": constant.TICKS_CENTER - quarter,
          "range_max": constant.TICKS_CENTER + quarter,
      }
      for i, name in enumerate(constant.JOINTS)
  }
  fake_camera = types.SimpleNamespace(
      config=types.SimpleNamespace(
          width=constant.CAMERA_WIDTH, height=constant.CAMERA_HEIGHT
      )
  )
  return real_device.create_device_layer(
      real_device.SO101FollowerDevice(object(), calibration),
      real.OpenCVCameraDevice(fake_camera, name=constant.WRIST_CAMERA),
      control_period_seconds=0.02,
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
      # Bounds are deliberately not compared: the real clamp comes from each
      # arm's calibrated travel, the sim clamp from the MJCF joint ranges.
      self.assertEqual(sim_spec[key].shape, spec.shape, key)
      self.assertEqual(sim_spec[key].dtype, spec.dtype, key)
      self.assertIs(type(sim_spec[key]), type(spec), key)


class SimFollowerTest(absltest.TestCase):

  def test_goals_move_the_arm(self):
    layer = _sim_device_layer()
    layer.begin_stepping()
    goal = dict(
        zip(
            constant.GOAL_POSITION_KEYS.values(),
            np.asarray(constant.FOLLOWER_HOME),
            strict=True,
        )
    )

    for _ in range(75):  # 1.5 seconds at 50 Hz.
      measured = layer.step(goal)

    for joint, key in constant.JOINT_POSITION_KEYS.items():
      if joint == constant.GRIPPER_JOINT:
        continue  # Percentage scale; checked below.
      np.testing.assert_allclose(
          measured[key],
          goal[f"follower/goal_position/{joint}"],
          atol=0.05,
          err_msg=joint,
      )

  def test_gripper_round_trips_through_the_percent_scale(self):
    layer = _sim_device_layer()
    layer.begin_stepping()
    goal = dict(
        zip(
            constant.GOAL_POSITION_KEYS.values(),
            np.asarray(constant.FOLLOWER_HOME),
            strict=True,
        )
    )
    goal[constant.GOAL_POSITION_KEYS[constant.GRIPPER_JOINT]] = 40.0

    for _ in range(75):
      measured = layer.step(goal)

    gripper = measured[constant.JOINT_POSITION_KEYS[constant.GRIPPER_JOINT]]
    self.assertBetween(float(gripper), 30.0, 50.0)


if __name__ == "__main__":
  absltest.main()
