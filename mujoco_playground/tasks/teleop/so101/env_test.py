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
"""Tests for the sim and real SO101 teleop environment."""

import os
import pathlib
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from mujoco_playground import wrapper as wrapper_lib
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device
from mujoco_playground.robots.so101.real import get_calibration
from mujoco_playground.robots.so101.sim import device as sim_device
from mujoco_playground.robots.so101.sim import scene as so101_scene
from mujoco_playground.tasks.teleop.so101 import env as teleop_env
import numpy as np
from reaf.simulation.torch import flat_scene as build
import torch


def setUpModule():
  os.environ.setdefault("MUJOCO_GL", "osmesa")


class _FakeBus:
  """Stands in for a raw FeetechMotorsBus."""

  def __init__(self):
    self.positions = {n: constant.TICKS_CENTER for n in constant.JOINTS}
    self.written = None

  def sync_read(self, register, normalize=True):
    assert register == "Present_Position" and not normalize
    return dict(self.positions)

  def sync_write(self, register, values, normalize=True):
    assert register == "Goal_Position" and not normalize
    self.written = dict(values)
    self.positions.update(values)


class _FakeArm:
  """Stands in for a connected lerobot robot."""

  def __init__(self):
    self.bus = _FakeBus()

  def disconnect(self):
    pass


def _mock_calibration() -> dict[str, dict[str, int]]:
  return {
      name: {
          "id": i + 1,
          "zero": constant.TICKS_CENTER,
          "range_min": 512,
          "range_max": 3584,
      }
      for i, name in enumerate(constant.JOINTS)
  }


def _write_calibration(role: str) -> None:
  get_calibration.write(role, _mock_calibration())


class TeleopEnvTest(absltest.TestCase):
  """Tests common contract across both real and simulated environments."""

  def _make_real_env(
      self,
  ) -> tuple[teleop_env.reaf_environment.Environment, _FakeArm]:
    arm = _FakeArm()
    layer = device.create_device_layer(
        device.SO101FollowerDevice(arm, _mock_calibration()),
        control_period_seconds=1e-3,
    )
    env = teleop_env._build_teleop_environment(  # pylint: disable=protected-access
        device_layer=layer, instruction="test", real_time_homing=False
    )
    return env, arm

  def _make_sim_env(self) -> teleop_env.reaf_environment.Environment:
    return teleop_env.create_sim_env(instruction="test")

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    patcher = mock.patch.object(
        get_calibration.paths, "local_dir", return_value=pathlib.Path(tmp.name)
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    _write_calibration("follower")

  def test_sim_and_real_action_spec_parity(self):
    real_env, _ = self._make_real_env()
    sim_env = self._make_sim_env()
    real_action_spec: Any = real_env.action_spec()
    sim_action_spec: Any = sim_env.action_spec()
    assert isinstance(real_action_spec, dict)
    assert isinstance(sim_action_spec, dict)

    self.assertEqual(real_action_spec.keys(), sim_action_spec.keys())
    for key, r_spec in real_action_spec.items():
      s_spec = sim_action_spec[key]
      self.assertEqual(r_spec.shape, s_spec.shape, msg=f"Mismatch for {key}")
      self.assertEqual(r_spec.dtype, s_spec.dtype, msg=f"Mismatch for {key}")

  def test_sim_and_real_observation_spec_parity(self):
    real_env, _ = self._make_real_env()
    sim_env = self._make_sim_env()
    real_obs_spec: Any = real_env.observation_spec()
    sim_obs_spec: Any = sim_env.observation_spec()
    assert isinstance(real_obs_spec, dict)
    assert isinstance(sim_obs_spec, dict)

    # Both must expose the follower joint positions, timestamp, and instruction.
    for key in constant.JOINT_POSITION_KEYS.values():
      self.assertIn(key, real_obs_spec)
      self.assertIn(key, sim_obs_spec)
      self.assertEqual(real_obs_spec[key].dtype, np.float32)
      self.assertEqual(sim_obs_spec[key].dtype, np.float32)

    self.assertIn(constant.TIMESTAMP, real_obs_spec)
    self.assertIn(constant.TIMESTAMP, sim_obs_spec)

  def test_reset_and_step_both_environments(self):
    real_env, _ = self._make_real_env()
    sim_env = self._make_sim_env()
    goal = {
        key: np.float32(constant.FOLLOWER_HOME[i])
        for i, key in enumerate(constant.GOAL_POSITION_KEYS.values())
    }

    for name, env in (("real", real_env), ("sim", sim_env)):
      with self.subTest(env=name):
        ts = env.reset()
        self.assertContainsSubset(
            tuple(constant.JOINT_POSITION_KEYS.values()), ts.observation
        )
        next_ts = env.step(goal)
        self.assertFalse(next_ts.last())

  def test_reset_drives_arm_home_in_both_environments(self):
    real_env, _ = self._make_real_env()
    sim_env = self._make_sim_env()

    for name, env in (("real", real_env), ("sim", sim_env)):
      with self.subTest(env=name):
        ts = env.reset()
        for i, (joint, key) in enumerate(constant.JOINT_POSITION_KEYS.items()):
          if joint == constant.GRIPPER_JOINT:
            continue
          np.testing.assert_allclose(
              ts.observation[key], constant.FOLLOWER_HOME[i], atol=0.1
          )


if __name__ == "__main__":
  absltest.main()
