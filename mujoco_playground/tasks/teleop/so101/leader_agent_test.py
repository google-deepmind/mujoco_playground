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

"""Tests for leader_agent.py."""

from absl.testing import absltest
import dm_env
from dm_env import specs
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device
from mujoco_playground.tasks.teleop.so101 import leader_agent
import numpy as np

_ACTION_KEYS = tuple(constant.GOAL_POSITION_KEYS.values())

# +-135 degrees around the encoder center: wide enough for LEADER_HOME.
_CALIBRATION = {
    name: {
        "id": i + 1,
        "zero": constant.TICKS_CENTER,
        "range_min": 512,
        "range_max": 3584,
    }
    for i, name in enumerate(constant.JOINTS)
}


class _FakeBus:
  """Stands in for the leader's raw FeetechMotorsBus."""

  def __init__(self, arrives: bool = True):
    self.positions = {n: constant.TICKS_CENTER for n in constant.JOINTS}
    self.arrives = arrives
    self.torque_enabled = None

  def sync_read(self, register, normalize=True):
    assert register == "Present_Position" and not normalize
    return dict(self.positions)

  def sync_write(self, register, values, normalize=True):
    assert register == "Goal_Position" and not normalize
    if self.arrives:
      self.positions.update(values)  # The fake arm arrives instantly.

  def enable_torque(self, *unused_args, **unused_kwargs):
    self.torque_enabled = True

  def disable_torque(self, *unused_args, **unused_kwargs):
    self.torque_enabled = False


class _FakeLeader:

  def __init__(self, arrives: bool = True):
    self.bus = _FakeBus(arrives)


def _set_pose(bus: _FakeBus, radians: float, gripper: float = 50.0) -> None:
  """Poses the fake leader: all arm joints at `radians`, gripper in percent."""
  pose = {n: radians for n in constant.JOINTS}
  pose[constant.GRIPPER_JOINT] = gripper
  device.write_goals(bus, _CALIBRATION, pose)


def _timestep() -> dm_env.TimeStep:
  return dm_env.restart(
      dict(zip(constant.JOINT_POSITION_KEYS.values(), np.zeros(6), strict=True))
  )


class SO101LeaderAgentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.leader = _FakeLeader()
    minimum = np.array([-np.pi / 2] * 5 + [0.0])
    maximum = np.array([np.pi / 2] * 5 + [100.0])
    self.action_spec = {
        key: specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=low, maximum=high
        )
        for key, low, high in zip(_ACTION_KEYS, minimum, maximum, strict=True)
    }
    self.agent = leader_agent.SO101LeaderAgent(
        self.leader,
        _CALIBRATION,
        self.action_spec,
        real_time_homing=False,
    )

  def test_initial_state_releases_torque(self):
    state = self.agent.initial_state()

    self.assertEqual(state, ())
    self.assertIs(self.leader.bus.torque_enabled, False)

  def test_step_spec(self):
    (action_spec, extra_spec), state_spec = self.agent.step_spec(
        ()  # pyrefly: ignore[bad-argument-type]
    )

    self.assertEqual(action_spec, self.action_spec)
    for key in constant.LEADER_RAW_KEYS.values():
      self.assertEqual(extra_spec[key].shape, ())
      self.assertEqual(extra_spec[key].dtype, np.float32)
    self.assertEqual(state_spec, ())

  def test_step_passes_the_leader_pose_through(self):
    _set_pose(self.leader.bus, np.radians(30.0), gripper=20.0)

    (action, extra), _ = self.agent.step(_timestep(), ())

    for key in _ACTION_KEYS[:-1]:
      np.testing.assert_allclose(action[key], np.radians(30.0), atol=1e-3)
    np.testing.assert_allclose(action[_ACTION_KEYS[-1]], 20.0, atol=0.1)
    # The raw (unclipped) leader pose rides along for logging.
    np.testing.assert_allclose(
        [extra[k] for k in list(constant.LEADER_RAW_KEYS.values())[:-1]],
        np.radians(30.0),
        atol=1e-3,
    )

  def test_step_clips_to_the_action_spec(self):
    _set_pose(self.leader.bus, np.radians(120.0))  # Beyond the +-90 spec.

    (action, _), _ = self.agent.step(_timestep(), ())

    for key in _ACTION_KEYS[:-1]:
      np.testing.assert_allclose(action[key], self.action_spec[key].maximum)

  def test_step_action_satisfies_the_action_spec(self):
    # The environment validates the action against the spec on every step, and
    # dtype counts: a float64 action is rejected outright.
    _set_pose(self.leader.bus, np.radians(30.0), gripper=20.0)

    (action, _), _ = self.agent.step(_timestep(), ())

    for key, spec in self.action_spec.items():
      spec.validate(action[key])

  def test_reset_drives_the_leader_home(self):
    self.agent.reset()

    self.assertIs(self.leader.bus.torque_enabled, True)
    arrived = device.read_joints(self.leader.bus, _CALIBRATION)
    for i, joint in enumerate(constant.JOINTS):
      # One encoder tick is ~1.5e-3 rad, but ~0.033 on the gripper's % scale.
      atol = 0.05 if joint == constant.GRIPPER_JOINT else 0.01
      np.testing.assert_allclose(
          arrived[joint], constant.LEADER_HOME[i], atol=atol, err_msg=joint
      )


if __name__ == "__main__":
  absltest.main()
