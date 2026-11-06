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

"""Tests that the real deployment environment matches the simulated one."""

from typing import Any
from typing import Mapping
from unittest import mock

from absl.testing import absltest
from dm_env import specs
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as real_device
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env as pick_cube_env
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import run as deploy_run
import numpy as np
from reaf.core import device as reaf_device
import torch

_EPISODE_LENGTH_SECS = 1.0

# A calibration whose zero sits at the encoder's center, over its full sweep.
CALIBRATION = {
    joint: {
        "zero": constant.TICKS_CENTER,
        "range_min": 0,
        "range_max": constant.ENCODER_MAX,
    }
    for joint in constant.JOINTS
}


class FakeBus:
  """A Feetech bus that parks every joint at its calibrated zero."""

  def __init__(self):
    self.goals: dict[str, int] = {}

  def sync_read(self, register: str, normalize: bool) -> dict[str, int]:
    del register, normalize
    return {joint: constant.TICKS_CENTER for joint in constant.JOINTS}

  def sync_write(
      self, register: str, values: Mapping[str, int], normalize: bool
  ) -> None:
    del register, normalize
    self.goals = dict(values)


class FakeRobot:
  """A connected lerobot follower, reduced to what the devices touch."""

  def __init__(self):
    self.bus = FakeBus()
    self.disconnected = False

  def disconnect(self) -> None:
    self.disconnected = True


class FakeCamera(reaf_device.Device):
  """A wrist camera streaming a constant frame."""

  def __init__(self, name: str, height: int = 64, width: int = 64):
    self._name = name
    self._frame = np.zeros((height, width, 3), dtype=np.uint8)
    self.closed = False

  @property
  def name(self) -> str:
    return self._name

  def measurements_spec(self) -> dict[str, specs.Array]:
    return {self._name: specs.Array(shape=self._frame.shape, dtype=np.uint8)}

  def get_measurements(self) -> Mapping[str, Any]:
    return {self._name: self._frame}

  def commands_spec(self) -> Mapping[str, Any]:
    return {}

  def set_commands(self, commands: Mapping[str, Any]) -> None:
    del commands

  def close(self) -> None:
    self.closed = True


def observation_shapes(timestep: Any) -> dict[str, tuple[int, ...]]:
  return {
      key: tuple(value.shape) for key, value in timestep.observation.items()
  }


class PickCubeEnvTest(absltest.TestCase):

  def create_real_env(
      self,
      robot: FakeRobot,
      camera: FakeCamera,
      control_freq_hz: float | None = None,
  ):
    with (
        mock.patch.object(
            real_device, "connect_follower", return_value=(robot, CALIBRATION)
        ),
        mock.patch.object(
            real_device, "create_wrist_camera_device", return_value=camera
        ),
    ):
      return pick_cube_env.create_real_env(
          target_pos=(0.25, 0.0, 0.15),
          episode_length_secs=_EPISODE_LENGTH_SECS,
          homing_duration_secs=0.0,
          control_freq_hz=control_freq_hz,
      )

  def test_real_matches_simulation(self):
    sim_env = pick_cube_env.create_sim_env(
        episode_length_secs=_EPISODE_LENGTH_SECS
    )
    real_env = self.create_real_env(
        FakeRobot(), FakeCamera(sim_constant.WRIST_CAM_RGB)
    )

    # Real device layer measurements are a subset of sim; shared keys must match.
    sim_meas = sim_env.device_layer.get_measurements()
    real_meas = real_env.device_layer.get_measurements()
    self.assertContainsSubset(real_meas.keys(), sim_meas.keys())
    for key in real_meas:
      self.assertEqual(sim_meas[key].shape, real_meas[key].shape, key)
      self.assertEqual(sim_meas[key].dtype, real_meas[key].dtype, key)

    # Both modes run the same deployment task layer, so the MDP the policy
    # sees has to be identical down to the shapes.
    self.assertEqual(
        observation_shapes(real_env.reset()),
        observation_shapes(sim_env.reset()),
    )
    self.assertEqual(real_env.action_spec(), sim_env.action_spec())

    action = {"actions": torch.zeros(1, len(constant.JOINTS))}
    self.assertEqual(
        observation_shapes(real_env.step(action)),
        observation_shapes(sim_env.step(action)),
    )

  def test_torch_policy_action_scale(self):
    action_spec = {"actions": specs.Array(shape=(6,), dtype=np.float32)}
    network = mock.Mock(
        return_value=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    )
    policy = deploy_run.TorchPolicy(network, action_spec, action_scale=0.5)
    timestep = mock.Mock(observation={})
    (actions, _), _ = policy.step(timestep)
    torch.testing.assert_close(
        actions["actions"],
        torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]),
    )

  def test_real_env_steps(self):
    robot = FakeRobot()
    env = self.create_real_env(robot, FakeCamera(sim_constant.WRIST_CAM_RGB))
    env.reset()

    env.step({"actions": torch.zeros(1, len(constant.JOINTS))})

    self.assertCountEqual(robot.bus.goals, constant.JOINTS)

  def test_closing_parks_the_arm_and_releases_the_hardware(self):
    robot = FakeRobot()
    camera = FakeCamera(sim_constant.WRIST_CAM_RGB)
    env = self.create_real_env(robot, camera)

    env.close()

    self.assertTrue(robot.disconnected)
    self.assertTrue(camera.closed)
    self.assertCountEqual(robot.bus.goals, constant.JOINTS)

  def test_real_env_forwards_camera_config_and_handles_640x480(self):
    robot = FakeRobot()
    camera = FakeCamera(sim_constant.WRIST_CAM_RGB, height=480, width=640)
    camera_cfg = real_device.CameraConfig(width=640, height=480)
    with (
        mock.patch.object(
            real_device, "connect_follower", return_value=(robot, CALIBRATION)
        ),
        mock.patch.object(
            real_device, "create_wrist_camera_device", return_value=camera
        ) as mock_create_cam,
    ):
      env = pick_cube_env.create_real_env(
          target_pos=(0.25, 0.0, 0.15),
          episode_length_secs=_EPISODE_LENGTH_SECS,
          homing_duration_secs=0.0,
          camera_config=camera_cfg,
      )

    mock_create_cam.assert_called_once_with(
        name=sim_constant.WRIST_CAM_RGB,
        config=camera_cfg,
    )
    timestep = env.reset()
    self.assertEqual(
        tuple(timestep.observation["camera"].shape), (1, 3, 16, 16)
    )


if __name__ == "__main__":
  absltest.main()
