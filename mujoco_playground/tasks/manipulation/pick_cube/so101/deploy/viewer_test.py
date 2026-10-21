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

"""Tests the deployment viewer and pacing loop."""

from collections.abc import Sequence
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import mujoco
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from mujoco_playground.tasks.manipulation.pick_cube.so101 import task as so101_task
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env as pick_cube_env
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import run as deploy_run
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import viewer
import numpy as np
from reaf.simulation.torch import flat_scene as build
from reaf.simulation.torch import viewer as reaf_viewer
import torch

_TARGET_POS = (0.25, 0.0, 0.15)
_TARGET_BODY = f"{pick_task.TARGET}/{pick_task.TARGET}"
_JOINT_POS_KEY = f"{constant.ROBOT_STATE}/{constant.JOINT_POS}"


def build_deploy_spec() -> mujoco.MjSpec:
  cfg = so101_task.so101_pick_cube_pixels_env(impl="cpp")
  deploy_cfg = build.make_deployment_config(
      cfg, observation_groups=pick_cube_env.OBSERVATION_GROUPS
  )
  return build.build_scene(deploy_cfg)


def mock_real_device_layer(
    joint_pos: Sequence[float] | np.ndarray | None = None,
) -> mock.MagicMock:
  device_layer = mock.MagicMock()
  device_layer.get_device.return_value = pick_cube_env.TargetDevice(_TARGET_POS)
  device_layer.get_measurements.return_value = (
      {}
      if joint_pos is None
      else {_JOINT_POS_KEY: torch.as_tensor(joint_pos).unsqueeze(0)}
  )
  return device_layer


class ViewerTest(parameterized.TestCase):
  """Tests how the deployment viewer configures scenes and bridges hardware."""

  @parameterized.parameters(None, ([0.30, -0.05, 0.18],))
  def test_configure_viewer_spec_shows_target_and_frames_camera(
      self, lookat: Sequence[float] | None
  ) -> None:
    spec = build_deploy_spec()
    viewer.configure_viewer_spec(spec, lookat=lookat)
    model = spec.compile()

    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)

    self.assertEqual(opt.geomgroup[model.geom(_TARGET_BODY).group], 1)
    np.testing.assert_allclose(
        cam.lookat, _TARGET_POS if lookat is None else lookat, atol=1e-6
    )
    self.assertAlmostEqual(cam.distance, 0.8)
    self.assertAlmostEqual(cam.azimuth, 135.0)
    self.assertAlmostEqual(cam.elevation, -25.0)

  @parameterized.parameters(None, ([0.1, -0.2, 0.3, -0.4, 0.5, 0.6],))
  def test_real_simulator_poses_arm_and_updates_target(
      self, joint_pos: Sequence[float] | None
  ) -> None:
    device_layer = mock_real_device_layer(joint_pos)
    sim = viewer.RealSimulator(device_layer)
    model = sim.mj_model
    mocap_id = model.body_mocapid[model.body(_TARGET_BODY).id]

    data = sim.data.obj
    expected_qpos = (
        np.zeros(len(constant.JOINTS)) if joint_pos is None else joint_pos
    )
    qpos = [
        data.qpos[
            model.joint(f"{sim_constant.ATTACHMENT_PREFIX}{joint}").qposadr[0]
        ]
        for joint in constant.JOINTS
    ]
    np.testing.assert_allclose(qpos, expected_qpos, atol=1e-6)
    np.testing.assert_allclose(data.mocap_pos[mocap_id], _TARGET_POS)

    new_target = np.array([0.30, -0.05, 0.18], dtype=np.float64)
    sim.data.mocap_pos[0, mocap_id].copy_(torch.as_tensor(new_target))
    _ = sim.data

    target_device = device_layer.get_device.return_value
    np.testing.assert_allclose(target_device.target_pos, new_target, atol=1e-6)
    np.testing.assert_allclose(
        target_device.get_measurements()[pick_task.TARGET_MOCAP_POS],
        new_target,
        atol=1e-6,
    )
    np.testing.assert_allclose(data.mocap_pos[mocap_id], new_target, atol=1e-6)

  @parameterized.parameters("sim", "real")
  def test_launch(self, mode: str) -> None:
    env = mock.MagicMock()
    env.device_layer = mock_real_device_layer()
    if mode == "sim":
      env.device_layer.sim.mj_spec = build_deploy_spec()
      env.device_layer.control_timestep = 0.02
    else:
      env.device_layer.sim = None
    policy = mock.MagicMock()
    logger = mock.MagicMock()

    with mock.patch.object(reaf_viewer, "launch") as mock_launch:
      viewer.launch(env, policy, loggers=[logger], gfx="web")

    mock_launch.assert_called_once()
    _, kwargs = mock_launch.call_args
    if mode == "sim":
      self.assertIs(kwargs["simulator"], env.device_layer.sim)
      self.assertEqual(kwargs["step_duration"], 0.02)
      self.assertEqual(env.device_layer.sim.mj_spec.geom(_TARGET_BODY).group, 2)
    else:
      self.assertIsInstance(kwargs["simulator"], viewer.RealSimulator)
      self.assertIsNone(kwargs["step_duration"])
    self.assertEqual(kwargs["reward_plot_keys"], ())
    logger.write.assert_called_once_with()


class RateLimiterTest(absltest.TestCase):
  """Tests the simulation rate limiter pacing operation."""

  def test_rate_limiter_paces_to_period(self) -> None:
    rate_limiter = deploy_run.RateLimiterOperation(period_seconds=0.005)
    rate_limiter.before_episode_reset()
    start = time.perf_counter()
    rate_limiter.inspect_timestep(dm_env.restart({}))
    self.assertGreaterEqual(time.perf_counter() - start, 0.004)


if __name__ == "__main__":
  absltest.main()
