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

"""Sim and real hardware environment builders for the agentic tasks.

Simulation and real hardware both wire the canonical `Follower` device
(`SO101Follower` / `YamFollower` in sim; `SO101FollowerDevice` /
`YamFollowerDevice` on hardware) under the `"follower"` device name, so the
Cartesian IK controller, observations, and agent harness are identical across
sim and real for every registered embodiment.
"""

from collections.abc import Sequence
import time
from typing import Any
from typing import Literal

from absl import logging
from mujoco_playground import real
from mujoco_playground import wrapper
from mujoco_playground._src.real.common import clock_device
from mujoco_playground.robots.so101.real import device as so101_real_device
from mujoco_playground.robots.yam.real import device as yam_real_device
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
from mujoco_playground.tasks.agentic.tabletop_v0 import task as agentic_task
import numpy as np
from reaf.core import device as reaf_device
from reaf.core import environment as reaf_environment
from reaf.simulation.torch import environment as torch_environment
from reaf.simulation.torch import flat_scene as build
from reaf.simulation.torch import functional_observation

_CONTROL_TIMESTEP = 0.02


def _interpolate_to_pose(
    follower: reaf_device.Device,
    joint_keys: tuple[str, ...],
    goal_keys: tuple[str, ...],
    target_qpos: Sequence[float] | np.ndarray,
    duration_secs: float,
    dt: float = _CONTROL_TIMESTEP,
) -> None:
  """Linearly interpolates the follower arm from its current pose to `target_qpos`."""
  meas = follower.get_measurements()
  q_start = np.array(
      [float(np.asarray(meas[k]).reshape(-1)[0]) for k in joint_keys],
      dtype=np.float32,
  )
  q_end = np.asarray(target_qpos, dtype=np.float32)
  steps = max(1, int(round(duration_secs / dt)))
  for s in range(1, steps + 1):
    alpha = s / steps
    q = (1.0 - alpha) * q_start + alpha * q_end
    follower.set_commands(
        {k: np.float32(q[i]) for i, k in enumerate(goal_keys)}
    )
    time.sleep(dt)


class HomingReset(torch_environment.EnvironmentReset):
  """Smoothly moves the real follower arm to `emb.home_qpos` on reset."""

  def __init__(
      self,
      emb: embodiment_lib.Embodiment,
      follower: reaf_device.Device | None = None,
      clock: clock_device.ClockDevice | None = None,
      duration_secs: float = 2.0,
      dt: float = _CONTROL_TIMESTEP,
  ):
    self._emb = emb
    self._follower = follower
    self._clock = clock
    self._duration_secs = duration_secs
    self._dt = dt

  def on_reset(
      self,
      config: Any = None,
      env_mask: Any = None,
      num_envs: int | None = None,
  ) -> None:
    del config, env_mask, num_envs
    follower = (
        self._follower
        if self._follower is not None
        else self._env.device_layer.get_device("follower")
    )
    _interpolate_to_pose(
        follower=follower,
        joint_keys=self._emb.joint_position_keys,
        goal_keys=self._emb.goal_position_keys,
        target_qpos=self._emb.home_qpos,
        duration_secs=self._duration_secs,
        dt=self._dt,
    )
    if self._clock is not None:
      self._clock.reset()


class Shutdown(reaf_environment.EnvironmentCloser):
  """Returns the real follower arm to `emb.rest_qpos` and closes hardware."""

  def __init__(
      self,
      follower: reaf_device.Device,
      emb: embodiment_lib.Embodiment,
      camera: Any = None,
      duration_secs: float = 2.0,
      dt: float = _CONTROL_TIMESTEP,
  ):
    self._follower = follower
    self._emb = emb
    self._camera = camera
    self._duration_secs = duration_secs
    self._dt = dt

  def close(self) -> None:
    try:
      _interpolate_to_pose(
          follower=self._follower,
          joint_keys=self._emb.joint_position_keys,
          goal_keys=self._emb.goal_position_keys,
          target_qpos=self._emb.rest_qpos,
          duration_secs=self._duration_secs,
          dt=self._dt,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to return %s to rest pose.", self._emb.name)
    finally:
      if self._camera is not None and hasattr(self._camera, "close"):
        self._camera.close()
      if hasattr(self._follower, "close"):
        self._follower.close()


def create_sim_env(
    *,
    task: str = "cube_in_bowl",
    robot: str = "so101",
    impl: Literal["cpp", "warp"] = "cpp",
    episode_length_secs: float = 1800.0,
) -> torch_environment.Environment:
  """Creates the simulated environment for a registered task and robot."""
  cfg = agentic_task.env_config(task=task, robot=robot, impl=impl)
  cfg.episode_length_secs = episode_length_secs
  return build.build_env(cfg)


def create_real_env(
    *,
    robot: str = "so101",
    episode_length_secs: float = 1800.0,
    homing_duration_secs: float = 2.0,
) -> torch_environment.Environment:
  """Creates the real-hardware agentic environment for a registered robot."""
  emb = embodiment_lib.get(robot)
  if emb.name == "so101":
    real_mod = so101_real_device
  elif emb.name == "yam":
    real_mod = yam_real_device
  else:
    raise NotImplementedError(f"No hardware driver for robot {robot!r} yet.")

  cfg = agentic_task.env_config(robot=emb.name, impl="cpp")
  cfg.episode_length_secs = episode_length_secs
  cfg.sim.xpu_device = "cpu"
  deploy_cfg = build.make_deployment_config(
      cfg, observation_groups=tuple(cfg.observation_adapter)
  )
  control_timestep = deploy_cfg.sim.control_timestep

  clock = clock_device.ClockDevice()
  follower = real_mod.create_follower_device()
  cam_w, cam_h = emb.camera_size
  camera = real_mod.create_wrist_camera_device(
      name=emb.wrist_rgb_key,
      config=real.OpenCVCameraConfig(
          width=cam_w,
          height=cam_h,
          fps=round(1.0 / control_timestep),
      ),
  )
  device_layer = real_mod.create_device_layer(
      clock,
      follower,
      camera,
      control_period_seconds=control_timestep,
  )

  return torch_environment.Environment(
      device_layer=wrapper.TorchDeviceLayerWrapper(
          device_layer,
          sim=None,
          control_timestep=control_timestep,
          xpu_device="cpu",
      ),
      task_layer=build.build_task_layer(deploy_cfg),
      observation_space_adapter=functional_observation.FunctionalObservationAdapter(
          observations=deploy_cfg.observation_adapter,
          xpu_device="cpu",
          nworld=1,
      ),
      environment_reset=HomingReset(
          emb=emb,
          follower=follower,
          clock=clock,
          duration_secs=homing_duration_secs,
          dt=control_timestep,
      ),
      environment_closer=Shutdown(
          follower=follower,
          emb=emb,
          camera=camera,
          duration_secs=homing_duration_secs,
          dt=control_timestep,
      ),
  )
