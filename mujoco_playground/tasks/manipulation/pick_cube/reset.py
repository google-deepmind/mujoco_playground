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

"""Resetter for SO101 Pick Cube task."""

from typing import Any

from reaf.simulation.torch import environment
from reaf.simulation.torch import noise
import torch


def reset_joints(
    arm,
    arm_state,
    env_mask,
    nworld: int,
    device: torch.device,
    perturbation: float,
) -> None:
  """Reset robot joints to home position with uniform perturbation."""
  perturb = noise.sample_uniform(
      (nworld, arm_state.home_qpos.shape[0]),
      device,
      low=-perturbation,
      high=perturbation,
  )
  qpos = arm_state.home_qpos.unsqueeze(0) + perturb
  arm.set_ctrl(env_mask, arm.home_ctrl)
  arm_state.set_qpos(env_mask, qpos)
  arm_state.set_qvel(env_mask, arm_state.home_qvel)


def reset_free_body(
    body,
    env_mask,
    nworld: int,
    device: torch.device,
    pos_range: tuple[tuple[float, float], ...],
) -> None:
  """Reset free body position with random yaw orientation."""
  body.set_pos(env_mask, _sample_pos_3d(nworld, device, pos_range))
  yaw = noise.sample_uniform((nworld, 1), device, low=-torch.pi, high=torch.pi)
  body.set_quat(env_mask, _quat_from_yaw(yaw, device))
  body.set_vel(env_mask, torch.zeros(nworld, 6, device=device))


def clamp_qpos_to_limits(model, data, env_mask) -> None:
  """Clamp qpos to jnt_range for all limited joints."""
  ltd = model.jnt_limited.bool()
  ids = model.jnt_qposadr[ltd]
  jnt_range = model.jnt_range
  data.qpos[:, ids] = torch.where(
      env_mask.unsqueeze(1),
      data.qpos[:, ids].clamp(jnt_range[:, ltd, 0], jnt_range[:, ltd, 1]),
      data.qpos[:, ids],
  )


class PickCubeResetter(environment.EnvironmentReset):
  """Resetter for pick task with mid-episode target resampling."""

  def __init__(
      self,
      object_pos_range: tuple[tuple[float, float], ...] = (
          (-0.15, 0.15),
          (-0.15, 0.15),
          (0.025, 0.025),
      ),
      target_pos_range: tuple[tuple[float, float], ...] = (
          (-0.15, 0.15),
          (-0.15, 0.15),
          (0.15, 0.25),
      ),
      resampling_time_range: tuple[float, float] = (8.0, 12.0),
      joint_perturbation: float = 0.1,
  ):
    self._object_pos_range = object_pos_range
    self._target_pos_range = target_pos_range
    self._resampling_time_range = resampling_time_range
    self._joint_perturbation = joint_perturbation

    self._time_since_resample: torch.Tensor = None  # pyrefly: ignore[bad-assignment]
    self._next_resample_time: torch.Tensor = None  # pyrefly: ignore[bad-assignment]

  def on_startup(self) -> None:
    """Initialize resampling timers once at startup."""
    nworld, xpu_device = self._env.nworld, self._env.xpu_device
    self._time_since_resample = torch.zeros((nworld,), device=xpu_device)
    self._next_resample_time = _sample_resample_times(
        nworld, xpu_device, self._resampling_time_range
    )

  def on_reset(
      self,
      config: Any,  # gdmr_env.ResetOptions,
      env_mask: torch.Tensor,
  ) -> None:
    """Randomize robot joints, object pose, and target pose."""
    # Reset robot, object, and target.
    arm = self._env.device_layer.get_device("robot")
    arm_state = self._env.device_layer.get_device("robot_state")
    nworld = self._env.nworld
    xpu_device = self._env.xpu_device
    reset_joints(
        arm, arm_state, env_mask, nworld, xpu_device, self._joint_perturbation
    )
    obj = self._env.device_layer.get_device("object")
    reset_free_body(obj, env_mask, nworld, xpu_device, self._object_pos_range)
    target = self._env.device_layer.get_device("target")
    # reset target position
    target.set_pos(
        env_mask, _sample_pos_3d(nworld, xpu_device, self._target_pos_range)
    )

    sim = self._env.device_layer.sim
    clamp_qpos_to_limits(sim.model, sim.data, env_mask)
    # Reset resampling timers.
    self._time_since_resample[:] = torch.where(
        env_mask, 0.0, self._time_since_resample
    )
    self._next_resample_time[:] = torch.where(
        env_mask,
        _sample_resample_times(nworld, xpu_device, self._resampling_time_range),
        self._next_resample_time,
    )

  def on_step(self) -> None:
    """Check and resample targets mid-episode."""
    if self._time_since_resample is None:
      return

    dt = self._env.device_layer.control_timestep
    self._time_since_resample += dt

    resample_mask = self._time_since_resample >= self._next_resample_time
    nworld = self._env.nworld
    target = self._env.device_layer.get_device("target")
    target.set_pos(
        resample_mask,
        _sample_pos_3d(nworld, self._env.xpu_device, self._target_pos_range),
    )

    self._time_since_resample[:] = torch.where(
        resample_mask, 0.0, self._time_since_resample
    )
    self._next_resample_time[:] = torch.where(
        resample_mask,
        _sample_resample_times(
            nworld, self._env.xpu_device, self._resampling_time_range
        ),
        self._next_resample_time,
    )


def _sample_pos_3d(
    nworld: int, device: torch.device, r: tuple[tuple[float, float], ...]
) -> torch.Tensor:
  """Sample nworld 3D positions uniformly within per-dimension ranges."""
  low = torch.tensor([r[0][0], r[1][0], r[2][0]], device=device)
  high = torch.tensor([r[0][1], r[1][1], r[2][1]], device=device)
  return noise.sample_uniform((nworld, 3), device, low=low, high=high)


def _quat_from_yaw(yaw: torch.Tensor, device: torch.device) -> torch.Tensor:
  """Quaternion for z-rotation: [cos(yaw/2), 0, 0, sin(yaw/2)]."""
  nworld = yaw.shape[0]
  zeros = torch.zeros(nworld, 1, device=device)
  return torch.cat([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], -1)


def _sample_resample_times(
    nworld: int, device: torch.device, time_range: tuple[float, float]
) -> torch.Tensor:
  """Sample nworld resample times uniformly within the time range."""
  low, high = time_range
  return noise.sample_uniform((nworld,), device, low=low, high=high)
