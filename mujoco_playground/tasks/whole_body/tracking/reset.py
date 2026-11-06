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

"""Resetters for the whole-body tracking task."""

from typing import Any

from reaf.simulation.torch import environment
from reaf.simulation.torch import math as pg_math
import torch


def _range_tensor(
    range_dict: dict[str, tuple[float, float]], device: torch.device
) -> torch.Tensor:
  range_list = [
      range_dict.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
  ]
  return torch.tensor(range_list, dtype=torch.float32, device=device)


class MotionResetter(environment.EnvironmentReset):
  """Drives reference state initialization (RSI) and clip-end looping.

  On episode reset (`on_reset`), samples new reference motion time steps and
  teleports reset environments to the perturbed reference state. On each step
  (`on_step`), advances the reference motion time and teleports any environment
  whose trajectory clip reached the end.
  """

  def __init__(
      self,
      *,
      joint_names: tuple[str, ...],
      soft_joint_pos_limits: torch.Tensor,
      pose_range: dict[str, tuple[float, float]] | None = None,
      velocity_range: dict[str, tuple[float, float]] | None = None,
      joint_position_range: tuple[float, float] = (-0.1, 0.1),
      root_joint_name: str = "floating_base_joint",
      command_name: str = "motion",
  ) -> None:
    self._joint_names = tuple(joint_names)
    self._soft_limits_init = soft_joint_pos_limits
    self._pose_range = dict(pose_range or {})
    self._velocity_range = dict(velocity_range or {})
    self._joint_position_range = joint_position_range
    self._root_joint_name = root_joint_name
    self._command_name = command_name

    self._motion_ref: Any = None
    self._joint_q_adr: torch.Tensor | None = None
    self._joint_v_adr: torch.Tensor | None = None
    self._free_q_adr: int = 0
    self._free_v_adr: int = 0
    self._soft_limits: torch.Tensor | None = None
    self._pose_ranges: torch.Tensor | None = None
    self._vel_ranges: torch.Tensor | None = None

  def _get_motion_ref(self) -> Any:
    if self._motion_ref is None:
      self._motion_ref = self._env.task_layer.get_component_by_name(
          self._command_name
      )
    return self._motion_ref

  def on_startup(self) -> None:
    if self._env is None or self._joint_q_adr is not None:
      return
    sim = self._env.device_layer.sim
    mj_model = sim.mj_model
    device = sim.data.qpos.device

    q_adr, v_adr = [], []
    for jn in self._joint_names:
      j = mj_model.joint(jn)
      q_adr.append(int(j.qposadr[0]))
      v_adr.append(int(j.dofadr[0]))
    self._joint_q_adr = torch.tensor(q_adr, dtype=torch.long, device=device)
    self._joint_v_adr = torch.tensor(v_adr, dtype=torch.long, device=device)

    root = mj_model.joint(self._root_joint_name)
    self._free_q_adr = int(root.qposadr[0])
    self._free_v_adr = int(root.dofadr[0])

    self._soft_limits = self._soft_limits_init.to(device=device)
    self._pose_ranges = _range_tensor(self._pose_range, device)
    self._vel_ranges = _range_tensor(self._velocity_range, device)

  def _teleport_to_reference(self, env_mask: torch.Tensor) -> None:
    sim = self._env.device_layer.sim
    data = sim.data
    device = data.qpos.device
    n = self._env.nworld
    motion_ref = self._get_motion_ref()
    t = motion_ref.time_steps

    root_pos = motion_ref.motion.body_pos_w[t, 0].clone()
    root_ori = motion_ref.motion.body_quat_w[t, 0].clone()
    root_lin_vel = motion_ref.motion.body_lin_vel_w[t, 0].clone()
    root_ang_vel = motion_ref.motion.body_ang_vel_w[t, 0].clone()

    assert self._pose_ranges is not None
    assert self._vel_ranges is not None
    assert self._soft_limits is not None
    assert self._joint_q_adr is not None
    assert self._joint_v_adr is not None

    rand_pose = pg_math.sample_uniform_range(
        self._pose_ranges[:, 0], self._pose_ranges[:, 1], (n, 6), device=device
    )
    root_pos += rand_pose[:, 0:3]
    ori_delta = pg_math.quat_from_euler_xyz(
        rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5]
    )
    root_ori = pg_math.quat_mul(ori_delta, root_ori)

    rand_vel = pg_math.sample_uniform_range(
        self._vel_ranges[:, 0], self._vel_ranges[:, 1], (n, 6), device=device
    )
    root_lin_vel += rand_vel[:, :3]
    root_ang_vel += rand_vel[:, 3:]

    joint_pos = motion_ref.motion.joint_pos[t].clone()
    joint_vel = motion_ref.motion.joint_vel[t]
    joint_pos += pg_math.sample_uniform_range(
        self._joint_position_range[0],
        self._joint_position_range[1],
        tuple(joint_pos.shape),
        device=device,
    )
    joint_pos = torch.clip(
        joint_pos,
        self._soft_limits[:, 0],
        self._soft_limits[:, 1],
    )

    fq, fv = self._free_q_adr, self._free_v_adr
    mask_col = env_mask.unsqueeze(1)
    q_dtype = data.qpos.dtype
    v_dtype = data.qvel.dtype

    data.qpos[:, self._joint_q_adr] = torch.where(
        mask_col, joint_pos.to(q_dtype), data.qpos[:, self._joint_q_adr]
    )
    data.qvel[:, self._joint_v_adr] = torch.where(
        mask_col, joint_vel.to(v_dtype), data.qvel[:, self._joint_v_adr]
    )
    data.qpos[:, fq : fq + 3] = torch.where(
        mask_col, root_pos.to(q_dtype), data.qpos[:, fq : fq + 3]
    )
    data.qpos[:, fq + 3 : fq + 7] = torch.where(
        mask_col, root_ori.to(q_dtype), data.qpos[:, fq + 3 : fq + 7]
    )
    data.qvel[:, fv : fv + 3] = torch.where(
        mask_col, root_lin_vel.to(v_dtype), data.qvel[:, fv : fv + 3]
    )
    data.qvel[:, fv + 3 : fv + 6] = torch.where(
        mask_col,
        pg_math.quat_rotate_inverse(root_ori, root_ang_vel).to(v_dtype),
        data.qvel[:, fv + 3 : fv + 6],
    )

  def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
    del config
    motion_ref = self._get_motion_ref()
    motion_ref.sample_time_steps(env_mask)
    motion_ref.clear_fail_latch(env_mask)
    self._teleport_to_reference(env_mask)

  def on_step(self) -> None:
    if self._env is None:
      return
    motion_ref = self._get_motion_ref()
    ended = motion_ref.advance_time_steps()
    if ended.any():
      motion_ref.sample_time_steps(ended)
      self._teleport_to_reference(ended)
      self._env.device_layer.forward()


class PushRobotResetter(environment.EnvironmentReset):
  """Applies random root-velocity pushes at sampled time intervals.

  Per-environment timers sampled uniformly from `interval_range_s` trigger a
  push that adds a uniform 6-DoF velocity delta to the root in the world frame
  (converting angular velocity to the free joint's body-local frame on write).
  """

  def __init__(
      self,
      velocity_range: dict[str, tuple[float, float]],
      interval_range_s: tuple[float, float] = (1.0, 3.0),
      control_timestep: float = 0.02,
      root_joint_name: str = "floating_base_joint",
  ) -> None:
    self._velocity_range = dict(velocity_range)
    self._interval_range_s = interval_range_s
    self._control_timestep = control_timestep
    self._root_joint_name = root_joint_name
    self._timers: torch.Tensor | None = None
    self._free_q_adr: int | None = None
    self._free_v_adr: int | None = None
    self._ranges: torch.Tensor | None = None

  def _sample_timers(self, n: int, device: torch.device) -> torch.Tensor:
    lo, hi = self._interval_range_s
    return torch.rand(n, device=device) * (hi - lo) + lo

  def on_startup(self) -> None:
    if self._env is None or self._timers is not None:
      return
    sim = self._env.device_layer.sim
    root = sim.mj_model.joint(self._root_joint_name)
    self._free_q_adr = int(root.qposadr[0])
    self._free_v_adr = int(root.dofadr[0])
    device = sim.data.qpos.device
    self._timers = self._sample_timers(self._env.nworld, device)
    self._ranges = _range_tensor(self._velocity_range, device)

  def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
    del config
    if self._timers is None:
      return
    self._timers[:] = torch.where(
        env_mask,
        self._sample_timers(self._env.nworld, self._timers.device),
        self._timers,
    )

  def on_step(self) -> None:
    if self._env is None or self._timers is None or self._ranges is None:
      return
    self._timers -= self._control_timestep
    push_mask = self._timers < 1e-6
    if not push_mask.any():
      return
    device = self._timers.device
    nworld = self._env.nworld
    sim = self._env.device_layer.sim
    data = sim.data
    fq, fv = self._free_q_adr, self._free_v_adr
    assert fq is not None and fv is not None

    vel_delta = pg_math.sample_uniform_range(
        self._ranges[:, 0], self._ranges[:, 1], (nworld, 6), device=device
    )
    root_quat = data.qpos[:, fq + 3 : fq + 7].float()
    lin_vel_w = data.qvel[:, fv : fv + 3].float()
    ang_vel_w = pg_math.quat_rotate(
        root_quat, data.qvel[:, fv + 3 : fv + 6].float()
    )
    lin_vel_w = lin_vel_w + vel_delta[:, :3]
    ang_vel_w = ang_vel_w + vel_delta[:, 3:]

    mask_col = push_mask.unsqueeze(1)
    data.qvel[:, fv : fv + 3] = torch.where(
        mask_col, lin_vel_w.to(data.qvel.dtype), data.qvel[:, fv : fv + 3]
    )
    data.qvel[:, fv + 3 : fv + 6] = torch.where(
        mask_col,
        pg_math.quat_rotate_inverse(root_quat, ang_vel_w).to(data.qvel.dtype),
        data.qvel[:, fv + 3 : fv + 6],
    )
    self._timers[:] = torch.where(
        push_mask, self._sample_timers(nworld, device), self._timers
    )
