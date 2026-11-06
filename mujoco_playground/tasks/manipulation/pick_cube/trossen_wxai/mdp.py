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

"""MDP functions and custom resetter for Trossen Pick Cube task."""

from typing import Any

import mujoco
from mujoco_playground.tasks.manipulation.pick_cube import reset as pick_cube_reset
from reaf.simulation.torch import noise
import torch
import torch.nn.functional as F


class TrossenPickCubeResetter(pick_cube_reset.PickCubeResetter):
  """Custom resetter for Trossen to ensure symmetric gripper randomization."""

  def on_reset(
      self,
      config: Any,
      env_mask: torch.Tensor,
  ) -> None:
    """Randomize robot joints, object pose, and target pose."""
    arm = self._env.device_layer.get_device("robot")
    arm.set_ctrl(env_mask, arm.home_ctrl)

    # Reset arm joints.
    arm_state = self._env.device_layer.get_device("arm_state")
    nworld = self._env.nworld
    xpu_device = self._env.xpu_device
    perturb_arm = noise.sample_uniform(
        (nworld, 6),
        xpu_device,
        low=-self._joint_perturbation,
        high=self._joint_perturbation,
    )
    qpos_arm = arm_state.home_qpos.unsqueeze(0) + perturb_arm
    arm_state.set_qpos(env_mask, qpos_arm)
    arm_state.set_qvel(env_mask, arm_state.home_qvel)

    # Reset gripper joints symmetrically.
    gripper_state = self._env.device_layer.get_device("gripper_state")
    perturb_gripper_scalar = noise.sample_uniform(
        (nworld, 1), xpu_device, low=-0.005, high=0.005
    )
    perturb_gripper = perturb_gripper_scalar.repeat(1, 2)
    qpos_gripper = gripper_state.home_qpos.unsqueeze(0) + perturb_gripper
    gripper_state.set_qpos(env_mask, qpos_gripper)
    gripper_state.set_qvel(env_mask, gripper_state.home_qvel)

    # Reset object.
    obj = self._env.device_layer.get_device("object")
    pick_cube_reset.reset_free_body(
        obj, env_mask, nworld, xpu_device, self._object_pos_range
    )

    # Reset target.
    r = self._target_pos_range
    low = torch.tensor([r[0][0], r[1][0], r[2][0]], device=xpu_device)
    high = torch.tensor([r[0][1], r[1][1], r[2][1]], device=xpu_device)
    sampled_pos = noise.sample_uniform(
        (nworld, 3), xpu_device, low=low, high=high
    )
    target = self._env.device_layer.get_device("target")
    target.set_pos(env_mask, sampled_pos)

    sim = self._env.device_layer.sim
    pick_cube_reset.clamp_qpos_to_limits(sim.model, sim.data, env_mask)

    # Reset resampling timers.
    assert self._time_since_resample is not None
    assert self._next_resample_time is not None
    self._time_since_resample[:] = torch.where(
        env_mask, 0.0, self._time_since_resample
    )
    low, high = self._resampling_time_range
    self._next_resample_time[:] = torch.where(
        env_mask,
        noise.sample_uniform((nworld,), xpu_device, low=low, high=high),
        self._next_resample_time,
    )
