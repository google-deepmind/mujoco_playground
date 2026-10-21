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

"""Resetter for the agentic tasks: home pose + scattered objects."""

from collections.abc import Sequence
from typing import Any

from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
from mujoco_playground.tasks.manipulation.pick_cube import reset as pick_reset
from reaf.simulation.torch import environment
from reaf.simulation.torch import noise
import torch


class ScatterObjectsResetter(environment.EnvironmentReset):
  """Resets the arm to the home pose and scatters objects without overlap.

  Objects are scattered in the embodiment's spawn rectangle, further limited
  to a disc around its base axis that a top-down grasp can reach.

  Args:
    objects: (device name, rest height z, randomize yaw, footprint radius) per
      free object. Two objects are kept at least the sum of their radii plus
      `margin` apart in XY.
    robot: Embodiment name (see `embodiment.EMBODIMENTS`).
    margin: Extra XY clearance between footprints (m).
  """

  def __init__(
      self,
      objects: Sequence[tuple[str, float, bool, float]],
      robot: str = "so101",
      margin: float = 0.02,
  ):
    self._objects = tuple(objects)
    self._emb = embodiment_lib.get(robot)
    self._margin = margin

  def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
    del config
    n, xpu_device = self._env.nworld, self._env.xpu_device
    sim = self._env.device_layer.sim
    emb = self._emb
    prefix = embodiment_lib.ATTACHMENT_PREFIX
    mj_model = sim.mj_model
    qpos_ids = torch.tensor(
        [int(mj_model.joint(f"{prefix}{j}").qposadr[0]) for j in emb.mj_joints],
        dtype=torch.long,
        device=xpu_device,
    )
    dof_ids = torch.tensor(
        [int(mj_model.joint(f"{prefix}{j}").dofadr[0]) for j in emb.mj_joints],
        dtype=torch.long,
        device=xpu_device,
    )
    act_ids = torch.tensor(
        [int(mj_model.actuator(f"{prefix}{j}").id) for j in emb.joints],
        dtype=torch.long,
        device=xpu_device,
    )
    mj_home = torch.tensor(
        emb.follower_to_mj_qpos(emb.home_qpos),
        dtype=sim.data.qpos.dtype,
        device=xpu_device,
    )
    mask_q = env_mask.unsqueeze(-1).expand(n, len(qpos_ids))
    sim.data.qpos[:, qpos_ids] = torch.where(
        mask_q, mj_home.unsqueeze(0).expand(n, -1), sim.data.qpos[:, qpos_ids]
    )
    sim.data.qvel[:, dof_ids] = torch.where(
        mask_q,
        torch.zeros(
            (n, len(dof_ids)), dtype=sim.data.qvel.dtype, device=xpu_device
        ),
        sim.data.qvel[:, dof_ids],
    )
    follower = self._env.device_layer.get_device("follower")
    prev_ctrl = sim.data.ctrl[:, act_ids].clone()
    follower.set_substep_commands(
        sim,
        {
            k: torch.full(
                (n,), float(v), dtype=torch.float32, device=xpu_device
            )
            for k, v in zip(emb.goal_position_keys, emb.home_qpos)
        },
    )
    if not env_mask.all():
      mask_a = env_mask.unsqueeze(-1).expand(n, len(act_ids))
      sim.data.ctrl[:, act_ids] = torch.where(
          mask_a, sim.data.ctrl[:, act_ids], prev_ctrl
      )

    placed: list[tuple[torch.Tensor, float]] = []
    # Largest footprint first so the small objects fit around it.
    for name, rest_z, random_yaw, radius in sorted(
        self._objects, key=lambda o: -o[3]
    ):
      xy = self._sample_xy(n, xpu_device, placed, radius)
      pos = torch.cat([xy, torch.full((n, 1), rest_z, device=xpu_device)], -1)
      if random_yaw:
        yaw = noise.sample_uniform(
            (n, 1), xpu_device, low=-torch.pi, high=torch.pi
        )
      else:
        yaw = torch.zeros((n, 1), device=xpu_device)
      zeros = torch.zeros((n, 1), device=xpu_device)
      quat = torch.cat(
          [torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], -1
      )
      obj = self._env.device_layer.get_device(name)
      obj.set_pos(env_mask, pos)
      obj.set_quat(env_mask, quat)
      obj.set_vel(env_mask, torch.zeros((n, 6), device=xpu_device))
      placed.append((xy, radius))

    sim = self._env.device_layer.sim
    pick_reset.clamp_qpos_to_limits(sim.model, sim.data, env_mask)

  def _sample_xy(
      self,
      n: int,
      device: torch.device,
      placed: Sequence[tuple[torch.Tensor, float]],
      radius: float,
  ) -> torch.Tensor:
    """Rejection-samples XY positions whose footprint clears `placed`."""
    xy = _sample_xy(n, device, self._emb)
    for _ in range(100):
      bad = torch.zeros(n, dtype=torch.bool, device=device)
      for other, other_radius in placed:
        min_dist = radius + other_radius + self._margin
        bad |= torch.linalg.norm(xy - other, dim=-1) < min_dist
      if not bad.any():
        break
      xy[bad] = _sample_xy(int(bad.sum().item()), device, self._emb)
    return xy


def _sample_xy(
    n: int,
    device: torch.device,
    emb: embodiment_lib.Embodiment,
) -> torch.Tensor:
  """Samples XY in the spawn rectangle, within `max_reach` of the base."""
  low = torch.tensor([emb.spawn_x[0], emb.spawn_y[0]], device=device)
  high = torch.tensor([emb.spawn_x[1], emb.spawn_y[1]], device=device)
  base = torch.tensor(emb.base_xy, device=device)
  xy = noise.sample_uniform((n, 2), device, low=low, high=high)
  for _ in range(100):
    far = torch.linalg.norm(xy - base, dim=-1) > emb.max_reach
    if not far.any():
      break
    k = int(far.sum().item())
    xy[far] = noise.sample_uniform((k, 2), device, low=low, high=high)
  return xy
