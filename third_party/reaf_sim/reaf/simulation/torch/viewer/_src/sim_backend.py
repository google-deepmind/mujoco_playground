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

"""Simulation-side backend for reaf-sim Studio Viewer."""

from collections.abc import Sequence
from typing import Any
from typing import Generic

import dm_env
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import types
from reaf.simulation.viewer import sim_plugin
import torch
import tree


class TorchSimBackend(Generic[gdmr_types.PolicyDataType]):
  """Torch backend bridging Simulator to SimPlugin."""

  def __init__(
      self,
      environment: gdmr_env.Environment,
      policy: gdmr_policy.Policy[gdmr_types.PolicyDataType],
      simulator: simulator_module.Simulator,
      xpu_device: str,
      world_id: int,
      *,
      loggers: Sequence[Any] = (),
  ) -> None:
    self._environment = environment
    self._policy = policy
    self._policy_state = self._policy.initial_state()
    self._simulator = simulator
    self._xpu_device = xpu_device
    self._world_id = world_id
    self._loggers = list(loggers)
    self._timestep: dm_env.TimeStep | None = None

  def reward_spec(self) -> Any:
    """Returns the single-world reward spec for self._world_id."""
    return tree.map_structure(
        lambda s: dm_env.specs.Array(
            shape=s.shape[1:], dtype=s.dtype, name=s.name
        ),
        self._environment.reward_spec(),
    )

  def _to_world_timestep(
      self, batch_timestep: dm_env.TimeStep
  ) -> dm_env.TimeStep:
    """Extracts a single-world dm_env.TimeStep for self._world_id."""

    def to_world(x: torch.Tensor | np.ndarray) -> np.ndarray:
      x = torch.as_tensor(x)
      return (x[self._world_id] if x.ndim else x).detach().cpu().numpy()

    reward = (
        None
        if batch_timestep.reward is None
        else tree.map_structure(to_world, batch_timestep.reward)
    )
    return dm_env.TimeStep(
        step_type=dm_env.StepType(
            int(batch_timestep.step_type[self._world_id])
        ),
        reward=reward,
        discount=batch_timestep.discount,
        observation=batch_timestep.observation,
    )

  def reset(self) -> dm_env.TimeStep:
    """Resets the REAF environment and writes/resets episodic loggers."""
    # Episode ended - write loggers if resetting from a previous episode.
    if self._timestep is not None and self._loggers:
      for logger in self._loggers:
        logger.write()
    self._timestep = self._environment.reset()
    if self._loggers:
      for logger in self._loggers:
        logger.reset(self._timestep)
    return self._to_world_timestep(self._timestep)

  def step(self) -> dm_env.TimeStep:
    """Steps the policy and environment and records to episodic loggers."""
    assert self._timestep is not None
    with torch.no_grad():
      if self._timestep.step_type[self._world_id] == dm_env.StepType.FIRST:
        self._policy_state = self._policy.initial_state()
      (action, policy_extra), self._policy_state = self._policy.step(
          self._timestep, self._policy_state
      )
      self._timestep = self._environment.step(action)
    for logger in self._loggers:
      logger.record_action_and_next_timestep(
          action, self._timestep, policy_extra
      )
    return self._to_world_timestep(self._timestep)

  def sync_to_viewer(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Copies the state of `self._world_id` from the simulation into (model, data).

    Also syncs domain-randomized model fields and runs mj_forward so that
    derived quantities (xpos, xquat, ...) are up to date.

    Args:
      model: Destination MuJoCo model.
      data: Destination MuJoCo data.
    """
    sim = self._simulator
    world_id = self._world_id
    if sim.impl == types.Impl.CPP:
      # CPP impl: copy state using mj_getState/mj_setState.
      mjt_state = (
          int(mujoco.mjtState.mjSTATE_PHYSICS)
          | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
          | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
      )
      state = np.empty(mujoco.mj_stateSize(model, mjt_state))
      mujoco.mj_getState(
          model,
          sim.data.obj,  # pyrefly: ignore[bad-argument-type]
          state,
          mjt_state,  # pyrefly: ignore[bad-argument-type]
      )
      mujoco.mj_setState(model, data, state, mjt_state)
    else:
      # WARP impl: use mjw.get_data_into to copy from GPU.
      mjw.get_data_into(
          data,
          model,
          sim.data.obj,  # pyrefly: ignore[bad-argument-type]
          world_id=world_id,
      )

    # sync domain randomized model fields.
    model.geom_size[:] = sim.model.geom_size[world_id].cpu().numpy()
    model.geom_rgba[:] = sim.model.geom_rgba[world_id].cpu().numpy()
    model.light_pos[:] = sim.model.light_pos[world_id].cpu().numpy()
    model.light_dir[:] = sim.model.light_dir[world_id].cpu().numpy()
    model.light_diffuse[:] = sim.model.light_diffuse[world_id].cpu().numpy()
    model.light_ambient[:] = sim.model.light_ambient[world_id].cpu().numpy()
    model.cam_pos[:] = sim.model.cam_pos[world_id].cpu().numpy()
    model.cam_quat[:] = sim.model.cam_quat[world_id].cpu().numpy()
    model.cam_fovy[:] = sim.model.cam_fovy[world_id].cpu().numpy()
    model.cam_intrinsic[:] = sim.model.cam_intrinsic[world_id].cpu().numpy()
    model.site_pos[:] = sim.model.site_pos[world_id].cpu().numpy()
    model.site_quat[:] = sim.model.site_quat[world_id].cpu().numpy()

    # Run mj_forward to stabilize xfrc_applied computation of applyPerturbForce.
    mujoco.mj_forward(model, data)

  def apply_perturbation(self, xfrc_applied_cpu: np.ndarray) -> None:
    """Applies perturbation forces from viewer MjData to the Torch simulator."""
    xfrc_applied = torch.as_tensor(xfrc_applied_cpu, device=self._xpu_device)
    self._simulator.data.xfrc_applied[self._world_id].copy_(xfrc_applied)

  def apply_mocap(self, mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> None:
    """Applies mocap poses from viewer MjData to the Torch simulator."""
    data = self._simulator.data
    data.mocap_pos[self._world_id].copy_(
        torch.as_tensor(mocap_pos, device=self._xpu_device)
    )
    data.mocap_quat[self._world_id].copy_(
        torch.as_tensor(mocap_quat, device=self._xpu_device)
    )

  def set_instruction(self, instruction: str) -> None:
    """Forwards a text instruction to the environment if supported."""
    sim_plugin.set_env_instruction(self._environment, instruction)
