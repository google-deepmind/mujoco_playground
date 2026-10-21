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

"""Shared visual ghost robot entity and updater for trajectory tracking."""

from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import Any

from dm_env import specs
import mujoco
from mujoco_playground import asset
from reaf.simulation import torch as device
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import substep_commands_processor
import torch

# Namespace of the attached ghost robot in the compiled scene.
GHOST_PREFIX = "ghost/"


class GhostDevice(device.ArticulatedDevice):
  """Visual ghost robot entity for trajectory visual cues.

  Spawns a semi-transparent ghost robot body configured with no collisions
  and standard gravity compensation, ideal for imitating target reference
  frames.
  """

  def configure_spec(  # pyrefly: ignore[bad-override]
      self,
      spec: mujoco.MjSpec,
      *,
      xml_path: str | None = None,
  ) -> mujoco.MjSpec:
    super().configure_spec(spec)
    if xml_path is None:
      raise ValueError(
          "xml_path must be provided in build_params for GhostDevice."
      )
    ghost_spec = asset.get(xml_path).spec()
    while ghost_spec.actuators:
      ghost_spec.delete(ghost_spec.actuators[0])
    while ghost_spec.keys:
      ghost_spec.delete(ghost_spec.keys[0])
    for mat in ghost_spec.materials:
      mat.rgba[3] = 0.3
    for geom in ghost_spec.worldbody.find_all("geom"):
      geom.contype = 0
      geom.conaffinity = 0
      if geom.rgba is not None:
        geom.rgba[3] = 0.3
      else:
        geom.rgba = [0.5, 0.5, 0.5, 0.3]
    for body in ghost_spec.worldbody.find_all("body"):
      body.gravcomp = 1.0
    self._joint_names = [
        f"{GHOST_PREFIX}{j.name}_ghost" for j in ghost_spec.joints if j.name
    ]
    frame = spec.worldbody.add_frame()
    spec.attach(ghost_spec, prefix=GHOST_PREFIX, frame=frame, suffix="_ghost")
    return spec


class GhostUpdater(substep_commands_processor.SubstepCommandsProcessor):
  """Substep command processor updating visual ghost robot poses.

  Syncs the transparent ghost robot with the reference state currently
  emitted by MotionReferenceProducer (via a SharedMotionState handle), so the
  ghost follows the per-env motion time including RSI and adaptive
  sampling.
  """

  def __init__(
      self,
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      shared_state: Any = None,
      ghost_device_name: str = "ghost_robot",
  ):
    """Initializes the ghost updater.

    Args:
      nworld: Number of parallel simulation environments.
      xpu_device: Hardware device executor pointer (e.g., 'cuda:0').
      shared_state: motion_command.SharedMotionState updated by
        MotionReferenceProducer each step.
      ghost_device_name: Unique name identifier of the visual ghost entity.
    """
    del nworld
    super().__init__()
    self._xpu_device = xpu_device
    self._shared_state = shared_state
    self._ghost_name = ghost_device_name
    self._ghost_qpos_indices = None
    self._ghost_qvel_indices = None

  @property
  def name(self) -> str:
    return "ghost_updater"

  def produced_substep_commands_keys(self) -> set[str]:
    return set()

  def consumed_substep_commands_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def initialize(
      self,
      simulator: simulator_module.Simulator,
  ) -> None:
    mj_model = simulator.mj_model
    data = simulator.data
    ghost_qpos_indices = []
    ghost_qvel_indices = []
    for i in range(mj_model.njnt):
      joint_name = mj_model.joint(i).name
      if joint_name.endswith("_ghost"):
        joint = mj_model.joint(i)
        qpos_size = 7 if joint.type == mujoco.mjtJoint.mjJNT_FREE else 1
        qvel_size = 6 if joint.type == mujoco.mjtJoint.mjJNT_FREE else 1
        qpos_adr = joint.qposadr[0]
        dof_adr = joint.dofadr[0]
        for idx in range(qpos_adr, qpos_adr + qpos_size):
          ghost_qpos_indices.append(idx)
        for idx in range(dof_adr, dof_adr + qvel_size):
          ghost_qvel_indices.append(idx)
    self._ghost_qpos_indices = torch.tensor(
        ghost_qpos_indices, dtype=torch.long, device=data.qpos.device
    )
    self._ghost_qvel_indices = torch.tensor(
        ghost_qvel_indices, dtype=torch.long, device=data.qvel.device
    )

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    pass

  def process_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      substep_commands: MutableMapping[str, torch.Tensor],
  ) -> None:
    del substep_commands
    state = self._shared_state
    if self._ghost_qpos_indices is None or state.root_pos is None:
      return
    full_ghost_qpos = torch.cat(
        [state.root_pos, state.root_quat, state.joint_pos], dim=-1
    )
    data = simulator.data
    data.qpos[:, self._ghost_qpos_indices] = full_ghost_qpos
    data.qvel[:, self._ghost_qvel_indices] = 0.0
