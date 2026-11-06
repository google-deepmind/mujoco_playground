# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entities for actuated robots."""

from collections.abc import Mapping
from typing import Any
from typing import override

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class ActuatorDevice(device.Device):
  """Actuate joints, tendons, or sites using existing MJCF actuators."""

  def __init__(
      self,
      name: str = "",
      xpu_device: str = "cuda:0",
      *,
      targets: list[str],
      measurements: list[str] | None = None,
      home_ctrl: torch.Tensor | list[float] | None = None,
  ):
    """Initialize runtime control logic.

    Args:
      name: Entity name for lookup.
      xpu_device: Torch device.
      targets: List of actuator names in the compiled MjModel.
      measurements: Which measurements to expose. Options: "joint_pos",
        "joint_vel", "joint_range", "actuator_force", "tendon_length",
        "tendon_velocity", "ctrl". Default: [].
      home_ctrl: Default home control values.
    """
    self._name = name
    self._xpu_device = xpu_device
    self._target_names = targets
    self._measurements = list(measurements) if measurements is not None else []
    self._num_targets = len(targets)

    if home_ctrl is not None:
      if not isinstance(home_ctrl, torch.Tensor):
        home_ctrl = torch.tensor(
            home_ctrl, dtype=torch.float32, device=xpu_device
        )
      self._home_ctrl = home_ctrl.to(xpu_device)
    else:
      self._home_ctrl = None

    # Indices resolved in initialize()
    self._joint_qpos_ids: torch.Tensor | None = None
    self._joint_qvel_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._out_ids: torch.Tensor | None = None
    self._tendon_ids: torch.Tensor | None = None
    self._joints: list[Any] = []

  @property
  @override
  def name(self) -> str:
    return self._name

  @property
  def num_targets(self) -> int:
    return self._num_targets

  @property
  def home_ctrl(self) -> torch.Tensor:
    """Home control values. Shape: (num_ctrl,)."""
    if self._home_ctrl is None:
      raise RuntimeError(
          f"ActuatorDevice {self._name!r} must be initialized via"
          " initialize(simulator) before accessing home_ctrl."
      )
    return self._home_ctrl

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Resolve joint, actuator, and tendon indices."""
    self._data = simulator.data
    mj_model = simulator.mj_model
    mj_spec = simulator.mj_spec

    joint_qpos_ids = []
    joint_qvel_ids = []
    ctrl_ids = []
    out_ids = []
    tendon_ids = []
    self._joints = []

    for target_name in self._target_names:
      act = mj_model.actuator(target_name)
      # Map actuator control and output spans (e.g. multi-input <dcmotor>).
      ctrl_adr = int(mj_model.actuator_ctrladr[act.id])
      ctrl_ids.extend(
          range(ctrl_adr, ctrl_adr + int(mj_model.actuator_ctrlnum[act.id]))
      )
      out_adr = int(mj_model.actuator_outadr[act.id])
      out_ids.extend(
          range(out_adr, out_adr + int(mj_model.actuator_outnum[act.id]))
      )

      trntype = mujoco.mjtTrn(mj_model.actuator_trntype[act.id])
      if trntype in (
          mujoco.mjtTrn.mjTRN_JOINT,
          mujoco.mjtTrn.mjTRN_JOINTINPARENT,
      ):
        joint_id = mj_model.actuator_trnid[act.id, 0]
        joint_qpos_ids.append(mj_model.jnt_qposadr[joint_id])
        joint_qvel_ids.append(mj_model.jnt_dofadr[joint_id])
        joint_name = mj_model.joint(joint_id).name
        self._joints.append(mj_spec.joint(joint_name))
      elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
        tendon_id = mj_model.actuator_trnid[act.id, 0]
        tendon_ids.append(tendon_id)

    # Store as tensors on the entity's device
    self._joint_qpos_ids = torch.tensor(
        joint_qpos_ids, dtype=torch.long, device=self._xpu_device
    )
    self._joint_qvel_ids = torch.tensor(
        joint_qvel_ids, dtype=torch.long, device=self._xpu_device
    )
    self._ctrl_ids = torch.tensor(
        ctrl_ids, dtype=torch.long, device=self._xpu_device
    )
    if self._home_ctrl is None:
      self._home_ctrl = torch.zeros(len(ctrl_ids), device=self._xpu_device)
    elif self._home_ctrl.shape != (len(ctrl_ids),):
      raise ValueError(
          f"home_ctrl shape {tuple(self._home_ctrl.shape)} does not match"
          f" resolved control width ({len(ctrl_ids)},) for targets"
          f" {self._target_names}."
      )
    self._out_ids = torch.tensor(
        out_ids, dtype=torch.long, device=self._xpu_device
    )
    self._tendon_ids = torch.tensor(
        tendon_ids, dtype=torch.long, device=self._xpu_device
    )

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    assert self._ctrl_ids is not None, (
        f"ActuatorDevice {self._name!r} must be initialized via"
        " initialize(simulator) before querying specs."
    )
    return {
        f"{self._name}/ctrl": specs.Array(
            shape=(self._ctrl_ids.numel(),), dtype=np.float32
        ),
    }

  @override
  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    if (
        self._ctrl_ids is None
        or self._out_ids is None
        or self._joint_qpos_ids is None
        or self._joint_qvel_ids is None
        or self._tendon_ids is None
    ):
      raise AssertionError(
          f"ActuatorDevice {self._name!r} must be initialized via"
          " initialize(simulator) before querying specs."
      )
    result = {}
    for m in self._measurements:
      # Use resolved control/output widths for multi-input/output actuators.
      if m in ("ctrl", "home_ctrl"):
        shape = (self._ctrl_ids.numel(),)
      elif m == "actuator_force":
        shape = (self._out_ids.numel(),)
      elif m == "joint_pos":
        shape = (self._joint_qpos_ids.numel(),)
      elif m == "joint_vel":
        shape = (self._joint_qvel_ids.numel(),)
      elif m == "joint_range":
        shape = (len(self._joints), 2)
      elif m in ("tendon_length", "tendon_velocity"):
        shape = (self._tendon_ids.numel(),)
      else:
        raise KeyError(m)
      result[f"{self._name}/{m}"] = specs.Array(shape=shape, dtype=np.float32)
    return result

  @override
  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    ctrl = consumed_substep_commands[f"{self._name}/ctrl"]
    simulator.data.ctrl[:, self._ctrl_ids] = ctrl.to(
        dtype=simulator.data.ctrl.dtype
    )

  @override
  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    data = simulator.data

    def get(m: str) -> torch.Tensor:
      if m == "joint_pos":
        return data.qpos[:, self._joint_qpos_ids]
      elif m == "joint_vel":
        return data.qvel[:, self._joint_qvel_ids]
      elif m == "actuator_force":
        return data.actuator_force[:, self._out_ids]
      elif m == "tendon_length":
        return data.ten_length[:, self._tendon_ids]
      elif m == "tendon_velocity":
        return data.ten_velocity[:, self._tendon_ids]
      elif m == "joint_range":
        bound_model = simulator.model.bind(self._joints)
        return bound_model.range
      elif m == "ctrl":
        return data.ctrl[:, self._ctrl_ids]
      elif m == "home_ctrl":
        return self.home_ctrl
      raise KeyError(m)

    return {f"{self._name}/{m}": get(m) for m in self._measurements}

  # Puppet API - resetter calls these.

  def set_ctrl(self, env_mask: torch.Tensor | None, ctrl: torch.Tensor) -> None:
    """Set actuator control for specified environments."""
    if env_mask is None:
      self._data.ctrl[:, self._ctrl_ids] = ctrl
    else:
      curr = self._data.ctrl[:, self._ctrl_ids]
      self._data.ctrl[:, self._ctrl_ids] = torch.where(
          env_mask.unsqueeze(1), ctrl, curr
      )
