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
    self._bound_model = None
    self._empty_joint_range: torch.Tensor | None = None
    self._act_setpoint_ids: torch.Tensor | None = None
    self._act_setpoint_local_ctrl_ids: torch.Tensor | None = None
    self._act_setpoint_ctrl_ids: torch.Tensor | None = None
    self._act_setpoint_act_obj_ids: torch.Tensor | None = None
    self._act_setpoint_ctrllimited: torch.Tensor | None = None
    self._act_setpoint_actlimited: torch.Tensor | None = None
    self._has_setpoint_ctrllimited: bool = False
    self._has_setpoint_actlimited: bool = False
    self._act_zero_ids: torch.Tensor | None = None

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
    self._model = simulator.model
    self._data = simulator.data
    mj_model = simulator.mj_model
    mj_spec = simulator.mj_spec

    joint_qpos_ids = []
    joint_qvel_ids = []
    ctrl_ids = []
    out_ids = []
    tendon_ids = []
    act_setpoint_ids = []
    act_setpoint_local_ctrl_ids = []
    act_setpoint_ctrl_ids = []
    act_setpoint_act_obj_ids = []
    act_setpoint_ctrllimited = []
    act_setpoint_actlimited = []
    act_zero_ids = []
    self._joints = []
    clamp_ctrl_enabled = not bool(
        int(mj_model.opt.disableflags)
        & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    )

    for target_name in self._target_names:
      act = mj_model.actuator(target_name)
      local_ctrl_start = len(ctrl_ids)
      # Map actuator control and output spans (e.g. multi-input <dcmotor>).
      ctrl_adr = int(mj_model.actuator_ctrladr[act.id])
      ctrl_num = int(mj_model.actuator_ctrlnum[act.id])
      ctrl_ids.extend(range(ctrl_adr, ctrl_adr + ctrl_num))
      out_adr = int(mj_model.actuator_outadr[act.id])
      out_ids.extend(
          range(out_adr, out_adr + int(mj_model.actuator_outnum[act.id]))
      )

      # Precompute activation slots to set (setpoint) or zero in set_ctrl.
      act_adr = int(mj_model.actuator_actadr[act.id])
      act_num = int(mj_model.actuator_actnum[act.id])
      if act_adr >= 0 and act_num > 0:
        dyntype = mujoco.mjtDyn(mj_model.actuator_dyntype[act.id])
        setpoint_slot = -1
        if ctrl_num > 0:
          if dyntype in (
              mujoco.mjtDyn.mjDYN_FILTER,
              mujoco.mjtDyn.mjDYN_FILTEREXACT,
          ):
            setpoint_slot = act_adr + act_num - 1
          elif (
              dyntype == mujoco.mjtDyn.mjDYN_PID
              and mj_model.actuator_dynprm[act.id, 1] > 0.0
          ):
            setpoint_slot = act_adr
          elif (
              dyntype == mujoco.mjtDyn.mjDYN_DCMOTOR
              and mj_model.actuator_dynprm[act.id, 7] > 0.0
          ):
            setpoint_slot = act_adr
        for slot in range(act_adr, act_adr + act_num):
          if slot == setpoint_slot:
            act_setpoint_ids.append(slot)
            act_setpoint_local_ctrl_ids.append(local_ctrl_start)
            act_setpoint_ctrl_ids.append(ctrl_adr)
            act_setpoint_act_obj_ids.append(act.id)
            act_setpoint_ctrllimited.append(
                clamp_ctrl_enabled
                and bool(mj_model.actuator_ctrllimited[ctrl_adr])
            )
            act_setpoint_actlimited.append(
                dyntype != mujoco.mjtDyn.mjDYN_DCMOTOR
                and bool(mj_model.actuator_actlimited[act.id])
            )
          else:
            act_zero_ids.append(slot)

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
    self._bound_model = None
    self._empty_joint_range = None
    if "joint_range" in self._measurements and self._joints:
      self._bound_model = simulator.model.bind(self._joints)
      # Trigger lazy tiling of jnt_range before CUDA graph capture.
      _ = self._bound_model.range
    elif "joint_range" in self._measurements:
      self._empty_joint_range = torch.empty(
          (simulator.nworld, 0, 2), dtype=torch.float32, device=self._xpu_device
      )
    self._act_setpoint_ids = None
    self._act_setpoint_local_ctrl_ids = None
    self._act_setpoint_ctrl_ids = None
    self._act_setpoint_act_obj_ids = None
    self._act_setpoint_ctrllimited = None
    self._act_setpoint_actlimited = None
    self._has_setpoint_ctrllimited = False
    self._has_setpoint_actlimited = False
    if act_setpoint_ids:
      self._act_setpoint_ids = torch.tensor(
          act_setpoint_ids, dtype=torch.long, device=self._xpu_device
      )
      self._act_setpoint_local_ctrl_ids = torch.tensor(
          act_setpoint_local_ctrl_ids, dtype=torch.long, device=self._xpu_device
      )
      self._act_setpoint_ctrl_ids = torch.tensor(
          act_setpoint_ctrl_ids, dtype=torch.long, device=self._xpu_device
      )
      self._act_setpoint_act_obj_ids = torch.tensor(
          act_setpoint_act_obj_ids, dtype=torch.long, device=self._xpu_device
      )
      self._has_setpoint_ctrllimited = any(act_setpoint_ctrllimited)
      if self._has_setpoint_ctrllimited:
        self._act_setpoint_ctrllimited = torch.tensor(
            act_setpoint_ctrllimited, dtype=torch.bool, device=self._xpu_device
        )
        # Trigger lazy tiling before CUDA graph capture.
        _ = simulator.model.actuator_ctrlrange
      self._has_setpoint_actlimited = any(act_setpoint_actlimited)
      if self._has_setpoint_actlimited:
        self._act_setpoint_actlimited = torch.tensor(
            act_setpoint_actlimited, dtype=torch.bool, device=self._xpu_device
        )
        # Trigger lazy tiling before CUDA graph capture.
        _ = simulator.model.actuator_actrange
    self._act_zero_ids = (
        torch.tensor(act_zero_ids, dtype=torch.long, device=self._xpu_device)
        if act_zero_ids
        else None
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
  def get_measurements(
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
        if self._bound_model is None:
          assert self._empty_joint_range is not None
          return self._empty_joint_range
        return self._bound_model.range
      elif m == "ctrl":
        return data.ctrl[:, self._ctrl_ids]
      elif m == "home_ctrl":
        return self.home_ctrl
      raise KeyError(m)

    return {f"{self._name}/{m}": get(m) for m in self._measurements}

  # Puppet API - resetter calls these.

  def set_ctrl(self, env_mask: torch.Tensor | None, ctrl: torch.Tensor) -> None:
    """Set actuator control and reset stateful activations for environments."""
    if env_mask is None:
      self._data.ctrl[:, self._ctrl_ids] = ctrl
    else:
      curr = self._data.ctrl[:, self._ctrl_ids]
      self._data.ctrl[:, self._ctrl_ids] = torch.where(
          env_mask.unsqueeze(1), ctrl, curr
      )

    if self._act_setpoint_ids is not None:
      setpoint_ctrl = ctrl[..., self._act_setpoint_local_ctrl_ids].to(
          dtype=self._data.act.dtype
      )
      if self._has_setpoint_ctrllimited:
        ctrlrange = self._model.actuator_ctrlrange[
            :, self._act_setpoint_ctrl_ids
        ]
        clamped = torch.clamp(
            setpoint_ctrl, ctrlrange[..., 0], ctrlrange[..., 1]
        )
        setpoint_ctrl = torch.where(
            self._act_setpoint_ctrllimited, clamped, setpoint_ctrl
        )
      if self._has_setpoint_actlimited:
        actrange = self._model.actuator_actrange[
            :, self._act_setpoint_act_obj_ids
        ]
        clamped = torch.clamp(setpoint_ctrl, actrange[..., 0], actrange[..., 1])
        setpoint_ctrl = torch.where(
            self._act_setpoint_actlimited, clamped, setpoint_ctrl
        )
      if env_mask is None:
        self._data.act[:, self._act_setpoint_ids] = setpoint_ctrl
      else:
        curr_act = self._data.act[:, self._act_setpoint_ids]
        self._data.act[:, self._act_setpoint_ids] = torch.where(
            env_mask.unsqueeze(1), setpoint_ctrl, curr_act
        )

    if self._act_zero_ids is not None:
      if env_mask is None:
        self._data.act[:, self._act_zero_ids] = 0.0
      else:
        curr_zero = self._data.act[:, self._act_zero_ids]
        self._data.act[:, self._act_zero_ids] = torch.where(
            env_mask.unsqueeze(1), 0.0, curr_zero
        )
