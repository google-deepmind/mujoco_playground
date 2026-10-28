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
from typing import Literal
from typing import override

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch

# Type aliases - strings only for serialization
ControlMode = Literal["position", "velocity", "motor"]
TransmissionType = Literal["joint", "tendon", "site"]

_TRNTYPE_MAP = {
    "joint": mujoco.mjtTrn.mjTRN_JOINT,
    "tendon": mujoco.mjtTrn.mjTRN_TENDON,
    "site": mujoco.mjtTrn.mjTRN_SITE,
}


class ActuatorDevice(device.Device):
  """Actuate joints, tendons, or sites with position, velocity, or motor control."""

  @override
  def configure_spec(  # pyrefly: ignore[bad-override]
      self,
      spec: mujoco.MjSpec,
      *,
      mode: ControlMode = "position",
      stiffness: dict[str, float] | float = 100.0,
      damping: dict[str, float] | float = 10.0,
      forcerange: dict[str, float] | float | None = 30.0,
      ctrlrange: (
          dict[str, tuple[float, float]] | tuple[float, float] | None
      ) = None,
      armature: dict[str, float] | float | None = None,
      frictionloss: dict[str, float] | float | None = None,
      gear: dict[str, float | list[float]] | float | list[float] = 1.0,
      use_existing: bool = False,
  ) -> mujoco.MjSpec:
    """Configure MjSpec with actuators (build time).

    Args:
      spec: The MjSpec to modify.
      mode: Control mode - "position", "velocity", "motor", or mujoco.mjtGain.*.
      stiffness: PD stiffness (kp). Only used for position mode.
      damping: PD damping (kd). Used for position and velocity modes.
      forcerange: Max actuator force. None = no limit.
      ctrlrange: Ctrl limits as (min, max) per target or uniform. When provided,
        sets ctrllimited=True on the actuator. None = no limit.
      armature: Reflected rotor inertia. None = don't modify (joint/tendon
        only).
      frictionloss: Dry friction. None = don't modify (joint/tendon only).
      gear: Actuator gear ratio. For sites, can be 6-element list [trans_x,
        trans_y, trans_z, rot_x, rot_y, rot_z]. For joints/tendons, scalar or
        gear[0].
      use_existing: If True, skip reconfiguration of actuators that already
        exist in the MJCF. This preserves the original gainprm, biasprm, gear,
        and other parameters defined in the XML.
    """
    spec = super().configure_spec(spec)
    targets = self._target_names
    transmission = self._transmission
    trntype = _TRNTYPE_MAP[transmission]

    for target_name in targets:
      # Find or create actuator
      act = spec.actuator(target_name)
      # Fallback manual search because of MuJoCo MjSpec.actuator bug.
      # TODO(btaba): Fix the bug upstream and remove this fallback.
      # see the test case test_configure_spec_existing_velocity_actuator in
      # third_party/py/reaf/simulation/torch/common/actuator_entity_test.py
      if act is None:
        for a in spec.actuators:
          if a.name == target_name:
            act = a
            break

      if act is None:
        act = spec.add_actuator(name=target_name, target=target_name)
      elif use_existing:
        continue  # Keep actuator as configured in MJCF
      # TODO(team): Use act.set_to_position, act.set_to_velocity, and
      # act.set_to_motor so reconfiguring existing actuators resets all
      # parameters cleanly instead of raising NotImplementedError.
      elif (
          act.gaintype != mujoco.mjtGain.mjGAIN_FIXED
          or act.dyntype != mujoco.mjtDyn.mjDYN_NONE
      ):
        raise NotImplementedError(
            f"Actuator {target_name!r} has non-standard gain"
            f" ({mujoco.mjtGain(act.gaintype).name}) or dynamics"
            f" ({mujoco.mjtDyn(act.dyntype).name}); pass use_existing=True to"
            " preserve its MJCF configuration."
        )

      # Get per-target or uniform values
      kp = stiffness[target_name] if isinstance(stiffness, dict) else stiffness
      kd = damping[target_name] if isinstance(damping, dict) else damping
      gr = gear[target_name] if isinstance(gear, dict) else gear

      if trntype == mujoco.mjtTrn.mjTRN_JOINT:
        target_obj = spec.joint(target_name)
        if target_obj is None:
          raise ValueError(f"Joint not found: {target_name}")
      elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
        target_obj = spec.tendon(target_name)
        if target_obj is None:
          raise ValueError(f"Tendon not found: {target_name}")
      elif trntype == mujoco.mjtTrn.mjTRN_SITE:
        target_obj = spec.site(target_name)
        if target_obj is None:
          raise ValueError(f"Site not found: {target_name}")
        target_obj = None  # Set to None so we skip armature/frictionloss below
      else:
        raise ValueError(f"Unknown transmission type: {transmission}")

      if target_obj is not None:
        if armature is not None:
          arm = (
              armature[target_name] if isinstance(armature, dict) else armature
          )
          target_obj.armature = arm
        if frictionloss is not None:
          fl = (
              frictionloss[target_name]
              if isinstance(frictionloss, dict)
              else frictionloss
          )
          target_obj.frictionloss = fl

      # Set transmission type and gear
      act.trntype = trntype
      if isinstance(gr, list):
        # Full 6-element gear for sites
        for i, g in enumerate(gr[:6]):
          act.gear[i] = g
      else:
        # Scalar gear for joint/tendon
        act.gear[0] = gr

      act.dyntype = mujoco.mjtDyn.mjDYN_NONE
      act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      if mode == "position":
        # <position> actuator
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kd
        # no ctrl limits (position target can exceed joint range)
        act.ctrllimited = False
      elif mode == "velocity":
        # <velocity> actuator - ctrl=velocity target
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kd
        act.biasprm[2] = -kd
        # TODO: either pipe through ctrlrange explicitly or use inheritrange
        act.ctrllimited = False
      elif mode == "motor":
        # <motor> actuator
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.ctrllimited = True
      else:
        raise ValueError(f"Unknown mode: {mode}")

      # Set force limits
      if forcerange is not None:
        fr = (
            forcerange[target_name]
            if isinstance(forcerange, dict)
            else forcerange
        )
        act.forcelimited = True
        act.forcerange = [-fr, fr]
        if mode == "motor":
          act.ctrlrange = [-fr, fr]  # Motor: ctrl=force
      else:
        act.forcelimited = False
        if mode == "motor":
          act.ctrllimited = False

      if ctrlrange is not None:
        cr = (
            ctrlrange[target_name] if isinstance(ctrlrange, dict) else ctrlrange
        )
        act.ctrllimited = True
        act.ctrlrange = list(cr)

    return spec

  def __init__(
      self,
      name: str = "",
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      *,
      targets: list[str],
      transmission: TransmissionType = "joint",
      measurements: list[str] | None = None,
      home_ctrl: torch.Tensor | list[float] | None = None,
  ):
    """Initialize runtime control logic.

    Args:
      name: Entity name for lookup.
      nworld: Number of parallel worlds.
      xpu_device: Torch device.
      targets: List of target names (must match configure_spec).
      transmission: Transmission type - must match configure_spec.
      measurements: Which measurements to expose. Options: "joint_pos",
        "joint_vel", "joint_range", "actuator_force", "tendon_length",
        "tendon_velocity", "ctrl". Default: [].
      home_ctrl: Default home control values.
    """
    self._name = name
    self._nworld = nworld
    self._xpu_device = xpu_device
    self._target_names = targets
    self._transmission = transmission
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
    self._actuator_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._out_ids: torch.Tensor | None = None
    self._tendon_ids: torch.Tensor | None = None
    self._joint_range: torch.Tensor | None = None  # Static (njoint, 2)
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

    joint_ids = []
    joint_qpos_ids = []
    joint_qvel_ids = []
    actuator_ids = []
    ctrl_ids = []
    out_ids = []
    tendon_ids = []
    self._joints = []

    for target_name in self._target_names:
      act = mj_model.actuator(target_name)
      actuator_ids.append(act.id)
      # Map actuator control and output spans (e.g. multi-input <dcmotor>).
      ctrl_adr = int(mj_model.actuator_ctrladr[act.id])
      ctrl_ids.extend(
          range(ctrl_adr, ctrl_adr + int(mj_model.actuator_ctrlnum[act.id]))
      )
      out_adr = int(mj_model.actuator_outadr[act.id])
      out_ids.extend(
          range(out_adr, out_adr + int(mj_model.actuator_outnum[act.id]))
      )

      if self._transmission == "joint":
        joint_id = mj_model.actuator_trnid[act.id, 0]
        joint_ids.append(joint_id)
        joint_qpos_ids.append(mj_model.jnt_qposadr[joint_id])
        joint_qvel_ids.append(mj_model.jnt_dofadr[joint_id])
        joint_name = mj_model.joint(joint_id).name
        self._joints.append(mj_spec.joint(joint_name))
      elif self._transmission == "tendon":
        ten = mj_model.tendon(target_name)
        tendon_ids.append(ten.id)

    # Store as tensors on the entity's device
    self._joint_qpos_ids = torch.tensor(
        joint_qpos_ids, dtype=torch.long, device=self._xpu_device
    )
    self._joint_qvel_ids = torch.tensor(
        joint_qvel_ids, dtype=torch.long, device=self._xpu_device
    )
    self._actuator_ids = torch.tensor(
        actuator_ids, dtype=torch.long, device=self._xpu_device
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

    # Store static joint range if transmission is joint
    if self._transmission == "joint" and "joint_range" in self._measurements:
      jnt_range = mj_model.jnt_range[joint_ids]
      self._joint_range = torch.tensor(
          jnt_range, dtype=torch.float32, device=self._xpu_device
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
    assert self._ctrl_ids is not None and self._out_ids is not None, (
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
      else:
        shape = (self._num_targets,)
        if m == "joint_range":
          shape += (2,)
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
