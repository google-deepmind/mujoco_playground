# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Free-floating and mocap entities for reaf-sim."""

from collections.abc import Mapping
from typing import Union

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import tensors
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


class FreeDevice(device.Device):
  """Physics-driven 6-DOF free-floating body (cube, ball, etc)."""

  def __init__(
      self,
      *,
      name: str = "",
      body_name: str,
      joint_name: str,
      measurements: list[str] | None = None,
      reference_site_name: str | None = None,
  ):
    """Initialize FreeDevice.

    Args:
      name: Entity name for lookup.
      body_name: Name of the body in the MjSpec.
      joint_name: Name of the freejoint in the MjSpec.
      measurements: Which measurements to expose. Options: "xpos", "xquat",
        "joint_vel". Default: ["xpos", "xquat"].
      reference_site_name: Optional name of the site to measure relative to.
    """
    self._name = name
    self._body_name = body_name
    self._joint_name = joint_name
    self._measurements = measurements or ["xpos", "xquat"]
    self._reference_site_name = reference_site_name

    # Set in initialize().
    self._simulator: simulator_module.Simulator | None = None
    self._data: types.Data | None = None
    self._body_id: int | None = None
    self._joint_id: int | None = None
    self._qpos_adr: int | None = None
    self._qvel_adr: int | None = None
    self._sensor_pos_adr: int | None = None
    self._sensor_pos_dim: int | None = None
    self._sensor_quat_adr: int | None = None
    self._sensor_quat_dim: int | None = None
    self._xpu_device: torch.device | None = None
    self._nworld: int | None = None

  @property
  def name(self) -> str:
    return self._name

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
  ) -> mujoco.MjSpec:
    spec = super().configure_spec(spec)
    if self._reference_site_name is not None:
      self._sensor_pos_name = f"{self._name}_framepos"
      spec.add_sensor(
          name=self._sensor_pos_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          objname=self._body_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
      self._sensor_quat_name = f"{self._name}_framequat"
      spec.add_sensor(
          name=self._sensor_quat_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          objname=self._body_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
    return spec

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Resolve body/joint IDs from compiled model."""
    self._simulator = simulator
    data = simulator.data
    mj_model = simulator.mj_model
    self._data = data
    self._body_id = mj_model.body(self._body_name).id
    self._joint_id = mj_model.joint(self._joint_name).id
    self._qpos_adr = mj_model.jnt_qposadr[self._joint_id]
    self._qvel_adr = mj_model.jnt_dofadr[self._joint_id]

    if self._reference_site_name is not None:
      sensor_pos = mj_model.sensor(self._sensor_pos_name)
      self._sensor_pos_adr = int(sensor_pos.adr[0])
      self._sensor_pos_dim = int(sensor_pos.dim[0])

      sensor_quat = mj_model.sensor(self._sensor_quat_name)
      self._sensor_quat_adr = int(sensor_quat.adr[0])
      self._sensor_quat_dim = int(sensor_quat.dim[0])

    self._xpu_device = data.qpos.device
    self._nworld = data.qpos.shape[0]

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """No-op. State is controlled by resetter via set_pos/set_quat/set_vel."""
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Spec for configured substep measurements."""
    shapes = {"xpos": (3,), "xquat": (4,), "joint_vel": (6,)}
    return {
        f"{self._name}/{m}": specs.Array(shape=shapes[m], dtype=np.float32)
        for m in self._measurements
    }

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Return configured substep measurements."""
    data = simulator.data

    def get(m: str) -> torch.Tensor:
      if m == "xpos":
        if self._reference_site_name is not None:
          assert self._sensor_pos_adr is not None
          assert self._sensor_pos_dim is not None
          adr = self._sensor_pos_adr
          dim = self._sensor_pos_dim
          return data.sensordata[:, adr : adr + dim]
        return data.xpos[:, self._body_id]
      elif m == "xquat":
        if self._reference_site_name is not None:
          assert self._sensor_quat_adr is not None
          assert self._sensor_quat_dim is not None
          adr = self._sensor_quat_adr
          dim = self._sensor_quat_dim
          return data.sensordata[:, adr : adr + dim]
        return data.xquat[:, self._body_id]
      elif m == "joint_vel":
        assert self._qvel_adr is not None
        return data.qvel[:, self._qvel_adr : self._qvel_adr + 6]
      raise KeyError(m)

    return {f"{self._name}/{m}": get(m) for m in self._measurements}

  # Public setter methods.

  def set_pos(self, env_mask: torch.Tensor | None, pos: torch.Tensor) -> None:
    """Set position for specified environments."""
    assert self._data is not None and self._qpos_adr is not None
    s = slice(self._qpos_adr, self._qpos_adr + 3)
    tensors.masked_fill(self._data.qpos[:, s], env_mask, pos)

  def set_quat(self, env_mask: torch.Tensor | None, quat: torch.Tensor) -> None:
    """Set quaternion for specified environments."""
    assert self._data is not None and self._qpos_adr is not None
    s = slice(self._qpos_adr + 3, self._qpos_adr + 7)
    tensors.masked_fill(self._data.qpos[:, s], env_mask, quat)

  def set_vel(self, env_mask: torch.Tensor | None, vel: torch.Tensor) -> None:
    """Set 6-DOF velocity (linear + angular) for specified environments."""
    assert self._data is not None and self._qvel_adr is not None
    s = slice(self._qvel_adr, self._qvel_adr + 6)
    tensors.masked_fill(self._data.qvel[:, s], env_mask, vel)


class MocapDevice(device.Device):
  """Fixed-base kinematic marker controlled via mocap (target position).

  A puppet entity: exposes set_pos/set_quat for external control.
  The resetter owns sampling logic and calls these methods.
  """

  def __init__(
      self,
      *,
      name: str = "",
      body_name: str,
      reference_site_name: str | None = None,
  ):
    """Initialize MocapDevice.

    Args:
      name: Entity name for lookup.
      body_name: Name of the mocap body in the MjSpec.
      reference_site_name: Optional name of the site to measure relative to.
    """
    self._name = name
    self._body_name = body_name
    self._reference_site_name = reference_site_name

    # Set in initialize().
    self._data: types.Data | None = None
    self._mocap_id: int | None = None
    self._xpu_device: torch.device | None = None
    self._nworld: int | None = None
    self._sensor_pos_adr: int | None = None
    self._sensor_pos_dim: int | None = None
    self._sensor_quat_adr: int | None = None
    self._sensor_quat_dim: int | None = None

  def configure_spec(  # pyrefly: ignore[bad-override]
      self,
      spec: mujoco.MjSpec,
      *,
      geom_group: int | None = None,
  ) -> mujoco.MjSpec:
    """Configures MjSpec sensors and optional geom group for the mocap body."""
    spec = super().configure_spec(spec)
    if geom_group is not None:
      body = spec.body(self._body_name)
      if body is not None:
        for g in body.geoms:
          g.group = geom_group
    if self._reference_site_name is not None:
      self._sensor_pos_name = f"{self._name}_mocap_framepos"
      spec.add_sensor(
          name=self._sensor_pos_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          objname=self._body_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
      self._sensor_quat_name = f"{self._name}_mocap_framequat"
      spec.add_sensor(
          name=self._sensor_quat_name,
          type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          objname=self._body_name,
          reftype=mujoco.mjtObj.mjOBJ_SITE,
          refname=self._reference_site_name,
      )
    return spec

  @property
  def name(self) -> str:
    return self._name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Resolve mocap ID from compiled model."""
    self._simulator = simulator
    self._data = simulator.data
    body_id = simulator.mj_model.body(self._body_name).id
    self._mocap_id = simulator.mj_model.body_mocapid[body_id]
    if self._mocap_id < 0:
      raise ValueError(f"Body '{self._body_name}' is not a mocap body.")

    self._xpu_device = simulator.data.mocap_pos.device
    self._nworld = simulator.data.mocap_pos.shape[0]

    if self._reference_site_name is not None:
      sensor_pos = simulator.mj_model.sensor(self._sensor_pos_name)
      self._sensor_pos_adr = int(sensor_pos.adr[0])
      self._sensor_pos_dim = int(sensor_pos.dim[0])

      sensor_quat = simulator.mj_model.sensor(self._sensor_quat_name)
      self._sensor_quat_adr = int(sensor_quat.adr[0])
      self._sensor_quat_dim = int(sensor_quat.dim[0])

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """No-op. Mocap position is controlled by resetter."""
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Spec for mocap_pos and mocap_quat measurements."""
    return {
        f"{self._name}/mocap_pos": specs.Array(shape=(3,), dtype=np.float32),
        f"{self._name}/mocap_quat": specs.Array(shape=(4,), dtype=np.float32),
    }

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Return mocap position and quaternion."""
    data = simulator.data
    measurements = {}
    if self._reference_site_name is not None:
      assert (
          self._sensor_pos_adr is not None and self._sensor_pos_dim is not None
      )
      adr = self._sensor_pos_adr
      dim = self._sensor_pos_dim
      measurements[f"{self._name}/mocap_pos"] = data.sensordata[
          :, adr : adr + dim
      ]

      assert (
          self._sensor_quat_adr is not None
          and self._sensor_quat_dim is not None
      )
      adr = self._sensor_quat_adr
      dim = self._sensor_quat_dim
      measurements[f"{self._name}/mocap_quat"] = data.sensordata[
          :, adr : adr + dim
      ]
    else:
      measurements[f"{self._name}/mocap_pos"] = data.mocap_pos[
          :, self._mocap_id
      ]
      measurements[f"{self._name}/mocap_quat"] = data.mocap_quat[
          :, self._mocap_id
      ]

    return measurements

  # Public setter methods.

  def set_pos(self, env_mask: torch.Tensor | None, pos: torch.Tensor) -> None:
    """Set mocap position for specified environments."""
    assert self._data is not None
    tensors.masked_fill(self._data.mocap_pos[:, self._mocap_id], env_mask, pos)

  def set_quat(self, env_mask: torch.Tensor | None, quat: torch.Tensor) -> None:
    """Set mocap quaternion for specified environments."""
    assert self._data is not None
    tensors.masked_fill(
        self._data.mocap_quat[:, self._mocap_id], env_mask, quat
    )
