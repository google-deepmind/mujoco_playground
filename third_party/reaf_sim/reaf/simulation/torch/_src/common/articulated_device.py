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

"""Entity for articulated robots (kinematic tree state)."""

from collections.abc import Mapping
from typing import override

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


def _get_dim(tensor: torch.Tensor) -> int:
  return tensor.shape[1] if tensor.ndim > 1 else 1


class ArticulatedDevice(device.Device):
  """Manages the kinematic state of a robot subtree."""

  def __init__(
      self,
      name: str = "",
      nworld: int = 1,
      xpu_device: str = "cuda:0",
      measurements: list[str] | None = None,
      home_qpos: torch.Tensor | list[float] | None = None,
      home_qvel: torch.Tensor | list[float] | None = None,
      joint_names: list[str] | None = None,
  ):
    del nworld
    self._name = name
    self._xpu_device = xpu_device
    self._measurements = measurements or []
    self._bound_joints = None
    self._joints = None
    self._joint_names = joint_names

    if home_qpos is not None:
      if not isinstance(home_qpos, torch.Tensor):
        home_qpos = torch.tensor(
            home_qpos, dtype=torch.float32, device=xpu_device
        )
      self._home_qpos = home_qpos.to(xpu_device)
    else:
      self._home_qpos = None

    if home_qvel is not None:
      if not isinstance(home_qvel, torch.Tensor):
        home_qvel = torch.tensor(
            home_qvel, dtype=torch.float32, device=xpu_device
        )
      self._home_qvel = home_qvel.to(xpu_device)
    else:
      self._home_qvel = None

  @property
  @override
  def name(self) -> str:
    return self._name

  @property
  def home_qpos(self) -> torch.Tensor:
    """Home position for joints. Shape: (nq,)."""
    assert self._home_qpos is not None
    return self._home_qpos

  @property
  def home_qvel(self) -> torch.Tensor:
    """Home velocity for joints. Shape: (nv,)."""
    assert self._home_qvel is not None
    return self._home_qvel

  @override
  def initialize(self, simulator: simulator_module.Simulator) -> None:
    self._simulator = simulator
    self._data = simulator.data
    mj_spec = simulator.mj_spec

    if self._joint_names is not None:
      self._joints = [mj_spec.joint(jn) for jn in self._joint_names]
    else:
      self._joints = [j for j in mj_spec.joints if j.name]
    if not self._joints:
      raise ValueError("No joints found in spec.")

    self._bound_joints = simulator.data.bind(simulator.model, self._joints)

    nq = _get_dim(self._bound_joints.qpos)
    nv = _get_dim(self._bound_joints.qvel)
    if self._home_qpos is None:
      self._home_qpos = torch.zeros(nq, device=self._xpu_device)
    if self._home_qvel is None:
      self._home_qvel = torch.zeros(nv, device=self._xpu_device)

  @override
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  @override
  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    result = {}
    assert self._bound_joints is not None
    qpos_size = _get_dim(self._bound_joints.qpos)
    qvel_size = _get_dim(self._bound_joints.qvel)
    for m in self._measurements:
      if m == "joint_pos":
        result[f"{self._name}/{m}"] = specs.Array(
            shape=(qpos_size,), dtype=np.float32
        )
      elif m == "joint_vel":
        result[f"{self._name}/{m}"] = specs.Array(
            shape=(qvel_size,), dtype=np.float32
        )
      elif m == "home_qpos":
        result[f"{self._name}/{m}"] = specs.Array(
            shape=(qpos_size,), dtype=np.float32
        )
      elif m == "home_qvel":
        result[f"{self._name}/{m}"] = specs.Array(
            shape=(qvel_size,), dtype=np.float32
        )
      elif m == "joint_range":
        result[f"{self._name}/{m}"] = specs.Array(
            shape=(qpos_size, 2), dtype=np.float32
        )
    return result

  @override
  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    result = {}
    assert self._bound_joints is not None
    if "joint_pos" in self._measurements:
      result[f"{self._name}/joint_pos"] = self._bound_joints.qpos
    if "joint_vel" in self._measurements:
      result[f"{self._name}/joint_vel"] = self._bound_joints.qvel
    if "home_qpos" in self._measurements:
      assert self._home_qpos is not None
      result[f"{self._name}/home_qpos"] = self._home_qpos
    if "home_qvel" in self._measurements:
      assert self._home_qvel is not None
      result[f"{self._name}/home_qvel"] = self._home_qvel
    if "joint_range" in self._measurements:
      bound_model = simulator.model.bind(self._joints)
      result[f"{self._name}/joint_range"] = bound_model.range
    return result

  def set_qpos(self, env_mask: torch.Tensor | None, qpos: torch.Tensor) -> None:
    """Set joint positions for specified environments."""
    assert self._bound_joints is not None
    self._bound_joints.set("qpos", qpos, env_mask)

  def set_qvel(self, env_mask: torch.Tensor | None, qvel: torch.Tensor) -> None:
    """Set joint velocities for specified environments."""
    assert self._bound_joints is not None
    self._bound_joints.set("qvel", qvel, env_mask)
