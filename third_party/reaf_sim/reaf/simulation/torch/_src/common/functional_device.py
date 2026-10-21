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
"""Entity that wraps callable functions for measurements."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch


class MeasurementDevice(device.Device):
  """Entity that wraps callable functions for measurements."""

  def __init__(
      self,
      *,
      name: str = "",
      measurements: Mapping[str, "types.Term"],
  ):
    """Initialize MeasurementDevice.

    Args:
      name: Entity name.
      measurements: Dict mapping measurement names to Term objects.
    """
    self._name = name
    self._measurements = measurements
    self._specs = {
        f"{name}/{k}": specs.Array(shape=term.output_shape, dtype=np.float32)
        for k, term in measurements.items()
    }

  @property
  def name(self) -> str:
    return self._name

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Forward initialize to terms."""
    for term in self._measurements.values():
      if hasattr(term.func, "initialize"):
        term.func.initialize(
            simulator.mj_spec,
            simulator.mj_model,
            simulator.model,
            simulator.data,
        )

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._specs

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Compute measurements by calling the provided functions."""
    return {
        f"{self._name}/{name}": term.func(simulator.model, simulator.data)
        for name, term in self._measurements.items()
    }


def joint(joint_name: str, extractor: Callable[[Any], torch.Tensor]):
  """Returns a measurement function for a joint feature."""

  def func(_model: types.Model, _data: types.Data) -> torch.Tensor:
    return extractor(func.bound_joint)  # pyrefly: ignore[missing-attribute]

  func.bound_joint = None  # pyrefly: ignore[missing-attribute]

  def initialize(
      mj_spec: mujoco.MjSpec,
      mj_model: mujoco.MjModel,
      model: types.Model,
      data: types.Data,
  ):
    del mj_model
    func.bound_joint = data.bind(model, [mj_spec.joint(joint_name)])  # pyrefly: ignore[missing-attribute]

  func.initialize = initialize  # pyrefly: ignore[missing-attribute]
  return func


def joint_pos(joint_name: str):
  """Returns a measurement function for a joint's position."""
  return joint(joint_name, lambda b: b.qpos)


def joint_vel(joint_name: str):
  """Returns a measurement function for a joint's velocity."""
  return joint(joint_name, lambda b: b.qvel)
