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

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.builders.flat_scene import config
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
      measurements: Mapping[str, types.Term | Callable[..., torch.Tensor]],
  ):
    """Initialize MeasurementDevice.

    Args:
      name: Entity name.
      measurements: Dict mapping measurement names to Term objects or callables.
    """
    self._name = name
    self._measurements: dict[str, types.Term] = {
        k: config.Term(v) if isinstance(v, Callable) else v
        for k, v in measurements.items()
    }
    self._specs: dict[str, specs.Array] = {
        f"{name}/{k}": specs.Array(shape=term.output_shape, dtype=np.float32)
        for k, term in self._measurements.items()
        if term.output_shape is not None
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
    """Move tensor params to simulator device and infer missing output shapes."""
    for k, term in self._measurements.items():
      term.params = {
          pk: (
              pv.to(simulator.xpu_device)
              if isinstance(pv, torch.Tensor)
              else pv
          )
          for pk, pv in term.params.items()
      }
      if term.output_shape is None:
        out = term.func(simulator.model, simulator.data, **term.params)
        self._specs[f"{self._name}/{k}"] = specs.Array(
            shape=tuple(out.shape[1:]), dtype=np.float32
        )

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._specs

  def get_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Compute measurements by calling the provided functions."""
    return {
        f"{self._name}/{name}": term.func(
            simulator.model, simulator.data, **term.params
        )
        for name, term in self._measurements.items()
    }
