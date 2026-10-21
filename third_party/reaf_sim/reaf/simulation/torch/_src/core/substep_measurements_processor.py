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
"""Abstract class for substep measurements manipulation in reaf-sim."""

import abc
from collections.abc import Mapping
from collections.abc import MutableMapping

from dm_env import specs
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class SubstepMeasurementsProcessor(abc.ABC):
  """reaf-sim specific substep measurements processor."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Returns a unique string identifier for this object."""

  @abc.abstractmethod
  def reset(self, env_mask: torch.Tensor | None = None):
    """Resets the internal state of this processor."""

  def initialize(
      self,
      simulator: simulator_module.Simulator,
  ) -> None:
    """Initializes the processor with the compiled model."""
    pass

  @abc.abstractmethod
  def produced_substep_measurements_spec(
      self,
  ) -> Mapping[str, specs.Array]:
    """Spec of the substep measurements produced by this processor."""

  @abc.abstractmethod
  def consumed_substep_measurements_keys(self) -> set[str]:
    """Keys of the substep measurements consumed by this processor."""

  @abc.abstractmethod
  def process_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      substep_measurements: MutableMapping[str, torch.Tensor],
  ) -> None:
    """Processes substep measurements by modifying the dictionary in place.

    Args:
      simulator: the simulator bound to the DeviceLayer.
      substep_measurements: mutable dictionary of substep measurements.
        Processors can insert, update, or remove measurement keys in place.
    """
