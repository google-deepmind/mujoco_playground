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
"""Represents the simulation's internal clock."""

from collections.abc import Mapping

from dm_env import specs
import numpy as np
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class ClockDevice(device_module.Device):
  """Provides the current simulation time at each step."""

  def __init__(
      self,
      name: str = "",
      time_output_name: str = "time",
  ):
    """Initializes the ClockDevice.

    Args:
      name: The name of this entity.
      time_output_name: The desired name for the time measurement output.
    """
    self._name = name
    self._output_name = time_output_name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Initialize entity with compiled model. ClockDevice has no state."""
    del simulator

  @property
  def name(self) -> str:
    """Returns the name of this entity."""
    return self._name

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """This entity has no state to reset."""
    del simulator, env_mask

  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    """This entity does not accept any commands."""
    return {}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    """This entity does not process any commands."""
    del simulator, consumed_substep_commands

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Defines the structure of the time measurement.

    Returns:
      A dictionary specifying a single scalar float32 for the time.
    """
    return {
        self._output_name: specs.Array(
            shape=(),
            dtype=np.float32,
            name=self._output_name,
        )
    }

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Returns the current simulation time."""
    return {self._output_name: simulator.data.time}
