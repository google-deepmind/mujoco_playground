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
"""Basic component to interface with the simulation."""

import abc
from collections.abc import Mapping

from dm_env import specs
import mujoco
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch


class Device(abc.ABC):
  """A REAF Device representing a simulation participant."""

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
  ) -> mujoco.MjSpec:
    """Validate/configure the spec for this device."""
    return spec

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Initializes the device with the compiled simulator."""
    del simulator

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Returns the instance name of this device."""

  @abc.abstractmethod
  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Resets the device state in the simulation."""

  # ---------------------------------------------------------------------------
  # Substep interface
  # ---------------------------------------------------------------------------

  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    """Returns the substep commands spec for this device."""
    return {}

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    """Returns the substep measurements spec for this device."""
    return {}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    """Sets the substep commands for this device."""
    del simulator, consumed_substep_commands

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Returns the substep measurements provided by this device."""
    del simulator
    return {}
