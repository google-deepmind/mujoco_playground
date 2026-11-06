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

  def on_substep(self, simulator: simulator_module.Simulator) -> None:
    """Advances per-substep state after every physics substep.

    E.g. pushes the current reading into a substep history buffer.

    Args:
      simulator: The simulator, after the physics substep.
    """
    del simulator

  def on_step(self, simulator: simulator_module.Simulator) -> None:
    """Advances per-control-step state on the final substep.

    Runs on the last physics substep right after `on_substep()` and before
    `get_measurements()` and substep processors, holding the prior step's value
    across earlier substeps (zero-order hold). Does not run after resets:
    `on_begin_stepping()` sets this state for reset worlds.

    Args:
      simulator: The simulator, after the last physics substep.
    """
    del simulator

  def on_begin_stepping(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Records the reset state of the `env_mask` worlds.

    `reset()` runs before the environment resetter, `Simulator.set_const()`,
    and `Simulator.forward()`, so `simulator.data` does not hold the reset
    state yet. This hook runs after them and before the reset state is read,
    e.g. to fill history buffers with the reset reading. Worlds outside
    `env_mask` must keep their state.

    Args:
      simulator: The simulator, holding the final reset state.
      env_mask: Boolean mask of the reset worlds. None: all worlds.
    """
    del simulator, env_mask

  def get_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Returns the measurements for the current state.

    Should not advance any state: `DeviceLayer` can call this method any number
    of times and it should return valid measurements at any point in the step
    lifecycle.

    Args:
      simulator: The simulator.

    Returns:
      The measurements, batched over worlds. They may be views of simulator or
      device state, so callers must not modify them in place, and must copy
      them to keep them across simulator updates.
    """
    del simulator
    return {}

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Deprecated: implement `on_substep()`, `on_step()`, and `get_measurements()`.

    `DeviceLayer` calls overrides after every physics substep and after resets.

    Args:
      simulator: The simulator.

    Returns:
      The measurements, batched over worlds.
    """
    # TODO: b/570300769 - Remove once downstream devices are migrated.
    self.on_substep(simulator)
    return self.get_measurements(simulator)
