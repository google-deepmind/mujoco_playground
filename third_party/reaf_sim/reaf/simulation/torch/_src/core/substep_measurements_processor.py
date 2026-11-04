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

  def on_substep(
      self,
      simulator: simulator_module.Simulator,
      substep_measurements: Mapping[str, torch.Tensor],
  ) -> None:
    """Advances per-substep state after every physics substep.

    E.g. pushes the substep reading into a delay buffer.

    Args:
      simulator: The simulator, after the physics substep.
      substep_measurements: The measurements of the substep, as processed by the
        preceding processors. They may be views of simulator or device state, so
        copy them into processor-owned buffers to keep them (e.g. in a history),
        and do not modify them in place.
    """
    del simulator, substep_measurements

  def on_begin_stepping(
      self,
      simulator: simulator_module.Simulator,
      measurements: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Records the reset state of the `env_mask` worlds.

    Runs after `reset()`, once the reset state is final and before it is
    processed, e.g. to fill delay buffers with the reset reading. Worlds outside
    `env_mask` must keep their state.

    Args:
      simulator: The simulator, holding the final reset state.
      measurements: The measurements of the reset state, as processed by the
        preceding processors. They may be views of simulator or device state, so
        copy them into processor-owned buffers to keep them (e.g. in a history),
        and do not modify them in place.
      env_mask: Boolean mask of the reset worlds. None: all worlds.
    """
    del simulator, measurements, env_mask

  def process_measurements(
      self,
      simulator: simulator_module.Simulator,
      measurements: MutableMapping[str, torch.Tensor],
  ) -> None:
    """Transforms measurements in place.

    Should not advance any state: `DeviceLayer` may call this method more than
    once per substep (e.g. after resets or in `update_measurements()`).

    Args:
      simulator: The simulator.
      measurements: Mutable dictionary of measurements. Processors can insert,
        update, or remove measurement keys in place, but must not modify the
        tensors of measurements in place (they may be views of simulator or
        device state); they may write into tensors they own (e.g. `out=`).
    """
    del simulator, measurements

  def process_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      substep_measurements: MutableMapping[str, torch.Tensor],
  ) -> None:
    """Processes substep measurements by modifying the dictionary in place.

    Deprecated: implement `on_substep()` and `process_measurements()` instead.
    `DeviceLayer` calls overrides after every physics substep and after resets,
    and emits a `DeprecationWarning` for them.

    Args:
      simulator: the simulator bound to the DeviceLayer.
      substep_measurements: mutable dictionary of substep measurements.
        Processors can insert, update, or remove measurement keys in place.
    """
    # TODO: b/570300769 - Remove once downstream processors are migrated.
    self.on_substep(simulator, substep_measurements)
    self.process_measurements(simulator, substep_measurements)
