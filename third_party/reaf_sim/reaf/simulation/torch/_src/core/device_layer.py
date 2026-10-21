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
"""Simulation device layer."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import itertools
import math
from typing import Any
from typing import Optional

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.simulation.torch._src.core import clock as clock_module
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import profile
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import substep_commands_processor as substeps_cmd_module
from reaf.simulation.torch._src.core import substep_measurements_processor as substep_meas_module
from reaf.simulation.torch._src.core import types
import torch
import tree


class DeviceLayer:
  """Simulation device layer.

  Coordinates:
    1. The simulator (MjSpec -> Model & Data, physics stepping).
    2. All Devices.
    3. All substep command and measurement processors.
    4. The substep loop.
  """

  def __init__(
      self,
      *,
      simulator: simulator_module.Simulator,
      devices: Sequence[device_module.Device] = (),
      control_timestep: float,
      substep_commands_processors: Sequence[
          substeps_cmd_module.SubstepCommandsProcessor
      ] = (),
      substep_measurements_processors: Sequence[
          substep_meas_module.SubstepMeasurementsProcessor
      ] = (),
  ):
    self._simulator = simulator
    self._nworld = simulator.nworld
    self._xpu_device = simulator.xpu_device
    self._devices = devices
    self._substep_commands_processors = substep_commands_processors
    self._substep_measurements_processors = substep_measurements_processors

    # Compute the number of substeps for the simulation.
    if len(self.sim.model.opt.timestep.unique()) > 1:
      raise AssertionError(
          "Only models with a unique opt.timestep are supported."
      )

    # NOTE: Avoid rounding errors when calculating the number of substeps.
    simulation_timestep_seconds = self.sim.model.opt.timestep[0].item()
    timestep_ratio = control_timestep / simulation_timestep_seconds
    if not math.isclose(round(timestep_ratio), timestep_ratio, rel_tol=1e-3):
      raise ValueError(
          f"Control timestep ({control_timestep}) must be a multiple of"
          f" the simulation timestep ({simulation_timestep_seconds})."
      )
    self._n_substeps = round(timestep_ratio)
    self._control_timestep = self._n_substeps * simulation_timestep_seconds

    # Initialize the simulation clock and inject it into the simulator.
    self._clock = clock_module.Clock(
        sim_dt=simulation_timestep_seconds,
        control_dt=self._control_timestep,
        n_substeps=self._n_substeps,
        nworld=self._nworld,
        xpu_device=self._xpu_device,
    )
    self.sim.set_clock(self._clock)

    # Initialize devices
    for dev in self._devices:
      dev.initialize(self.sim)

    # We make a set of all processors so that these objects only get
    # their resets called once. This is important for e.g. when having a single
    # object that derives from both processors.
    self._resettable_processors = []
    unique_ids = set()
    for processor in itertools.chain(
        self._substep_commands_processors,
        self._substep_measurements_processors,
    ):
      processor.initialize(self.sim)
      processor_id = id(processor)
      if processor_id not in unique_ids:
        unique_ids.add(processor_id)
        self._resettable_processors.append(processor)

    # Build specs by aggregating from devices and processors.
    self._build_specs()

  def _build_specs(self) -> None:
    # Build measurements spec
    measurements_spec = {}
    for dev in self._devices:
      substep_spec = dev.substep_measurements_spec()

      # Expand spec to nworld
      original_spec = {**substep_spec}
      updated_spec = {}
      for key, spec in original_spec.items():
        updated_spec[key] = spec.replace(shape=(self._nworld,) + spec.shape)
      measurements_spec.update(updated_spec)

    for processor in self._substep_measurements_processors:
      available_keys = measurements_spec.keys()
      consumed_keys = processor.consumed_substep_measurements_keys()
      if not consumed_keys.issubset(available_keys):
        raise ValueError(
            f"Processor {processor.name} could not find keys: {consumed_keys}."
        )
      for key in consumed_keys:
        del measurements_spec[key]
      updated_spec = tree.map_structure(
          lambda spec: spec.replace(shape=(self._nworld,) + spec.shape),
          processor.produced_substep_measurements_spec(),
      )
      measurements_spec.update(updated_spec)

    self._measurements_spec = measurements_spec
    self._measurements = {
        key: torch.empty(
            spec.shape,
            dtype=types.NUMPY_TO_TORCH_DTYPE[spec.dtype],
            device=self._xpu_device,
        )
        for key, spec in measurements_spec.items()
    }

    # Build commands spec
    commands_spec = {}
    for dev in self._devices:
      original_spec = dev.substep_commands_spec()
      updated_spec = tree.map_structure(
          lambda spec: spec.replace(shape=(self._nworld,) + spec.shape),
          original_spec,
      )
      commands_spec.update(updated_spec)

    for processor in reversed(self._substep_commands_processors):
      produced_keys = processor.produced_substep_commands_keys()
      consumed_keys = commands_spec.keys()
      if not produced_keys.issubset(consumed_keys):
        raise ValueError(
            f"Processor {processor.name} produced unused keys: {produced_keys}."
        )
      for key in produced_keys:
        del commands_spec[key]
      updated_spec = tree.map_structure(
          lambda spec: spec.replace(shape=(self._nworld,) + spec.shape),
          processor.consumed_substep_commands_spec(),
      )
      commands_spec.update(updated_spec)

    self._commands_spec = commands_spec

  # DeviceLayer Protocol properties
  @property
  def xpu_device(self) -> str:
    return self._xpu_device

  @property
  def nworld(self) -> int:
    return self._nworld

  @property
  def devices(self) -> Sequence[device_module.Device]:
    return self._devices

  @property
  def sim(self) -> simulator_module.Simulator:
    return self._simulator

  def forward(self) -> None:
    self._simulator.forward()

  def commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return self._commands_spec

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._measurements_spec

  def begin_stepping(
      self, env_mask: torch.Tensor | None = None
  ) -> Mapping[str, torch.Tensor]:
    del env_mask
    # Collect initial measurements
    self._update_measurements()

    return self._measurements

  def end_stepping(self, env_mask: torch.Tensor | None = None) -> None:
    pass

  @profile.trace(color=profile.Section.PHYSICS)
  def step(
      self,
      commands: Mapping[str, torch.Tensor],
  ) -> Mapping[str, torch.Tensor]:
    # Substep loop
    for _ in range(self._n_substeps):
      self._substep(commands)

    self._clock.tick_step()
    profile.frame_mark()
    return self._measurements

  @profile.trace(color=profile.Section.OBSERVATIONS)
  def update_measurements(self) -> None:
    self._update_measurements()

  def get_measurements(self) -> Mapping[str, torch.Tensor]:
    return self._measurements

  @property
  def control_timestep(self) -> float:
    return self._control_timestep

  @property
  def clock(self) -> clock_module.Clock:
    return self._clock

  def has_device(self, name: str) -> bool:
    return any(dev.name == name for dev in self._devices)

  def get_device(self, name: str) -> device_module.Device:
    for dev in self._devices:
      if dev.name == name:
        return dev
    raise ValueError(f"Device with name '{name}' not found.")

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    self.sim.reset(env_mask)
    for dev in self._devices:
      dev.reset(self.sim, env_mask)
    for processor in self._resettable_processors:
      processor.reset(env_mask)
    self._measurements.clear()

  @profile.trace(color=profile.Section.PHYSICS)
  def _substep(
      self, commands: Optional[Mapping[str, torch.Tensor]] = None
  ) -> None:
    self._propagate_commands(commands)
    self.sim.step()
    self._update_measurements()

  @profile.trace(color=profile.Section.COMMANDS)
  def _propagate_commands(
      self, commands: Optional[Mapping[str, torch.Tensor]] = None
  ) -> None:
    current_substep_commands: dict[str, Any] = (
        dict(commands) if commands is not None else {}
    )

    for processor in self._substep_commands_processors:
      processor.process_substep_commands(self.sim, current_substep_commands)

    for dev in self._devices:
      dev.set_substep_commands(self.sim, current_substep_commands)

  @profile.trace(color=profile.Section.OBSERVATIONS)
  def _update_measurements(self) -> None:
    measurements: dict[str, Any] = {}
    for dev in self._devices:
      produced_substep_measurements = dev.get_substep_measurements(self.sim)
      measurements.update(produced_substep_measurements)

    for processor in self._substep_measurements_processors:
      processor.process_substep_measurements(self.sim, measurements)

    self._measurements.update(measurements)
