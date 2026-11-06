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
import warnings

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.simulation.torch._src.core import clock as clock_module
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import profile
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import substep_commands_processor as substeps_cmd_module
from reaf.simulation.torch._src.core import substep_measurements_processor as substep_meas_module
import torch
import tree

_DEVICE_MEASUREMENT_METHODS = (
    "on_substep",
    "on_step",
    "on_begin_stepping",
    "get_measurements",
)
_PROCESSOR_MEASUREMENT_METHODS = (
    "on_substep",
    "on_begin_stepping",
    "process_measurements",
)


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
    self._check_legacy_components()

  def _check_legacy_components(self) -> None:
    """Identifies legacy devices and processors that use deprecated methods."""
    # TODO: b/570300769 - Warn about legacy devices once the built-in devices
    # are migrated, and remove the legacy paths once downstream callers are
    # migrated.
    self._legacy_device_ids = {
        id(dev)
        for dev in self._devices
        if _is_legacy(
            dev,
            device_module.Device,
            "get_substep_measurements",
            _DEVICE_MEASUREMENT_METHODS,
        )
    }
    self._legacy_processor_ids = set()
    base = substep_meas_module.SubstepMeasurementsProcessor
    for processor in self._substep_measurements_processors:
      if _is_legacy(
          processor,
          base,
          "process_substep_measurements",
          _PROCESSOR_MEASUREMENT_METHODS,
      ):
        self._legacy_processor_ids.add(id(processor))
      if _overrides(processor, base, "process_substep_measurements"):
        warnings.warn(
            f"{type(processor).__name__} overrides the deprecated"
            " `SubstepMeasurementsProcessor.process_substep_measurements()`."
            " Implement `on_substep()` and `process_measurements()` instead.",
            DeprecationWarning,
            stacklevel=3,
        )

  def _build_specs(self) -> None:
    # Build measurements spec
    measurements_spec = {}
    self._expected_ndim: dict[str, int] = {}
    for dev in self._devices:
      substep_spec = dev.substep_measurements_spec()

      # Expand spec to nworld
      original_spec = {**substep_spec}
      updated_spec = {}
      for key, spec in original_spec.items():
        self._expected_ndim[key] = 1 + len(spec.shape)
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

    for key, spec in measurements_spec.items():
      self._expected_ndim.setdefault(key, len(spec.shape))

    self._measurements_spec = measurements_spec
    self._output_buffers: dict[str, torch.Tensor] = {}
    self._latest_measurements: dict[str, torch.Tensor] = {}

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
    """Records the reset state, then reads, updating only `env_mask` rows."""
    measurements: dict[str, torch.Tensor] = {}
    for dev in self._devices:
      if id(dev) in self._legacy_device_ids:
        measurements.update(dev.get_substep_measurements(self.sim))
      else:
        dev.on_begin_stepping(self.sim, env_mask)
        measurements.update(dev.get_measurements(self.sim))

    for processor in self._substep_measurements_processors:
      if id(processor) in self._legacy_processor_ids:
        processor.process_substep_measurements(self.sim, measurements)
      else:
        processor.on_begin_stepping(self.sim, measurements, env_mask)
        processor.process_measurements(self.sim, measurements)

    self._latest_measurements = measurements
    self._commit_output_buffers(measurements, env_mask=env_mask)
    return self._output_buffers

  def end_stepping(self, env_mask: torch.Tensor | None = None) -> None:
    pass

  @profile.trace(color=profile.Section.PHYSICS)
  def step(
      self,
      commands: Mapping[str, torch.Tensor],
  ) -> Mapping[str, torch.Tensor]:
    """Runs one control step and returns the measurements of all worlds.

    Args:
      commands: The commands for this control step.

    Returns:
      The measurements of all worlds. The same tensors are returned and
      overwritten in place by every `step()`, `begin_stepping()`, and
      `update_measurements()`, so callers that keep measurements across these
      calls (e.g. for a history) must `.clone()` them.
    """
    for i in range(self._n_substeps):
      self._substep(commands, is_last_substep=i == self._n_substeps - 1)

    self._commit_output_buffers(self._latest_measurements, env_mask=None)
    self._clock.tick_step()
    profile.frame_mark()
    return self._output_buffers

  @profile.trace(color=profile.Section.OBSERVATIONS)
  def update_measurements(self) -> None:
    """Reads all devices and processes their measurements, without hooks."""
    measurements: dict[str, torch.Tensor] = {}
    for dev in self._devices:
      if id(dev) in self._legacy_device_ids:
        measurements.update(dev.get_substep_measurements(self.sim))
      else:
        measurements.update(dev.get_measurements(self.sim))

    for processor in self._substep_measurements_processors:
      if id(processor) in self._legacy_processor_ids:
        processor.process_substep_measurements(self.sim, measurements)
      else:
        processor.process_measurements(self.sim, measurements)

    self._latest_measurements = measurements
    self._commit_output_buffers(measurements, env_mask=None)

  def get_measurements(self) -> Mapping[str, torch.Tensor]:
    """Returns the latest measurements of all worlds.

    The tensors are overwritten in place by every `step()`, `begin_stepping()`,
    and `update_measurements()`, so callers that keep them across these calls
    must `.clone()` them.
    """
    return self._output_buffers

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

  @profile.trace(color=profile.Section.PHYSICS)
  def _substep(
      self,
      commands: Optional[Mapping[str, torch.Tensor]] = None,
      *,
      is_last_substep: bool = False,
  ) -> None:
    """Runs one physics substep and the per-substep hooks and processing."""
    self._propagate_commands(commands)
    self.sim.step()
    self._substep_measurements(is_last_substep=is_last_substep)

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
  def _substep_measurements(self, *, is_last_substep: bool = False) -> None:
    """Advances per-substep state and reads and processes measurements."""
    measurements = self._latest_measurements
    measurements.clear()
    for dev in self._devices:
      if id(dev) in self._legacy_device_ids:
        measurements.update(dev.get_substep_measurements(self.sim))
      else:
        dev.on_substep(self.sim)
        if is_last_substep:
          dev.on_step(self.sim)
        measurements.update(dev.get_measurements(self.sim))

    for processor in self._substep_measurements_processors:
      if id(processor) in self._legacy_processor_ids:
        processor.process_substep_measurements(self.sim, measurements)
      else:
        processor.on_substep(self.sim, measurements)
        processor.process_measurements(self.sim, measurements)

  def _commit_output_buffers(
      self,
      measurements: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Copies `measurements` into `_output_buffers`, masked by `env_mask`."""
    if not self._output_buffers.keys() <= measurements.keys():
      missing = sorted(self._output_buffers.keys() - measurements.keys())
      raise KeyError(f"Missing measurement keys: {missing}")
    for key, val in measurements.items():
      buf = self._output_buffers.get(key)
      if buf is None:
        # Runs once per key, when its buffer is allocated.
        if key not in self._expected_ndim:
          raise KeyError(
              f"Measurement {key!r} is not in the spec of any device or"
              " processor."
          )
        self._output_buffers[key] = val.clone(
            memory_format=torch.contiguous_format
        )
        continue
      if env_mask is None or val.ndim < self._expected_ndim[key]:
        buf.copy_(val)
      else:
        mask = env_mask.to(device=val.device).view(
            self._nworld, *([1] * (val.ndim - 1))
        )
        torch.where(mask, val, buf, out=buf)


def _is_legacy(
    obj: object,
    base: type[object],
    deprecated_method: str,
    methods: Sequence[str],
) -> bool:
  """Whether `obj` overrides `deprecated_method` or none of `methods`.

  Args:
    obj: The device or processor.
    base: Its base class.
    deprecated_method: The deprecated method that `methods` replace.
    methods: The methods that replace `deprecated_method`.

  Returns:
    Whether `obj` is polled through `deprecated_method`.

  Raises:
    TypeError: If `obj` overrides both `deprecated_method` and some of
      `methods`, since `DeviceLayer` would silently skip the latter.
  """
  overridden = [method for method in methods if _overrides(obj, base, method)]
  if not _overrides(obj, base, deprecated_method):
    return not overridden

  if overridden:
    names = ", ".join(f"`{method}()`" for method in overridden)
    raise TypeError(
        f"{type(obj).__name__} overrides both the deprecated"
        f" `{deprecated_method}()` and {names}, which would not be called."
        " Implement only the new methods; if a base class overrides"
        f" `{deprecated_method}()`, keep overriding it instead until the base"
        " class is migrated."
    )
  return True


def _overrides(obj: object, base: type[object], name: str) -> bool:
  """Whether `type(obj)` overrides `base.name` (instance attributes don't)."""
  base_method = getattr(base, name)
  return getattr(type(obj), name, base_method) is not base_method
