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

"""Device-layer wrappers, the counterparts of the environment wrappers."""

from typing import Any
from typing import Mapping
from typing import Protocol

from dm_env import specs
import numpy as np
from reaf.simulation.torch import clock
from reaf.simulation.torch import protocols
import torch


class DeviceLayer(Protocol):
  """Protocol for a generic (hardware or sim) DeviceLayer.

  TODO(b/541412703): move this to the core REAF library
  """

  def measurements_spec(self) -> Mapping[str, Any]:
    ...

  def commands_spec(self) -> Mapping[str, Any]:
    ...

  def begin_stepping(self) -> Mapping[str, Any]:
    ...

  def step(self, commands: Mapping[str, Any]) -> Mapping[str, Any]:
    ...

  def end_stepping(self) -> None:
    ...


class DeviceLayerWrapper(DeviceLayer):
  """Base class for device-layer wrappers."""

  def __init__(self, device_layer: DeviceLayer | protocols.TorchDeviceLayer):
    self._device_layer = device_layer

  def __getattr__(self, name: str) -> Any:
    if name.startswith("__"):
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    return getattr(self._device_layer, name)

  @property
  def unwrapped(self) -> Any:
    """Returns the base device layer (recursively unwrapping wrappers)."""
    if hasattr(self._device_layer, "unwrapped"):
      return self._device_layer.unwrapped
    return self._device_layer

  def measurements_spec(self) -> Any:
    return self._device_layer.measurements_spec()

  def commands_spec(self) -> Any:
    return self._device_layer.commands_spec()

  def begin_stepping(self) -> Any:
    return self._device_layer.begin_stepping()

  def step(self, commands: Any) -> Any:
    return self._device_layer.step(commands)

  def end_stepping(self) -> None:
    self._device_layer.end_stepping()


class SingleDeviceLayerWrapper(DeviceLayerWrapper):
  """Presents a batched device layer (nworld=1) as unbatched."""

  def __init__(self, device_layer: protocols.TorchDeviceLayer):
    if device_layer.nworld != 1:
      raise ValueError(f"Expected nworld=1, got {device_layer.nworld}.")
    super().__init__(device_layer)
    self._device_layer: protocols.TorchDeviceLayer = device_layer

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return {
        key: _squeeze_spec(spec)
        for key, spec in self._device_layer.measurements_spec().items()
    }

  def commands_spec(self) -> Mapping[str, specs.Array]:
    return {
        key: _squeeze_spec(spec)
        for key, spec in self._device_layer.commands_spec().items()
    }

  def begin_stepping(self) -> Mapping[str, torch.Tensor]:
    return _squeeze(self._device_layer.begin_stepping())

  def step(self, commands: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    batched = {
        key: torch.as_tensor(value).reshape(1)
        for key, value in commands.items()
    }
    return _squeeze(self._device_layer.step(batched))


class NumpyDeviceLayerWrapper(DeviceLayerWrapper):
  """Presents a torch device layer as numpy, conforming to its specs.

  This lifts a simulated device layer up to the interface of a real
  `Environment` so a hardware deployment can be run on a simulated backend.
  """

  def begin_stepping(self) -> Mapping[str, np.ndarray]:
    return self._conform(_to_numpy(self._device_layer.begin_stepping()))

  def step(self, commands: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    as_torch = {key: torch.as_tensor(value) for key, value in commands.items()}
    return self._conform(_to_numpy(self._device_layer.step(as_torch)))

  def _conform(
      self, measurements: Mapping[str, np.ndarray]
  ) -> dict[str, np.ndarray]:
    """Casts measurements to the dtype their spec advertises."""
    measurements_spec = self.measurements_spec()
    return {
        key: (
            # `copy=False` makes this a no-op when the dtypes already agree,
            # which is the common case on the warp backend.
            value.astype(measurements_spec[key].dtype, copy=False)
            if key in measurements_spec
            else value
        )
        for key, value in measurements.items()
    }


def _squeeze(measurements: Mapping[str, torch.Tensor]) -> dict[str, Any]:
  return {key: value[0] for key, value in measurements.items()}


def _squeeze_spec(spec: specs.Array) -> specs.Array:
  """Drops the leading nworld dimension from a spec."""
  if isinstance(spec, specs.BoundedArray):
    return specs.BoundedArray(
        shape=spec.shape[1:],
        dtype=spec.dtype,
        minimum=spec.minimum,
        maximum=spec.maximum,
    )
  return specs.Array(shape=spec.shape[1:], dtype=spec.dtype)


def _batch_spec(spec: specs.Array) -> specs.Array:
  """Prepends an nworld=1 dimension to a spec."""
  return spec.replace(shape=(1,) + spec.shape)


def _to_numpy(measurements: Mapping[str, torch.Tensor]) -> dict[str, Any]:
  return {key: value.cpu().numpy() for key, value in measurements.items()}


class TorchDeviceLayerWrapper(DeviceLayerWrapper, protocols.TorchDeviceLayer):
  """Presents an unbatched numpy device layer as a batched torch one.

  This lifts a real device layer up to the interface for a simulated
  `Environment`, so a policy trained in simulation runs on real hardware.
  """

  def __init__(
      self,
      device_layer: DeviceLayer,
      *,
      control_timestep: float,
      sim: protocols.Simulator | None = None,
      xpu_device: str = "cpu",
  ):
    super().__init__(device_layer)
    self._device_layer: Any = device_layer
    self._sim = sim
    self._control_timestep = control_timestep
    self._xpu_device = xpu_device
    self._last_measurements: Mapping[str, torch.Tensor] = {}

  @property
  def sim(self) -> protocols.Simulator | None:
    return self._sim

  @property
  def nworld(self) -> int:
    return 1

  @property
  def xpu_device(self) -> str:
    return self._xpu_device

  @property
  def control_timestep(self) -> float:
    return self._control_timestep

  @property
  def clock(self) -> clock.Clock | None:
    return self._sim.clock if self._sim is not None else None

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return {
        key: _batch_spec(spec)
        for key, spec in self._device_layer.measurements_spec().items()
    }

  def commands_spec(self) -> Mapping[str, Any]:
    return {
        key: _batch_spec(spec)
        for key, spec in self._device_layer.commands_spec().items()
    }

  def begin_stepping(
      self, env_mask: torch.Tensor | None = None
  ) -> Mapping[str, torch.Tensor]:
    del env_mask  # There is only one world.
    self._last_measurements = self._to_torch(
        self._device_layer.begin_stepping()
    )
    return self._last_measurements

  def step(
      self, commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    unbatched = {key: value[0].cpu().numpy() for key, value in commands.items()}
    self._last_measurements = self._to_torch(self._device_layer.step(unbatched))
    return self._last_measurements

  def end_stepping(self, env_mask: torch.Tensor | None = None) -> None:
    del env_mask  # There is only one world.
    self._device_layer.end_stepping()

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """No-op: the hardware is homed by the environment's reset hooks."""
    del env_mask

  def forward(self) -> None:
    """No-op: hardware needs no equivalent of `mj_forward`."""

  def has_device(self, name: str) -> bool:
    """Returns True if a device with the specified name exists."""
    for device in self._device_layer.device_coordinator.get_devices():
      if device.name == name:
        return True
    return False

  def get_device(self, name: str) -> Any:
    """The wrapped device called `name`, as the torch device layer does it."""
    for device in self._device_layer.device_coordinator.get_devices():
      if device.name == name:
        return device
    raise ValueError(f"Device with name '{name}' not found.")

  def get_measurements(self) -> Mapping[str, torch.Tensor]:
    """Returns the latest measurements from the real device layer."""
    return self._last_measurements

  def _to_torch(
      self, measurements: Mapping[str, np.ndarray]
  ) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=self._xpu_device).unsqueeze(0)
        for key, value in measurements.items()
    }
