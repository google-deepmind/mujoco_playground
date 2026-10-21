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
"""Protocols for reaf-sim simulation."""

from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from reaf.simulation.torch._src.core import clock as clock_module
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch


@runtime_checkable
class Simulator(Protocol):
  """Protocol defining the reaf-sim simulation interface."""

  @property
  def clock(self) -> clock_module.Clock:
    """Simulation clock tracking step and substep counts."""
    ...

  def set_clock(self, clock: clock_module.Clock) -> None:
    """Sets the simulation clock."""
    ...

  @property
  def mj_model(self) -> mujoco.MjModel:
    """Underlying compiled CPU MuJoCo model."""
    ...

  @property
  def mj_data(self) -> mujoco.MjData:
    """Underlying CPU MuJoCo data buffer."""
    ...

  @property
  def model(self) -> types.Model:
    """Torch-wrapped model attributes across batch dimensions."""
    ...

  @property
  def data(self) -> types.Data:
    """Torch-wrapped dynamic state tensors across batch dimensions."""
    ...

  @property
  def nworld(self) -> int:
    """Number of parallel worlds / batch dimension."""
    ...

  @property
  def xpu_device(self) -> str:
    """Device hosting tensors (e.g. 'cuda:0', 'cpu')."""
    ...

  def forward(self) -> None:
    """Computes forward kinematics and dynamics."""
    ...

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets simulator state for specified environments."""
    ...

  def step(self) -> None:
    """Executes a single low-level physics step."""
    ...

  def step_n(self, n_substeps: int) -> None:
    """Executes `n_substeps` low-level physics steps."""
    ...

  def set_const(self) -> torch_wrapper.SetConstLevel:
    """Recomputes derived model constants for modified model fields."""
    ...


@runtime_checkable
class TorchDeviceLayer(Protocol):
  """Protocol defining the reaf-sim simulation DeviceLayer interface."""

  def forward(self) -> None:
    """Advances/recomputes simulation kinematic state to match current positions."""
    ...

  @property
  def nworld(self) -> int:
    """Number of parallel worlds / batch dimension."""
    ...

  @property
  def xpu_device(self) -> str:
    """Device hosting tensors (e.g. 'cuda:0', 'cpu')."""
    ...

  @property
  def sim(self) -> Simulator | None:
    """Underlying simulator instance, or `None` for hardware device layers."""
    ...

  @property
  def control_timestep(self) -> float:
    """Control timestep in seconds."""
    ...

  @property
  def clock(self) -> clock_module.Clock | None:
    """Simulation clock, or `None` for hardware device layers."""
    ...

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    """Returns the batched measurements specification."""
    ...

  def commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    """Returns the batched commands specification."""
    ...

  def begin_stepping(
      self, env_mask: torch.Tensor | None = None
  ) -> Mapping[str, torch.Tensor]:
    """Prepares and returns initial measurements before stepping."""
    ...

  def step(
      self, commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Steps the device layer with commands and returns measurements."""
    ...

  def end_stepping(self, env_mask: torch.Tensor | None = None) -> None:
    """Completes the stepping phase."""
    ...

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets simulator and device states."""
    ...

  def get_measurements(self) -> Mapping[str, torch.Tensor]:
    """Returns the latest measurements dictionary."""
    ...

  def get_device(self, name: str) -> Any:
    """Returns the device with the specified name."""
    ...

  def has_device(self, name: str) -> bool:
    """Returns True if a device with the specified name exists."""
    ...
