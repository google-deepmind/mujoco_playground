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
"""Discrete simulation clock for reaf-sim."""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class Clock:
  """Discrete physics and environment clock.

  Attributes:
    sim_dt: Physics simulation timestep (seconds).
    control_dt: Environment control timestep (seconds).
    n_substeps: Number of physics substeps per control step.
    nworld: Number of parallel environments / worlds.
    xpu_device: Device on which episode counters are allocated.
    global_step: Monotonically increasing count of completed control steps.
    global_substep: Monotonically increasing count of physics substeps since
      simulation start.
    episode_step: Tensor of shape (nworld,) containing control steps since last
      reset for each environment.
    episode_substep: Tensor of shape (nworld,) containing physics substeps since
      last reset for each environment.
  """

  sim_dt: float
  control_dt: float
  n_substeps: int
  nworld: int
  xpu_device: str | torch.device = "cpu"

  # Monotonic global counters (never reset during simulation lifetime)
  global_step: int = 0
  global_substep: int = 0

  # Per-environment episode counters (reset on env_mask)
  episode_step: torch.Tensor = dataclasses.field(init=False)
  episode_substep: torch.Tensor = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    self.episode_step = torch.zeros(
        self.nworld, dtype=torch.int64, device=self.xpu_device
    )
    self.episode_substep = torch.zeros(
        self.nworld, dtype=torch.int64, device=self.xpu_device
    )

  @property
  def global_time(self) -> float:
    """Current continuous simulation time in seconds."""
    return self.global_substep * self.sim_dt

  @property
  def episode_time(self) -> torch.Tensor:
    """Current continuous episode time in seconds for each environment."""
    return self.episode_substep * self.sim_dt

  def tick_substep(self) -> None:
    """Advances the physics substep counters by 1."""
    self.global_substep += 1
    self.episode_substep += 1

  def tick_step(self) -> None:
    """Advances the control step counters by 1."""
    self.global_step += 1
    self.episode_step += 1

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets episode step and substep counters for specified environments.

    Args:
      env_mask: Boolean tensor of shape (nworld,) indicating which environments
        to reset. If None, resets all environments.
    """
    if env_mask is None:
      self.episode_step.zero_()
      self.episode_substep.zero_()
    else:
      self.episode_step[env_mask] = 0
      self.episode_substep[env_mask] = 0
