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

"""Truncates when the episode time reaches the specified timeout."""

from collections.abc import Mapping
import math

import numpy as np
from reaf.simulation.torch._src.core import termination_checker
import torch


class TimeoutTerminationChecker(termination_checker.TerminationChecker):
  """Truncates when the episode time reaches the specified timeout."""

  def __init__(
      self,
      timeout: float,
      time_feature_name: str,
      xpu_device: str = "cuda:0",
  ):
    self._timeout_secs = float(timeout)
    self._timeout = torch.tensor(
        timeout, dtype=torch.float32, device=xpu_device
    )
    self._time_feature_name = time_feature_name

  def name(self) -> str:
    return "timeout_termination"

  @property
  def timeout(self) -> float:
    """Returns the episode timeout in seconds."""
    return self._timeout_secs

  def max_steps(self, control_timestep: float) -> int:
    """Returns the first control step at which `check_termination` truncates.

    Pass `env.device_layer.control_timestep` so the result matches the simulated
    float32 timestep.
    """
    if not math.isfinite(control_timestep) or control_timestep <= 0:
      raise ValueError(
          f"control_timestep must be finite and > 0, got {control_timestep}."
      )
    with np.errstate(over="ignore"):
      timeout_f32 = np.float32(self._timeout_secs)
    if not np.isfinite(timeout_f32) or timeout_f32 <= 0:
      raise ValueError(
          "timeout must be finite in float32 and > 0, got"
          f" {self._timeout_secs}."
      )
    # Mirror the float32 `time >= timeout` comparison so the result matches the
    # truncation step even when `timeout` is not a multiple of `control_timestep`.
    steps = max(1, math.ceil(self._timeout_secs / control_timestep) - 1)
    while np.float32(steps * control_timestep) < timeout_f32:
      steps += 1
    return steps

  def check_termination(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    return torch.where(
        required_features[self._time_feature_name] >= self._timeout,
        termination_checker.TerminationResult.TRUNCATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def required_features_keys(self) -> set[str]:
    return {self._time_feature_name}
