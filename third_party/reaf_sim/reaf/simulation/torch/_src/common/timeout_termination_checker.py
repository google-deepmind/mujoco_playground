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

"""Truncates when the episode time exceeds the specified timeout."""

from collections.abc import Mapping

from reaf.simulation.torch._src.core import termination_checker
import torch


class TimeoutTerminationChecker(termination_checker.TerminationChecker):
  """Truncates when the episode time exceeds the specified timeout."""

  def __init__(
      self,
      timeout: float,
      time_feature_name: str,
      xpu_device: str = 'cuda:0',
  ):
    self._timeout = torch.tensor(
        timeout, dtype=torch.float32, device=xpu_device
    )
    self._time_feature_name = time_feature_name

  def name(self) -> str:
    return 'timeout_termination'

  @property
  def timeout(self) -> float:
    """Returns the episode timeout in seconds."""
    return self._timeout.item()

  def check_termination(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    return torch.where(
        required_features[self._time_feature_name] > self._timeout,
        termination_checker.TerminationResult.TRUNCATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def required_features_keys(self) -> set[str]:
    return {self._time_feature_name}
