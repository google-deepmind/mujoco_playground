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
"""Computes episode metrics."""

import abc
from collections.abc import Mapping

import torch


class MetricProvider(abc.ABC):
  """Computes episode metrics from per-step task features."""

  @abc.abstractmethod
  def update_metrics(
      self,
      features: Mapping[str, torch.Tensor],
  ) -> None:
    """Evaluates step metrics and updates internal episode accumulators.

    Modifies internal state; should only be called once per control step by
    `TaskLayer.update_metrics` (via `Environment.step`).
    """

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Mutates state: clears episode accumulators for the masked worlds."""
    del env_mask

  @abc.abstractmethod
  def get_episode_metrics(
      self,
      env_mask: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Returns 0-D mean reduced episode metrics over masked worlds.

    Read-only; does not modify internal state.
    """
