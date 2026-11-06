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
"""Computes the reward."""

from collections.abc import Mapping
from typing import Any

from reaf.core import reward_provider as core_reward_provider
import torch

RewardValue = torch.Tensor | Mapping[str, Any]
RewardSpec = core_reward_provider.RewardSpec

# TODO(reaf): this file needs to be tested.


class _RewardProvider(core_reward_provider._RewardProvider):  # pylint: disable=protected-access
  """Computes the reward."""

  def compute_reward(  # pyrefly: ignore[bad-override]
      self,
      required_features: Mapping[str, torch.Tensor],
  ) -> RewardValue:
    """Computes the reward."""
    ...

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    """Resets the internal state of the reward provider."""
    ...


class RewardProvider(_RewardProvider, core_reward_provider.RewardProvider):
  """Computes the reward."""
